from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import torch
import ultralytics
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ebc_qp_config import EBCQPConfig, SOURCE_SHA256
from src.ebc_qp_protocol import (
    E1_CONTROLLED_AMP_GROWTH_INTERVAL,
    E1_CONTROLLED_AMP_SCALE,
    E1_EXPECTED_OPTIMIZER_ATTEMPTS,
    build_initial_state,
    dataset_signature,
    write_d2_subset,
)
from src.rtdetr_ebc_qp import EBCQPDetectionModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create immutable artifacts for an EBC-QP paired run.")
    parser.add_argument("--data", default="VisDrone.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, choices=(0, 1, 2))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_protocol(args.data, output_dir=args.output_dir, seed=args.seed)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def build_e1_training_contract() -> dict:
    return {
        "epochs": 10,
        "fraction": 1.0,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "device": "0",
        "amp": True,
        "controlled_amp_scale": E1_CONTROLLED_AMP_SCALE,
        "controlled_amp_growth_interval": E1_CONTROLLED_AMP_GROWTH_INTERVAL,
        "expected_optimizer_attempts": E1_EXPECTED_OPTIMIZER_ATTEMPTS,
        "save_period": -1,
        "retained_zero_based_epoch_checkpoints": [7, 8, 9],
        "deterministic": True,
        "nbs": 64,
        "nms": False,
        "max_det": 300,
        "optimizer": "MuSGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
    }


def prepare_protocol(data_file: str | Path, *, output_dir: Path, seed: int) -> dict:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = check_det_dataset(str(data_file))
    dataset_root = Path(data["path"]).resolve()
    train_images = _collect_images(data["train"])

    subset_file = output_dir / "d2-train-10pct.txt"
    subset = write_d2_subset(train_images, root=dataset_root, output=subset_file, fraction=0.10)
    d2_data_file = output_dir / "VisDrone-d2-10pct.yaml"
    d2_data = {
        "path": str(dataset_root),
        "train": str(subset_file),
        "val": data["val"],
        "names": data["names"],
    }
    _write_locked_yaml(d2_data_file, d2_data)

    dataset = dataset_signature(dataset_root)
    mapping_sha = _json_sha256(data["names"])
    environment = _environment_record()
    git_commit = _git_commit()
    experiment_payload = {
        "dataset": dataset,
        "category_mapping_sha256": mapping_sha,
        "subset": subset,
        "data_sha256": _file_sha256(d2_data_file),
        "source_sha256": SOURCE_SHA256,
        "git_commit": git_commit,
        "environment": environment,
        "e1_training": build_e1_training_contract(),
        "tsgr_config": EBCQPConfig(
            lambda_p2=0.1,
            lambda_ebc=0.0,
            lambda_quality=0.0,
            query_injection_enabled=False,
            p2_c2_grad_scale=0.1,
            contribution_separated_aux_gradients=True,
        ).as_dict(),
    }
    experiment_signature = _json_sha256(experiment_payload)
    frozen_experiment = {
        "format_version": 1,
        "experiment_signature": experiment_signature,
        "payload": experiment_payload,
    }
    _write_locked_text(
        output_dir / "e1-experiment-signature.json",
        json.dumps(frozen_experiment, indent=2, sort_keys=True) + "\n",
    )
    metadata = {
        "seed": seed,
        "dataset": dataset,
        "category_mapping_sha256": mapping_sha,
        "subset": subset,
        "source_sha256": SOURCE_SHA256,
        "git_commit": git_commit,
        "environment": environment,
        "experiment_signature": experiment_signature,
    }
    initial_state_file = output_dir / f"initial-state-seed{seed}.pt"
    artifact = _create_initial_state(seed=seed, nc=data["nc"], channels=data["channels"], metadata=metadata)
    _save_locked_initial_state(initial_state_file, artifact)

    manifest = {
        "format_version": 1,
        "seed": seed,
        "experiment_signature": experiment_signature,
        "dataset": dataset,
        "category_mapping_sha256": mapping_sha,
        "subset": {**subset, "path": str(subset_file)},
        "data": {"path": str(d2_data_file), "sha256": _file_sha256(d2_data_file)},
        "initial_state": {"path": str(initial_state_file), "sha256": _file_sha256(initial_state_file)},
        "source_sha256": SOURCE_SHA256,
        "git_commit": metadata["git_commit"],
        "environment": metadata["environment"],
    }
    manifest["signature"] = _json_sha256(manifest)
    manifest_file = output_dir / f"protocol-seed{seed}.json"
    _write_locked_text(manifest_file, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _create_initial_state(*, seed: int, nc: int, channels: int, metadata: dict) -> dict:
    torch.manual_seed(seed)
    control = RTDETRDetectionModel("rtdetr-l.yaml", nc=nc, ch=channels, verbose=False)
    torch.manual_seed(seed + 10_000)
    method = EBCQPDetectionModel(
        ROOT / "configs" / "rtdetr-l-ebc-qp.yaml",
        nc=nc,
        ch=channels,
        verbose=False,
    )
    metadata = {
        **metadata,
        "control_parameters": sum(parameter.numel() for parameter in control.parameters()),
        "method_parameters": sum(parameter.numel() for parameter in method.parameters()),
        "innovation_seed": seed + 10_000,
    }
    return build_initial_state(control.state_dict(), method.state_dict(), metadata=metadata)


def _collect_images(train: str | list[str]) -> list[Path]:
    roots = train if isinstance(train, list) else [train]
    images = []
    for root in roots:
        path = Path(root)
        if not path.is_dir():
            raise ValueError(f"protocol preparation requires a training image directory: {path}")
        images.extend(
            candidate for candidate in path.rglob("*") if candidate.is_file() and candidate.suffix[1:].lower() in IMG_FORMATS
        )
    if not images:
        raise ValueError("training image directory is empty")
    return images


def _write_locked_yaml(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    YAML.save(temporary, dict(payload))
    content = temporary.read_text(encoding="utf-8")
    temporary.unlink()
    _write_locked_text(path, content)


def _save_locked_initial_state(path: Path, artifact: dict) -> None:
    if path.exists():
        current = torch.load(path, map_location="cpu", weights_only=False)
        if current.get("fingerprints") != artifact.get("fingerprints") or current.get("metadata") != artifact.get("metadata"):
            raise FileExistsError(f"refusing to replace changed initial state: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(path)


def _write_locked_text(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"refusing to replace changed protocol artifact: {path}")
    path.write_text(content, encoding="utf-8")


def _environment_record() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _json_sha256(payload: object) -> str:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256(content).hexdigest().upper()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    main()
