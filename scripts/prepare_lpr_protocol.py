"""Create immutable data and initialization artifacts for paired LPR runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lpr_protocol import (
    CATEGORY_NAMES,
    EXPECTED_COMMON_FINGERPRINTS,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SOURCE_SHA256,
    EXPECTED_SUBSET_SHA256,
    build_initial_state,
    category_mapping_sha256,
    current_environment,
    dataset_signature,
    environment_violations,
    file_sha256,
    select_hashed_subset,
    source_violations,
    subset_signature,
)
from src.rtdetr_lpr import LPRRTDETRDetectionModel
from ultralytics.nn.tasks import RTDETRDetectionModel


def _write_locked(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to replace changed protocol artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _yaml_text(payload: dict) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)


def prepare_data_files(dataset_root: Path, output_dir: Path, *, fraction: float = 0.10) -> dict:
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    train_images = sorted((dataset_root / "images" / "train").glob("*.jpg"))
    selected = select_hashed_subset(train_images, root=dataset_root, fraction=fraction)
    subset = {
        "count": len(selected),
        "fraction": fraction,
        "sha256": subset_signature(selected, root=dataset_root),
    }
    subset_path = output_dir / "train-10pct.txt"
    _write_locked(subset_path, "".join(f"{path.resolve()}\n" for path in selected))
    names = {index: name for index, name in enumerate(CATEGORY_NAMES)}
    screen_data_path = output_dir / "VisDrone-screen-10pct.yaml"
    formal_data_path = output_dir / "VisDrone-formal-full.yaml"
    _write_locked(
        screen_data_path,
        _yaml_text(
            {
                "path": str(dataset_root),
                "train": str(subset_path),
                "val": str(dataset_root / "images" / "val"),
                "names": names,
            }
        ),
    )
    _write_locked(
        formal_data_path,
        _yaml_text(
            {
                "path": str(dataset_root),
                "train": str(dataset_root / "images" / "train"),
                "val": str(dataset_root / "images" / "val"),
                "names": names,
            }
        ),
    )
    return {
        "subset": subset,
        "subset_path": subset_path,
        "screen_data_path": screen_data_path,
        "formal_data_path": formal_data_path,
    }


def validate_data_authority(dataset: dict, subset: dict) -> None:
    expected_dataset = {"file_count": 14038, "sha256": EXPECTED_DATASET_SHA256}
    if dataset != expected_dataset:
        raise ValueError(f"dataset does not match frozen authority: expected={expected_dataset}, actual={dataset}")
    expected_subset = {"count": 647, "sha256": EXPECTED_SUBSET_SHA256}
    actual_subset = {"count": subset.get("count"), "sha256": subset.get("sha256")}
    if actual_subset != expected_subset:
        raise ValueError(f"subset does not match frozen authority: expected={expected_subset}, actual={actual_subset}")


def create_initial_state_artifact(*, seed: int, nc: int = 10, channels: int = 3) -> dict:
    torch.manual_seed(seed)
    control = RTDETRDetectionModel("rtdetr-l.yaml", nc=nc, ch=channels, verbose=False)
    torch.manual_seed(seed + 10_000)
    method = LPRRTDETRDetectionModel(
        "rtdetr-l.yaml",
        nc=nc,
        ch=channels,
        verbose=False,
        lpr_seed=seed + 10_000,
    )
    return build_initial_state(
        control.state_dict(),
        method.state_dict(),
        metadata={
            "seed": seed,
            "innovation_seed": seed + 10_000,
            "control_parameters": sum(parameter.numel() for parameter in control.parameters()),
            "lpr_parameters": sum(parameter.numel() for parameter in method.parameters()),
        },
    )


def _save_locked_state(path: Path, artifact: dict) -> None:
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=False)
        if existing.get("fingerprints") != artifact.get("fingerprints") or existing.get("metadata") != artifact.get(
            "metadata"
        ):
            raise FileExistsError(f"refusing to replace changed initial-state artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(artifact, temporary)
    os.replace(temporary, path)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _json_sha256(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256(data).hexdigest().upper()


def prepare_protocol(dataset_root: Path, output_dir: Path, *, seed: int) -> dict:
    environment = current_environment()
    violations = environment_violations(environment)
    if violations:
        raise ValueError(f"environment does not match frozen authority: {violations}")
    source_drift = source_violations()
    if source_drift:
        raise ValueError(f"Ultralytics source does not match frozen authority: {source_drift}")
    dataset = dataset_signature(dataset_root)
    data_files = prepare_data_files(dataset_root, output_dir)
    validate_data_authority(dataset, data_files["subset"])

    state_path = Path(output_dir).resolve() / f"initial-state-seed{seed}.pt"
    artifact = create_initial_state_artifact(seed=seed)
    expected_common = EXPECTED_COMMON_FINGERPRINTS[seed]
    if artifact["fingerprints"]["common"] != expected_common:
        raise ValueError(
            f"common initial state is not the Linux authority for seed {seed}: "
            f"expected={expected_common}, actual={artifact['fingerprints']['common']}"
        )
    artifact["metadata"].update(
        {
            "dataset": dataset,
            "subset": data_files["subset"],
            "environment": environment,
            "category_mapping_sha256": category_mapping_sha256(CATEGORY_NAMES),
        }
    )
    _save_locked_state(state_path, artifact)
    manifest = {
        "format_version": 1,
        "seed": seed,
        "git_commit": _git_commit(),
        "dataset_root": str(Path(dataset_root).resolve()),
        "dataset": dataset,
        "subset": {
            **data_files["subset"],
            "path": str(data_files["subset_path"]),
            "file_sha256": file_sha256(data_files["subset_path"]),
        },
        "category_mapping_sha256": category_mapping_sha256(CATEGORY_NAMES),
        "environment": environment,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "data": {
            "screen": {
                "path": str(data_files["screen_data_path"]),
                "sha256": file_sha256(data_files["screen_data_path"]),
            },
            "formal": {
                "path": str(data_files["formal_data_path"]),
                "sha256": file_sha256(data_files["formal_data_path"]),
            },
        },
        "initial_state": {
            "path": str(state_path),
            "sha256": file_sha256(state_path),
            "fingerprints": artifact["fingerprints"],
        },
    }
    manifest["signature"] = _json_sha256(manifest)
    manifest_path = Path(output_dir).resolve() / f"protocol-seed{seed}.json"
    _write_locked(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare immutable strict LPR paired-protocol artifacts.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(prepare_protocol(args.dataset_root, args.output_dir, seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
