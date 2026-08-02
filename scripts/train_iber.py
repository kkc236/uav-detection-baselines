"""Train the fixed seed0 30-epoch IBER-BE B3 screen with a frozen detector."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.iber_protocol import (  # noqa: E402
    DESIGN_VERSION,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    PRIVATE_OPTIMIZER,
    PRIVATE_SEED,
    PROTOCOL_SHA256,
    RUNTIME_AMENDMENT_SHA256,
    SCREEN_CONTRACT,
    SCREEN_EPOCHS,
    SCREEN_TRAIN_COUNT,
    SCREEN_VAL_COUNT,
    execution_environment,
    file_sha256,
    module_state_sha256,
    validate_screen_contract,
)
from src.itber_metrics import correction_rms  # noqa: E402
from src.lpr_protocol import (  # noqa: E402
    CATEGORY_NAMES,
    category_mapping_sha256,
    dataset_signature,
    select_hashed_subset,
    subset_signature,
)
from src.rtdetr_iber import FrozenIBERAdapter  # noqa: E402


TRAINING_CONSTANTS = {
    "stage": "screen",
    "probe": "b3",
    "seed": 0,
    "private_seed": PRIVATE_SEED,
    "epochs": SCREEN_EPOCHS,
    "train_images": SCREEN_TRAIN_COUNT,
    "val_images": SCREEN_VAL_COUNT,
    "imgsz": 640,
    "batch": 8,
    "workers": 8,
    "amp": True,
    "amp_scale": 128.0,
    "save_period": 1,
    "optimizer": "AdamW",
    "lr": float(PRIVATE_OPTIMIZER["lr"]),
    "weight_decay": float(PRIVATE_OPTIMIZER["weight_decay"]),
    "betas": tuple(PRIVATE_OPTIMIZER["betas"]),
    "clip_grad_norm": float(PRIVATE_OPTIMIZER["clip"]),
    "on_the_fly_evidence": True,
    "max_det": 300,
    "nms": False,
}
AUGMENTATION = {
    "mosaic": 1.0,
    "close_mosaic": 10,
    "mixup": 0.0,
    "scale": 0.5,
    "translate": 0.1,
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "cutmix": 0.0,
    "copy_paste": 0.0,
}
REQUIRED_CHECKPOINT_KEYS = {
    "format_version",
    "design_version",
    "stage",
    "probe",
    "seed",
    "epoch",
    "baseline_sha256",
    "dataset_sha256",
    "subset_sha256",
    "source_commit",
    "runtime_amendment_sha256",
    "protocol_sha256",
    "refiner",
    "optimizer",
    "scaler",
    "rng",
}
EXPECTED_CATEGORY_SHA256 = (
    "1455A1F5D9FA9988799815B36CF2CE6D5044B5BD7CAD7CC614D6B5E5059EF2A6"
)


def build_private_optimizer(module: torch.nn.Module) -> torch.optim.AdamW:
    """Build the only optimizer allowed for the private B3 head."""
    parameters = [value for value in module.parameters() if value.requires_grad]
    if not parameters:
        raise ValueError("IBER-BE refiner has no trainable private parameters")
    return torch.optim.AdamW(
        parameters,
        lr=TRAINING_CONSTANTS["lr"],
        betas=TRAINING_CONSTANTS["betas"],
        weight_decay=TRAINING_CONSTANTS["weight_decay"],
    )


def highest_contiguous_verified_epoch(
    rows: Sequence[Mapping[str, Any]],
) -> int:
    """Return the verified ledger tip, rejecting gaps and unverified rows."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("publication ledger must be a sequence")
    for expected_epoch, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError("publication ledger rows must be mappings")
        if row.get("completed_epoch") != expected_epoch:
            raise ValueError("publication ledger epochs must be contiguous")
        if row.get("verified") is not True:
            raise ValueError("publication ledger row is not verified")
    if len(rows) > SCREEN_EPOCHS:
        raise ValueError("publication ledger exceeds the fixed 30 epochs")
    return len(rows)


def validate_resume_checkpoint(
    artifact: Mapping[str, Any],
    *,
    source_commit: str,
    highest_verified_epoch: int,
) -> None:
    """Reject any resume that is not the exact remotely verified ledger tip."""
    if not isinstance(artifact, Mapping):
        raise ValueError("invalid IBER-BE resume artifact")
    missing = REQUIRED_CHECKPOINT_KEYS - set(artifact)
    violations: dict[str, Any] = {
        f"required.{name}": {"expected": "present", "actual": None}
        for name in sorted(missing)
    }
    expected = {
        "format_version": 1,
        "design_version": DESIGN_VERSION,
        "stage": TRAINING_CONSTANTS["stage"],
        "probe": TRAINING_CONSTANTS["probe"],
        "seed": TRAINING_CONSTANTS["seed"],
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "subset_sha256": EXPECTED_SUBSET_SHA256,
        "source_commit": source_commit.lower(),
        "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
    }
    for name, value in expected.items():
        if artifact.get(name) != value:
            violations[name] = {"expected": value, "actual": artifact.get(name)}
    epoch = artifact.get("epoch")
    if type(epoch) is not int or not 1 <= epoch <= SCREEN_EPOCHS:
        violations["epoch"] = {
            "expected": f"1..{SCREEN_EPOCHS}",
            "actual": epoch,
        }
    if type(highest_verified_epoch) is not int or highest_verified_epoch != epoch:
        violations["highest_verified_epoch"] = {
            "expected": epoch,
            "actual": highest_verified_epoch,
        }
    for name in ("refiner", "optimizer", "scaler", "rng"):
        if not isinstance(artifact.get(name), Mapping):
            violations[name] = {
                "expected": "mapping",
                "actual": type(artifact.get(name)).__name__,
            }
    if "detector" in artifact:
        violations["detector"] = {"expected": "absent", "actual": "present"}
    if violations:
        raise ValueError(
            "invalid IBER-BE resume " + ", ".join(sorted(violations))
        )


def atomic_save_private_checkpoint(
    path: str | Path, artifact: Mapping[str, Any]
) -> Path:
    """Durably replace one private-only checkpoint without partial exposure."""
    destination = Path(path)
    if "detector" in artifact:
        raise ValueError("IBER-BE checkpoints must not serialize the detector")
    missing = REQUIRED_CHECKPOINT_KEYS - set(artifact)
    if missing:
        raise ValueError(
            "IBER-BE checkpoint missing " + ", ".join(sorted(missing))
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(dict(artifact), temporary)
        with temporary.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--gate1-decision", type=Path, required=True)
    parser.add_argument("--publication-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args(argv)


def _source_commit() -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip().lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("IBER-BE source commit is invalid")
    return value


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _assert_detector_frozen(detector: torch.nn.Module) -> None:
    trainable = [name for name, value in detector.named_parameters() if value.requires_grad]
    training = [name for name, module in detector.named_modules() if module.training]
    if trainable or training:
        raise RuntimeError(
            "IBER-BE detector is not frozen: "
            f"trainable={trainable[:5]}, training={training[:5]}"
        )


def _json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"invalid JSONL row {line_number} in {path}")
        rows.append(value)
    return rows


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _atomic_write_bytes(path, encoded)


def _record_epoch_row(
    path: Path,
    payload: Mapping[str, Any],
    *,
    epoch: int,
    verified_tip: int,
) -> None:
    """Replace only the single unpublished tail row and keep verified rows immutable."""
    if epoch != verified_tip + 1:
        raise ValueError("IBER-BE evidence row is not the next unverified epoch")
    rows = _json_rows(path)
    if len(rows) < verified_tip:
        raise ValueError("IBER-BE local evidence is missing a verified row")
    rows = rows[:verified_tip]
    rows.append(dict(payload))
    encoded = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    ).encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _load_publication_ledger(output_root: Path) -> list[dict[str, Any]]:
    rows = _json_rows(output_root / "publication-ledger.jsonl")
    highest_contiguous_verified_epoch(rows)
    for expected_epoch, row in enumerate(rows, start=1):
        expected = {
            "design_version": DESIGN_VERSION,
            "stage": "screen",
            "probe": "b3",
            "seed": 0,
        }
        for name, value in expected.items():
            if row.get(name) != value:
                raise ValueError(
                    f"publication ledger epoch {expected_epoch} has invalid {name}"
                )
        checkpoint = row.get("checkpoint")
        if not isinstance(checkpoint, Mapping) or not isinstance(
            checkpoint.get("sha256"), str
        ):
            raise ValueError(
                f"publication ledger epoch {expected_epoch} lacks checkpoint authority"
            )
    return rows


def _validate_gate1_decision(path: Path, *, source_commit: str) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "design_version": DESIGN_VERSION,
        "stage": "gate1_decision",
        "status": "passed",
    }
    violations: list[str] = [
        name for name, value in expected.items() if payload.get(name) != value
    ]
    engineering = payload.get("engineering")
    conditions = payload.get("conditions")
    if not isinstance(engineering, Mapping) or not engineering or not all(
        value is True for value in engineering.values()
    ):
        violations.append("engineering")
    if not isinstance(conditions, Mapping) or not conditions or not all(
        value is True for value in conditions.values()
    ):
        violations.append("conditions")
    reports = payload.get("reports")
    if not isinstance(reports, Mapping) or set(reports) != {"b0", "b1", "b2", "b3"}:
        violations.append("reports")
    else:
        for arm, report in reports.items():
            authority = report.get("cache_authority", {})
            expected_authority = {
                "baseline_sha256": EXPECTED_BASELINE_SHA256,
                "dataset_sha256": EXPECTED_DATASET_SHA256,
                "subset_sha256": EXPECTED_SUBSET_SHA256,
                "source_commit": source_commit,
                "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
            }
            for name, value in expected_authority.items():
                if authority.get(name) != value:
                    violations.append(f"reports.{arm}.cache_authority.{name}")
    if violations:
        raise ValueError(
            "invalid IBER-BE Gate-1 decision " + ", ".join(sorted(set(violations)))
        )
    return file_sha256(path)


def _build_train_loader(dataset_root: Path, output_root: Path):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.dataset import RTDETRDataset

    image_paths = sorted((dataset_root / "images" / "train").glob("*.jpg"))
    selected = select_hashed_subset(image_paths, root=dataset_root, fraction=0.10)
    if len(image_paths) != 6471 or len(selected) != SCREEN_TRAIN_COUNT:
        raise ValueError(
            f"IBER-BE train count mismatch: full={len(image_paths)}, subset={len(selected)}"
        )
    if subset_signature(selected, root=dataset_root) != EXPECTED_SUBSET_SHA256:
        raise ValueError("IBER-BE fixed subset SHA256 mismatch")
    val_count = len(list((dataset_root / "images" / "val").glob("*.jpg")))
    if val_count != SCREEN_VAL_COUNT:
        raise ValueError(f"IBER-BE validation image count mismatch: {val_count}")

    subset_path = output_root / "fixed-train647.txt"
    subset_payload = (
        "\n".join(str(path.resolve()) for path in selected) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(subset_path, subset_payload)
    image_source = str(subset_path.resolve())
    overrides = {
        "task": "detect",
        "mode": "train",
        "imgsz": TRAINING_CONSTANTS["imgsz"],
        "batch": TRAINING_CONSTANTS["batch"],
        "workers": TRAINING_CONSTANTS["workers"],
        "cache": False,
        "rect": False,
        "single_cls": False,
        "classes": None,
        "fraction": 1.0,
        "seed": TRAINING_CONSTANTS["seed"],
        "deterministic": True,
        **AUGMENTATION,
    }
    cfg = get_cfg(overrides=overrides)
    data = {
        "path": str(dataset_root.resolve()),
        "train": image_source,
        "val": str((dataset_root / "images" / "val").resolve()),
        "names": {index: name for index, name in enumerate(CATEGORY_NAMES)},
        "nc": len(CATEGORY_NAMES),
        "channels": 3,
    }
    dataset = RTDETRDataset(
        img_path=image_source,
        imgsz=TRAINING_CONSTANTS["imgsz"],
        batch_size=TRAINING_CONSTANTS["batch"],
        augment=True,
        hyp=cfg,
        rect=False,
        cache=None,
        single_cls=False,
        prefix="iber-screen: ",
        classes=None,
        data=data,
        fraction=1.0,
    )
    loader = build_dataloader(
        dataset,
        batch=TRAINING_CONSTANTS["batch"],
        workers=TRAINING_CONSTANTS["workers"],
        shuffle=True,
        rank=-1,
        drop_last=False,
    )
    return loader, cfg


def _move_training_batch(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    moved = {
        name: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for name, value in batch.items()
    }
    moved["img"] = moved["img"].float().div_(255)
    return moved


def _correction_diagnostics(adapter: FrozenIBERAdapter) -> dict[str, float]:
    output = adapter.last_output
    matches = adapter.last_match_indices
    if output is None or matches is None:
        raise RuntimeError("missing IBER-BE output diagnostics")
    batch, queries = output.effective_correction.shape[:2]
    matched = torch.zeros(
        (batch, queries),
        device=output.effective_correction.device,
        dtype=torch.bool,
    )
    for image_index, (source, _target) in enumerate(matches):
        matched[image_index, source.to(device=matched.device, dtype=torch.long)] = True
    correction = output.effective_correction.float()
    values = {
        "matched_correction_rms": float(
            correction_rms(correction, matched).detach().cpu()
        ),
        "unmatched_correction_rms": float(
            correction_rms(correction, ~matched).detach().cpu()
        ),
        "gate_mean": float(output.gates.float().mean().detach().cpu()),
        "gate_p95": float(
            torch.quantile(output.gates.float(), 0.95).detach().cpu()
        ),
        "residual_mean": float(output.residuals.float().mean().detach().cpu()),
        "residual_rms": float(
            output.residuals.float().square().mean().sqrt().detach().cpu()
        ),
        "f3_embedding_rms": float(
            output.f3_boundary_features.float().square().mean().sqrt().detach().cpu()
        ),
        "rgb_embedding_rms": float(
            output.rgb_boundary_features.float().square().mean().sqrt().detach().cpu()
        ),
        "boundary_embedding_rms": float(
            output.boundary_features.float().square().mean().sqrt().detach().cpu()
        ),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("non-finite IBER-BE correction diagnostics")
    return values


def _restore_rng(artifact: Mapping[str, Any]) -> None:
    rng = artifact["rng"]
    random.setstate(rng["python"])
    numpy_state = rng["numpy"]
    if not isinstance(numpy_state, Mapping) or not torch.is_tensor(
        numpy_state.get("state")
    ):
        raise ValueError("IBER-BE checkpoint has an unsafe NumPy RNG state")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["state"].detach().cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])


def _rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _reseed_loader_for_epoch(loader: Any, epoch: int) -> None:
    if type(epoch) is not int or epoch < 1:
        raise ValueError("IBER-BE epoch must be positive")
    loader.close()
    loader.generator.manual_seed(6148914691236517204 + epoch)
    loader.reset()


def _optimizer_evidence(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict[str, Any]:
    groups = []
    for group in optimizer.param_groups:
        groups.append(
            {
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "betas": [float(value) for value in group["betas"]],
                "parameter_count": sum(value.numel() for value in group["params"]),
            }
        )
    steps = []
    exp_avg_square_sum = 0.0
    exp_avg_elements = 0
    for state in optimizer.state.values():
        if "step" in state:
            step = state["step"]
            steps.append(float(step.detach().cpu()) if torch.is_tensor(step) else float(step))
        if "exp_avg" in state:
            value = state["exp_avg"].detach().float()
            exp_avg_square_sum += float(value.square().sum().cpu())
            exp_avg_elements += value.numel()
    evidence = {
        "name": type(optimizer).__name__,
        "groups": groups,
        "state_parameter_count": len(optimizer.state),
        "step_min": min(steps) if steps else 0.0,
        "step_max": max(steps) if steps else 0.0,
        "exp_avg_rms": (
            math.sqrt(exp_avg_square_sum / exp_avg_elements)
            if exp_avg_elements
            else 0.0
        ),
        "amp_scale": float(scaler.get_scale()),
    }
    numeric = [
        evidence["step_min"],
        evidence["step_max"],
        evidence["exp_avg_rms"],
        evidence["amp_scale"],
    ]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise FloatingPointError("non-finite IBER-BE optimizer evidence")
    return evidence


def _published_record(output_root: Path, epoch: int) -> dict[str, Any] | None:
    rows = _load_publication_ledger(output_root)
    if epoch > len(rows):
        return None
    return rows[epoch - 1]


def _record_evaluation(
    output_root: Path,
    evaluation_path: Path,
    *,
    epoch: int,
    verified_tip: int,
) -> dict[str, Any]:
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("design_version") != DESIGN_VERSION:
        raise RuntimeError("IBER-BE evaluation design identity mismatch")
    evaluated_epoch = evaluation.get("epoch", evaluation.get("checkpoint_epoch"))
    if evaluated_epoch != epoch:
        raise RuntimeError("IBER-BE evaluation epoch mismatch")
    row = {
        "design_version": DESIGN_VERSION,
        "stage": "screen",
        "probe": "b3",
        "seed": 0,
        "epoch": epoch,
        "evaluation": evaluation,
    }
    _record_epoch_row(
        output_root / "results.jsonl",
        row,
        epoch=epoch,
        verified_tip=verified_tip,
    )
    return row


def _protect_completed_epoch(
    *,
    baseline_checkpoint: Path,
    dataset_root: Path,
    gate1_decision: Path,
    output_root: Path,
    checkpoint: Path,
    publication_config: Path,
    epoch: int,
    verified_tip: int,
) -> None:
    record = _published_record(output_root, epoch)
    evaluation = output_root / "evaluations" / f"epoch-{epoch:04d}.json"
    checkpoint_sha = file_sha256(checkpoint).lower()
    if record is not None:
        if (
            not evaluation.is_file()
            or record.get("checkpoint", {}).get("sha256") != checkpoint_sha
        ):
            raise RuntimeError("IBER-BE epoch publication did not verify local evidence")
        return
    if not evaluation.is_file():
        evaluate_command = [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "evaluate_iber.py"),
            "--baseline-checkpoint",
            str(baseline_checkpoint),
            "--private-checkpoint",
            str(checkpoint),
            "--dataset-root",
            str(dataset_root),
            "--gate1-decision",
            str(gate1_decision),
            "--output",
            str(evaluation),
        ]
        subprocess.run(evaluate_command, cwd=REPOSITORY_ROOT, check=True)
    _record_evaluation(
        output_root,
        evaluation,
        epoch=epoch,
        verified_tip=verified_tip,
    )
    publish_command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "publish_iber_epoch.py"),
        "--run-dir",
        str(output_root),
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(publication_config),
    ]
    subprocess.run(publish_command, cwd=REPOSITORY_ROOT, check=True)
    rows = _load_publication_ledger(output_root)
    if highest_contiguous_verified_epoch(rows) != epoch:
        raise RuntimeError("IBER-BE epoch publication did not verify")
    record = rows[epoch - 1]
    if record.get("checkpoint", {}).get("sha256") != checkpoint_sha:
        raise RuntimeError("IBER-BE epoch publication did not verify checkpoint")


def _mean(values: Sequence[float], *, name: str) -> float:
    if not values:
        raise RuntimeError(f"IBER-BE epoch has no {name} values")
    result = math.fsum(values) / len(values)
    if not math.isfinite(result):
        raise FloatingPointError(f"non-finite IBER-BE {name}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    contract = validate_screen_contract(dict(SCREEN_CONTRACT))
    if contract["status"] != "passed_with_runtime_amendment":
        raise RuntimeError("frozen IBER-BE screen contract is internally invalid")
    _seed_everything(TRAINING_CONSTANTS["seed"])
    source_commit = _source_commit()
    baseline = args.baseline_checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    publication_config = args.publication_config.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not publication_config.is_file():
        raise FileNotFoundError(publication_config)

    baseline_sha = file_sha256(baseline)
    dataset_sha = str(dataset_signature(dataset_root)["sha256"])
    category_sha = category_mapping_sha256(CATEGORY_NAMES)
    if baseline_sha != EXPECTED_BASELINE_SHA256:
        raise ValueError("IBER-BE baseline SHA256 mismatch")
    if dataset_sha != EXPECTED_DATASET_SHA256:
        raise ValueError("IBER-BE dataset SHA256 mismatch")
    if category_sha != EXPECTED_CATEGORY_SHA256:
        raise ValueError("IBER-BE category mapping SHA256 mismatch")
    gate1_decision_sha = _validate_gate1_decision(
        args.gate1_decision.resolve(), source_commit=source_commit
    )
    ledger_rows = _load_publication_ledger(output_root)
    verified_tip = highest_contiguous_verified_epoch(ledger_rows)
    if args.resume_checkpoint is None and verified_tip:
        raise ValueError(
            "IBER-BE run already has verified epochs; resume checkpoint is required"
        )

    from ultralytics import RTDETR

    device = torch.device(f"cuda:{args.device}")
    detector = RTDETR(str(baseline)).model.to(device).eval()
    detector.requires_grad_(False)
    loader, dataset_hyp = _build_train_loader(dataset_root, output_root)
    with FrozenIBERAdapter.from_detector(
        detector,
        private_seed=PRIVATE_SEED,
        probe="b3",
        image_size=TRAINING_CONSTANTS["imgsz"],
        rho=0.05,
    ).to(device).train() as adapter:
        detector.eval()
        detector.requires_grad_(False)
        _assert_detector_frozen(detector)
        detector_sha_before = module_state_sha256(detector)
        optimizer = build_private_optimizer(adapter.refiner)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=TRAINING_CONSTANTS["amp"],
            init_scale=TRAINING_CONSTANTS["amp_scale"],
            growth_interval=2**31 - 1,
        )
        start_epoch = 1
        if args.resume_checkpoint is not None:
            resume_path = args.resume_checkpoint.resolve()
            if verified_tip < 1:
                raise ValueError("IBER-BE has no remotely verified epoch to resume")
            ledger_checkpoint = ledger_rows[-1].get("checkpoint", {})
            if ledger_checkpoint.get("sha256") != file_sha256(resume_path).lower():
                raise ValueError("IBER-BE resume checkpoint SHA256 is not the ledger tip")
            artifact = torch.load(
                resume_path,
                map_location=device,
                weights_only=True,
            )
            validate_resume_checkpoint(
                artifact,
                source_commit=source_commit,
                highest_verified_epoch=verified_tip,
            )
            if artifact.get("detector_sha_after") != detector_sha_before:
                raise ValueError("IBER-BE detector resume authority mismatch")
            adapter.refiner.load_state_dict(artifact["refiner"], strict=True)
            optimizer.load_state_dict(artifact["optimizer"])
            scaler.load_state_dict(artifact["scaler"])
            _restore_rng(artifact)
            start_epoch = artifact["epoch"] + 1

        checkpoint_root = output_root / "checkpoints"
        diagnostics_path = output_root / "diagnostics.jsonl"
        if start_epoch > SCREEN_EPOCHS:
            return 0
        for epoch in range(start_epoch, SCREEN_EPOCHS + 1):
            if epoch > SCREEN_EPOCHS - AUGMENTATION["close_mosaic"]:
                loader.dataset.close_mosaic(hyp=copy.copy(dataset_hyp))
            _reseed_loader_for_epoch(loader, epoch)
            loss_accumulator: dict[str, list[float]] = {}
            diagnostic_values: dict[str, list[float]] = {}
            batch_count = 0
            for raw_batch in loader:
                batch_count += 1
                batch = _move_training_batch(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=TRAINING_CONSTANTS["amp"],
                ):
                    losses = adapter.training_step(batch)
                if not bool(torch.isfinite(losses.total.detach())):
                    raise FloatingPointError("non-finite IBER-BE private loss")
                scaler.scale(losses.total).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    adapter.refiner.parameters(),
                    max_norm=TRAINING_CONSTANTS["clip_grad_norm"],
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise FloatingPointError("non-finite IBER-BE private gradient")
                if any(value.grad is not None for value in detector.parameters()):
                    raise RuntimeError("frozen IBER-BE detector received a gradient")
                scaler.step(optimizer)
                scaler.update()
                if float(scaler.get_scale()) != TRAINING_CONSTANTS["amp_scale"]:
                    raise RuntimeError("fixed IBER-BE amp_scale changed")
                for name, value in vars(losses).items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        loss_accumulator.setdefault(name, []).append(
                            float(value.detach().float().cpu())
                        )
                for name, value in _correction_diagnostics(adapter).items():
                    diagnostic_values.setdefault(name, []).append(value)

            detector_sha_after = module_state_sha256(detector)
            if detector_sha_after != detector_sha_before:
                raise RuntimeError(
                    "frozen IBER-BE detector state changed during private training"
                )
            optimizer_evidence = _optimizer_evidence(optimizer, scaler)
            diagnostic = {
                "design_version": DESIGN_VERSION,
                "stage": "screen",
                "probe": "b3",
                "seed": 0,
                "epoch": epoch,
                "batch_count": batch_count,
                "losses": {
                    name: _mean(values, name=f"loss.{name}")
                    for name, values in loss_accumulator.items()
                },
                "diagnostics": {
                    name: _mean(values, name=name)
                    for name, values in diagnostic_values.items()
                },
                "detector_sha_before": detector_sha_before,
                "detector_sha_after": detector_sha_after,
                "optimizer_evidence": optimizer_evidence,
                "amp_scale": float(scaler.get_scale()),
            }
            artifact = {
                "format_version": 1,
                "design_version": DESIGN_VERSION,
                "stage": "screen",
                "probe": "b3",
                "seed": 0,
                "private_seed": PRIVATE_SEED,
                "epoch": epoch,
                "baseline_sha256": baseline_sha,
                "dataset_sha256": dataset_sha,
                "subset_sha256": EXPECTED_SUBSET_SHA256,
                "category_sha256": category_sha,
                "source_commit": source_commit,
                "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
                "protocol_sha256": PROTOCOL_SHA256,
                "gate1_decision_sha256": gate1_decision_sha,
                "execution_environment": execution_environment(),
                "detector_sha_before": detector_sha_before,
                "detector_sha_after": detector_sha_after,
                "refiner": adapter.refiner.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": _rng_state(),
                "diagnostic": diagnostic,
                "optimizer_evidence": optimizer_evidence,
                "training_constants": dict(TRAINING_CONSTANTS),
                "augmentation": dict(AUGMENTATION),
            }
            checkpoint = atomic_save_private_checkpoint(
                checkpoint_root / f"epoch-{epoch:04d}.pt",
                artifact,
            )
            last_temporary = checkpoint_root / "last.pt.tmp"
            shutil.copy2(checkpoint, last_temporary)
            os.replace(last_temporary, checkpoint_root / "last.pt")
            epoch_root = output_root / "epochs" / f"epoch-{epoch:04d}"
            _atomic_write_json(epoch_root / "diagnostic.json", diagnostic)
            _atomic_write_json(
                epoch_root / "detector-fingerprint.json",
                {
                    "epoch": epoch,
                    "detector_sha_before": detector_sha_before,
                    "detector_sha_after": detector_sha_after,
                },
            )
            _atomic_write_json(
                epoch_root / "optimizer-evidence.json",
                {"epoch": epoch, "optimizer_evidence": optimizer_evidence},
            )
            _record_epoch_row(
                diagnostics_path,
                diagnostic,
                epoch=epoch,
                verified_tip=verified_tip,
            )
            _protect_completed_epoch(
                baseline_checkpoint=baseline,
                dataset_root=dataset_root,
                gate1_decision=args.gate1_decision.resolve(),
                output_root=output_root,
                checkpoint=checkpoint,
                publication_config=publication_config,
                epoch=epoch,
                verified_tip=verified_tip,
            )
            verified_tip = epoch
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
