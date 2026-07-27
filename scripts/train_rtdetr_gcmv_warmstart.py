from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import torch

from src.gcmv_plec_protocol import validate_runtime_environment
from src.gcmv_warmstart import (
    EXPECTED_BASELINE_SHA256,
    build_module_artifact,
    sha256_file,
)
from src.rtdetr_gcmv_warmstart import (
    DETECTOR_LR,
    GCMVWarmStartCalibrationTrainer,
    GCMVWarmStartControlTrainer,
    GCMVWarmStartTrainer,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "configs" / "rtdetr-l-gcmv-plec.yaml"
EXPECTED_DATA_YAML_SHA256 = (
    "7EB91FCEF62A687A26A8EF76E9075B97"
    "93B52BC8BB110E4235FACF3E2B958324"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one stage of the mature-baseline GCMV diagnostic."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("calibration", "control", "method"),
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--module-artifact",
        required=True,
        help=(
            "Calibration output path, or the calibrated module input path "
            "for Method. Control records but does not consume this path."
        ),
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    return parser


def build_settings(args: argparse.Namespace) -> dict:
    if str(args.device) != "0" or int(args.seed) != 0:
        raise ValueError("warm-start diagnostic requires seed0 on device0")
    calibration = args.stage == "calibration"
    return {
        "model": str(Path(args.model)),
        "pretrained": False,
        "data": str(Path(args.data)),
        "project": str(Path(args.project)),
        "name": args.name,
        "epochs": 1 if calibration else 10,
        "fraction": 1.0,
        "batch": 8,
        "workers": 8,
        "device": args.device,
        "seed": args.seed,
        "imgsz": 640,
        "optimizer": "MuSGD",
        "lr0": DETECTOR_LR,
        "lrf": 1.0,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 0.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "nbs": 64,
        "cos_lr": False,
        "amp": True,
        "deterministic": True,
        "val": False,
        "plots": False,
        "save": not calibration,
        "save_period": -1,
        "exist_ok": False,
        "cache": False,
        "rect": False,
        "multi_scale": 0.0,
        "close_mosaic": 1 if calibration else 10,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "bgr": 0.0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "patience": 100,
    }


def trainer_class(
    args: argparse.Namespace,
) -> type[GCMVWarmStartTrainer]:
    return {
        "calibration": GCMVWarmStartCalibrationTrainer,
        "control": GCMVWarmStartControlTrainer,
        "method": GCMVWarmStartTrainer,
    }[args.stage]


def validate_inputs(args: argparse.Namespace) -> dict:
    source_commit = str(args.source_commit).lower()
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("source commit must be an exact 40-character SHA-1")
    baseline = Path(args.baseline).resolve()
    data = Path(args.data).resolve()
    module = Path(args.module_artifact).resolve()
    if not baseline.is_file():
        raise FileNotFoundError(baseline)
    baseline_sha = sha256_file(baseline)
    if baseline_sha != EXPECTED_BASELINE_SHA256:
        raise ValueError(
            "formal RTX4090 baseline checksum mismatch: "
            f"{baseline_sha}"
        )
    if not data.is_file():
        raise FileNotFoundError(data)
    data_sha = sha256_file(data)
    if data_sha != EXPECTED_DATA_YAML_SHA256:
        raise ValueError(f"formal full-data YAML checksum mismatch: {data_sha}")
    if args.stage == "calibration":
        if module.exists():
            raise FileExistsError(module)
    elif args.stage == "method" and not module.is_file():
        raise FileNotFoundError(module)
    return {
        "source_commit": source_commit,
        "baseline": {
            "path": baseline.as_posix(),
            "sha256": baseline_sha,
        },
        "data": {
            "path": data.as_posix(),
            "yaml_sha256": data_sha,
            "dataset_sha256": (
                "FD92E9FF4B3B58FCDD5A32F7E770FC3"
                "398E566B627DB0E188CB5FF9F3B7BBDAB"
            ),
            "train_images": 6471,
            "val_images": 548,
        },
        "module_artifact": module.as_posix(),
        "environment": validate_runtime_environment(),
    }


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_training_completion(
    trainer: GCMVWarmStartTrainer,
    *,
    stage: str,
) -> dict:
    attempts = int(trainer.plec_optimizer_attempts)
    accumulate = int(trainer.accumulate)
    expected = (
        len(trainer.train_loader) * int(trainer.epochs)
    ) // accumulate
    if attempts != expected:
        raise RuntimeError(
            "optimizer attempts drift: "
            f"expected={expected} actual={attempts}"
        )
    amp_min = float(trainer.plec_amp_scale_min)
    amp_max = float(trainer.plec_amp_scale_max)
    if amp_min != 128.0 or amp_max != 128.0:
        raise RuntimeError(
            f"fixed AMP scale drift: min={amp_min} max={amp_max}"
        )
    gamma = float(
        torch.tanh(trainer.model.gcmv_injector.peg.rho)
        .detach()
        .float()
        .cpu()
        .item()
    )
    if not math.isfinite(gamma):
        raise FloatingPointError("trained PEG gamma is non-finite")
    if stage == "calibration" and gamma != 0.0:
        raise RuntimeError(f"calibration gamma drift: {gamma}")
    return {
        "optimizer_attempts": attempts,
        "expected_optimizer_attempts": expected,
        "accumulate": accumulate,
        "amp_scale_min": amp_min,
        "amp_scale_max": amp_max,
        "trained_gamma": gamma,
        "gamma_materially_open": abs(gamma) >= 1e-4,
    }


def write_manifest(
    save_dir: str | Path,
    *,
    status: str,
    stage: str,
    protocol: dict,
    settings: dict,
    runtime: dict | None = None,
    error: str | None = None,
) -> Path:
    if status not in {"running", "completed", "failed"}:
        raise ValueError(f"unsupported status: {status}")
    payload = {
        "schema_version": "gcmv-ei-warmstart-stage/v1",
        "status": status,
        "stage": stage,
        "protocol": protocol,
        "settings": settings,
        "runtime": runtime or {},
    }
    if error is not None:
        payload["error"] = error
    destination = Path(save_dir) / "gcmv_warmstart_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def main() -> None:
    args = build_parser().parse_args()
    protocol = validate_inputs(args)
    settings = build_settings(args)
    module_path = Path(args.module_artifact).resolve()
    calibrated_input = (
        module_path if args.stage == "method" else None
    )
    trainer = trainer_class(args)(
        overrides=settings,
        baseline_checkpoint_path=args.baseline,
        calibrated_module_path=calibrated_input,
    )
    write_manifest(
        trainer.save_dir,
        status="running",
        stage=args.stage,
        protocol=protocol,
        settings=settings,
    )
    try:
        trainer.train()
        runtime = validate_training_completion(
            trainer,
            stage=args.stage,
        )
        runtime["baseline_load"] = trainer.baseline_summary
        if args.stage == "calibration":
            _atomic_torch_save(
                module_path,
                build_module_artifact(trainer.model),
            )
            runtime["module_artifact"] = {
                "path": module_path.as_posix(),
                "sha256": sha256_file(module_path),
            }
    except BaseException as exc:
        write_manifest(
            trainer.save_dir,
            status="failed",
            stage=args.stage,
            protocol=protocol,
            settings=settings,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    write_manifest(
        trainer.save_dir,
        status="completed",
        stage=args.stage,
        protocol=protocol,
        settings=settings,
        runtime=runtime,
    )


if __name__ == "__main__":
    main()
