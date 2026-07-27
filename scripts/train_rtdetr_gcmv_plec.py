from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import torch
import yaml

from src.ascv_loc_protocol import subset_signature
from src.gcmv_plec_protocol import (
    EXPECTED_DATA_YAML_SHA256,
    EXPECTED_INITIAL_STATE_SHA256,
    EXPECTED_OPTIMIZER_ATTEMPTS,
    EXPECTED_SUBSET_COUNT,
    EXPECTED_SUBSET_FILE_SHA256,
    EXPECTED_SUBSET_SHA256,
    sha256_file,
    validate_plec_initial_state_artifact,
    validate_runtime_environment,
)
from src.rtdetr_gcmv_plec import (
    GCMVPLECControlTrainer,
    GCMVPLECTrainer,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "configs" / "rtdetr-l-gcmv-plec.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded GCMV-EI server screen."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--initial-state", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--source-commit",
        required=True,
        help="Exact 40-character Git commit deployed for this run.",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    parser.add_argument(
        "--control",
        action="store_true",
        help="Train the matched stock arm without local-view PLEC fusion.",
    )
    return parser


def build_settings(args: argparse.Namespace) -> dict:
    if str(args.device) != "0":
        raise ValueError("GCMV-EI seed0 screen requires device 0")
    return {
        "model": str(Path(args.model)),
        "pretrained": False,
        "data": str(Path(args.data)),
        "project": str(Path(args.project)),
        "name": args.name,
        "epochs": 10,
        "fraction": 1.0,
        "batch": 8,
        "workers": 8,
        "device": args.device,
        "seed": args.seed,
        "imgsz": 640,
        "optimizer": "MuSGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "nbs": 64,
        "amp": True,
        "deterministic": True,
        "val": False,
        "plots": False,
        "save": True,
        "save_period": -1,
        "exist_ok": False,
        "cache": False,
        "rect": False,
        "multi_scale": 0.0,
        "close_mosaic": 10,
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
    }


def validate_screen_inputs(
    args: argparse.Namespace,
    *,
    check_environment: bool = True,
) -> dict:
    if int(args.seed) != 0 or str(args.device) != "0":
        raise ValueError("GCMV-EI first screen requires seed=0 on device 0")
    initial_state = Path(args.initial_state).resolve()
    data_path = Path(args.data).resolve()
    if not initial_state.is_file():
        raise FileNotFoundError(initial_state)
    actual_initial = sha256_file(initial_state)
    if actual_initial != EXPECTED_INITIAL_STATE_SHA256[0]:
        raise ValueError(
            "GCMV initial-state checksum mismatch: "
            f"{actual_initial}"
        )
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    actual_yaml = sha256_file(data_path)
    if actual_yaml != EXPECTED_DATA_YAML_SHA256:
        raise ValueError(f"GCMV data YAML checksum mismatch: {actual_yaml}")

    artifact = torch.load(
        initial_state,
        map_location="cpu",
        weights_only=False,
    )
    validate_plec_initial_state_artifact(artifact, seed=0)
    payload = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GCMV data YAML must contain a mapping")
    subset_path = Path(payload.get("train", "")).resolve()
    dataset_root = Path(payload.get("path", "")).resolve()
    if not subset_path.is_file():
        raise FileNotFoundError(subset_path)
    actual_subset_file = sha256_file(subset_path)
    if actual_subset_file != EXPECTED_SUBSET_FILE_SHA256:
        raise ValueError(
            f"GCMV subset-file checksum mismatch: {actual_subset_file}"
        )
    semantic = subset_signature(subset_path, root=dataset_root)
    expected_semantic = {
        "count": EXPECTED_SUBSET_COUNT,
        "sha256": EXPECTED_SUBSET_SHA256,
    }
    if semantic != expected_semantic:
        raise ValueError(f"GCMV subset semantic drift: {semantic}")
    environment = (
        validate_runtime_environment()
        if check_environment
        else None
    )
    return {
        "initial_state_sha256": actual_initial,
        "data_yaml_sha256": actual_yaml,
        "subset_file_sha256": actual_subset_file,
        "subset": semantic,
        "environment": environment,
    }


def trainer_class(
    args: argparse.Namespace,
) -> type[GCMVPLECTrainer]:
    return GCMVPLECControlTrainer if args.control else GCMVPLECTrainer


def validate_training_completion(trainer: GCMVPLECTrainer) -> None:
    attempts = int(trainer.plec_optimizer_attempts)
    if attempts != EXPECTED_OPTIMIZER_ATTEMPTS:
        raise RuntimeError(
            "GCMV optimizer attempts drift: "
            f"expected={EXPECTED_OPTIMIZER_ATTEMPTS} actual={attempts}"
        )
    amp_min = float(trainer.plec_amp_scale_min)
    amp_max = float(trainer.plec_amp_scale_max)
    if amp_min != 128.0 or amp_max != 128.0:
        raise RuntimeError(
            "GCMV AMP scale drift: "
            f"expected=128 min={amp_min} max={amp_max}"
        )


def build_run_manifest(
    *,
    args: argparse.Namespace,
    inputs: dict,
    trainer: GCMVPLECTrainer,
    status: str,
    error: str | None = None,
) -> dict:
    source_commit = str(args.source_commit).lower()
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("source commit must be an exact 40-character SHA-1")
    if status not in {"running", "completed", "failed"}:
        raise ValueError(f"unsupported GCMV run status: {status}")
    manifest = {
        "schema_version": 1,
        "status": status,
        "arm": "control" if args.control else "method",
        "seed": int(args.seed),
        "source_commit": source_commit,
        "protocol": inputs,
        "runtime": {
            "optimizer_attempts": int(trainer.plec_optimizer_attempts),
            "expected_optimizer_attempts": EXPECTED_OPTIMIZER_ATTEMPTS,
            "amp_scale_min": float(trainer.plec_amp_scale_min),
            "amp_scale_max": float(trainer.plec_amp_scale_max),
        },
    }
    if error is not None:
        manifest["error"] = error
    return manifest


def write_run_manifest(save_dir: str | Path, manifest: dict) -> Path:
    destination = Path(save_dir) / "gcmv_protocol_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def main() -> None:
    args = build_parser().parse_args()
    inputs = validate_screen_inputs(args)
    trainer = trainer_class(args)(
        overrides=build_settings(args),
        initial_state_path=args.initial_state,
    )
    write_run_manifest(
        trainer.save_dir,
        build_run_manifest(
            args=args,
            inputs=inputs,
            trainer=trainer,
            status="running",
        ),
    )
    try:
        trainer.train()
        validate_training_completion(trainer)
    except BaseException as exc:
        write_run_manifest(
            trainer.save_dir,
            build_run_manifest(
                args=args,
                inputs=inputs,
                trainer=trainer,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise
    write_run_manifest(
        trainer.save_dir,
        build_run_manifest(
            args=args,
            inputs=inputs,
            trainer=trainer,
            status="completed",
        ),
    )


if __name__ == "__main__":
    main()
