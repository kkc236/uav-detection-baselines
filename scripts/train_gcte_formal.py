"""Launch the frozen 100-epoch RT-DETR detector stage for ACR-EG."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DATA = "/mnt/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml"
DEFAULT_MODEL = "rtdetr-l.yaml"
DEFAULT_CONFIG = ROOT / "configs" / "rtdetr-l-acr-eg.yaml"
DEFAULT_BASELINE = "/home/ubuntu/matched-baseline-best-epoch-0100.pt"
MATURE_BASELINE_SHA256 = (
    "54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B"
)
TRAINER_METADATA_KEYS = frozenset(
    {
        "gcte_config",
        "baseline_checkpoint",
        "baseline_sha256",
        "gcte_enabled",
        "gcte_forward_integration",
        "gcte_acr_eg_off",
        "gcte_off",
        "amp_scale",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen 100-epoch RT-DETR detector stage for ACR-EG."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/gcte-formal")
    parser.add_argument("--name", default="acr-eg-rtdetr-formal-100")
    parser.add_argument("--module", default="")
    parser.add_argument("--module-sha256", default="")
    parser.add_argument("--baseline-checkpoint", default=DEFAULT_BASELINE)
    parser.add_argument("--baseline-sha256", default=MATURE_BASELINE_SHA256)
    parser.add_argument(
        "--resume",
        default="",
        help="Downloaded last.pt/epoch checkpoint to resume after interruption.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_settings(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs != 100:
        raise ValueError("GCTE_FORMAL_EPOCHS_MUST_BE_100")
    if args.imgsz != 640 or args.batch != 8 or args.workers != 8:
        raise ValueError("GCTE_FORMAL_INPUT_PROTOCOL_DRIFT")
    if args.device != "0" or args.seed != 0:
        raise ValueError("GCTE_FORMAL_DEVICE_OR_SEED_DRIFT")
    baseline_sha256 = str(args.baseline_sha256).upper()
    if (
        len(baseline_sha256) != 64
        or any(character not in "0123456789ABCDEF" for character in baseline_sha256)
    ):
        raise ValueError("GCTE_FORMAL_BASELINE_SHA256_INVALID")
    from src.acr_eg_integration import load_acr_eg_config

    gcte_config = load_acr_eg_config(args.config)
    return {
        "model": str(Path(args.config).resolve()),
        "data": str(Path(args.data).resolve()),
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "device": "0",
        "seed": 0,
        "project": str(Path(args.project).resolve()),
        "name": args.name,
        "gcte_config": str(Path(args.config).resolve()),
        "baseline_checkpoint": (
            str(Path(args.baseline_checkpoint).resolve())
            if args.baseline_checkpoint
            else ""
        ),
        "baseline_sha256": baseline_sha256,
        "gcte_enabled": gcte_config.enabled,
        "gcte_forward_integration": gcte_config.forward_integration,
        "gcte_acr_eg_off": gcte_config.acr_eg_off,
        "gcte_off": gcte_config.gcte_off,
        "exist_ok": False,
        "pretrained": False,
        "resume": str(Path(args.resume).resolve()) if args.resume else False,
        "cache": False,
        "amp": True,
        "amp_scale": 128.0,
        "compile": False,
        "deterministic": True,
        "fraction": 1.0,
        "nbs": 64,
        "nms": False,
        "max_det": 300,
        "save": True,
        "save_period": 1,
        "optimizer": "MuSGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "cos_lr": False,
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
        "plots": False,
        "val": False,
    }


def build_trainer_overrides(settings: dict[str, Any]) -> dict[str, Any]:
    """Remove evidence-only metadata while preserving every trainer setting."""

    return {
        key: value
        for key, value in settings.items()
        if key not in TRAINER_METADATA_KEYS
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def main() -> None:
    args = build_parser().parse_args()
    settings = build_settings(args)
    if args.model != DEFAULT_MODEL:
        raise ValueError("GCTE_FORMAL_MODEL_CONFIG_DRIFT")

    if not args.dry_run:
        from src.rtdetr_acr_eg import (
            ACREGFormalTrainer,
            validate_acr_eg_resume_checkpoint,
        )

        baseline = Path(args.baseline_checkpoint).resolve()
        if not baseline.is_file():
            raise FileNotFoundError(baseline)
        actual_baseline_sha256 = sha256(baseline.read_bytes()).hexdigest().upper()
        if actual_baseline_sha256 != settings["baseline_sha256"]:
            raise ValueError("GCTE_FORMAL_BASELINE_SHA256_MISMATCH")
        if args.resume:
            import torch

            resume_path = Path(args.resume).resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(resume_path)
            payload = torch.load(resume_path, map_location="cpu", weights_only=False)
            validate_acr_eg_resume_checkpoint(payload)
            del payload

    protocol_path = Path(settings["project"]) / f"{args.name}.protocol.json"
    if protocol_path.exists():
        raise FileExistsError(protocol_path)
    run_path = Path(settings["project"]) / args.name
    if run_path.exists():
        raise FileExistsError(run_path)
    protocol = {
        "schema_version": "gcte-acr-eg-formal-training/v2",
        "source_commit": os.environ.get("GCTE_SOURCE_COMMIT", "unknown"),
        "module_path": str(Path(args.module).resolve()) if args.module else "",
        "module_sha256": args.module_sha256,
        "settings": settings,
        "resume_contract": {
            "enabled": bool(args.resume),
            "fixed_amp_scale": 128.0,
            "fixed_amp_growth_interval": 2**31 - 1,
            "fixed_scale_reconstruction": bool(args.resume),
            "note": (
                "fixed-scale scaler has no dynamic growth state; reconstruction "
                "at the same exact scale is the frozen protocol"
            ),
        },
    }
    _write_json(protocol_path, protocol)
    print(f"GCTE_FORMAL_PROTOCOL {protocol_path}", flush=True)
    if args.dry_run:
        return

    os.environ["GCTE_ACR_EG_BASELINE"] = str(baseline)
    os.environ["GCTE_ACR_EG_YAML"] = settings["gcte_config"]
    train_settings = build_trainer_overrides(settings)
    trainer = ACREGFormalTrainer(overrides=train_settings)
    trainer.train()


if __name__ == "__main__":
    main()
