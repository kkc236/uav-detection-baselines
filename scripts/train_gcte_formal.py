"""Launch the frozen 100-epoch RT-DETR detector stage for ACR-EG."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DATA = "/mnt/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml"
DEFAULT_MODEL = "rtdetr-l.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen 100-epoch RT-DETR detector stage for ACR-EG."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
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
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_settings(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs != 100:
        raise ValueError("GCTE_FORMAL_EPOCHS_MUST_BE_100")
    if args.imgsz != 640 or args.batch != 8 or args.workers != 8:
        raise ValueError("GCTE_FORMAL_INPUT_PROTOCOL_DRIFT")
    if args.device != "0" or args.seed != 0:
        raise ValueError("GCTE_FORMAL_DEVICE_OR_SEED_DRIFT")
    return {
        "model": args.model,
        "data": str(Path(args.data).resolve()),
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "device": "0",
        "seed": 0,
        "project": str(Path(args.project).resolve()),
        "name": args.name,
        "exist_ok": False,
        "pretrained": False,
        "resume": False,
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
    protocol_path = Path(settings["project"]) / f"{args.name}.protocol.json"
    if protocol_path.exists():
        raise FileExistsError(protocol_path)
    protocol = {
        "schema_version": "gcte-formal-training/v1",
        "source_commit": os.environ.get("GCTE_SOURCE_COMMIT", "unknown"),
        "module_path": str(Path(args.module).resolve()) if args.module else "",
        "module_sha256": args.module_sha256,
        "settings": settings,
    }
    _write_json(protocol_path, protocol)
    print(f"GCTE_FORMAL_PROTOCOL {protocol_path}", flush=True)
    if args.dry_run:
        return

    from ultralytics import RTDETR

    from src.gcte_formal_trainer import GCTEFormalTrainer

    if args.model != DEFAULT_MODEL:
        raise ValueError("GCTE_FORMAL_MODEL_CONFIG_DRIFT")
    train_settings = dict(settings)
    train_settings.pop("model")
    train_settings.pop("amp_scale")
    model = RTDETR(DEFAULT_MODEL)
    model.train(trainer=GCTEFormalTrainer, **train_settings)


if __name__ == "__main__":
    main()
