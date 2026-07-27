from __future__ import annotations

import argparse
from pathlib import Path

from src.rtdetr_gcmv_plec import (
    GCMVPLECControlTrainer,
    GCMVPLECTrainer,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "configs" / "rtdetr-l-gcmv-plec.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded GCMV PLEC server screen."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--pretrained-weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fraction", type=float, default=0.03)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--control",
        action="store_true",
        help="Train the matched stock arm without local-view PLEC fusion.",
    )
    return parser


def build_settings(args: argparse.Namespace) -> dict:
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 0 < args.fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    if args.batch <= 0 or args.workers < 0:
        raise ValueError("batch must be positive and workers non-negative")
    return {
        "model": str(Path(args.model)),
        "pretrained": str(Path(args.pretrained_weights)),
        "data": str(Path(args.data)),
        "project": str(Path(args.project)),
        "name": args.name,
        "epochs": args.epochs,
        "fraction": args.fraction,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "imgsz": 640,
        "optimizer": "MuSGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 1.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "nbs": 8,
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
        "close_mosaic": 0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "bgr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
    }


def trainer_class(
    args: argparse.Namespace,
) -> type[GCMVPLECTrainer]:
    return GCMVPLECControlTrainer if args.control else GCMVPLECTrainer


def main() -> None:
    args = build_parser().parse_args()
    trainer = trainer_class(args)(overrides=build_settings(args))
    trainer.train()


if __name__ == "__main__":
    main()
