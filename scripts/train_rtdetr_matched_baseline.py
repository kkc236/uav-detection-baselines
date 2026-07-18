from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.models.rtdetr.train import RTDETRTrainer


RUN_NAME = "scratch-rtdetr-l-btdse-matched-baseline-100ep"


class MatchedBaselineTrainer(RTDETRTrainer):
    """Stock RT-DETR trainer with relocatable, fixed-protocol resume."""

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if not self.resume:
            return
        for key in (
            "project",
            "name",
            "batch",
            "workers",
            "device",
            "save_period",
            "amp",
            "optimizer",
            "lr0",
            "lrf",
            "momentum",
            "weight_decay",
            "warmup_epochs",
        ):
            if key in overrides:
                setattr(self.args, key, overrides[key])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the fixed-protocol BTD-SE-matched stock RT-DETR-L baseline.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "matched-baseline")
    parser.add_argument("--name", default=RUN_NAME)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", help="Resume from a validated checkpoint without changing the protocol.")
    parser.add_argument("--smoke", action="store_true", help="Run one epoch on one percent of the training set.")
    return parser


def build_settings(args: argparse.Namespace) -> dict:
    settings = {
        "model": "rtdetr-l.yaml",
        "data": "VisDrone.yaml",
        "epochs": 1 if args.smoke else 100,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "device": args.device,
        "project": str(args.project.resolve()),
        "name": f"{args.name}-smoke" if args.smoke else args.name,
        "exist_ok": True,
        "pretrained": False,
        "cache": False,
        "amp": True,
        "deterministic": True,
        "seed": 0,
        "nbs": 64,
        "nms": False,
        "max_det": 300,
        "save": True,
        "save_period": 1,
        "optimizer": "auto",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "scale": 0.5,
        "translate": 0.1,
        "plots": True,
        "val": True,
    }
    if args.smoke:
        settings["fraction"] = 0.01
    if args.resume:
        settings["resume"] = str(Path(args.resume).resolve())
    return settings


def main() -> None:
    args = build_parser().parse_args()
    trainer = MatchedBaselineTrainer(overrides=build_settings(args))
    trainer.train()


if __name__ == "__main__":
    main()
