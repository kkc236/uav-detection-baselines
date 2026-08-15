from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import RTDETR

from src.experiment_config import DEFAULT_RTDETR_MODEL, build_train_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RT-DETR baseline on VisDrone.")
    parser.add_argument("--model", default=DEFAULT_RTDETR_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="rtdetr-l-scratch-visdrone")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = build_train_settings(
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
    )
    model = RTDETR(str(settings.pop("model")))
    model.train(**settings)


if __name__ == "__main__":
    main()
