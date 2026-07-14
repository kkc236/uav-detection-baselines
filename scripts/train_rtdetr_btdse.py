from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rtdetr_btdse import BTDSETrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train scratch RT-DETR-L with BTD-SE V2.5-S on VisDrone.")
    parser.add_argument("--model", default="configs/rtdetr-l-btdse.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--name", default="scratch-rtdetr-l-btdse-100ep")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "btdse")
    parser.add_argument("--lambda-background", type=float, default=0.1)
    parser.add_argument("--lambda-saliency", type=float, default=0.1)
    parser.add_argument("--resume", help="Resume from a complete last.pt or epochN.pt checkpoint.")
    parser.add_argument("--smoke", action="store_true", help="Run one epoch on 1% of training images.")
    return parser


def build_settings(args: argparse.Namespace) -> dict:
    epochs = 1 if args.smoke else args.epochs
    fraction = 0.01 if args.smoke else args.fraction
    name = f"{args.name}-smoke" if args.smoke else args.name
    settings = {
        "model": args.model,
        "data": "VisDrone.yaml",
        "epochs": epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "fraction": fraction,
        "project": str(args.project.resolve()),
        "name": name,
        "exist_ok": True,
        "pretrained": False,
        "cache": False,
        "amp": True,
        "deterministic": True,
        "seed": 0,
        "nms": False,
        "max_det": 300,
        "save": True,
        "save_period": 1,
        "plots": True,
        "val": True,
    }
    if args.resume:
        settings["resume"] = str(Path(args.resume).resolve())
    return settings


def write_epoch_diagnostics(trainer: BTDSETrainer) -> None:
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    module = model.btdse

    def stats(tensor: torch.Tensor | None) -> dict[str, float] | None:
        if tensor is None:
            return None
        detached = tensor.detach().float()
        return {
            "min": float(detached.min().cpu()),
            "mean": float(detached.mean().cpu()),
            "max": float(detached.max().cpu()),
        }

    record = {
        "epoch": int(trainer.epoch + 1),
        "background_reliability": stats(module.last_background_reliability),
        "saliency": stats(module.last_saliency),
        "normalizer": stats(module.last_normalizer),
        "gamma": stats(module.gamma),
        "auxiliary_losses": {
            key: float(value.cpu()) for key, value in model.last_auxiliary_losses.items()
        },
        "cuda_peak_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2) if torch.cuda.is_available() else 0.0,
    }
    path = Path(trainer.save_dir) / "btdse_diagnostics.jsonl"
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    settings = build_settings(args)
    trainer = BTDSETrainer(
        overrides=settings,
        lambda_background=args.lambda_background,
        lambda_saliency=args.lambda_saliency,
    )
    trainer.add_callback("on_train_epoch_end", write_epoch_diagnostics)
    trainer.train()


if __name__ == "__main__":
    main()
