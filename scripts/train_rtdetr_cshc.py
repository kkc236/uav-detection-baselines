"""Explicitly gated training entrypoint for CSHC RT-DETR experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rtdetr_cshc import CSHCTrainer


MODE_SETTINGS = {
    "smoke": {"epochs": 1, "fraction": 0.01, "name": "rtdetr-l-cshc-smoke"},
    "screen": {"epochs": 5, "fraction": 0.10, "name": "rtdetr-l-cshc-screen"},
    "formal": {"epochs": 100, "fraction": 1.0, "name": "rtdetr-l-cshc-formal"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CSHC RT-DETR only under an explicit gated mode.")
    parser.add_argument("--model", default="configs/rtdetr-l-cshc.yaml")
    parser.add_argument("--data", default="VisDrone.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "cshc")
    parser.add_argument("--name")
    parser.add_argument("--lambda-c2-candidate", type=float, default=0.25)
    parser.add_argument("--resume")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="1 epoch using 1% of train images.")
    mode.add_argument("--screen", action="store_true", help="5 epochs using 10% of train images.")
    mode.add_argument("--formal", action="store_true", help="100 epochs using all train images; requires prior approval.")
    return parser


def _selected_mode(args: argparse.Namespace) -> str:
    return next(mode for mode in MODE_SETTINGS if getattr(args, mode))


def build_settings(args: argparse.Namespace) -> dict:
    mode = _selected_mode(args)
    limits = MODE_SETTINGS[mode]
    settings = {
        "model": args.model,
        "data": args.data,
        "epochs": limits["epochs"],
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "fraction": limits["fraction"],
        "project": str(args.project.resolve()),
        "name": args.name or limits["name"],
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


def write_epoch_diagnostics(trainer: CSHCTrainer) -> None:
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    candidates = model.cshc_decoder.last_candidates
    if candidates is None:
        return
    logits = candidates.objectness_logits.detach().float()
    record = {
        "epoch": int(trainer.epoch + 1),
        "c2_candidate_loss": float(model.last_auxiliary_losses["c2_candidate_loss"].cpu()),
        "c2_logit_min": float(logits.min().cpu()),
        "c2_logit_mean": float(logits.mean().cpu()),
        "c2_logit_max": float(logits.max().cpu()),
        "cuda_peak_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2) if torch.cuda.is_available() else 0.0,
    }
    path = Path(trainer.save_dir) / "cshc_diagnostics.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    trainer = CSHCTrainer(overrides=build_settings(args), lambda_c2_candidate=args.lambda_c2_candidate)
    trainer.add_callback("on_train_epoch_end", write_epoch_diagnostics)
    trainer.train()


if __name__ == "__main__":
    main()
