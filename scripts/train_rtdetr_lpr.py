"""Train LPR-RT-DETR under the frozen VisDrone screening protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lpr_head import LocalizationPriorRefiner
from src.rtdetr_lpr import LPRTrainer


FROZEN_PROTOCOL = {
    "model": "rtdetr-l.yaml",
    "data": "VisDrone.yaml",
    "imgsz": 640,
    "batch": 8,
    "fraction": 1.0,
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
    "plots": True,
    "val": True,
    "mosaic": 1.0,
    "mixup": 0.0,
    "scale": 0.5,
    "translate": 0.1,
    "perspective": 0.0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train output-isolated LPR-RT-DETR-L on VisDrone.")
    parser.add_argument("--epochs", type=int, choices=(10, 100), default=10)
    parser.add_argument("--max-logit-delta", type=float, default=0.5)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "lpr")
    parser.add_argument("--name")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Run one epoch on 1%% for engineering validation.")
    return parser


def build_settings(args: argparse.Namespace) -> dict:
    if args.max_logit_delta <= 0:
        raise ValueError("--max-logit-delta must be positive")
    if args.resume is not None and args.epochs != 100:
        raise ValueError("A passing checkpoint may only be resumed to total epoch 100")
    if args.epochs == 100 and args.resume is None:
        raise ValueError("The frozen 100-epoch protocol must resume a passing 10-epoch checkpoint")
    if args.smoke and args.resume is not None:
        raise ValueError("Smoke validation must start from scratch")

    name = args.name or "scratch-rtdetr-l-lpr-v1-10ep"
    settings = {
        **FROZEN_PROTOCOL,
        "epochs": 1 if args.smoke else args.epochs,
        "workers": args.workers,
        "device": args.device,
        "project": str(args.project.resolve()),
        "name": f"{name}-smoke" if args.smoke else name,
        "exist_ok": True,
    }
    if args.smoke:
        settings["fraction"] = 0.01
    if args.resume is not None:
        settings["resume"] = str(args.resume.resolve())
    return settings


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _lpr_refiners(trainer) -> list[LocalizationPriorRefiner]:
    model = _unwrap_model(trainer.model)
    return [module for module in model.modules() if isinstance(module, LocalizationPriorRefiner)]


def reset_peak_memory(_trainer) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def reset_lpr_batch_gradient(trainer) -> None:
    trainer._lpr_grad_sq = 0.0


def register_lpr_gradient_hooks(trainer) -> None:
    trainer._lpr_hook_handles = []
    trainer._lpr_grad_sq = 0.0

    def record(gradient: torch.Tensor) -> torch.Tensor:
        trainer._lpr_grad_sq += float(gradient.detach().float().square().sum().cpu())
        return gradient

    for refiner in _lpr_refiners(trainer):
        for parameter in refiner.parameters():
            trainer._lpr_hook_handles.append(parameter.register_hook(record))


def remove_lpr_gradient_hooks(trainer) -> None:
    for handle in getattr(trainer, "_lpr_hook_handles", []):
        handle.remove()
    trainer._lpr_hook_handles = []


def _available(values: Iterable[torch.Tensor | None]) -> list[torch.Tensor]:
    return [value for value in values if value is not None]


def capture_lpr_epoch_state(trainer) -> None:
    refiners = _lpr_refiners(trainer)
    residual_means = _available(refiner.last_residual_mean for refiner in refiners)
    residual_maxima = _available(refiner.last_residual_max for refiner in refiners)
    trainer._lpr_epoch_state = {
        "gates": [float((0.5 * torch.tanh(refiner.alpha.detach())).cpu()) for refiner in refiners],
        "residual_mean": (
            float(torch.stack([value.float().cpu() for value in residual_means]).mean())
            if residual_means
            else float("nan")
        ),
        "residual_max": (
            float(torch.stack([value.float().cpu() for value in residual_maxima]).max())
            if residual_maxima
            else float("nan")
        ),
        "lpr_grad_norm": math.sqrt(float(getattr(trainer, "_lpr_grad_sq", 0.0))),
    }


def _map75(trainer) -> float:
    return float(trainer.validator.metrics.box.map75)


def write_lpr_diagnostics(trainer) -> None:
    state = getattr(trainer, "_lpr_epoch_state", None)
    if state is None:
        raise RuntimeError("LPR epoch state was not captured before validation")
    record = {
        "epoch": int(trainer.epoch + 1),
        "map75": _map75(trainer),
        **state,
        "cuda_peak_mib": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 2) if torch.cuda.is_available() else 0.0
        ),
    }
    path = Path(trainer.save_dir) / "lpr_diagnostics.jsonl"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    _atomic_write(path, existing + json.dumps(record, allow_nan=False, ensure_ascii=True) + "\n")


def write_protocol_manifest(trainer, max_logit_delta: float) -> None:
    manifest = {
        "protocol": FROZEN_PROTOCOL,
        "epochs": int(trainer.args.epochs),
        "workers": int(trainer.args.workers),
        "device": str(trainer.args.device),
        "max_logit_delta": float(max_logit_delta),
    }
    _atomic_write(
        Path(trainer.save_dir) / "lpr_protocol.json",
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


def main() -> None:
    args = build_parser().parse_args()
    trainer = LPRTrainer(overrides=build_settings(args), max_logit_delta=args.max_logit_delta)
    trainer.add_callback("on_train_start", register_lpr_gradient_hooks)
    trainer.add_callback("on_train_start", lambda current: write_protocol_manifest(current, args.max_logit_delta))
    trainer.add_callback("on_train_epoch_start", reset_peak_memory)
    trainer.add_callback("on_train_batch_start", reset_lpr_batch_gradient)
    trainer.add_callback("on_train_epoch_end", capture_lpr_epoch_state)
    trainer.add_callback("on_fit_epoch_end", write_lpr_diagnostics)
    trainer.add_callback("teardown", remove_lpr_gradient_hooks)
    trainer.train()


if __name__ == "__main__":
    main()
