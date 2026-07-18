from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gpu_adaptive_batch import load_adaptive_state, save_adaptive_state
from src.rtdetr_vsf_rmr import MatchedBaselineTrainer, VSFRMRTrainer


EXIT_PLANNED_RESTART = 75
FROZEN_PROTOCOL = {
    "model": "rtdetr-l.yaml",
    "epochs": 100,
    "imgsz": 640,
    "batch": 8,
    "workers": 8,
    "save_period": 1,
    "optimizer": "auto",
    "lr0": 0.01,
    "momentum": 0.937,
    "fraction": 1.0,
    "amp": True,
}


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def run_name_for_variant(variant: str) -> str:
    if variant == "baseline":
        return "scratch-rtdetr-l-vsf-matched-baseline-100ep"
    if variant == "vsf-rmr":
        return "scratch-rtdetr-l-vsf-rmr-100ep"
    raise ValueError(f"Unknown training variant: {variant}")


def trainer_class_for_variant(variant: str):
    if variant == "baseline":
        return MatchedBaselineTrainer
    if variant == "vsf-rmr":
        return VSFRMRTrainer
    raise ValueError(f"Unknown training variant: {variant}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the matched RT-DETR-L baseline or standalone VSF-RMR.")
    parser.add_argument("--variant", choices=("baseline", "vsf-rmr"), default="vsf-rmr")
    parser.add_argument("--model", default="rtdetr-l.yaml")
    parser.add_argument("--epochs", type=int, default=FROZEN_PROTOCOL["epochs"])
    parser.add_argument("--imgsz", type=int, default=FROZEN_PROTOCOL["imgsz"])
    parser.add_argument("--batch", type=int, default=FROZEN_PROTOCOL["batch"])
    parser.add_argument("--workers", type=int, default=FROZEN_PROTOCOL["workers"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--save-period", type=int, default=FROZEN_PROTOCOL["save_period"])
    parser.add_argument("--optimizer", default=FROZEN_PROTOCOL["optimizer"])
    parser.add_argument("--lr0", type=float, default=FROZEN_PROTOCOL["lr0"])
    parser.add_argument("--momentum", type=float, default=FROZEN_PROTOCOL["momentum"])
    parser.add_argument("--fraction", type=float, default=FROZEN_PROTOCOL["fraction"])
    parser.add_argument("--name")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "vsf-rmr")
    parser.add_argument("--amp", type=parse_bool, default=FROZEN_PROTOCOL["amp"])
    parser.add_argument("--fixed-protocol", action="store_true")
    parser.add_argument("--lambda-vsf", type=float, default=0.1)
    parser.add_argument("--resume", help="Resume from a validated last.pt or epochN.pt checkpoint.")
    parser.add_argument("--state", type=Path, help="Adaptive supervisor state updated after model saves.")
    parser.add_argument("--smoke", action="store_true", help="Run one epoch on 1% of training images.")
    return parser


def build_settings(args: argparse.Namespace) -> dict:
    actual = {
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "save_period": args.save_period,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "momentum": args.momentum,
        "fraction": args.fraction,
        "amp": args.amp,
    }
    drift = {key: (actual[key], expected) for key, expected in FROZEN_PROTOCOL.items() if actual[key] != expected}
    if drift:
        raise ValueError(f"Arguments violate the frozen matched protocol: {drift}")

    name = args.name or run_name_for_variant(args.variant)
    settings = {
        "model": args.model,
        "data": "VisDrone.yaml",
        "epochs": 1 if args.smoke else args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "fraction": 0.01 if args.smoke else args.fraction,
        "project": str(args.project.resolve()),
        "name": f"{name}-smoke" if args.smoke else name,
        "exist_ok": True,
        "pretrained": False,
        "cache": False,
        "amp": args.amp,
        "deterministic": True,
        "seed": 0,
        "nbs": 64,
        "nms": False,
        "max_det": 300,
        "save": True,
        "save_period": args.save_period,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": 0.01,
        "momentum": args.momentum,
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
    if args.resume:
        settings["resume"] = str(Path(args.resume).resolve())
    return settings


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def reset_peak_memory(_trainer) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _scalar(value: torch.Tensor | float) -> float:
    return float(value.detach().float().cpu()) if isinstance(value, torch.Tensor) else float(value)


def write_epoch_diagnostics(trainer) -> None:
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    result = getattr(model, "last_vsf_result", None)
    if result is None:
        return
    diagnostics = model.last_vsf_diagnostics
    record = {
        "epoch": int(trainer.epoch + 1),
        "batch": int(trainer.batch_size),
        "amp": bool(trainer.amp),
        "local_loss": _scalar(result.local),
        "global_loss": _scalar(result.global_),
        "target_count": int(result.target_count.detach().cpu()),
        **{key: _scalar(value) for key, value in diagnostics.items()},
        "cuda_peak_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2) if torch.cuda.is_available() else 0.0,
    }
    path = Path(trainer.save_dir) / "vsf_rmr_diagnostics.jsonl"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    _atomic_write(path, existing + json.dumps(record, ensure_ascii=True) + "\n")


def update_adaptive_state_after_save(trainer, state_path: Path) -> None:
    state = load_adaptive_state(state_path)
    previous_batch = state.current_batch
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 1.0
    state.checkpoint = str(Path(trainer.last).resolve())
    state.record_epoch(completed_epoch=int(trainer.epoch) + 1, peak_gib=peak_gib, total_gib=total_gib)
    save_adaptive_state(state_path, state)
    run_dir = Path(trainer.save_dir).resolve()
    save_adaptive_state(run_dir / "adaptive_state.json", state)

    record = {
        "epoch": state.completed_epoch,
        "batch_before": previous_batch,
        "batch_after": state.current_batch,
        "amp": state.amp_enabled,
        "peak_gib": state.last_peak_gib,
        "event": state.last_event,
    }
    for history in {state_path.parent / "batch_history.jsonl", run_dir / "batch_history.jsonl"}:
        existing = history.read_text(encoding="utf-8") if history.exists() else ""
        _atomic_write(history, existing + json.dumps(record, ensure_ascii=True) + "\n")
    if state.current_batch != previous_batch:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(EXIT_PLANNED_RESTART)


def main() -> None:
    args = build_parser().parse_args()
    trainer_class = trainer_class_for_variant(args.variant)
    trainer_kwargs = {"overrides": build_settings(args)}
    if trainer_class is VSFRMRTrainer:
        trainer_kwargs["lambda_vsf"] = args.lambda_vsf
    trainer = trainer_class(**trainer_kwargs)
    trainer.add_callback("on_train_epoch_start", reset_peak_memory)
    if args.variant == "vsf-rmr":
        trainer.add_callback("on_train_epoch_end", write_epoch_diagnostics)
    if args.state is not None:
        state_path = args.state.resolve()
        trainer.add_callback("on_model_save", lambda current: update_adaptive_state_after_save(current, state_path))
    trainer.train()


if __name__ == "__main__":
    main()
