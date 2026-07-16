from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rtdetr_ioqc_sa import IOQCSATrainer
from src.gpu_adaptive_batch import load_adaptive_state, save_adaptive_state


EXIT_PLANNED_RESTART = 75


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train scratch RT-DETR-L with standalone IOQC-SA on VisDrone.")
    parser.add_argument("--model", default="rtdetr-l.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--name", default="scratch-rtdetr-l-ioqc-sa-100ep")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "ioqc-sa")
    parser.add_argument("--amp", type=parse_bool, default=True)
    parser.add_argument("--lambda-competition", type=float, default=0.05)
    parser.add_argument("--lambda-alignment", type=float, default=0.05)
    parser.add_argument("--density-threshold", type=float, default=1.0)
    parser.add_argument("--duplicate-threshold", type=float, default=0.10)
    parser.add_argument("--resume", help="Resume from a complete last.pt or epochN.pt checkpoint.")
    parser.add_argument("--state", type=Path, help="Optional adaptive supervisor state updated after model saves.")
    parser.add_argument("--smoke", action="store_true", help="Run one epoch on 1% of training images.")
    return parser


def build_settings(args: argparse.Namespace) -> dict:
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
        "name": f"{args.name}-smoke" if args.smoke else args.name,
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
        "plots": True,
        "val": True,
    }
    if args.resume:
        settings["resume"] = str(Path(args.resume).resolve())
    return settings


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def update_ioqc_progress(trainer: IOQCSATrainer) -> None:
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    model.set_ioqc_progress(int(trainer.epoch), int(trainer.epochs))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def write_epoch_diagnostics(trainer: IOQCSATrainer) -> None:
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    result = model.last_ioqc_result
    if result is None:
        return

    def scalar(value: torch.Tensor | float) -> float:
        return float(value.detach().float().cpu()) if isinstance(value, torch.Tensor) else float(value)

    record = {
        "epoch": int(trainer.epoch + 1),
        "batch": int(trainer.batch_size),
        "amp": bool(trainer.amp),
        "active_weight": scalar(model.last_ioqc_diagnostics["active_weight"]),
        "competition_loss": scalar(result.competition),
        "alignment_loss": scalar(result.alignment),
        "competition_contribution": scalar(model.last_ioqc_diagnostics["competition_contribution"]),
        "alignment_contribution": scalar(model.last_ioqc_diagnostics["alignment_contribution"]),
        "dense_targets": int(result.dense_count.detach().cpu()),
        "duplicates": int(result.duplicate_count.detach().cpu()),
        "valid_queries": int(result.valid_query_count.detach().cpu()),
        "p3_mass_min": scalar(result.p3_mass_min),
        "p3_mass_max": scalar(result.p3_mass_max),
        "cuda_peak_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2) if torch.cuda.is_available() else 0.0,
    }
    path = Path(trainer.save_dir) / "ioqc_sa_diagnostics.jsonl"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    _atomic_write(path, existing + json.dumps(record, ensure_ascii=True) + "\n")


def update_adaptive_state_after_save(trainer: IOQCSATrainer, state_path: Path) -> None:
    state = load_adaptive_state(state_path)
    old_batch = state.current_batch
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 1.0
    state.checkpoint = str(Path(trainer.last).resolve())
    state.record_epoch(completed_epoch=int(trainer.epoch) + 1, peak_gib=peak_gib, total_gib=total_gib)
    save_adaptive_state(state_path, state)

    history = state_path.parent / "batch_history.jsonl"
    record = {
        "epoch": state.completed_epoch,
        "batch_before": old_batch,
        "batch_after": state.current_batch,
        "amp": state.amp_enabled,
        "peak_gib": state.last_peak_gib,
        "event": state.last_event,
    }
    existing = history.read_text(encoding="utf-8") if history.exists() else ""
    _atomic_write(history, existing + json.dumps(record, ensure_ascii=True) + "\n")
    if state.current_batch != old_batch:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(EXIT_PLANNED_RESTART)


def main() -> None:
    args = build_parser().parse_args()
    trainer = IOQCSATrainer(
        overrides=build_settings(args),
        lambda_competition=args.lambda_competition,
        lambda_alignment=args.lambda_alignment,
        density_threshold=args.density_threshold,
        duplicate_threshold=args.duplicate_threshold,
    )
    trainer.add_callback("on_train_epoch_start", update_ioqc_progress)
    trainer.add_callback("on_train_epoch_end", write_epoch_diagnostics)
    if args.state is not None:
        state_path = args.state.resolve()
        trainer.add_callback("on_model_save", lambda current: update_adaptive_state_after_save(current, state_path))
    trainer.train()


if __name__ == "__main__":
    main()
