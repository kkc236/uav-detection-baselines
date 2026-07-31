"""Train strict paired stock/LPR RT-DETR arms on VisDrone."""

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
from src.checkpoint_recovery import validate_checkpoint
from src.lpr_protocol import (
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    current_environment,
    dataset_signature,
    environment_violations,
)
from src.rtdetr_lpr import FixedPairedControlTrainer, LPRTrainer


FROZEN_PROTOCOL = {
    "model": "rtdetr-l.yaml",
    "imgsz": 640,
    "batch": 8,
    "workers": 8,
    "device": "0",
    "fraction": 1.0,
    "pretrained": False,
    "cache": False,
    "amp": True,
    "deterministic": True,
    "nbs": 64,
    "nms": False,
    "max_det": 300,
    "save": True,
    "save_period": -1,
    "optimizer": "MuSGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.0,
    "cos_lr": False,
    "plots": True,
    "val": True,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "mixup": 0.0,
    "scale": 0.5,
    "translate": 0.1,
    "perspective": 0.0,
    "degrees": 0.0,
    "shear": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "cutmix": 0.0,
    "copy_paste": 0.0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a strict paired stock or LPR RT-DETR-L arm.")
    parser.add_argument("--variant", choices=("control", "lpr"), required=True)
    parser.add_argument("--stage", choices=("screen", "formal"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "lpr")
    parser.add_argument("--name")
    parser.add_argument("--preflight", action="store_true", help="Run a non-comparable one-epoch engineering check.")
    return parser


def validate_launch_authority(
    args: argparse.Namespace,
    manifest: dict,
    actual_environment: dict,
    current_dataset: dict,
) -> None:
    violations = environment_violations(actual_environment)
    if violations:
        raise ValueError(f"environment does not match frozen authority: {violations}")
    if int(manifest.get("seed", -1)) != args.seed:
        raise ValueError(f"protocol seed mismatch: expected {args.seed}, got {manifest.get('seed')}")
    expected_dataset = {"file_count": 14038, "sha256": EXPECTED_DATASET_SHA256}
    if manifest.get("dataset") != expected_dataset or current_dataset != expected_dataset:
        raise ValueError(
            f"dataset does not match frozen authority: manifest={manifest.get('dataset')}, current={current_dataset}"
        )
    subset = manifest.get("subset", {})
    if subset.get("count") != 647 or subset.get("sha256") != EXPECTED_SUBSET_SHA256:
        raise ValueError(f"subset does not match frozen authority: {subset}")
    expected_state = Path(manifest.get("initial_state", {}).get("path", "")).resolve()
    if args.initial_state.resolve() != expected_state:
        raise ValueError("initial-state path does not match paired protocol manifest")


def validate_resume_authority(
    args: argparse.Namespace,
    authority: dict,
    environment: dict,
) -> None:
    """Reject checkpoints from a different paired arm or scientific protocol."""
    if args.resume is None:
        return
    if args.preflight:
        raise ValueError("preflight may not resume a scientific checkpoint")
    checkpoint = args.resume.resolve()
    valid, reason = validate_checkpoint(checkpoint)
    if not valid:
        raise ValueError(f"resume checkpoint is invalid: {reason}")
    if checkpoint.parent.name != "weights":
        raise ValueError("resume checkpoint must be inside its run weights directory")
    runtime_path = checkpoint.parent.parent / "lpr_protocol.json"
    if not runtime_path.is_file():
        raise ValueError(f"resume protocol manifest is missing: {runtime_path}")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    expected = {
        "protocol": FROZEN_PROTOCOL,
        "authority": authority,
        "environment": environment,
        "variant": args.variant,
        "stage": args.stage,
        "seed": args.seed,
        "epochs": 10 if args.stage == "screen" else 100,
        "initial_state": str(args.initial_state.resolve()),
    }
    for field, value in expected.items():
        if runtime.get(field) != value:
            raise ValueError(
                f"resume {field} does not match frozen authority: "
                f"expected={value!r}, actual={runtime.get(field)!r}"
            )


def build_settings(args: argparse.Namespace, manifest: dict) -> dict:
    if args.preflight and args.resume is not None:
        raise ValueError("preflight may not resume a scientific checkpoint")
    data_key = "screen" if args.stage == "screen" else "formal"
    epochs = 10 if args.stage == "screen" else 100
    name = args.name or f"{args.stage}-seed{args.seed}-{args.variant}-lpr-v1"
    settings = {
        **FROZEN_PROTOCOL,
        "data": manifest["data"][data_key]["path"],
        "epochs": 1 if args.preflight else epochs,
        "seed": args.seed,
        "project": str(args.project.resolve()),
        "name": f"{name}-preflight" if args.preflight else name,
        "exist_ok": False,
    }
    if args.preflight:
        settings["fraction"] = 0.02
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


def write_protocol_manifest(trainer, authority: dict, args: argparse.Namespace, environment: dict) -> None:
    runtime_manifest = {
        "protocol": FROZEN_PROTOCOL,
        "authority": authority,
        "environment": environment,
        "variant": args.variant,
        "stage": args.stage,
        "seed": args.seed,
        "epochs": int(trainer.args.epochs),
        "workers": int(trainer.args.workers),
        "device": str(trainer.args.device),
        "initial_state": str(args.initial_state.resolve()),
    }
    _atomic_write(
        Path(trainer.save_dir) / "lpr_protocol.json",
        json.dumps(runtime_manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


def main() -> None:
    args = build_parser().parse_args()
    authority = json.loads(args.protocol_manifest.read_text(encoding="utf-8"))
    environment = current_environment()
    current_dataset = dataset_signature(Path(authority["dataset_root"]))
    validate_launch_authority(args, authority, environment, current_dataset)
    validate_resume_authority(args, authority, environment)
    if not args.initial_state.is_file():
        raise FileNotFoundError(f"missing paired initial state: {args.initial_state}")

    trainer_kwargs = {
        "overrides": build_settings(args, authority),
        "initial_state_path": args.initial_state,
    }
    if args.variant == "lpr":
        trainer = LPRTrainer(**trainer_kwargs, experiment_seed=args.seed)
    else:
        trainer = FixedPairedControlTrainer(**trainer_kwargs)

    if args.variant == "lpr":
        trainer.add_callback("on_train_start", register_lpr_gradient_hooks)
        trainer.add_callback("on_train_batch_start", reset_lpr_batch_gradient)
        trainer.add_callback("on_train_epoch_end", capture_lpr_epoch_state)
        trainer.add_callback("on_fit_epoch_end", write_lpr_diagnostics)
        trainer.add_callback("teardown", remove_lpr_gradient_hooks)
    trainer.add_callback(
        "on_train_start",
        lambda current: write_protocol_manifest(current, authority, args, environment),
    )
    trainer.add_callback("on_train_epoch_start", reset_peak_memory)
    trainer.train()


if __name__ == "__main__":
    main()
