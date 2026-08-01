"""Train strict paired seed0 stock/LPR-G RT-DETR-L arms on VisDrone."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint_recovery import validate_checkpoint
from scripts.sync_experiment_checkpoint import (
    prune_local_epoch_checkpoints,
    validate_token_file,
)
from src.lpr_g_audit import (
    common_model_fingerprint,
    common_optimizer_fingerprint,
    write_epoch_audit,
)
from src.lpr_g_publication import PublicationConfig, publish_with_retry
from src.lpr_g_protocol import validate_lpr_g_initial_state_file
from src.lpr_protocol import (
    EXPECTED_DATASET_SHA256,
    EXPECTED_SOURCE_SHA256,
    EXPECTED_SUBSET_SHA256,
    current_environment,
    dataset_signature,
    environment_violations,
    source_violations,
)
from src.rtdetr_lpr_g import LPRGControlTrainer, LPRGTrainer


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


PRIVATE_DIAGNOSTIC_FIELDS = (
    "quality_mean",
    "quality_rms",
    "quality_p05",
    "quality_p50",
    "quality_p95",
    "gate_mean",
    "gate_rms",
    "gate_p05",
    "gate_p50",
    "gate_p95",
    "residual_mean",
    "residual_rms",
    "residual_p05",
    "residual_p50",
    "residual_p95",
    "loss_bbox_refine",
    "loss_giou_refine",
    "lpr_g_gradient_norm",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a strict paired seed0 stock or LPR-G v2 RT-DETR-L arm."
    )
    parser.add_argument("--variant", choices=("control", "lprg"), required=True)
    parser.add_argument("--stage", choices=("screen", "formal"), required=True)
    parser.add_argument("--seed", type=int, choices=(0,), required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "lpr-g")
    parser.add_argument("--name")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--repo", default="kkc236/uav-detection-baselines")
    parser.add_argument(
        "--repo-url",
        default="https://github.com/kkc236/uav-detection-baselines.git",
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-branch", default="codex/lpr-rtdetr")
    parser.add_argument("--results-branch", default="training-results")
    parser.add_argument(
        "--results-repo",
        type=Path,
        default=Path.home() / "uav-training-results-lpr-g",
    )
    parser.add_argument("--asset-prefix", required=True)
    parser.add_argument("--retain", type=int, default=3)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run a non-comparable one-epoch engineering check.",
    )
    return parser


def _scientific_epochs(stage: str) -> int:
    return 50 if stage == "screen" else 100


def build_settings(args: argparse.Namespace, manifest: dict) -> dict:
    """Build exact seed0 settings before any trainer can mutate runtime state."""
    if int(getattr(args, "seed", -1)) != 0:
        raise ValueError("LPR-G v2 experiments are frozen to seed0")
    preflight = bool(getattr(args, "preflight", False))
    resume = getattr(args, "resume", None)
    if preflight and resume is not None:
        raise ValueError("preflight may not resume a scientific checkpoint")
    stage = str(args.stage)
    if stage not in {"screen", "formal"}:
        raise ValueError(f"unknown LPR-G stage: {stage}")
    variant = str(getattr(args, "variant", ""))
    if variant not in {"control", "lprg"}:
        raise ValueError(f"unknown LPR-G variant: {variant}")
    data_key = "screen" if stage == "screen" else "formal"
    epochs = _scientific_epochs(stage)
    name = getattr(args, "name", None) or f"{stage}-seed0-{variant}-lpr-g-v2"
    settings = {
        **FROZEN_PROTOCOL,
        "data": manifest["data"][data_key]["path"],
        "epochs": 1 if preflight else epochs,
        "seed": 0,
        "project": str(Path(args.project).resolve()),
        "name": f"{name}-preflight" if preflight else name,
        "exist_ok": False,
    }
    if preflight:
        settings["fraction"] = 0.02
    if resume is not None:
        settings["resume"] = str(Path(resume).resolve())
    return settings


def validate_launch_authority(
    args: argparse.Namespace,
    manifest: dict,
    actual_environment: dict,
    current_dataset: dict,
) -> None:
    """Reject environment, source, data, seed, or artifact drift before training."""
    if args.seed != 0:
        raise ValueError("LPR-G v2 experiments are frozen to seed0")
    if manifest.get("format_version") != 2 or manifest.get("seed") != 0:
        raise ValueError("LPR-G protocol manifest must be format v2 seed0")
    violations = environment_violations(actual_environment)
    if violations:
        raise ValueError(f"environment does not match frozen authority: {violations}")
    source_drift = source_violations()
    if source_drift or manifest.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "Ultralytics source does not match frozen authority: "
            f"manifest={manifest.get('source_sha256')}, current={source_drift}"
        )
    expected_dataset = {"file_count": 14038, "sha256": EXPECTED_DATASET_SHA256}
    if manifest.get("dataset") != expected_dataset or current_dataset != expected_dataset:
        raise ValueError(
            "dataset does not match frozen authority: "
            f"manifest={manifest.get('dataset')}, current={current_dataset}"
        )
    subset = manifest.get("subset", {})
    if subset.get("count") != 647 or subset.get("sha256") != EXPECTED_SUBSET_SHA256:
        raise ValueError(f"subset does not match frozen authority: {subset}")
    expected_state = Path(manifest.get("initial_state", {}).get("path", "")).resolve()
    if args.initial_state.resolve() != expected_state:
        raise ValueError("initial-state path does not match LPR-G protocol manifest")


def validate_resume_authority(
    args: argparse.Namespace,
    authority: dict,
    environment: dict,
) -> None:
    """Reject checkpoints from a different LPR-G arm or scientific protocol."""
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
    runtime_path = checkpoint.parent.parent / "lpr_g_protocol.json"
    if not runtime_path.is_file():
        raise ValueError(f"resume protocol manifest is missing: {runtime_path}")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    expected = {
        "protocol": FROZEN_PROTOCOL,
        "authority": authority,
        "environment": environment,
        "variant": args.variant,
        "stage": args.stage,
        "seed": 0,
        "epochs": _scientific_epochs(args.stage),
        "initial_state": str(args.initial_state.resolve()),
        "publication": publication_authority(args),
    }
    for field, value in expected.items():
        if runtime.get(field) != value:
            raise ValueError(
                f"resume {field} does not match frozen authority: "
                f"expected={value!r}, actual={runtime.get(field)!r}"
            )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("diagnostic scalar tensor must contain one value")
        value = value.detach().float().cpu().item()
    result = float(value)
    return result if math.isfinite(result) else None


def _distribution(prefix: str, value: torch.Tensor | None) -> dict[str, float | None]:
    fields = {f"{prefix}_{suffix}": None for suffix in ("mean", "rms", "p05", "p50", "p95")}
    if value is None or value.numel() == 0:
        return fields
    data = value.detach().float().reshape(-1).cpu()
    if not bool(torch.isfinite(data).all()):
        raise FloatingPointError(f"non-finite {prefix} diagnostics")
    fields.update(
        {
            f"{prefix}_mean": float(data.mean()),
            f"{prefix}_rms": float(data.square().mean().sqrt()),
            f"{prefix}_p05": float(torch.quantile(data, 0.05)),
            f"{prefix}_p50": float(torch.quantile(data, 0.50)),
            f"{prefix}_p95": float(torch.quantile(data, 0.95)),
        }
    )
    return fields


def _stock_losses(trainer) -> dict[str, float | None]:
    values = getattr(trainer, "tloss", None)
    if values is None:
        return {"loss_giou": None, "loss_class": None, "loss_bbox": None}
    if isinstance(values, torch.Tensor):
        values = values.detach().float().reshape(-1).cpu().tolist()
    else:
        values = list(values)
    names = ("loss_giou", "loss_class", "loss_bbox")
    return {name: _float(values[index]) if index < len(values) else None for index, name in enumerate(names)}


def _map75(trainer) -> float:
    return float(trainer.validator.metrics.box.map75)


def _private_diagnostics(model) -> dict[str, float | None]:
    decoder = model.model[-1].decoder
    refiner = decoder.lpr_g_refiner
    losses = getattr(model, "last_lpr_g_losses", {})
    return {
        **_distribution("quality", refiner.last_quality),
        **_distribution("gate", refiner.last_gate),
        **_distribution("residual", refiner.last_residual),
        "loss_bbox_refine": _float(losses.get("loss_bbox_refine")),
        "loss_giou_refine": _float(losses.get("loss_giou_refine")),
    }


def write_lpr_g_diagnostics(trainer, *, variant: str) -> dict[str, Any] | None:
    """Atomically append one comparable epoch diagnostic row."""
    epoch = int(trainer.epoch + 1)
    if epoch > int(trainer.args.epochs):
        return None
    if variant not in {"control", "lprg"}:
        raise ValueError(f"unknown LPR-G diagnostic variant: {variant}")
    model = _unwrap_model(trainer.model)
    private = _private_diagnostics(model) if variant == "lprg" else {
        field: None for field in PRIVATE_DIAGNOSTIC_FIELDS if field != "lpr_g_gradient_norm"
    }
    norms = getattr(trainer, "last_gradient_norms", {})
    record: dict[str, Any] = {
        "epoch": epoch,
        "map75": _map75(trainer),
        **_stock_losses(trainer),
        **private,
        "gradient_norm": _float(norms.get("gradient_norm")),
        "lpr_g_gradient_norm": (
            _float(norms.get("lpr_g_gradient_norm")) if variant == "lprg" else None
        ),
        "cuda_peak_mib": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 2)
            if torch.cuda.is_available()
            else 0.0
        ),
        "common_model_sha256": None,
        "common_optimizer_sha256": None,
    }
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is not None and isinstance(model, torch.nn.Module):
        record["common_model_sha256"] = common_model_fingerprint(model)
        record["common_optimizer_sha256"] = common_optimizer_fingerprint(model, optimizer)
    path = Path(trainer.save_dir) / "lpr_g_diagnostics.jsonl"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    _atomic_write(path, existing + json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
    return record


def validate_runtime_model(trainer) -> None:
    model = _unwrap_model(trainer.model)
    head = model.model[-1]
    if int(getattr(head, "nc", -1)) != 10:
        raise RuntimeError(f"paired RT-DETR head must have 10 classes, got {getattr(head, 'nc', None)}")
    if int(getattr(head, "num_queries", -1)) != 300:
        raise RuntimeError(
            "paired RT-DETR head must have exactly 300 queries, "
            f"got {getattr(head, 'num_queries', None)}"
        )


def reset_peak_memory(_trainer) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def publication_authority(args: argparse.Namespace) -> dict[str, Any]:
    """Return credential-free publication settings frozen into the run manifest."""
    prefix = f"{args.asset_prefix}-preflight" if args.preflight else args.asset_prefix
    return {
        "token_file": str(args.token_file.resolve()),
        "repo": args.repo,
        "repo_url": args.repo_url,
        "tag": args.tag,
        "source_branch": args.source_branch,
        "results_branch": args.results_branch,
        "results_repo": str(args.results_repo.resolve()),
        "asset_prefix": prefix,
        "retain": int(args.retain),
    }


def publication_config(trainer, args: argparse.Namespace) -> PublicationConfig:
    authority = publication_authority(args)
    return PublicationConfig(
        repo=authority["repo"],
        repo_url=authority["repo_url"],
        source_branch=authority["source_branch"],
        results_branch=authority["results_branch"],
        tag=authority["tag"],
        asset_prefix=authority["asset_prefix"],
        run_name=Path(trainer.save_dir).name,
        token_file=Path(authority["token_file"]),
        results_repo=Path(authority["results_repo"]),
        variant=args.variant,
        stage=args.stage,
        retain=authority["retain"],
    )


def write_common_state_audit(trainer) -> dict | None:
    epoch = int(trainer.epoch + 1)
    if epoch > int(trainer.args.epochs):
        return None
    return write_epoch_audit(
        Path(trainer.save_dir) / "common_state_audit.jsonl",
        epoch=epoch,
        model=trainer.model,
        optimizer=trainer.optimizer,
    )


def publish_current_epoch(trainer, *, args: argparse.Namespace) -> dict:
    """Publish the exact just-saved epoch and prune only after remote verification."""
    checkpoint = Path(trainer.save_dir) / "weights" / f"epoch{trainer.epoch}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"exact epoch checkpoint was not saved: {checkpoint}")
    record = publish_with_retry(
        Path(trainer.save_dir),
        checkpoint,
        publication_config(trainer, args),
    )
    expected_epoch = int(trainer.epoch + 1)
    if int(record.get("completed_epoch", -1)) != expected_epoch:
        raise RuntimeError(
            f"published epoch mismatch: expected={expected_epoch}, "
            f"actual={record.get('completed_epoch')}"
        )
    prune_local_epoch_checkpoints(
        checkpoint.parent,
        retain=int(args.retain),
    )
    return record


def write_protocol_manifest(
    trainer,
    authority: dict,
    args: argparse.Namespace,
    environment: dict,
) -> None:
    runtime_manifest = {
        "protocol": FROZEN_PROTOCOL,
        "authority": authority,
        "environment": environment,
        "variant": args.variant,
        "stage": args.stage,
        "seed": 0,
        "epochs": int(trainer.args.epochs),
        "workers": int(trainer.args.workers),
        "device": str(trainer.args.device),
        "initial_state": str(args.initial_state.resolve()),
        "publication": publication_authority(args),
    }
    _atomic_write(
        Path(trainer.save_dir) / "lpr_g_protocol.json",
        json.dumps(runtime_manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


def main() -> None:
    args = build_parser().parse_args()
    args.token_file = args.token_file.resolve()
    args.results_repo = args.results_repo.resolve()
    validate_token_file(args.token_file)
    authority = json.loads(args.protocol_manifest.read_text(encoding="utf-8"))
    environment = current_environment()
    current_dataset = dataset_signature(Path(authority["dataset_root"]))
    validate_launch_authority(args, authority, environment, current_dataset)
    validate_lpr_g_initial_state_file(
        args.initial_state,
        manifest_record=authority.get("initial_state", {}),
    )
    validate_resume_authority(args, authority, environment)

    trainer_kwargs = {
        "overrides": build_settings(args, authority),
        "initial_state_path": args.initial_state,
    }
    if args.variant == "lprg":
        trainer = LPRGTrainer(**trainer_kwargs, experiment_seed=0)
    else:
        trainer = LPRGControlTrainer(**trainer_kwargs)

    trainer.add_callback("on_train_start", validate_runtime_model)
    trainer.add_callback(
        "on_train_start",
        lambda current: write_protocol_manifest(current, authority, args, environment),
    )
    trainer.add_callback("on_train_epoch_start", reset_peak_memory)
    trainer.add_callback(
        "on_model_save",
        lambda current: write_lpr_g_diagnostics(current, variant=args.variant),
    )
    trainer.add_callback("on_model_save", write_common_state_audit)
    trainer.add_callback(
        "on_model_save",
        lambda current: publish_current_epoch(current, args=args),
    )
    trainer.train()


if __name__ == "__main__":
    main()
