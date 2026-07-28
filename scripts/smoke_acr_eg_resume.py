"""Run one real VisDrone batch through the resumed integrated ACR-EG model."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_gcte_formal import (
    DEFAULT_CONFIG,
    DEFAULT_DATA,
    MATURE_BASELINE_SHA256,
    build_settings,
    build_trainer_overrides,
)
from src.acr_eg_release import inspect_acr_eg_checkpoint
from src.acr_eg_smoke import inspect_acr_eg_gradients, validate_multiview_batch
from src.github_checkpoint_sync import sha256_file


EPOCH8_SHA256 = "802D72326F4B8FEE55C0FF8818A5B96B7445CBEE34F5C1ED9002A6D3E6771FE6"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit one real batch before formal ACR-EG continuation."
    )
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--baseline-sha256", default=MATURE_BASELINE_SHA256)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--resume-sha256", default=EPOCH8_SHA256)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", default="acr-eg-integrated-resume-smoke")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-start-epoch", type=int, default=9)
    return parser


def _formal_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model="rtdetr-l.yaml",
        config=args.config,
        data=args.data,
        epochs=100,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        project=args.project,
        name=args.name,
        module="",
        module_sha256="",
        baseline_checkpoint=args.baseline_checkpoint,
        baseline_sha256=args.baseline_sha256,
        resume=args.resume,
        dry_run=False,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _assert_source_commit(expected: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("ACR_EG_SMOKE_SOURCE_COMMIT_INVALID")
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    if actual != expected:
        raise RuntimeError("ACR_EG_SMOKE_SOURCE_CHECKOUT_MISMATCH")


def main() -> None:
    args = build_parser().parse_args()
    if args.expected_start_epoch != 9:
        raise ValueError("ACR_EG_SMOKE_START_EPOCH_PROTOCOL_DRIFT")
    _assert_source_commit(args.source_commit)
    settings = build_settings(_formal_args(args))
    baseline = Path(args.baseline_checkpoint).resolve()
    resume = Path(args.resume).resolve()
    if sha256_file(baseline).upper() != str(args.baseline_sha256).upper():
        raise ValueError("ACR_EG_SMOKE_BASELINE_SHA256_MISMATCH")
    if sha256_file(resume).upper() != str(args.resume_sha256).upper():
        raise ValueError("ACR_EG_SMOKE_RESUME_SHA256_MISMATCH")
    checkpoint_metadata = inspect_acr_eg_checkpoint(
        resume,
        expected_completed_epoch=args.expected_start_epoch,
    )

    run_path = Path(settings["project"]) / args.name
    if run_path.exists():
        raise FileExistsError(run_path)
    os.environ["GCTE_ACR_EG_BASELINE"] = str(baseline)
    os.environ["GCTE_ACR_EG_YAML"] = settings["gcte_config"]

    from ultralytics.utils.torch_utils import unwrap_model
    from src.rtdetr_acr_eg import ACREGDetectionModel, ACREGFormalTrainer

    trainer = ACREGFormalTrainer(overrides=build_trainer_overrides(settings))
    trainer._setup_train()
    model = unwrap_model(trainer.model)
    if not isinstance(model, ACREGDetectionModel):
        raise RuntimeError("ACR_EG_SMOKE_MODEL_IDENTITY_MISMATCH")
    state = model.state_dict()
    acr_keys = [key for key in state if key.startswith("acr_eg.")]
    if len(acr_keys) != 48:
        raise RuntimeError("ACR_EG_SMOKE_STATE_IDENTITY_MISMATCH")
    if trainer.start_epoch != args.expected_start_epoch:
        raise RuntimeError("ACR_EG_SMOKE_START_EPOCH_MISMATCH")
    optimizer_state = trainer.optimizer.state_dict().get("state", {})
    if not optimizer_state:
        raise RuntimeError("ACR_EG_SMOKE_OPTIMIZER_STATE_EMPTY")
    scaler_state = trainer.scaler.state_dict()
    if float(trainer.scaler.get_scale()) != 128.0:
        raise RuntimeError("ACR_EG_SMOKE_SCALER_SCALE_MISMATCH")
    if int(scaler_state.get("growth_interval", -1)) != 2**31 - 1:
        raise RuntimeError("ACR_EG_SMOKE_SCALER_GROWTH_INTERVAL_MISMATCH")

    raw_batch = next(iter(trainer.train_loader))
    batch_evidence = validate_multiview_batch(raw_batch)
    torch.cuda.reset_peak_memory_stats(trainer.device)
    trainer.optimizer.zero_grad(set_to_none=True)
    model.train()
    batch = trainer.preprocess_batch(raw_batch)
    with torch.amp.autocast("cuda", enabled=bool(trainer.amp)):
        loss, loss_items = trainer.model(batch)
        total_loss = loss.sum()
    if not torch.isfinite(total_loss):
        raise FloatingPointError("ACR_EG_SMOKE_LOSS_NONFINITE")
    if not torch.isfinite(loss_items).all():
        raise FloatingPointError("ACR_EG_SMOKE_LOSS_ITEMS_NONFINITE")
    if model.last_acr_eg_output is None:
        raise RuntimeError("ACR_EG_SMOKE_MULTIVIEW_FORWARD_NOT_EXECUTED")
    trainer.scaler.scale(total_loss).backward()
    gradient_evidence = inspect_acr_eg_gradients(
        model,
        expected_parameter_count=48,
    )
    torch.cuda.synchronize(trainer.device)
    peak_vram = int(torch.cuda.max_memory_allocated(trainer.device))
    if peak_vram <= 0:
        raise RuntimeError("ACR_EG_SMOKE_GPU_MEMORY_UNUSED")

    evidence = {
        "schema_version": "gcte-acr-eg-resume-smoke/v1",
        "state": "passed",
        "source_commit": args.source_commit,
        "model": {
            "type": type(model).__name__,
            "state_key_count": len(state),
            "acr_eg_key_count": len(acr_keys),
        },
        "resume": {
            "checkpoint": str(resume),
            "sha256": checkpoint_metadata.sha256,
            "checkpoint_epoch": checkpoint_metadata.checkpoint_epoch,
            "start_epoch": trainer.start_epoch,
            "optimizer_state_entries": len(optimizer_state),
            "scaler_scale": float(trainer.scaler.get_scale()),
            "scaler_growth_interval": int(scaler_state["growth_interval"]),
            "updates": checkpoint_metadata.updates,
        },
        "batch": batch_evidence,
        "forward_backward": {
            "loss": float(total_loss.detach().cpu()),
            "loss_items": [
                float(value) for value in loss_items.detach().cpu().reshape(-1)
            ],
            **gradient_evidence,
            "peak_vram_bytes": peak_vram,
        },
    }
    evidence_path = (
        args.evidence.resolve()
        if args.evidence is not None
        else Path(trainer.save_dir) / "resume-smoke-evidence.json"
    )
    _write_json(evidence_path, evidence)
    print(f"ACR_EG_RESUME_SMOKE_PASSED {evidence_path}", flush=True)


if __name__ == "__main__":
    main()
