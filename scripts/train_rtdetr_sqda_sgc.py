from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint_recovery import validate_checkpoint
from src.rtdetr_sqda_sgc import BASELINE_SHA256, SQDASGCTrainer, sha256_file
from ultralytics.utils.torch_utils import unwrap_model


RUN_NAMES = {
    "g1": "sqda-sgc-g1-seed0-3ep",
    "g1r": "sqda-sgc-g1r-seed0-3ep",
    "g2": "sqda-sgc-g2-seed0-10ep",
    "formal": "sqda-sgc-formal-seed0-100ep",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a pre-registered frozen-stock SQDA-SGC RT-DETR-L gate."
    )
    parser.add_argument("--gate", choices=("g1", "g1r", "g2", "formal"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--target-epochs", type=int)
    return parser


def build_settings(args: argparse.Namespace) -> dict:
    return {
        "model": "rtdetr-l.yaml",
        "data": str(args.data.expanduser().resolve()),
        "epochs": (
            args.target_epochs
            if args.target_epochs is not None
            else 100
            if args.gate == "formal"
            else 10
            if args.gate == "g2"
            else 3
        ),
        "imgsz": 640,
        "batch": 8,
        "workers": args.workers,
        "device": args.device,
        "project": str(args.project.expanduser().resolve()),
        "name": RUN_NAMES[args.gate],
        "exist_ok": True,
        "pretrained": False,
        "resume": (
            str(args.resume_from.expanduser().resolve())
            if args.resume_from is not None
            else False
        ),
        "cache": False,
        "amp": True,
        "deterministic": True,
        "seed": 0,
        "nbs": 64,
        "nms": False,
        "max_det": 300,
        "save": True,
        "save_period": 1,
        "optimizer": "AdamW",
        "lr0": 1e-4,
        "lrf": 1.0,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "warmup_epochs": 0.5,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "cos_lr": False,
        "freeze": list(range(29)),
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "scale": 0.5,
        "translate": 0.1,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "plots": True,
        "val": True,
    }


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest().upper()


def prepare_manifest(
    args: argparse.Namespace,
    settings: dict,
) -> Path:
    checkpoint = args.checkpoint.expanduser().resolve()
    data = args.data.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not data.is_file():
        raise FileNotFoundError(data)
    actual_baseline_sha = sha256_file(checkpoint)
    if actual_baseline_sha != BASELINE_SHA256:
        raise ValueError(
            f"baseline SHA256 mismatch: expected {BASELINE_SHA256}, got {actual_baseline_sha}"
        )

    resume_from = args.resume_from.expanduser().resolve() if args.resume_from else None
    if resume_from is not None:
        valid, reason = validate_checkpoint(resume_from)
        if not valid:
            raise ValueError(f"resume checkpoint is not valid: {resume_from} ({reason})")

    run_dir = args.project.expanduser().resolve() / RUN_NAMES[args.gate]
    if run_dir.exists() and any(run_dir.iterdir()):
        if resume_from is None:
            raise FileExistsError(
                f"refusing to overwrite non-empty pre-registered run directory: {run_dir}"
            )
        manifest_path = run_dir / "run-manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"resume run is missing its manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("gate") != args.gate:
            raise ValueError("resume gate does not match the existing run manifest")
        if manifest.get("baseline", {}).get("sha256") != actual_baseline_sha:
            raise ValueError("resume baseline SHA256 does not match the existing run")
        manifest["resume_from"] = str(resume_from)
        manifest["resume_checkpoint_sha256"] = sha256_file(resume_from)
        manifest["resume_updated_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest_path
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run-manifest.json"
    packages = {}
    for package in ("torch", "torchvision", "ultralytics"):
        packages[package] = importlib.metadata.version(package)
    manifest = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": args.gate,
        "git_sha": _git_sha(),
        "baseline": {
            "path": str(checkpoint),
            "sha256": actual_baseline_sha,
        },
        "dataset": {
            "yaml": str(data),
            "yaml_sha256": sha256_file(data),
        },
        "settings": settings,
        "settings_sha256": _json_hash(settings),
        "packages": packages,
        "python": sys.version,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    inherited_artifacts = {
        "g0-equivalence.json": args.project.expanduser().resolve() / "g0-equivalence.json",
        "top300-diagnostic.json": args.project.expanduser().resolve() / "top300-diagnostic.json",
        "cuda-smoke.json": args.project.expanduser().resolve() / "cuda-smoke.json",
        "cuda-smoke-g1r.json": args.project.expanduser().resolve() / "cuda-smoke-g1r.json",
        "input-preflight.json": args.project.expanduser().resolve() / "input-preflight.json",
    }
    for name, source in inherited_artifacts.items():
        if source.is_file():
            shutil.copy2(source, run_dir / name)
    return manifest_path


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def record_epoch_diagnostics(trainer: SQDASGCTrainer) -> None:
    model = unwrap_model(trainer.model)
    diagnostics = getattr(model, "last_sqda_diagnostics", None) or {}
    adapter = model.sqda_sgc
    payload = {
        "completed_epoch": int(trainer.epoch) + 1,
        "module_gradient_norm_before_clip": trainer.last_module_gradient_norm,
        "layer_scale": float(adapter.layer_scale.detach().cpu()),
        "context_strength": float(adapter.context_strength.detach().cpu()),
    }
    for key in (
        "sampling_validity",
        "context_reliability",
        "semantic_similarity",
        "geometry_similarity",
        "context_similarity",
        "residual_norm",
        "group_gates",
    ):
        value = diagnostics.get(key)
        if value is not None:
            value = value.detach().float()
            payload[f"{key}_mean"] = float(value.mean().cpu())
            payload[f"{key}_max"] = float(value.max().cpu())
    destination = Path(trainer.save_dir) / "sqda_sgc_diagnostics.jsonl"
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def record_stage_status(trainer: SQDASGCTrainer) -> None:
    payload = {
        "gate": getattr(
            trainer,
            "sqda_gate",
            "g1" if int(trainer.epochs) == 3 else "g2",
        ),
        "completed_epoch": int(trainer.epoch) + 1,
        "target_epochs": int(trainer.epochs),
        "metrics": {
            key: float(value)
            for key, value in (trainer.metrics or {}).items()
            if isinstance(value, (int, float))
        },
        "fitness": float(trainer.fitness) if trainer.fitness is not None else None,
        "best_fitness": float(trainer.best_fitness),
    }
    _write_json_atomic(Path(trainer.save_dir) / "stage-status.json", payload)


def main() -> None:
    args = build_parser().parse_args()
    settings = build_settings(args)
    manifest_path = prepare_manifest(args, settings)
    trainer = SQDASGCTrainer(
        overrides=settings,
        baseline_checkpoint=args.checkpoint,
        baseline_sha256=BASELINE_SHA256,
        manifest_path=manifest_path,
    )
    trainer.sqda_gate = args.gate
    trainer.add_callback("on_train_epoch_end", record_epoch_diagnostics)
    trainer.add_callback("on_fit_epoch_end", record_stage_status)
    trainer.train()


if __name__ == "__main__":
    main()
