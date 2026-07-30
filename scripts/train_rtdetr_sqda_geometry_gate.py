from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint_recovery import validate_checkpoint
from src.rtdetr_sqda_sgc import (
    BASELINE_SHA256,
    SQDAGeometryTrustTrainer,
    sha256_file,
)
from ultralytics.utils.torch_utils import unwrap_model


RUN_NAMES = {
    "g1": "sqda-geometry-gate-g1-seed0-3ep",
    "g2": "sqda-geometry-gate-g2-seed0-10ep",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train only SQDA's new geometry-trust MLP from a retained G2 adapter."
    )
    parser.add_argument("--gate", choices=tuple(RUN_NAMES), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume-from", type=Path)
    return parser


def build_settings(args: argparse.Namespace) -> dict:
    return {
        "model": "rtdetr-l.yaml",
        "data": str(args.data.expanduser().resolve()),
        "epochs": 3 if args.gate == "g1" else 10,
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
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest().upper()


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def prepare_manifest(args: argparse.Namespace, settings: dict) -> Path:
    checkpoint = args.checkpoint.expanduser().resolve()
    adapter_checkpoint = args.adapter_checkpoint.expanduser().resolve()
    data = args.data.expanduser().resolve()
    if not checkpoint.is_file() or not adapter_checkpoint.is_file() or not data.is_file():
        raise FileNotFoundError("baseline checkpoint, inherited adapter checkpoint, and data YAML are required")
    if sha256_file(checkpoint) != BASELINE_SHA256:
        raise ValueError("baseline SHA256 mismatch")
    resume_from = args.resume_from.expanduser().resolve() if args.resume_from else None
    if resume_from is not None:
        valid, reason = validate_checkpoint(resume_from)
        if not valid:
            raise ValueError(f"resume checkpoint is not valid: {resume_from} ({reason})")
    run_dir = args.project.expanduser().resolve() / RUN_NAMES[args.gate]
    manifest_path = run_dir / "run-manifest.json"
    if run_dir.exists() and any(run_dir.iterdir()):
        if resume_from is None:
            raise FileExistsError(f"refusing to overwrite non-empty run directory: {run_dir}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"resume run is missing its manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("gate") != args.gate:
            raise ValueError("resume gate does not match existing run manifest")
        if manifest.get("inherited_adapter", {}).get("sha256") != sha256_file(adapter_checkpoint):
            raise ValueError("resume inherited adapter SHA256 does not match existing run")
        manifest["resume_from"] = str(resume_from)
        manifest["resume_checkpoint_sha256"] = sha256_file(resume_from)
        manifest["resume_updated_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(manifest_path, manifest)
        return manifest_path

    run_dir.mkdir(parents=True, exist_ok=True)
    packages = {
        package: importlib.metadata.version(package)
        for package in ("torch", "torchvision", "ultralytics")
    }
    manifest = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": args.gate,
        "git_sha": _git_sha(),
        "baseline": {"path": str(checkpoint), "sha256": BASELINE_SHA256},
        "inherited_adapter": {
            "path": str(adapter_checkpoint),
            "sha256": sha256_file(adapter_checkpoint),
        },
        "dataset": {"yaml": str(data), "yaml_sha256": sha256_file(data)},
        "settings": settings,
        "settings_sha256": _json_hash(settings),
        "training_scope": "sqda_sgc.geometry_trust only",
        "packages": packages,
        "python": sys.version,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def _state_hash(state: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if not hasattr(value, "detach"):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest().upper()


def record_frozen_tensor_baseline(trainer: SQDAGeometryTrustTrainer) -> None:
    model = unwrap_model(trainer.model)
    trainer.geometry_frozen_audit = {
        "stock": _state_hash(model.model.state_dict()),
        "inherited_adapter": _state_hash(
            {
                key: value
                for key, value in model.sqda_sgc.state_dict().items()
                if not key.startswith("geometry_trust.")
            }
        ),
    }


def record_epoch_diagnostics(trainer: SQDAGeometryTrustTrainer) -> None:
    model = unwrap_model(trainer.model)
    frozen = getattr(trainer, "geometry_frozen_audit", None)
    if frozen is None:
        raise RuntimeError("frozen tensor audit was not initialized")
    current_stock = _state_hash(model.model.state_dict())
    current_inherited = _state_hash(
        {
            key: value
            for key, value in model.sqda_sgc.state_dict().items()
            if not key.startswith("geometry_trust.")
        }
    )
    if current_stock != frozen["stock"] or current_inherited != frozen["inherited_adapter"]:
        raise AssertionError("stock or inherited SQDA tensors changed during geometry-only training")
    diagnostics = getattr(model, "last_sqda_diagnostics", None) or {}
    payload = {
        "completed_epoch": min(int(trainer.epoch) + 1, int(trainer.epochs)),
        "module_gradient_norm_before_clip": trainer.last_module_gradient_norm,
        "frozen_stock_sha256": current_stock,
        "frozen_inherited_adapter_sha256": current_inherited,
    }
    for key in (
        "geometry_budget",
        "geometry_features",
        "pre_saturation_rms",
        "post_saturation_rms",
        "residual_norm",
    ):
        value = diagnostics.get(key)
        if value is not None:
            value = value.detach().float()
            payload[f"{key}_min"] = float(value.min().cpu())
            payload[f"{key}_mean"] = float(value.mean().cpu())
            payload[f"{key}_max"] = float(value.max().cpu())
    destination = Path(trainer.save_dir) / "geometry_gate_diagnostics.jsonl"
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    _write_json_atomic(
        Path(trainer.save_dir) / "frozen-tensor-audit.json",
        {"passed": True, **payload},
    )


def record_stage_status(trainer: SQDAGeometryTrustTrainer) -> None:
    payload = {
        "gate": trainer.geometry_gate,
        "completed_epoch": min(int(trainer.epoch) + 1, int(trainer.epochs)),
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
    trainer = SQDAGeometryTrustTrainer(
        overrides=settings,
        baseline_checkpoint=args.checkpoint,
        baseline_sha256=BASELINE_SHA256,
        adapter_checkpoint=args.adapter_checkpoint,
        adapter_sha256=sha256_file(args.adapter_checkpoint),
        manifest_path=manifest_path,
    )
    trainer.geometry_gate = args.gate
    trainer.add_callback("on_train_start", record_frozen_tensor_baseline)
    trainer.add_callback("on_train_epoch_end", record_epoch_diagnostics)
    trainer.add_callback("on_fit_epoch_end", record_stage_status)
    trainer.train()


if __name__ == "__main__":
    main()
