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

from src.rtdetr_sqda_sgc import BASELINE_SHA256, SQDASGCTrainer, sha256_file


RUN_NAMES = {
    "g1": "sqda-sgc-g1-seed0-3ep",
    "g2": "sqda-sgc-g2-seed0-10ep",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a pre-registered frozen-stock SQDA-SGC RT-DETR-L gate."
    )
    parser.add_argument("--gate", choices=("g1", "g2"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
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
        "resume": False,
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
        "warmup_bias_lr": 0.0,
        "cos_lr": False,
        "freeze": list(range(29)),
        "mosaic": 1.0,
        "mixup": 0.0,
        "scale": 0.5,
        "translate": 0.1,
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

    run_dir = args.project.expanduser().resolve() / RUN_NAMES[args.gate]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty pre-registered run directory: {run_dir}"
        )
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
    return manifest_path


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
    trainer.train()


if __name__ == "__main__":
    main()
