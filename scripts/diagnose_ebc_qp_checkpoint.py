from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rtdetr_ebc_qp import EBCQPDiagnosticsValidator, validate_ebc_qp_checkpoint_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Revalidate one EBC-QP checkpoint with read-only mechanism diagnostics.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("ebc_qp")
    model = checkpoint.get("ema")
    if model is None:
        model = checkpoint.get("model")
    if metadata is None or model is None:
        raise SystemExit("checkpoint must contain the EBC-QP metadata and model")
    model = model.float()
    validate_ebc_qp_checkpoint_metadata(metadata, model.ebc_config)
    model.set_ebc_progress(int(metadata["ebc_epoch"]))
    model.ebc_head.diagnostics_enabled = True

    validator = EBCQPDiagnosticsValidator(
        args={
            "model": str(args.checkpoint),
            "data": str(args.data),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "quantize": 16,
            "nms": False,
            "max_det": 300,
            "plots": False,
            "save_json": False,
            "split": "val",
        }
    )
    validator.attach_diagnostic_model(model)
    metrics = validator(model=model)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "ebc_epoch": int(metadata["ebc_epoch"]),
        "competition_active": bool(model.ebc_head.competition_active),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "mechanism": validator.mechanism_stats,
        "speed_ms_per_image": {key: float(value) for key, value in validator.speed.items()},
        "protocol": {
            "batch": args.batch,
            "workers": args.workers,
            "imgsz": args.imgsz,
            "nms": False,
            "max_det": 300,
            "quantize": 16,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, separators=(",", ":")))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    main()
