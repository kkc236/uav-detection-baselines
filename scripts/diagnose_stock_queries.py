from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rtdetr_ebc_qp import StockQueryDiagnosticsValidator, validate_ebc_qp_checkpoint_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure native RT-DETR stock tiny-query coverage and rank.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0", choices=("0",))
    parser.add_argument("--batch", type=int, default=8, choices=(8,))
    parser.add_argument("--workers", type=int, default=8, choices=(8,))
    parser.add_argument("--imgsz", type=int, default=640, choices=(640,))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to replace stock-query diagnostics: {args.output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema")
    if model is None:
        model = checkpoint.get("model")
    if model is None:
        raise SystemExit("checkpoint has no EMA or model")
    model = model.float()
    metadata = checkpoint.get("ebc_qp")
    if metadata is not None:
        if not hasattr(model, "ebc_config"):
            raise SystemExit("EBC-QP metadata/model mismatch")
        validate_ebc_qp_checkpoint_metadata(metadata, model.ebc_config)
        model.set_ebc_progress(int(metadata["ebc_epoch"]))

    validator = StockQueryDiagnosticsValidator(
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
        "format_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "stock_query": validator.stock_query_stats,
        "protocol": {
            "batch": args.batch,
            "workers": args.workers,
            "imgsz": args.imgsz,
            "nms": False,
            "max_det": 300,
            "rank_definition": "best 1-based global stock score rank per tiny GT; missing=N+1; normalized=rank/N",
        },
    }
    result["signature"] = _json_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, separators=(",", ":")))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _json_sha256(payload: object) -> str:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256(content).hexdigest().upper()


if __name__ == "__main__":
    main()
