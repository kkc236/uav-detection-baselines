"""Export CSHC raw C2 candidates and frozen BQP misses for the coverage gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cshc_coverage import export_from_checkpoint, load_frozen_ledger


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export raw pre-Top-300 CSHC candidates for the frozen BQP ledger.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True, help="Frozen BQP g0_images.jsonl input.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", default="configs/rtdetr-l-cshc.yaml")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=False)
    ledger_path = args.ledger.resolve()
    ledger = load_frozen_ledger(ledger_path)
    misses, candidates = export_from_checkpoint(
        checkpoint=args.checkpoint.resolve(),
        config=args.config,
        data=args.data.resolve(),
        ledger=ledger,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
    )
    misses_path = output / "frozen_stock_misses.jsonl"
    candidates_path = output / "new_c2_candidates.jsonl"
    _write_jsonl(misses_path, misses)
    _write_jsonl(candidates_path, candidates)
    manifest = {
        "protocol": "frozen_bqp_misses_vs_raw_pre_top300_c2_candidates",
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": _sha256(args.checkpoint.resolve())},
        "ledger": {"path": str(ledger_path), "sha256": _sha256(ledger_path)},
        "frozen_miss_count": len(misses),
        "candidate_count": len(candidates),
        "files": {
            "frozen_stock_misses": {"path": str(misses_path), "sha256": _sha256(misses_path)},
            "new_c2_candidates": {"path": str(candidates_path), "sha256": _sha256(candidates_path)},
        },
    }
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
