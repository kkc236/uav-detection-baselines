"""Audit coverage of frozen stock misses by *new* CSHC C2 candidates only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _validate_box(record: dict[str, Any]) -> tuple[float, float, float, float]:
    box = record.get("box")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("each record requires a four-value xyxy 'box' list")
    values = tuple(float(value) for value in box)
    if not (values[0] < values[2] and values[1] < values[3]):
        raise ValueError(f"box must be xyxy with positive area, got {box}")
    return values


def _iou_xyxy(left: list[float], right: list[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    intersection_w = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    intersection_h = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = intersection_w * intersection_h
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return intersection / union if union > 0.0 else 0.0


def summarize_new_candidate_coverage(
    missed: Iterable[dict[str, Any]], candidates: Iterable[dict[str, Any]], iou_threshold: float = 0.5
) -> dict[str, int | float]:
    """Class-aware one-to-one C2-candidate coverage of a frozen stock-miss list."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    misses = list(missed)
    proposals = list(candidates)
    for record in [*misses, *proposals]:
        if "image_id" not in record or "class_id" not in record:
            raise ValueError("each record requires image_id and class_id")
        _validate_box(record)

    by_key: dict[tuple[Any, Any], list[int]] = {}
    for index, proposal in enumerate(proposals):
        by_key.setdefault((proposal["image_id"], proposal["class_id"]), []).append(index)
    used: set[int] = set()
    covered = 0
    for miss in misses:
        best_index: int | None = None
        best_iou = iou_threshold
        for index in by_key.get((miss["image_id"], miss["class_id"]), []):
            if index in used:
                continue
            overlap = _iou_xyxy(miss["box"], proposals[index]["box"])
            if overlap >= best_iou:
                best_iou = overlap
                best_index = index
        if best_index is not None:
            used.add(best_index)
            covered += 1
    total = len(misses)
    return {
        "missed_tiny": total,
        "covered_by_new_candidates": covered,
        "coverage": covered / total if total else 0.0,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit frozen stock misses against new CSHC C2 candidates only.")
    parser.add_argument("--stock-misses", type=Path, required=True, help="Frozen JSONL of stock-model missed tiny GT boxes.")
    parser.add_argument("--c2-candidates", type=Path, required=True, help="JSONL exported before combined Top-300 selection.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stock_misses = args.stock_misses.resolve()
    c2_candidates = args.c2_candidates.resolve()
    result = summarize_new_candidate_coverage(
        _read_jsonl(stock_misses), _read_jsonl(c2_candidates), iou_threshold=args.iou_threshold
    )
    record = {
        "protocol": "new_c2_candidates_only_class_aware_one_to_one_iou",
        "iou_threshold": args.iou_threshold,
        "stock_misses": {"path": str(stock_misses), "sha256": _sha256(stock_misses)},
        "c2_candidates": {"path": str(c2_candidates), "sha256": _sha256(c2_candidates)},
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
