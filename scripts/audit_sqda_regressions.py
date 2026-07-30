from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_sqda_small_ap import build_coco_dataset
from src.sqda_error_audit import compare_error_summaries, summarize_detection_errors


CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"{resolved} must contain a JSON array of prediction objects")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit fixed-threshold SQDA detection regressions without training."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=548)
    parser.add_argument("--expected-annotations", type=int, default=38759)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = build_coco_dataset(args.images, args.labels)
    if len(dataset["images"]) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, found {len(dataset['images'])}")
    if len(dataset["annotations"]) != args.expected_annotations:
        raise RuntimeError(
            f"expected {args.expected_annotations} annotations, found {len(dataset['annotations'])}"
        )
    baseline = summarize_detection_errors(
        dataset,
        _load_predictions(args.baseline_predictions),
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    )
    candidate = summarize_detection_errors(
        dataset,
        _load_predictions(args.candidate_predictions),
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    )
    report = {
        "protocol": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "matching": "class-aware_score-descending_greedy",
            "training_signal": False,
            "images": len(dataset["images"]),
            "annotations": len(dataset["annotations"]),
        },
        "baseline": baseline,
        "candidate": candidate,
        "delta": compare_error_summaries(baseline, candidate),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
