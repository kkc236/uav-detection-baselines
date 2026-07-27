"""Seal and validate Fixed SADED on the current 4090 baseline before training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from scripts.evaluate_gcqf_g0 import (
    _dataset_key,
    _metric_row,
    _prediction_json,
    metric_deltas,
)
from src.gcqf_cache import VerifiedEvidenceCache
from src.gcqf_routing import route_gcqf_record
from src.sbr_artifacts import atomic_write_json, load_dataset, sha256_file
from src.sbr_metrics import evaluate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="G0-A: seal a strong Fixed-SADED anchor."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=548)
    parser.add_argument("--stage", default="G0-A")
    return parser


def anchor_gate(
    *,
    global_metrics: Mapping[str, Any],
    anchor_metrics: Mapping[str, Any],
    protected_exact: bool,
) -> dict[str, bool]:
    delta = metric_deltas(global_metrics, anchor_metrics)
    checks = {
        "map_anchor_gain": delta.get("mAP50-95", -1.0) >= 0.005,
        "tiny_anchor_gain": (
            delta.get("AP-tiny-SBR", -1.0) >= 0.010
        ),
        "tiny_recall_anchor_gain": (
            delta.get("tiny_recall", -1.0) >= 0.020
        ),
        "medium_anchor_budget": (
            delta.get("AP-medium-SBR", -1.0) >= -0.002
        ),
        "large_anchor_budget": (
            delta.get("AP-large-SBR", -1.0) >= -0.005
        ),
        "protected_global_exact": bool(protected_exact),
    }
    checks["advance_to_training"] = all(checks.values())
    return checks


def validate(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.stage != "G0-A" or args.expected_images != 548:
        raise ValueError("GCQF anchor validation protocol drift")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cache = VerifiedEvidenceCache(args.cache)
    records = list(cache.iter_records())
    dataset = load_dataset(args.data, split="val")
    if (
        len(records) != args.expected_images
        or dataset["image_count"] != args.expected_images
    ):
        raise RuntimeError("GCQF anchor image count drift")
    if (
        cache.manifest["dataset_signature"].upper()
        != dataset["dataset_signature"].upper()
    ):
        raise RuntimeError("GCQF anchor dataset signature drift")
    image_by_id = {
        image["relative_path"]: image for image in dataset["images"]
    }
    global_rows = []
    raw_rows = []
    anchor_rows = []
    reference_rows = []
    protected_exact = True
    for record in records:
        routed = route_gcqf_record(record, score_residual=None)
        key = _dataset_key(record.image_id)
        if key not in image_by_id:
            raise RuntimeError(f"anchor image identity drift: {key}")
        image = image_by_id[key]
        global_rows.append(_metric_row(image, routed.control))
        raw_rows.append(_metric_row(image, routed.raw_union))
        anchor_rows.append(_metric_row(image, routed.output))
        reference_rows.append(
            {
                "image_id": record.image_id,
                "predictions": _prediction_json(routed.output),
            }
        )
        protected_exact = protected_exact and bool(
            routed.invariants.get("protected_identity_exact")
        )
    metrics = {
        "Global": evaluate_dataset(global_rows),
        "Raw-Union": evaluate_dataset(raw_rows),
        "Fixed-SADED": evaluate_dataset(anchor_rows),
    }
    gate = anchor_gate(
        global_metrics=metrics["Global"],
        anchor_metrics=metrics["Fixed-SADED"],
        protected_exact=protected_exact,
    )
    reference_path = atomic_write_json(
        output / "anchor-reference.json",
        {
            "schema_version": "gcte-fixed-saded-anchor-reference/v1",
            "baseline_sha256": cache.manifest["baseline_sha256"],
            "dataset_signature": cache.manifest["dataset_signature"],
            "rows": reference_rows,
        },
    )
    result_path = atomic_write_json(
        output / "anchor-evaluation.json",
        {
            "schema_version": "gcte-fixed-saded-anchor-evaluation/v1",
            "cache_manifest_sha256": sha256_file(args.cache),
            "anchor_reference_sha256": sha256_file(reference_path),
            "metrics": metrics,
            "anchor_minus_global": metric_deltas(
                metrics["Global"],
                metrics["Fixed-SADED"],
            ),
            "raw_union_minus_global": metric_deltas(
                metrics["Global"],
                metrics["Raw-Union"],
            ),
            "gate": gate,
        },
    )
    print(
        f"GCQF_ANCHOR_COMPLETE advance={gate['advance_to_training']} "
        f"result={result_path}",
        flush=True,
    )
    return result_path, gate["advance_to_training"]


def main() -> None:
    path, passed = validate(build_parser().parse_args())
    print(path)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
