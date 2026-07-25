"""Frozen single-endpoint bridge for fresh-stock SADED postprocessing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.saded_stage import CACHE_ROW_KEYS, route_paired_caches


GT_FIELDS = {
    "gt_boxes",
    "gt_classes",
    "ignore_boxes",
    "annotations",
}


def route_single_cache(
    cache_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the frozen router to one checkpoint cache without GT."""

    if not cache_rows:
        raise ValueError("SADED single cache row count drift")
    for row in cache_rows:
        if (
            set(row) != CACHE_ROW_KEYS
            or not isinstance(row.get("image_id"), str)
            or not row["image_id"]
            or int(row.get("width", 0)) <= 0
            or int(row.get("height", 0)) <= 0
        ):
            raise ValueError("SADED single cache identity drift")
    paired, paired_invariants = route_paired_caches(
        cache_rows,
        cache_rows,
    )
    routed: list[dict[str, Any]] = []
    single_endpoint_exact = True
    per_image_passed = True
    for row in paired:
        control = row["arms"]["route_control"]
        duplicate = row["arms"]["route_treatment"]
        same_route = control == duplicate
        same_coverage = (
            row["coverage"]["control"] == row["coverage"]["treatment"]
        )
        control_invariants = row["invariants"]["control"]
        row_passed = (
            same_route
            and same_coverage
            and control_invariants.get("passed") is True
        )
        single_endpoint_exact = (
            single_endpoint_exact and same_route and same_coverage
        )
        per_image_passed = per_image_passed and row_passed
        routed.append(
            {
                "image_id": row["image_id"],
                "width": row["width"],
                "height": row["height"],
                "arms": {
                    "A": row["arms"]["A"],
                    "route_control": control,
                },
                "coverage": row["coverage"]["control"],
                "invariants": {
                    **control_invariants,
                    "single_endpoint_exact": same_route and same_coverage,
                    "passed": row_passed,
                },
            }
        )
    gt_fields_absent = all(
        not GT_FIELDS.intersection(row)
        and all(
            not GT_FIELDS.intersection(prediction)
            for predictions in row["arms"].values()
            for prediction in predictions
        )
        for row in routed
    )
    invariants = {
        "image_count": len(routed),
        "paired_bridge_passed": paired_invariants.get("passed") is True,
        "single_endpoint_exact": single_endpoint_exact,
        "per_image_passed": per_image_passed,
        "gt_fields_absent": gt_fields_absent,
    }
    invariants["passed"] = all(
        value is True
        for key, value in invariants.items()
        if key != "image_count"
    )
    return routed, invariants


__all__ = ["GT_FIELDS", "route_single_cache"]
