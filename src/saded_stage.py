"""Pure staged routing and gates for the frozen SADED experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from src.saded import ExpertCandidate, route_saded_image
from src.sbr_fusion import Detection
from src.tascv_protocol import FROZEN_SCREEN_GATE


PREDICTION_KEYS = {
    "box",
    "global_xyxy",
    "score",
    "class_id",
    "source_order",
    "query_index",
}
CACHE_ROW_KEYS = {
    "image_id",
    "width",
    "height",
    "full_predictions",
    "local_fused_predictions",
}
ROUTE_ARMS = ("A", "route_control", "route_treatment")


def _prediction(record: Mapping[str, Any]) -> Detection:
    if set(record) != PREDICTION_KEYS:
        raise ValueError("SADED prediction schema drift")
    detection = Detection(
        box=tuple(record["box"]),
        global_xyxy=tuple(record["global_xyxy"]),
        score=record["score"],
        class_id=record["class_id"],
        source_order=record["source_order"],
        query_index=record["query_index"],
    )
    box = detection.box
    provenance = detection.global_xyxy
    if (
        not detection._metadata_valid
        or provenance is None
        or detection.class_id < 0
        or detection.source_order < 0
        or detection.query_index < 0
        or not math.isfinite(detection.score)
        or detection.score < 0.0
        or detection.score > 1.0
        or not all(math.isfinite(value) for value in (*box, *provenance))
        or box[2] <= box[0]
        or box[3] <= box[1]
        or provenance[2] <= provenance[0]
        or provenance[3] <= provenance[1]
    ):
        raise ValueError("SADED prediction is invalid")
    return detection


def prediction_payload(detection: Detection) -> dict[str, Any]:
    if detection.global_xyxy is None:
        raise ValueError("SADED prediction is missing global provenance")
    return {
        "box": list(detection.box),
        "global_xyxy": list(detection.global_xyxy),
        "score": float(detection.score),
        "class_id": int(detection.class_id),
        "source_order": int(detection.source_order),
        "query_index": int(detection.query_index),
    }


def _candidates(
    predictions: Sequence[Mapping[str, Any]],
    *,
    image_id: str,
) -> tuple[ExpertCandidate, ...]:
    if len(predictions) > 300:
        raise ValueError("SADED cache exceeds max_det")
    return tuple(
        ExpertCandidate(
            detection=_prediction(record),
            image_id=image_id,
            original_index=index,
        )
        for index, record in enumerate(predictions)
    )


def route_paired_caches(
    baseline_rows: Sequence[Mapping[str, Any]],
    treatment_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Route matched baseline/treatment caches without any GT inputs."""

    if not baseline_rows or len(baseline_rows) != len(treatment_rows):
        raise ValueError("SADED paired cache row count drift")
    routed: list[dict[str, Any]] = []
    per_image_passed = True
    for baseline, treatment in zip(baseline_rows, treatment_rows):
        if (
            set(baseline) != CACHE_ROW_KEYS
            or set(treatment) != CACHE_ROW_KEYS
            or baseline["image_id"] != treatment["image_id"]
            or baseline["width"] != treatment["width"]
            or baseline["height"] != treatment["height"]
        ):
            raise ValueError("SADED paired cache identity drift")
        image_id = str(baseline["image_id"])
        width = int(baseline["width"])
        height = int(baseline["height"])
        baseline_full = _candidates(
            baseline["full_predictions"],
            image_id=image_id,
        )
        baseline_local = _candidates(
            baseline["local_fused_predictions"],
            image_id=image_id,
        )
        treatment_local = _candidates(
            treatment["local_fused_predictions"],
            image_id=image_id,
        )
        route_control = route_saded_image(
            image_id=image_id,
            width=width,
            height=height,
            baseline=baseline_full,
            local_fused=baseline_local,
        )
        route_treatment = route_saded_image(
            image_id=image_id,
            width=width,
            height=height,
            baseline=baseline_full,
            local_fused=treatment_local,
        )
        row_invariants = {
            "control": dict(route_control.invariants),
            "treatment": dict(route_treatment.invariants),
            "same_protected_prefix": (
                route_control.protected_baseline
                == route_treatment.protected_baseline
            ),
        }
        row_passed = (
            row_invariants["control"].get("passed") is True
            and row_invariants["treatment"].get("passed") is True
            and row_invariants["same_protected_prefix"] is True
        )
        row_invariants["passed"] = row_passed
        per_image_passed = per_image_passed and row_passed
        routed.append(
            {
                "image_id": image_id,
                "width": width,
                "height": height,
                "arms": {
                    "A": [
                        prediction_payload(candidate.detection)
                        for candidate in baseline_full
                    ],
                    "route_control": [
                        prediction_payload(detection)
                        for detection in route_control.predictions
                    ],
                    "route_treatment": [
                        prediction_payload(detection)
                        for detection in route_treatment.predictions
                    ],
                },
                "coverage": {
                    "control": dict(route_control.coverage),
                    "treatment": dict(route_treatment.coverage),
                },
                "invariants": row_invariants,
            }
        )
    invariants = {
        "image_count": len(routed),
        "paired_image_order_exact": [
            row["image_id"] for row in baseline_rows
        ]
        == [row["image_id"] for row in treatment_rows],
        "per_image_passed": per_image_passed,
        "gt_fields_absent": all(
            not {
                "gt_boxes",
                "gt_classes",
                "ignore_boxes",
                "annotations",
            }.intersection(row)
            for row in routed
        ),
    }
    invariants["passed"] = all(invariants.values())
    return routed, invariants


def _metric_delta(
    treatment: Mapping[str, Any],
    control: Mapping[str, Any],
    key: str,
) -> float:
    value = float(treatment[key]) - float(control[key])
    if not math.isfinite(value):
        raise ValueError("SADED metric delta is non-finite")
    return value


def screen_seed0_gate(
    *,
    route_control: Mapping[str, Any],
    route_treatment: Mapping[str, Any],
    invariants_passed: bool,
) -> dict[str, Any]:
    if not invariants_passed:
        return {
            "schema_version": "tascv-screen-seed0-adjudication/v1",
            "decision": "INVALID",
            "failures": ["evaluation_invariants_failed"],
        }
    gate = FROZEN_SCREEN_GATE["seed0"]
    keys = (
        "mAP50-95",
        "AP-tiny-SBR",
        "tiny_recall",
        "AP75",
        "AP-large-SBR",
    )
    try:
        deltas = {
            key: _metric_delta(route_treatment, route_control, key)
            for key in keys
        }
    except (KeyError, TypeError, ValueError):
        return {
            "schema_version": "tascv-screen-seed0-adjudication/v1",
            "decision": "INVALID",
            "failures": ["metric_schema_or_value_invalid"],
        }
    failures: list[str] = []
    if deltas["mAP50-95"] <= gate["mAP50-95"]:
        failures.append("mAP50-95_delta<=0")
    for key in ("AP-tiny-SBR", "tiny_recall", "AP75", "AP-large-SBR"):
        if deltas[key] < gate[key] - 1e-12:
            failures.append(f"{key}_delta<{gate[key]}")
    return {
        "schema_version": "tascv-screen-seed0-adjudication/v1",
        "decision": (
            "TASCV_SCREEN_SEED0_GO"
            if not failures
            else "TASCV_STOP"
        ),
        "failures": failures,
        "deltas": deltas,
        "thresholds": gate,
    }


__all__ = [
    "CACHE_ROW_KEYS",
    "PREDICTION_KEYS",
    "ROUTE_ARMS",
    "prediction_payload",
    "route_paired_caches",
    "screen_seed0_gate",
]
