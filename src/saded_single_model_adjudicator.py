"""Formal five-gate adjudication for single-model SADED."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
from typing import Any


SCHEMA_VERSION = "saded-single-model-formal-adjudication/v1"
FORMAL_THRESHOLDS = {
    "AP-tiny-SBR": 0.010,
    "mAP50-95": 0.003,
    "tiny_recall": 0.020,
    "AP75": -0.002,
    "AP-large-SBR": -0.005,
}


def _primary_values(metrics: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in FORMAL_THRESHOLDS:
        value = metrics.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"invalid primary metric: {key}")
        values[key] = float(value)
    return values


def adjudicate_single_model(
    *,
    arm_a: Mapping[str, Any],
    route_control: Mapping[str, Any],
    invariants_passed: bool,
) -> dict[str, Any]:
    """Apply the frozen formal gates to one unified SADED prediction set."""

    if invariants_passed is not True:
        return {
            "schema_version": SCHEMA_VERSION,
            "decision": "INVALID",
            "failures": ["evidence_invariants_failed"],
        }
    try:
        baseline = _primary_values(arm_a)
        candidate = _primary_values(route_control)
    except (AttributeError, TypeError, ValueError):
        return {
            "schema_version": SCHEMA_VERSION,
            "decision": "INVALID",
            "failures": ["metric_schema_or_value_invalid"],
        }
    deltas = {
        key: candidate[key] - baseline[key]
        for key in FORMAL_THRESHOLDS
    }
    gates = {
        key: deltas[key] >= threshold
        for key, threshold in FORMAL_THRESHOLDS.items()
    }
    failures = [
        f"{key}_delta<{threshold}"
        for key, threshold in FORMAL_THRESHOLDS.items()
        if not gates[key]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": (
            "SADED_SINGLE_SEED_GO"
            if not failures
            else "SADED_SINGLE_SEED_STOP"
        ),
        "failures": failures,
        "thresholds": dict(FORMAL_THRESHOLDS),
        "deltas": deltas,
        "gates": gates,
    }


__all__ = [
    "FORMAL_THRESHOLDS",
    "SCHEMA_VERSION",
    "adjudicate_single_model",
]
