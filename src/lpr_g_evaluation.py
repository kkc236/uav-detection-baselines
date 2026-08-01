"""Pure pre-registered decision logic for the LPR-G v2 screen."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _value(source: Mapping[str, Any], section: str, metric: str) -> float:
    value = float(source[section][metric])
    if not math.isfinite(value):
        raise ValueError(f"non-finite screen metric: {section}.{metric}")
    return value


def evaluate_screen_gate(
    control: Mapping[str, Any],
    method: Mapping[str, Any],
    ablation: Mapping[str, Any],
    activity: Mapping[str, Any],
    engineering_valid: bool,
) -> dict[str, Any]:
    """Evaluate all frozen scientific conditions without rounding any value."""
    delta = {
        section: {
            metric: _value(method, section, metric) - _value(control, section, metric)
            for metric in ("map", "ap75", "map50")
        }
        for section in ("final", "tail10")
    }
    refined_map_delta = float(ablation["refined"]["map"]) - float(
        ablation["stock"]["map"]
    )
    refined_ap75_delta = float(ablation["refined"]["ap75"]) - float(
        ablation["stock"]["ap75"]
    )
    activity_values = [
        float(value)
        for key, value in activity.items()
        if key != "finite" and isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    activity_finite = bool(activity.get("finite")) and all(
        math.isfinite(value) for value in activity_values
    )
    gate_p95 = float(activity.get("gate_p95", 0.0))
    residual_rms = float(activity.get("residual_rms", 0.0))

    conditions = {
        "final_map_win": delta["final"]["map"] > 0.0,
        "tail10_map_win": delta["tail10"]["map"] > 0.0,
        "ap75_support": (
            delta["final"]["ap75"] > 0.0 and delta["tail10"]["ap75"] >= 0.0
        )
        or (
            delta["tail10"]["ap75"] > 0.0 and delta["final"]["ap75"] >= 0.0
        ),
        "map50_floor": _value(method, "final", "map50")
        >= _value(control, "final", "map50") - 0.001
        and _value(method, "tail10", "map50")
        >= _value(control, "tail10", "map50") - 0.001,
        "same_checkpoint_refinement": refined_map_delta > 0.0 and refined_ap75_delta > 0.0,
        "refinement_active": activity_finite
        and gate_p95 > 1e-3
        and residual_rms > 0.0,
        # Efficiency targets are reported, but the approved design explicitly makes them non-blocking.
        "efficiency_reported": bool(activity.get("efficiency_measured", True)),
    }
    scientific_passed = all(conditions.values())
    if not engineering_valid:
        status = "engineering_invalid"
    elif scientific_passed:
        status = "passed"
    else:
        status = "scientific_failed"
    return {
        "passed": status == "passed",
        "status": status,
        "engineering_valid": bool(engineering_valid),
        "conditions": conditions,
        "delta": {
            **delta,
            "same_checkpoint": {
                "map": refined_map_delta,
                "ap75": refined_ap75_delta,
            },
        },
        "activity": dict(activity),
    }
