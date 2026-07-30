from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.sqda_error_audit import precision_recall_f1_curve, summarize_detection_errors


DIAGNOSTIC_MODES = ("full", "semantic_only", "geometry_only", "identity")
FIXED_AUDIT_CONFIDENCE = 0.25
FIXED_AUDIT_IOU = 0.50


def build_branch_summary(
    mode: str,
    dataset: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    coco_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Build read-only metrics for one retained-G2 counterfactual branch."""
    if mode not in DIAGNOSTIC_MODES:
        raise ValueError(f"unsupported diagnostic mode={mode!r}")
    return {
        "mode": mode,
        "training_signal": False,
        "coco": dict(coco_metrics),
        "error_at_0.25": summarize_detection_errors(
            dataset,
            predictions,
            confidence_threshold=FIXED_AUDIT_CONFIDENCE,
            iou_threshold=FIXED_AUDIT_IOU,
        ),
        "pr_f1_curve": precision_recall_f1_curve(
            dataset,
            predictions,
            iou_threshold=FIXED_AUDIT_IOU,
        ),
    }


def attach_baseline_threshold_metrics(
    summaries: Mapping[str, dict[str, Any]],
    dataset: Mapping[str, Any],
    predictions_by_mode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> float:
    """Evaluate every branch at the F1-optimal threshold selected only from full G2."""
    if "full" not in summaries or "full" not in predictions_by_mode:
        raise ValueError("full retained-G2 branch is required to select the baseline threshold")
    best = summaries["full"]["pr_f1_curve"]["best_f1"]
    threshold = best["confidence_threshold"]
    threshold = FIXED_AUDIT_CONFIDENCE if threshold is None else float(threshold)
    for mode, summary in summaries.items():
        if mode not in predictions_by_mode:
            raise ValueError(f"missing predictions for diagnostic mode={mode!r}")
        summary["fixed_baseline_threshold"] = {
            "confidence_threshold": threshold,
            "iou_threshold": FIXED_AUDIT_IOU,
            "error": summarize_detection_errors(
                dataset,
                predictions_by_mode[mode],
                confidence_threshold=threshold,
                iou_threshold=FIXED_AUDIT_IOU,
            ),
        }
    return threshold
