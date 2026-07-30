from __future__ import annotations

from collections.abc import Mapping
from typing import Any


RECALL_TOLERANCE = 0.0002


def _precision_recall(error: Mapping[str, Any]) -> tuple[float, float]:
    tp, fp, fn = (int(error[key]) for key in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def decide_g1_admission(diagnosis: Mapping[str, Any]) -> dict[str, Any]:
    """Admit G1 only when retained-G2 counterfactuals isolate geometry precision harm."""
    if diagnosis.get("training_signal") is not False:
        raise ValueError("admission requires a read-only diagnosis artifact")
    branches = diagnosis.get("branches")
    if not isinstance(branches, Mapping):
        raise ValueError("diagnosis is missing branches")
    try:
        full = branches["full"]["fixed_baseline_threshold"]
        semantic_only = branches["semantic_only"]["fixed_baseline_threshold"]
        full_error = full["error"]["all"]
        semantic_error = semantic_only["error"]["all"]
    except (KeyError, TypeError) as error:
        raise ValueError("diagnosis lacks full/semantic_only fixed-threshold errors") from error
    full_precision, full_recall = _precision_recall(full_error)
    semantic_precision, semantic_recall = _precision_recall(semantic_error)
    same_threshold = float(full["confidence_threshold"]) == float(
        semantic_only["confidence_threshold"]
    )
    criteria = {
        "same_frozen_baseline_threshold": same_threshold,
        "precision_non_decrease": semantic_precision >= full_precision,
        "recall_within_tolerance": semantic_recall >= full_recall - RECALL_TOLERANCE,
        "geometry_fp_excess": int(full_error["fp"]) > int(semantic_error["fp"]),
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "threshold": float(full["confidence_threshold"]),
        "full": {
            "precision": full_precision,
            "recall": full_recall,
            "tp": int(full_error["tp"]),
            "fp": int(full_error["fp"]),
            "fn": int(full_error["fn"]),
        },
        "semantic_only": {
            "precision": semantic_precision,
            "recall": semantic_recall,
            "tp": int(semantic_error["tp"]),
            "fp": int(semantic_error["fp"]),
            "fn": int(semantic_error["fn"]),
        },
        "recall_tolerance": RECALL_TOLERANCE,
        "training_signal": False,
    }
