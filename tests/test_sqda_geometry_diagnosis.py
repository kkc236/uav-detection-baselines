from __future__ import annotations

import pytest

from src.sqda_geometry_diagnosis import (
    DIAGNOSTIC_MODES,
    attach_baseline_threshold_metrics,
    build_branch_summary,
)


def _dataset() -> dict:
    return {
        "images": [{"id": "frame", "width": 64, "height": 64}],
        "annotations": [
            {
                "id": 1,
                "image_id": "frame",
                "category_id": 1,
                "bbox": [8.0, 8.0, 16.0, 16.0],
                "area": 16.0**2,
            }
        ],
        "categories": [{"id": 1, "name": "vehicle"}],
    }


def test_branch_summary_and_baseline_threshold_are_read_only() -> None:
    dataset = _dataset()
    full_predictions = [
        {
            "image_id": "frame",
            "category_id": 1,
            "bbox": [8.0, 8.0, 16.0, 16.0],
            "score": 0.75,
        }
    ]
    geometry_predictions = []
    summaries = {
        "full": build_branch_summary(
            "full",
            dataset,
            full_predictions,
            {"ap": 0.50, "ap_small": 0.50},
        ),
        "geometry_only": build_branch_summary(
            "geometry_only",
            dataset,
            geometry_predictions,
            {"ap": 0.0, "ap_small": 0.0},
        ),
    }

    threshold = attach_baseline_threshold_metrics(
        summaries,
        dataset,
        {"full": full_predictions, "geometry_only": geometry_predictions},
    )

    assert DIAGNOSTIC_MODES == ("full", "semantic_only", "geometry_only", "identity")
    assert threshold == pytest.approx(0.75)
    assert summaries["full"]["fixed_baseline_threshold"]["error"]["small"]["tp"] == 1
    assert summaries["geometry_only"]["fixed_baseline_threshold"]["error"]["small"]["fn"] == 1
    assert summaries["full"]["training_signal"] is False


def test_branch_summary_rejects_non_counterfactual_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        build_branch_summary("gated", _dataset(), [], {"ap": 0.0})


def test_diagnosis_cli_locks_all_four_read_only_modes() -> None:
    from scripts.diagnose_sqda_geometry_branches import build_parser

    options = {action.dest for action in build_parser()._actions}
    assert {
        "checkpoint",
        "adapter_checkpoint",
        "data",
        "images",
        "labels",
        "output",
    }.issubset(options)
    assert not {"mode", "epochs", "optimizer", "lr0", "residual_mode"}.intersection(options)
