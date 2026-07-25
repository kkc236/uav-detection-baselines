from __future__ import annotations

from copy import deepcopy

import pytest

from src.saded_stock_postprocess import route_single_cache


def _prediction(box, score, *, source=0, query=0):
    return {
        "box": list(box),
        "global_xyxy": list(box),
        "score": score,
        "class_id": 0,
        "source_order": source,
        "query_index": query,
    }


def _cache_row():
    return {
        "image_id": "image.jpg",
        "width": 640,
        "height": 640,
        "full_predictions": [
            _prediction((0, 0, 40, 40), 0.8),
            _prediction((100, 100, 110, 110), 0.4, query=1),
        ],
        "local_fused_predictions": [
            _prediction(
                (100, 100, 111, 111),
                0.5,
                source=1,
                query=2,
            )
        ],
    }


def test_single_cache_route_emits_only_one_baseline_and_candidate() -> None:
    rows, invariants = route_single_cache([_cache_row()])
    assert invariants["passed"] is True
    assert set(rows[0]["arms"]) == {"A", "route_control"}
    assert rows[0]["arms"]["A"][0]["box"] == [0, 0, 40, 40]
    assert rows[0]["arms"]["route_control"][0]["box"] == [0, 0, 40, 40]
    assert rows[0]["arms"]["route_control"][1]["box"] == [
        100,
        100,
        111,
        111,
    ]
    assert {
        "protected_baseline",
        "remaining_tiny_slots",
        "accepted_local",
        "capacity_rejected",
    }.issubset(rows[0]["coverage"])


def test_single_cache_route_rejects_empty_or_identity_drift() -> None:
    with pytest.raises(ValueError, match="row count"):
        route_single_cache([])
    bad = deepcopy(_cache_row())
    bad["image_id"] = ""
    with pytest.raises(ValueError, match="identity"):
        route_single_cache([bad])


def test_single_cache_route_contains_no_gt_or_duplicate_arm() -> None:
    rows, invariants = route_single_cache([_cache_row()])
    assert invariants["single_endpoint_exact"] is True
    assert invariants["gt_fields_absent"] is True
    assert "route_treatment" not in rows[0]["arms"]
    assert not {
        "gt_boxes",
        "gt_classes",
        "ignore_boxes",
        "annotations",
    }.intersection(rows[0])
