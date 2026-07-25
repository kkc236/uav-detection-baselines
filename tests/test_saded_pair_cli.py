from __future__ import annotations

from pathlib import Path

import pytest

from scripts.route_saded_pair import (
    _aggregate_capacity,
    _parse_checksums,
)
from scripts.evaluate_saded_stage import _three_way_deltas


def test_checksum_parser_requires_exact_unique_relative_artifacts(
    tmp_path: Path,
):
    path = tmp_path / "checksums.sha256"
    path.write_text(
        "a" * 64 + "  cache_manifest.json\n"
        + "b" * 64
        + "  predictions.jsonl.gz\n",
        encoding="utf-8",
    )
    assert _parse_checksums(path) == {
        "cache_manifest.json": "a" * 64,
        "predictions.jsonl.gz": "b" * 64,
    }

    path.write_text(
        "a" * 64 + "  ../escape.json\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum"):
        _parse_checksums(path)


def test_capacity_is_reported_for_both_routed_systems():
    rows = [
        {
            "image_id": "x.jpg",
            "coverage": {
                "control": {
                    "protected_baseline": 2,
                    "remaining_tiny_slots": 298,
                    "accepted_local": 3,
                    "capacity_rejected": 1,
                },
                "treatment": {
                    "protected_baseline": 2,
                    "remaining_tiny_slots": 298,
                    "accepted_local": 5,
                    "capacity_rejected": 0,
                },
            },
        }
    ]

    capacity = _aggregate_capacity(rows)

    assert set(capacity["systems"]) == {"route_control", "route_treatment"}
    assert (
        capacity["systems"]["route_control"]["accepted_local"]["total"]
        == 3
    )
    assert (
        capacity["systems"]["route_treatment"]["accepted_local"]["total"]
        == 5
    )


def test_evaluator_emits_primary_safety_and_attribution_deltas():
    keys = {
        "mAP50-95": 0.10,
        "AP-tiny-SBR": 0.20,
        "tiny_recall": 0.30,
        "AP75": 0.15,
        "AP-large-SBR": 0.25,
    }
    control = {**keys, "mAP50-95": 0.11}
    treatment = {
        **control,
        "mAP50-95": 0.12,
        "AP-tiny-SBR": 0.21,
    }

    deltas = _three_way_deltas(
        {
            "A": keys,
            "route_control": control,
            "route_treatment": treatment,
        }
    )

    assert set(deltas) == {
        "route_control_vs_A",
        "route_treatment_vs_A",
        "route_treatment_vs_route_control",
    }
    assert deltas["route_treatment_vs_route_control"]["mAP50-95"] == (
        pytest.approx(0.01)
    )
