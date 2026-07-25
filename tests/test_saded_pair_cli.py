from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.route_saded_pair import (
    _aggregate_capacity,
    _parse_checksums,
    _raw_view,
)
from scripts.evaluate_saded_stage import _three_way_deltas
from src.sbr_g0 import RawViewRecord
from src.sbr_geometry import LetterboxTransform


def _serialized_raw_view() -> dict:
    transform = LetterboxTransform(
        source_width=960,
        source_height=540,
        network_shape=(640, 640),
        gain_x=2 / 3,
        gain_y=2 / 3,
        pad_x=0.0,
        pad_y=140.0,
        resized_width=640,
        resized_height=360,
        auto=False,
        scale_fill=False,
        scaleup=False,
        center=True,
        padding_value=114,
    )
    original = RawViewRecord(
        image_id="x.jpg",
        width=960,
        height=540,
        arm="C",
        view_id="full",
        source_order=0,
        query_index=7,
        tile_bounds=None,
        transform=transform,
        network_xyxy=(10.0, 20.0, 30.0, 40.0),
        view_xyxy=(15.0, 5.0, 45.0, 35.0),
        global_xyxy=(15.0, 5.0, 45.0, 35.0),
        score=0.75,
        class_id=2,
    )
    return json.loads(json.dumps(original.to_dict()))


def test_raw_view_replays_cache_serializer_round_trip():
    record = _serialized_raw_view()

    replayed = _raw_view(record)

    assert json.loads(json.dumps(replayed.to_dict())) == record


@pytest.mark.parametrize("drift", ["missing", "extra"])
def test_raw_view_rejects_letterbox_transform_schema_drift(drift: str):
    record = _serialized_raw_view()
    if drift == "missing":
        del record["transform"]["network_width"]
    else:
        record["transform"]["unexpected"] = 1

    with pytest.raises(
        ValueError,
        match="letterbox-transform schema drift",
    ):
        _raw_view(record)


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
