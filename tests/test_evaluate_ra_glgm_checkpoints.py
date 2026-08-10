from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.evaluate_ra_glgm_checkpoints import (
    COCO_CATEGORY_IDS,
    IGNORE_DETECTION_IOF_THRESHOLD,
    MAX_DETECTIONS_PER_IMAGE,
    _coco_category_id,
    _bind_evaluation_row,
    _intersection_over_detection,
    _letterbox_geometry,
    _load_saved_predictions,
    _micro_precision_recall,
    _ordered_names,
    _validated_predictions,
)
from src.fdr_protocol import canonical_json_bytes


def test_coco_ids_match_ultralytics_non_coco_json_export() -> None:
    assert COCO_CATEGORY_IDS == tuple(range(1, 11))
    assert [_coco_category_id(index) for index in range(10)] == list(range(1, 11))


@pytest.mark.parametrize("class_id", [-1, 10])
def test_coco_id_conversion_rejects_foreign_classes(class_id: int) -> None:
    with pytest.raises(ValueError, match="VisDrone class"):
        _coco_category_id(class_id)


def test_prediction_validation_allows_fewer_than_max_det() -> None:
    geometry = _letterbox_geometry(640, 640)
    result = _validated_predictions(
        [{"image_id": "sample", "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.5}],
        {"sample": 1},
        {"sample": geometry},
        {"sample": []},
    )
    assert result == [
        {"image_id": 1, "category_id": 1, "bbox": [0.0, 0.0, 1.0, 1.0], "score": 0.5}
    ]


def test_area_evaluator_rejects_more_than_max_det() -> None:
    predictions = [
        {"image_id": "sample", "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.5}
        for _ in range(MAX_DETECTIONS_PER_IMAGE + 1)
    ]
    with pytest.raises(ValueError, match="exceeds max_det"):
        _validated_predictions(
            predictions,
            {"sample": 1},
            {"sample": _letterbox_geometry(640, 640)},
            {"sample": []},
        )


def test_category_mapping_must_preserve_exact_visdrone_order() -> None:
    expected = [
        "pedestrian",
        "people",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "awning-tricycle",
        "bus",
        "motor",
    ]
    assert _ordered_names({index: name for index, name in enumerate(expected)}) == expected


def test_non_square_boxes_use_centered_640_letterbox_geometry() -> None:
    geometry = _letterbox_geometry(1280, 720)

    # 1280x720 becomes 640x360 with 140 pixels of vertical padding.  Width
    # and height therefore share the same 0.5 scale instead of being stretched.
    assert geometry.xywh([128, 72, 256, 144]) == pytest.approx([64, 176, 128, 72])


def test_ignore_filter_uses_detection_iof_at_frozen_threshold() -> None:
    geometry = _letterbox_geometry(640, 640)
    ignored = [10.0, 10.0, 10.0, 10.0]
    assert IGNORE_DETECTION_IOF_THRESHOLD == 0.5
    assert _intersection_over_detection([10, 10, 20, 10], ignored) == pytest.approx(0.5)

    converted = _validated_predictions(
        [
            {"image_id": "sample", "category_id": 1, "bbox": [10, 10, 20, 10], "score": 0.9},
            {"image_id": "sample", "category_id": 1, "bbox": [30, 30, 5, 5], "score": 0.8},
        ],
        {"sample": 1},
        {"sample": geometry},
        {"sample": [ignored]},
    )
    assert len(converted) == 1
    assert converted[0]["bbox"] == pytest.approx([30, 30, 5, 5])


def test_saved_predictions_are_loaded_from_validator_output(tmp_path) -> None:
    rows = [{"image_id": "sample", "category_id": 1, "bbox": [0, 0, 1, 1]}]
    (tmp_path / "predictions.json").write_text(json.dumps(rows), encoding="utf-8")

    assert _load_saved_predictions(SimpleNamespace(save_dir=tmp_path)) == rows


def test_saved_predictions_reject_non_object_rows(tmp_path) -> None:
    (tmp_path / "predictions.json").write_text("[1]", encoding="utf-8")

    with pytest.raises(ValueError, match="list of objects"):
        _load_saved_predictions(SimpleNamespace(save_dir=tmp_path))


def test_evaluation_rows_form_a_canonical_forward_hash_chain() -> None:
    first = _bind_evaluation_row({"completed_epoch": 8, "map": 0.1}, "0" * 64)
    second = _bind_evaluation_row(
        {"completed_epoch": 9, "map": 0.2}, first["evaluation_row_sha256"]
    )

    assert first["previous_evaluation_row_sha256"] == "0" * 64
    assert second["previous_evaluation_row_sha256"] == first["evaluation_row_sha256"]
    unhashed = dict(second)
    recorded = unhashed.pop("evaluation_row_sha256")
    assert recorded == hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest().upper()


def test_micro_precision_recall_excludes_ignored_detections_and_ground_truths() -> None:
    metrics = _micro_precision_recall(
        [
            {
                "maxDet": 300,
                "aRng": [0.0, 1.0e10],
                "dtScores": [0.9, 0.8, 0.7],
                "dtMatches": [[1, 0, 0]],
                "dtIgnore": [[False, False, True]],
                "gtIds": [1, 2, 3],
                "gtMatches": [[1, 0, 0]],
                "gtIgnore": [False, False, True],
            }
        ]
    )
    assert metrics == {"precision": pytest.approx(0.5), "recall": pytest.approx(0.5)}
