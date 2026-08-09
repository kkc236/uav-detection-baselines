from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.evaluate_ra_glgm_checkpoints import (
    COCO_CATEGORY_IDS,
    MAX_DETECTIONS_PER_IMAGE,
    _coco_category_id,
    _load_saved_predictions,
    _normalized_640_area,
    _ordered_names,
    _validated_predictions,
)


def test_coco_ids_match_ultralytics_non_coco_json_export() -> None:
    assert COCO_CATEGORY_IDS == tuple(range(1, 11))
    assert [_coco_category_id(index) for index in range(10)] == list(range(1, 11))


@pytest.mark.parametrize("class_id", [-1, 10])
def test_coco_id_conversion_rejects_foreign_classes(class_id: int) -> None:
    with pytest.raises(ValueError, match="VisDrone class"):
        _coco_category_id(class_id)


def test_prediction_validation_allows_fewer_than_max_det() -> None:
    result = _validated_predictions(
        [{"image_id": "sample", "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.5}],
        {"sample": 1},
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
        _validated_predictions(predictions, {"sample": 1})


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


def test_tiny_small_area_is_normalized_to_640_not_source_resolution() -> None:
    # 0.02 x 0.02 is 163.84 square pixels on the frozen 640 canvas,
    # independently of the source-resolution image dimensions.
    assert _normalized_640_area(0.02, 0.02) == pytest.approx(163.84)


def test_saved_predictions_are_loaded_from_validator_output(tmp_path) -> None:
    rows = [{"image_id": "sample", "category_id": 1, "bbox": [0, 0, 1, 1]}]
    (tmp_path / "predictions.json").write_text(json.dumps(rows), encoding="utf-8")

    assert _load_saved_predictions(SimpleNamespace(save_dir=tmp_path)) == rows


def test_saved_predictions_reject_non_object_rows(tmp_path) -> None:
    (tmp_path / "predictions.json").write_text("[1]", encoding="utf-8")

    with pytest.raises(ValueError, match="list of objects"):
        _load_saved_predictions(SimpleNamespace(save_dir=tmp_path))
