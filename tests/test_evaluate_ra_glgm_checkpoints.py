from __future__ import annotations

import pytest

from scripts.evaluate_ra_glgm_checkpoints import (
    COCO_CATEGORY_IDS,
    MAX_DETECTIONS_PER_IMAGE,
    _coco_category_id,
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
