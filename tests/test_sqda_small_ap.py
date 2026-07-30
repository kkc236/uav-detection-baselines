from __future__ import annotations

import json

import pytest
from PIL import Image

from scripts.evaluate_sqda_small_ap import build_coco_dataset, evaluate_predictions


def test_small_ap_uses_original_image_area_and_max_det_300(tmp_path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (40, 40)).save(images / "frame.jpg")
    (labels / "frame.txt").write_text(
        "0 0.5 0.5 0.25 0.25\r\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            [
                {
                    "image_id": "frame",
                    "category_id": 1,
                    "bbox": [15.0, 15.0, 10.0, 10.0],
                    "score": 0.99,
                }
            ]
        ),
        encoding="utf-8",
    )

    dataset = build_coco_dataset(images, labels, class_count=1)
    metrics = evaluate_predictions(dataset, predictions)

    assert dataset["annotations"][0]["area"] == pytest.approx(100.0)
    assert metrics["max_dets"] == 300
    assert metrics["ap"] == pytest.approx(1.0)
    assert metrics["ap_small"] == pytest.approx(1.0)
