from __future__ import annotations

import pytest
from PIL import Image

from src.sqda_error_audit import (
    compare_error_summaries,
    precision_recall_f1_curve,
    summarize_detection_errors,
)


def _dataset(annotations: list[dict]) -> dict:
    return {
        "images": [{"id": "frame", "width": 640, "height": 640}],
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "vehicle"},
            {"id": 2, "name": "pedestrian"},
        ],
    }


def test_error_audit_uses_ground_truth_and_prediction_area_bins() -> None:
    dataset = _dataset(
        [
            {
                "id": 1,
                "image_id": "frame",
                "category_id": 1,
                "bbox": [10.0, 10.0, 16.0, 16.0],
                "area": 16.0**2,
            },
            {
                "id": 2,
                "image_id": "frame",
                "category_id": 1,
                "bbox": [100.0, 100.0, 64.0, 64.0],
                "area": 64.0**2,
            },
            {
                "id": 3,
                "image_id": "frame",
                "category_id": 2,
                "bbox": [300.0, 300.0, 120.0, 120.0],
                "area": 120.0**2,
            },
        ]
    )
    predictions = [
        {
            "image_id": "frame",
            "category_id": 1,
            "bbox": [10.0, 10.0, 16.0, 16.0],
            "score": 0.90,
        },
        {
            "image_id": "frame",
            "category_id": 1,
            "bbox": [500.0, 500.0, 64.0, 64.0],
            "score": 0.80,
        },
    ]

    report = summarize_detection_errors(dataset, predictions)

    assert report["small"]["tp"] == 1
    assert report["medium"]["fp"] == 1
    assert report["large"]["fn"] == 1
    assert report["small"]["mean_tp_score"] == pytest.approx(0.90)
    assert report["small"]["mean_tp_iou"] == pytest.approx(1.0)


def test_error_audit_matches_same_class_predictions_by_descending_score() -> None:
    dataset = _dataset(
        [
            {
                "id": 1,
                "image_id": "frame",
                "category_id": 1,
                "bbox": [100.0, 100.0, 64.0, 64.0],
                "area": 64.0**2,
            }
        ]
    )
    predictions = [
        {
            "image_id": "frame",
            "category_id": 1,
            "bbox": [100.0, 100.0, 64.0, 64.0],
            "score": 0.40,
        },
        {
            "image_id": "frame",
            "category_id": 1,
            "bbox": [100.0, 100.0, 64.0, 64.0],
            "score": 0.90,
        },
    ]

    report = summarize_detection_errors(dataset, predictions)

    assert report["medium"]["tp"] == 1
    assert report["medium"]["fp"] == 1
    assert report["medium"]["mean_tp_score"] == pytest.approx(0.90)
    assert report["medium"]["mean_fp_score"] == pytest.approx(0.40)


def test_error_audit_delta_never_converts_missing_means_to_numbers() -> None:
    baseline = {
        "small": {
            "tp": 2,
            "fp": 1,
            "fn": 3,
            "mean_tp_score": 0.5,
            "mean_fp_score": None,
            "mean_tp_iou": 0.7,
        }
    }
    candidate = {
        "small": {
            "tp": 3,
            "fp": 1,
            "fn": 2,
            "mean_tp_score": 0.6,
            "mean_fp_score": 0.4,
            "mean_tp_iou": 0.8,
        }
    }

    delta = compare_error_summaries(baseline, candidate)

    assert delta["small"]["tp"] == 1
    assert delta["small"]["fn"] == -1
    assert delta["small"]["mean_tp_score"] == pytest.approx(0.1)
    assert delta["small"]["mean_fp_score"] is None


def test_pr_curve_is_class_aware_and_selects_baseline_max_f1_threshold() -> None:
    dataset = _dataset(
        [
            {
                "id": 1,
                "image_id": "frame",
                "category_id": 1,
                "bbox": [10.0, 10.0, 16.0, 16.0],
                "area": 16.0**2,
            }
        ]
    )
    curve = precision_recall_f1_curve(
        dataset,
        [
            {
                "image_id": "frame",
                "category_id": 2,
                "bbox": [10.0, 10.0, 16.0, 16.0],
                "score": 0.95,
            },
            {
                "image_id": "frame",
                "category_id": 1,
                "bbox": [10.0, 10.0, 16.0, 16.0],
                "score": 0.90,
            },
        ],
    )

    assert curve["ground_truth"] == 1
    assert curve["points"][0]["precision"] == 0.0
    assert curve["points"][1]["recall"] == 1.0
    assert curve["best_f1"]["confidence_threshold"] == pytest.approx(0.90)
    assert curve["best_f1"]["f1"] == pytest.approx(2.0 / 3.0)


def test_error_audit_cli_writes_fixed_threshold_protocol_and_delta(tmp_path, monkeypatch) -> None:
    from scripts.audit_sqda_regressions import main

    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (64, 64)).save(images / "frame.jpg")
    (labels / "frame.txt").write_text(
        """0 0.25 0.25 0.25 0.25
0 0.75 0.75 0.25 0.25
""",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        '[{"image_id":"frame","category_id":1,"bbox":[8,8,16,16],"score":0.90}]',
        encoding="utf-8",
    )
    candidate.write_text(
        '[{"image_id":"frame","category_id":1,"bbox":[8,8,16,16],"score":0.90},'
        '{"image_id":"frame","category_id":1,"bbox":[40,40,16,16],"score":0.80}]',
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_sqda_regressions.py",
            "--images", str(images),
            "--labels", str(labels),
            "--baseline-predictions", str(baseline),
            "--candidate-predictions", str(candidate),
            "--output", str(output),
            "--expected-images", "1",
            "--expected-annotations", "2",
        ],
    )

    main()

    import json

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["protocol"]["confidence_threshold"] == pytest.approx(0.25)
    assert report["protocol"]["iou_threshold"] == pytest.approx(0.50)
    assert report["protocol"]["matching"] == "class-aware_score-descending_greedy"
    assert report["protocol"]["training_signal"] is False
    assert report["delta"]["small"]["tp"] == 1
    assert report["delta"]["small"]["fn"] == -1
