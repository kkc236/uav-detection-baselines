from __future__ import annotations

import pytest
import torch


def _image() -> dict:
    return {
        "relative_path": "frame.jpg",
        "width": 200,
        "height": 100,
        "gt_boxes": [[10.0, 20.0, 30.0, 40.0]],
        "gt_classes": [2],
        "ignore_boxes": [[0.0, 0.0, 5.0, 5.0]],
    }


def test_prediction_tensor_becomes_clipped_source_pixel_sbr_row() -> None:
    from src.acr_eg_live_evaluation import prediction_tensor_to_sbr_row

    predictions = torch.tensor(
        [
            [0.50, 0.50, 0.20, 0.40, 0.75, 3.0],
            [0.02, 0.02, 0.10, 0.10, 0.25, 1.0],
        ]
    )

    row = prediction_tensor_to_sbr_row(predictions, _image())

    assert row["image_id"] == "frame.jpg"
    assert torch.allclose(
        torch.tensor(row["pred_boxes"]),
        torch.tensor(
            [
                [80.0, 30.0, 120.0, 70.0],
                [0.0, 0.0, 14.0, 7.0],
            ]
        ),
    )
    assert row["pred_scores"] == pytest.approx([0.75, 0.25])
    assert row["pred_classes"] == [3, 1]
    assert row["pred_source"] == [0, 0]
    assert row["pred_query"] == [0, 1]
    assert row["gt_boxes"] == _image()["gt_boxes"]
    assert row["ignore_boxes"] == _image()["ignore_boxes"]
    assert row["effective_gain"] == 1.0


def test_prediction_tensor_rejects_nonfinite_or_wrong_layout() -> None:
    from src.acr_eg_live_evaluation import prediction_tensor_to_sbr_row

    for invalid in (
        torch.zeros(2, 5),
        torch.tensor([[0.5, 0.5, 0.2, 0.2, float("nan"), 1.0]]),
    ):
        try:
            prediction_tensor_to_sbr_row(invalid, _image())
        except ValueError:
            pass
        else:
            raise AssertionError("invalid prediction tensor was accepted")


def test_prediction_tensor_uses_recorded_network_to_source_homography() -> None:
    from src.acr_eg_live_evaluation import prediction_tensor_to_sbr_row

    image = dict(_image(), width=1280, height=640)
    network_to_source = torch.tensor(
        [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    predictions = torch.tensor([[0.5, 0.5, 0.25, 0.5, 0.9, 1.0]])

    row = prediction_tensor_to_sbr_row(
        predictions,
        image,
        network_to_source=network_to_source,
    )

    assert torch.allclose(
        torch.tensor(row["pred_boxes"]),
        torch.tensor([[480.0, 160.0, 800.0, 480.0]]),
    )


def test_numeric_deltas_include_only_shared_real_metrics() -> None:
    from src.acr_eg_live_evaluation import numeric_deltas

    baseline = {"mAP50-95": 0.2, "AP-tiny-SBR": 0.08, "counts": {}}
    method = {"mAP50-95": 0.23, "AP-tiny-SBR": 0.10, "extra": 4.0}

    assert numeric_deltas(baseline, method) == {
        "AP-tiny-SBR": 0.020000000000000004,
        "mAP50-95": 0.03,
    }


def test_build_result_records_live_endpoint_and_pass_fail() -> None:
    from src.acr_eg_live_evaluation import build_result

    result = build_result(
        baseline_metrics={"mAP50-95": 0.20, "AP-tiny-SBR": 0.08},
        method_metrics={"mAP50-95": 0.21, "AP-tiny-SBR": 0.09},
        checkpoint={"path": "epoch99.pt", "sha256": "A" * 64, "epoch": 99},
        baseline={"path": "baseline.pt", "sha256": "B" * 64},
        dataset={"image_count": 548, "signature": "C" * 64},
        runtime={"baseline_seconds": 1.0, "method_seconds": 5.0},
        source={"commit": "deadbeef"},
    )

    assert result["schema_version"] == "gcte-acr-eg-live-evaluation/v1"
    assert result["endpoint"] == "live-global-plus-four-local-views"
    assert result["deltas"]["mAP50-95"] == 0.009999999999999981
    assert result["decision"]["exceeds_baseline_mAP"] is True
    assert result["decision"]["tiny_improves"] is True
