from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.evaluate_gcmv_plec import (
    build_parser,
    decode_source_predictions,
    jsonable,
    metric_deltas,
)


def test_evaluation_parser_requires_paired_checkpoints(tmp_path):
    args = build_parser().parse_args(
        [
            "--control-checkpoint",
            "control.pt",
            "--method-checkpoint",
            "method.pt",
            "--data",
            "visdrone.yaml",
            "--output",
            str(tmp_path / "metrics.json"),
        ]
    )

    assert args.batch == 4
    assert args.control_checkpoint == "control.pt"
    assert args.method_checkpoint == "method.pt"


def test_decode_predictions_inverts_the_centered_letterbox():
    # Source 2000x1000 is scaled by 0.32 with y padding=160.
    prediction = torch.tensor(
        [
            [0.5, 0.5, 0.5, 0.25, 0.9, 3.0],
            [0.5, 0.5, 0.5, 0.25, 0.0001, 4.0],
        ]
    )

    decoded = decode_source_predictions(
        prediction,
        source_height=1000,
        source_width=2000,
        imgsz=640,
    )

    np.testing.assert_allclose(
        decoded["pred_boxes"],
        [[500.0, 250.0, 1500.0, 750.0]],
        atol=1e-5,
    )
    assert decoded["pred_scores"] == [pytest.approx(0.9)]
    assert decoded["pred_classes"] == [3]


def test_metric_deltas_are_method_minus_control():
    deltas = metric_deltas(
        {"mAP50-95": 0.2, "AP-tiny-SBR": 0.1},
        {"mAP50-95": 0.23, "AP-tiny-SBR": 0.14},
    )

    assert deltas == {
        "mAP50-95": pytest.approx(0.03),
        "AP-tiny-SBR": pytest.approx(0.04),
    }


def test_jsonable_normalizes_numpy_values_and_numeric_mapping_keys():
    assert jsonable({0.5: {np.int64(2): np.float32(0.25)}}) == {
        "0.5": {"2": pytest.approx(0.25)}
    }
