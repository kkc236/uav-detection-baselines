from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest
import torch
from ultralytics.nn.modules.head import RTDETRDecoder

from src.rtdetr_quality_oracle import (
    ALPHA_GRID,
    DEV_COUNT,
    MAP_GAIN_THRESHOLD,
    flattened_topk,
    oracle_topk,
    same_class_iou_quality,
)


def _assert_byte_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert torch.equal(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
    )


def test_frozen_oracle_constants_are_exact() -> None:
    assert ALPHA_GRID == (0.25, 0.5, 1.0, 2.0)
    assert DEV_COUNT == 129
    assert MAP_GAIN_THRESHOLD == Decimal("0.0050")


def test_same_class_iou_quality_uses_exact_per_class_maxima() -> None:
    boxes = torch.tensor(
        [
            [0.50, 0.50, 0.20, 0.20],
            [0.50, 0.50, 0.40, 0.40],
            [0.80, 0.80, 0.20, 0.20],
        ]
    )
    target_boxes = torch.tensor(
        [
            [0.50, 0.50, 0.20, 0.20],
            [0.20, 0.20, 0.20, 0.20],
            [0.80, 0.80, 0.20, 0.20],
        ]
    )
    target_classes = torch.tensor([1, 1, 2])

    quality = same_class_iou_quality(
        boxes, target_boxes, target_classes, num_classes=4
    )

    expected = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.25, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    torch.testing.assert_close(quality, expected, rtol=0, atol=1e-6)
    assert quality.shape == (3, 4)
    assert torch.isfinite(quality).all()
    assert torch.all((quality >= 0) & (quality <= 1))


def test_same_class_iou_quality_returns_zeros_for_empty_targets() -> None:
    quality = same_class_iou_quality(
        torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]),
        torch.empty(0, 4),
        torch.empty(0, dtype=torch.long),
        num_classes=3,
    )

    torch.testing.assert_close(quality, torch.zeros(2, 3), rtol=0, atol=0)
    assert torch.isfinite(quality).all()


@pytest.mark.parametrize(
    ("boxes", "target_boxes", "target_classes", "num_classes", "error"),
    [
        (torch.zeros(1, 2, 4), torch.zeros(1, 4), torch.tensor([0]), 2, ValueError),
        (torch.zeros(2, 4), torch.zeros(1, 3), torch.tensor([0]), 2, ValueError),
        (torch.zeros(2, 4), torch.zeros(1, 4), torch.tensor([[0]]), 2, ValueError),
        (torch.zeros(2, 4), torch.zeros(1, 4), torch.empty(0, dtype=torch.long), 2, ValueError),
        (torch.zeros(2, 4), torch.full((1, 4), float("nan")), torch.tensor([0]), 2, ValueError),
        (torch.zeros(2, 4), torch.zeros(1, 4), torch.tensor([2]), 2, ValueError),
        (torch.zeros(2, 4), torch.zeros(1, 4), torch.tensor([0.0]), 2, TypeError),
        (torch.zeros(2, 4), torch.zeros(1, 4), torch.tensor([0]), 0, ValueError),
    ],
)
def test_same_class_iou_quality_strictly_validates_inputs(
    boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    num_classes: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        same_class_iou_quality(
            boxes, target_boxes, target_classes, num_classes=num_classes
        )


def test_flattened_topk_is_byte_exact_with_ultralytics_8_4_90() -> None:
    fake_head = SimpleNamespace(num_queries=4, nc=3)
    boxes = torch.linspace(0.01, 0.99, 2 * 4 * 4).reshape(2, 4, 4)
    scores = torch.tensor(
        [
            [[0.01, 0.92, 0.11], [0.81, 0.21, 0.71], [0.61, 0.31, 0.51], [0.41, 0.99, 0.91]],
            [[0.93, 0.02, 0.12], [0.22, 0.82, 0.72], [0.32, 0.62, 0.52], [0.42, 0.98, 0.88]],
        ]
    )

    expected = RTDETRDecoder.postprocess(fake_head, boxes, scores)
    actual = flattened_topk(boxes, scores, num_classes=3, max_det=4)

    _assert_byte_equal(actual, expected)


def test_production_shapes_use_default_top_300_contract() -> None:
    fake_head = SimpleNamespace(num_queries=300, nc=10)
    boxes = torch.linspace(0.001, 0.999, 1 * 300 * 4).reshape(1, 300, 4)
    logits = torch.linspace(-5.0, 5.0, 1 * 300 * 10).reshape(1, 300, 10)
    scores = logits.sigmoid()
    qualities = torch.linspace(0.1, 1.0, 1 * 300 * 10).reshape(1, 300, 10)

    expected_stock = RTDETRDecoder.postprocess(fake_head, boxes, scores)
    stock = flattened_topk(boxes, scores, num_classes=10)
    oracle = oracle_topk(
        boxes,
        logits,
        qualities,
        alpha=0.5,
        num_classes=10,
    )
    quality = same_class_iou_quality(
        boxes[0],
        boxes[0, :2],
        torch.tensor([0, 9]),
        num_classes=10,
    )

    _assert_byte_equal(stock, expected_stock)
    assert stock.shape == oracle.shape == (1, 300, 6)
    assert quality.shape == (300, 10)


def test_flattened_topk_keeps_duplicate_queries_for_different_classes() -> None:
    boxes = torch.tensor(
        [[[0.10, 0.20, 0.30, 0.40], [0.50, 0.60, 0.70, 0.80]]]
    )
    scores = torch.tensor([[[0.99, 0.98, 0.10], [0.97, 0.96, 0.95]]])

    selected = flattened_topk(boxes, scores, num_classes=3, max_det=4)

    expected = torch.tensor(
        [
            [
                [0.10, 0.20, 0.30, 0.40, 0.99, 0.0],
                [0.10, 0.20, 0.30, 0.40, 0.98, 1.0],
                [0.50, 0.60, 0.70, 0.80, 0.97, 0.0],
                [0.50, 0.60, 0.70, 0.80, 0.96, 1.0],
            ]
        ]
    )
    torch.testing.assert_close(selected, expected, rtol=0, atol=0)
    assert selected.shape == (1, 4, 6)
    _assert_byte_equal(selected[0, 0, :4], boxes[0, 0])
    _assert_byte_equal(selected[0, 1, :4], boxes[0, 0])


@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0, 2.0])
def test_oracle_topk_uses_sigmoid_quality_power_then_flattened_topk(
    alpha: float,
) -> None:
    boxes = torch.tensor(
        [[[0.10, 0.20, 0.30, 0.40], [0.50, 0.60, 0.70, 0.80]]]
    )
    logits = torch.tensor([[[4.0, -4.0], [2.0, -2.0]]])
    qualities = torch.tensor([[[0.10, 0.50], [1.00, 0.25]]])
    expected_scores = logits.sigmoid() * qualities**alpha

    expected = flattened_topk(
        boxes, expected_scores, num_classes=2, max_det=3
    )
    actual = oracle_topk(
        boxes,
        logits,
        qualities,
        alpha=alpha,
        num_classes=2,
        max_det=3,
    )

    _assert_byte_equal(actual, expected)
    assert actual.shape == (1, 3, 6)


@pytest.mark.parametrize("alpha", [True, 0.0, 0.75, 4.0, float("nan")])
def test_oracle_topk_rejects_alpha_outside_frozen_grid(alpha: float) -> None:
    with pytest.raises(ValueError, match="ALPHA_GRID"):
        oracle_topk(
            torch.zeros(1, 1, 4),
            torch.zeros(1, 1, 1),
            torch.ones(1, 1, 1),
            alpha=alpha,
            num_classes=1,
            max_det=1,
        )


@pytest.mark.parametrize(
    ("boxes", "logits", "qualities", "error"),
    [
        (torch.zeros(1, 4), torch.zeros(1, 1, 1), torch.ones(1, 1, 1), ValueError),
        (torch.zeros(1, 1, 4), torch.zeros(1, 2, 1), torch.ones(1, 2, 1), ValueError),
        (torch.zeros(1, 1, 4), torch.zeros(1, 1, 2), torch.ones(1, 1, 1), ValueError),
        (torch.zeros(1, 1, 4), torch.full((1, 1, 1), float("inf")), torch.ones(1, 1, 1), ValueError),
        (torch.zeros(1, 1, 4), torch.zeros(1, 1, 1), torch.full((1, 1, 1), float("nan")), ValueError),
        (torch.zeros(1, 1, 4), torch.zeros(1, 1, 1), torch.full((1, 1, 1), 1.1), ValueError),
    ],
)
def test_oracle_topk_rejects_mismatched_or_nonfinite_inputs(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    qualities: torch.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        oracle_topk(
            boxes,
            logits,
            qualities,
            alpha=0.5,
            num_classes=1,
            max_det=1,
        )


@pytest.mark.parametrize(
    ("boxes", "scores"),
    [
        (torch.zeros(1, 4), torch.zeros(1, 1, 1)),
        (torch.zeros(1, 1, 4), torch.zeros(1, 1)),
        (torch.zeros(1, 2, 4), torch.zeros(1, 1, 1)),
        (torch.zeros(1, 1, 4), torch.zeros(1, 1, 2)),
        (torch.zeros(1, 1, 4), torch.full((1, 1, 1), float("nan"))),
    ],
)
def test_flattened_topk_rejects_invalid_shapes_or_scores(
    boxes: torch.Tensor, scores: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        flattened_topk(boxes, scores, num_classes=1, max_det=1)
