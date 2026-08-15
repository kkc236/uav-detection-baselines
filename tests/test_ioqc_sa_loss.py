from __future__ import annotations

import math

import pytest
import torch

from src.ioqc_sa_loss import (
    IOQCSATargets,
    compute_ioqc_sa_loss,
    ensure_finite_losses,
    target_scale,
)
from src.ioqc_sa_probe import P3SamplingStatistics


def dense_case(*, dtype: torch.dtype = torch.float32):
    gt_boxes = torch.tensor(
        [[0.40, 0.50, 0.10, 0.10], [0.52, 0.50, 0.10, 0.10]], dtype=dtype
    )
    targets = IOQCSATargets(
        boxes=gt_boxes,
        classes=torch.tensor([0, 0]),
        batch_indices=torch.tensor([0, 0]),
        groups=[2],
    )
    pred_boxes = torch.tensor(
        [
            [
                [0.40, 0.50, 0.10, 0.10],
                [0.52, 0.50, 0.10, 0.10],
                [0.40, 0.50, 0.10, 0.10],
                [0.90, 0.90, 0.05, 0.05],
            ]
        ],
        dtype=dtype,
    )
    pred_logits = torch.tensor([[[6.0], [6.0], [6.0], [-6.0]]], dtype=dtype)
    scale = 0.10 / math.sqrt(12.0)
    center = torch.tensor(
        [[[0.40, 0.50], [0.52, 0.50], [0.40, 0.50], [0.90, 0.90]]], dtype=dtype
    )
    extent = torch.full((1, 4, 2), scale, dtype=dtype)
    statistics = P3SamplingStatistics(
        center=center,
        extent=extent,
        p3_mass=torch.ones((1, 4), dtype=dtype),
        valid=torch.ones((1, 4), dtype=torch.bool),
        p3_shape=(80, 80),
    )
    matches = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    return pred_boxes, pred_logits, statistics, targets, matches


def compute(case=None):
    pred_boxes, pred_logits, statistics, targets, matches = case or dense_case()
    return compute_ioqc_sa_loss(
        pred_boxes=pred_boxes,
        pred_logits=pred_logits,
        statistics=statistics,
        targets=targets,
        match_indices=matches,
        density_threshold=1.0,
        duplicate_threshold=0.10,
    )


def test_identical_owner_and_duplicate_have_maximum_competition_and_zero_alignment():
    result = compute()

    torch.testing.assert_close(result.competition, torch.tensor(1.0))
    torch.testing.assert_close(result.alignment, torch.tensor(0.0), atol=1e-6, rtol=0)
    assert result.dense_count == 2
    assert result.duplicate_count == 1


def test_competition_reaches_zero_when_average_difference_is_one_target_scale():
    case = dense_case()
    statistics = case[2]
    scale = 0.10 / math.sqrt(12.0)
    statistics.center[0, 2] += scale
    statistics.extent[0, 2] += scale

    result = compute(case)

    torch.testing.assert_close(result.competition, torch.tensor(0.0), atol=1e-6, rtol=0)


def test_competition_stops_gradient_on_owner_but_updates_duplicate():
    case = dense_case()
    statistics = case[2]
    statistics.center.requires_grad_()
    statistics.extent.requires_grad_()
    statistics.center.data[0, 2, 0] += 0.005

    result = compute(case)
    result.competition.backward()

    assert statistics.center.grad is not None
    torch.testing.assert_close(statistics.center.grad[0, 0], torch.zeros(2))
    assert statistics.center.grad[0, 2].abs().sum() > 0


def test_top1_keeps_only_highest_quality_duplicate_per_ground_truth():
    case = dense_case()
    pred_boxes, pred_logits, statistics, targets, matches = case
    pred_boxes = torch.cat((pred_boxes, pred_boxes[:, 2:3]), dim=1)
    pred_boxes[0, 4, 0] += 0.01
    pred_logits = torch.cat((pred_logits, torch.tensor([[[2.0]]])), dim=1)
    statistics = P3SamplingStatistics(
        center=torch.cat((statistics.center, statistics.center[:, 2:3]), dim=1),
        extent=torch.cat((statistics.extent, statistics.extent[:, 2:3]), dim=1),
        p3_mass=torch.ones((1, 5)),
        valid=torch.ones((1, 5), dtype=torch.bool),
        p3_shape=(80, 80),
    )

    result = compute((pred_boxes, pred_logits, statistics, targets, matches))

    assert result.duplicate_count == 1


@pytest.mark.parametrize("mode", ["empty", "single", "not_dense", "no_duplicate", "zero_mass"])
def test_empty_valid_sets_return_exact_finite_graph_connected_zero(mode: str):
    case = dense_case()
    pred_boxes, pred_logits, statistics, targets, matches = case
    statistics.center.requires_grad_()
    if mode == "empty":
        targets = IOQCSATargets(
            boxes=torch.empty((0, 4)),
            classes=torch.empty((0,), dtype=torch.long),
            batch_indices=torch.empty((0,), dtype=torch.long),
            groups=[0],
        )
        matches = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
    elif mode == "single":
        targets = IOQCSATargets(
            boxes=targets.boxes[:1], classes=targets.classes[:1], batch_indices=targets.batch_indices[:1], groups=[1]
        )
        matches = [(torch.tensor([0]), torch.tensor([0]))]
    elif mode == "not_dense":
        targets.boxes[1, 0] = 0.90
    elif mode == "no_duplicate":
        pred_logits[0, 2, 0] = -20.0
    elif mode == "zero_mass":
        statistics.p3_mass[0, 2] = 0.0
        statistics.valid[0, 2] = False

    result = compute((pred_boxes, pred_logits, statistics, targets, matches))
    total = result.competition + result.alignment

    assert total.dtype == torch.float32
    assert result.competition.item() == 0.0
    if mode in {"no_duplicate", "zero_mass"}:
        # Duplicate selection is empty, while owner alignment remains independently active.
        assert result.alignment.abs().item() < 1e-6
    else:
        assert total.item() == 0.0
    assert torch.isfinite(total)
    total.backward()
    assert statistics.center.grad is not None


def test_tiny_target_scale_is_floored_by_one_p3_cell():
    boxes = torch.tensor([[0.5, 0.5, 1e-5, 2e-5]])

    raw, stable = target_scale(boxes, p3_shape=(80, 40))

    assert raw[0, 0] < 1 / 40
    assert raw[0, 1] < 1 / 80
    torch.testing.assert_close(stable[0], torch.tensor([1 / 40, 1 / 80]))


def test_half_inputs_produce_finite_float32_losses_and_gradients():
    case = dense_case(dtype=torch.float16)
    case[2].center.requires_grad_()
    case[2].extent.requires_grad_()

    result = compute(case)
    (result.competition + result.alignment).backward()

    assert result.competition.dtype == torch.float32
    assert result.alignment.dtype == torch.float32
    assert torch.isfinite(result.competition)
    assert torch.isfinite(result.alignment)
    assert torch.isfinite(case[2].center.grad).all()


def test_nonfinite_loss_guard_raises_explicit_marker():
    with pytest.raises(FloatingPointError, match="NONFINITE_LOSS"):
        ensure_finite_losses(comp=torch.tensor(float("nan")), align=torch.tensor(0.0))
