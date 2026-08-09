from __future__ import annotations

import math

import pytest
import torch

from src.bpdd_loss import (
    BPDDOptions,
    bpdd_distribution_loss,
    build_progressive_teachers,
    interpolated_edge_nll,
)


def _targets(matches: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.zeros((matches, 4), dtype=torch.long)
    right = torch.full((matches, 4), 0.25)
    left = 1.0 - right
    return indices, right, left


def test_options_freeze_safe_defaults_and_reject_invalid_values() -> None:
    assert BPDDOptions() == BPDDOptions(
        enabled=True,
        weight=0.5,
        temperature=0.5,
        margin=0.02,
        eps=1e-6,
    )
    for kwargs in (
        {"weight": -1.0},
        {"temperature": 0.0},
        {"margin": -0.1},
        {"eps": 0.0},
    ):
        with pytest.raises(ValueError):
            BPDDOptions(**kwargs)


def test_interpolated_edge_nll_matches_two_adjacent_bin_target() -> None:
    probabilities = torch.tensor([[[0.50, 0.25, 0.25]]]).expand(1, 4, 3)
    log_probabilities = probabilities.log()
    indices, right, left = _targets(matches=1)

    actual = interpolated_edge_nll(
        log_probabilities, indices, right, left
    )
    expected = torch.full((1, 4), -(0.75 * math.log(0.5) + 0.25 * math.log(0.25)))

    torch.testing.assert_close(actual, expected)


def test_progressive_teacher_uses_only_future_layers_and_actual_softmin() -> None:
    probabilities = torch.tensor(
        [
            [[[[0.10, 0.90]]]],
            [[[[0.80, 0.20]]]],
            [[[[0.60, 0.40]]]],
        ],
        dtype=torch.float32,
    ).reshape(3, 1, 1, 2)
    errors = -probabilities[..., 0].log()

    teachers, mixture_weights = build_progressive_teachers(
        probabilities, errors, temperature=0.5
    )

    assert len(teachers) == 2
    assert len(mixture_weights) == 2
    expected_l0_weights = torch.softmax(-errors[1:] / 0.5, dim=0)
    expected_l0_teacher = (
        expected_l0_weights.unsqueeze(-1) * probabilities[1:]
    ).sum(dim=0)
    torch.testing.assert_close(mixture_weights[0], expected_l0_weights)
    torch.testing.assert_close(teachers[0], expected_l0_teacher)
    torch.testing.assert_close(mixture_weights[1], torch.ones_like(errors[2:]))
    torch.testing.assert_close(teachers[1], probabilities[2])
    assert not teachers[0].requires_grad
    assert not mixture_weights[0].requires_grad


def test_better_future_teacher_backpropagates_only_into_student() -> None:
    logits = torch.tensor(
        [
            [[[-3.0, 3.0]] * 4],
            [[[3.0, -3.0]] * 4],
        ],
        requires_grad=True,
    )
    indices = torch.zeros((1, 4), dtype=torch.long)
    right = torch.zeros((1, 4))
    left = torch.ones((1, 4))

    result = bpdd_distribution_loss(
        logits,
        indices,
        right,
        left,
        options=BPDDOptions(margin=0.02),
    )
    result.loss.backward()

    assert result.loss.item() > 0
    assert result.statistics["active_edge_ratio"].item() == pytest.approx(1.0)
    assert result.statistics["mean_teacher_improvement"].item() > 0
    assert logits.grad is not None
    assert logits.grad[0].abs().sum() > 0
    torch.testing.assert_close(logits.grad[1], torch.zeros_like(logits.grad[1]))


def test_equal_or_worse_future_teacher_is_exact_noop() -> None:
    logits = torch.tensor(
        [
            [[[3.0, -3.0]] * 4],
            [[[-3.0, 3.0]] * 4],
        ],
        requires_grad=True,
    )
    indices = torch.zeros((1, 4), dtype=torch.long)
    right = torch.zeros((1, 4))
    left = torch.ones((1, 4))

    result = bpdd_distribution_loss(
        logits, indices, right, left, options=BPDDOptions()
    )
    result.loss.backward()

    torch.testing.assert_close(result.loss, torch.zeros_like(result.loss))
    torch.testing.assert_close(
        result.statistics["active_edge_ratio"],
        torch.zeros_like(result.statistics["active_edge_ratio"]),
    )
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))


def test_empty_matches_return_finite_graph_connected_zero() -> None:
    logits = torch.empty((6, 0, 4, 33), requires_grad=True)
    indices, right, left = _targets(matches=0)

    result = bpdd_distribution_loss(
        logits, indices, right, left, options=BPDDOptions()
    )
    result.loss.backward()

    assert result.loss.shape == ()
    assert torch.isfinite(result.loss)
    assert logits.grad is not None
    assert result.statistics["matched_queries"].item() == 0
    assert result.statistics["eligible_edges"].item() == 0


def test_weight_zero_is_exact_graph_connected_zero() -> None:
    logits = torch.randn((3, 2, 4, 33), requires_grad=True)
    indices, right, left = _targets(matches=2)

    result = bpdd_distribution_loss(
        logits, indices, right, left, options=BPDDOptions(weight=0.0)
    )
    result.loss.backward()

    torch.testing.assert_close(result.loss, torch.zeros_like(result.loss))
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))


def test_bpdd_promotes_half_precision_math_and_rejects_wrong_shapes() -> None:
    logits = torch.randn((3, 2, 4, 33), dtype=torch.float16, requires_grad=True)
    indices, right, left = _targets(matches=2)
    result = bpdd_distribution_loss(
        logits, indices, right, left, options=BPDDOptions()
    )
    assert result.loss.dtype == torch.float32
    assert torch.isfinite(result.loss)

    with pytest.raises(ValueError, match="corner_logits"):
        bpdd_distribution_loss(
            logits[:, :, 0], indices, right, left, options=BPDDOptions()
        )
    with pytest.raises(ValueError, match="target_indices"):
        bpdd_distribution_loss(
            logits,
            indices[:, :1],
            right,
            left,
            options=BPDDOptions(),
        )

