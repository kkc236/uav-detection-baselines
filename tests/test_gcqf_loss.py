import pytest
import torch

from src.gcqf_loss import compute_gcqf_loss


def _inputs():
    return {
        "adjusted_scores": torch.tensor(
            [[[0.9], [0.2], [0.8]]],
            requires_grad=True,
        ),
        "quality_targets": torch.tensor([[[1.0], [0.0], [0.75]]]),
        "canonical_queries": torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]],
            requires_grad=True,
        ),
        "equivariance_pairs": torch.tensor(
            [[0, 0, 1], [0, 0, 2]],
            dtype=torch.long,
        ),
        "score_residual": torch.tensor(
            [[[0.1], [0.0], [-0.2]]],
            requires_grad=True,
        ),
        "valid_mask": torch.tensor([[True, True, True]]),
        "anchor_mask": torch.tensor([[[True], [True], [False]]]),
    }


def test_quality_loss_prefers_calibrated_scores():
    inputs = _inputs()
    good = compute_gcqf_loss(**inputs)
    bad = compute_gcqf_loss(
        **{
            **inputs,
            "adjusted_scores": torch.tensor(
                [[[0.1], [0.8], [0.1]]]
            ),
        }
    )

    assert good.quality < bad.quality


def test_equivariance_loss_is_zero_for_same_query_and_positive_for_opposite():
    inputs = _inputs()
    same = compute_gcqf_loss(
        **{
            **inputs,
            "equivariance_pairs": torch.tensor(
                [[0, 0, 1]],
                dtype=torch.long,
            ),
        }
    )
    opposite = compute_gcqf_loss(
        **{
            **inputs,
            "equivariance_pairs": torch.tensor(
                [[0, 0, 2]],
                dtype=torch.long,
            ),
        }
    )

    assert same.equivariance.detach().item() == pytest.approx(0.0, abs=1e-7)
    assert opposite.equivariance.detach().item() > 1.9


def test_total_uses_frozen_loss_weights():
    result = compute_gcqf_loss(**_inputs())

    torch.testing.assert_close(
        result.total,
        result.quality
        + 0.1 * result.equivariance
        + 0.01 * result.residual_regularization,
    )
    assert result.equivariance_weight == 0.1
    assert result.residual_weight == 0.01


def test_empty_pairs_are_finite_and_zero():
    result = compute_gcqf_loss(
        **{
            **_inputs(),
            "equivariance_pairs": torch.empty((0, 3), dtype=torch.long),
        }
    )

    assert torch.isfinite(result.total)
    assert result.equivariance == 0


def test_loss_sends_gradients_to_all_gcqf_outputs_only():
    inputs = _inputs()
    inputs["quality_targets"].requires_grad_(True)

    result = compute_gcqf_loss(**inputs)
    result.total.backward()

    assert inputs["adjusted_scores"].grad is not None
    assert inputs["canonical_queries"].grad is not None
    assert inputs["score_residual"].grad is not None
    assert inputs["quality_targets"].grad is None


def test_loss_rejects_pairs_outside_query_range():
    inputs = _inputs()

    with pytest.raises(ValueError, match="pair index"):
        compute_gcqf_loss(
            **{
                **inputs,
                "equivariance_pairs": torch.tensor(
                    [[0, 0, 3]],
                    dtype=torch.long,
                ),
            }
        )
