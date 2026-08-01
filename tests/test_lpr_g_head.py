from __future__ import annotations

import torch

from src.lpr_g_head import QualityGatedRefiner, box_geometry_prior, detached_vfl_quality


def test_zero_heads_make_refined_boxes_exactly_equal_stock() -> None:
    refiner = QualityGatedRefiner(hidden_dim=8, private_seed=17)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    boxes = torch.rand(2, 5, 4).clamp(0.05, 0.95).requires_grad_()
    scores = torch.randn(2, 5, 3, requires_grad=True)

    refined = refiner(hidden, boxes, scores)

    torch.testing.assert_close(refined, boxes.detach(), rtol=0, atol=0)
    assert refiner.last_gate is not None
    assert refiner.last_quality is not None
    assert refiner.last_gate.shape == (2, 5, 1)
    assert refiner.last_quality.shape == (2, 5, 1)


def test_refinement_inputs_are_detached_from_private_loss() -> None:
    refiner = QualityGatedRefiner(hidden_dim=8, private_seed=18)
    with torch.no_grad():
        refiner.residual_head.bias.fill_(0.2)
    hidden = torch.randn(1, 4, 8, requires_grad=True)
    boxes = torch.full((1, 4, 4), 0.4, requires_grad=True)
    scores = torch.randn(1, 4, 3, requires_grad=True)

    refiner(hidden, boxes, scores).sum().backward()

    assert hidden.grad is None
    assert boxes.grad is None
    assert scores.grad is None
    assert refiner.residual_head.bias.grad is not None


def test_quality_and_geometry_are_stable_for_extreme_inputs() -> None:
    scores = torch.tensor([[[1000.0, -1000.0], [0.0, 0.0]]])
    boxes = torch.tensor([[[0.5, 0.5, 0.0, 1e-30], [0.5, 0.5, 1.0, 1.0]]])

    quality = detached_vfl_quality(scores)
    geometry = box_geometry_prior(boxes)

    assert quality.shape == (1, 2, 1)
    assert torch.isfinite(quality).all()
    assert torch.isfinite(geometry).all()
    assert geometry[..., 2:].min() >= -12
    assert geometry[..., 2:].max() <= 12


def test_gate_is_per_query_after_gate_head_learns() -> None:
    refiner = QualityGatedRefiner(hidden_dim=4, private_seed=19)
    with torch.no_grad():
        refiner.gate_head.weight.fill_(0.1)
        refiner.residual_head.bias.fill_(0.1)
    hidden = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 4.0, 9.0], [3.0, 1.0, 0.0, 2.0]]])
    boxes = torch.full((1, 3, 4), 0.4)
    scores = torch.zeros(1, 3, 2)

    refiner(hidden, boxes, scores)

    assert refiner.last_gate is not None
    assert torch.unique(refiner.last_gate).numel() > 1
