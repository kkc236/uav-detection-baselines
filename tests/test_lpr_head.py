from __future__ import annotations

import torch

from src.lpr_head import LocalizationPriorRefiner, box_geometry_prior


def test_geometry_prior_is_finite_for_tiny_boxes() -> None:
    boxes = torch.tensor([[[0.5, 0.25, 1e-12, 2e-12]]])

    prior = box_geometry_prior(boxes)

    assert prior.shape == (1, 1, 6)
    assert torch.isfinite(prior).all()
    torch.testing.assert_close(prior[..., :2], torch.tensor([[[0.0, -0.5]]]))


def test_zero_gate_is_bitwise_identity_and_alpha_gets_gradient() -> None:
    module = LocalizationPriorRefiner(hidden_dim=256, seed=3407)
    hidden = torch.randn(2, 5, 256, requires_grad=True)
    boxes = torch.rand(2, 5, 4).mul(0.8).add(0.1).requires_grad_()

    refined = module(hidden, boxes)

    assert torch.equal(refined, boxes)
    weights = torch.linspace(0.5, 1.5, refined.numel(), device=refined.device).reshape_as(refined)
    (refined * weights).sum().backward()
    assert module.alpha.grad is not None
    assert module.alpha.grad.abs().item() > 0


def test_positive_gate_keeps_refined_boxes_bounded_and_changes_output() -> None:
    module = LocalizationPriorRefiner(hidden_dim=32, seed=3408)
    module.alpha.data.fill_(0.4)
    hidden = torch.randn(2, 3, 32)
    boxes = torch.rand(2, 3, 4).mul(0.8).add(0.1)

    refined = module(hidden, boxes)

    assert torch.all(refined > 0)
    assert torch.all(refined < 1)
    assert not torch.equal(refined, boxes)


def test_refiner_construction_does_not_advance_global_rng() -> None:
    torch.manual_seed(17)
    expected = torch.rand(4)
    torch.manual_seed(17)

    LocalizationPriorRefiner(hidden_dim=256, seed=3407)
    actual = torch.rand(4)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
