from __future__ import annotations

import torch

from src.itber_head import ITBERRefiner


def _inputs(hidden_dim: int = 8, f3_channels: int = 4):
    hidden = torch.randn(2, 5, hidden_dim, requires_grad=True)
    stock = torch.tensor([0.5, 0.5, 0.2, 0.2]).view(1, 1, 4).expand(2, 5, 4).clone()
    stock.requires_grad_(True)
    box_l1 = (stock.detach() + torch.tensor([0.01, -0.01, 0.0, 0.0])).requires_grad_()
    box_l2 = (stock.detach() + torch.tensor([0.02, -0.02, 0.0, 0.0])).requires_grad_()
    scores = torch.randn(2, 5, 3, requires_grad=True)
    f3 = torch.randn(2, f3_channels, 20, 20, requires_grad=True)
    return hidden, box_l2, box_l1, stock, scores, f3


def _refiner(probe: str = "p3") -> ITBERRefiner:
    return ITBERRefiner(
        hidden_dim=8,
        f3_channels=4,
        private_seed=17,
        probe=probe,
        image_size=640,
        rho=0.05,
    )


def test_zero_output_heads_make_refined_boxes_exact_stock_identity() -> None:
    refiner = _refiner()

    output = refiner(*_inputs())

    torch.testing.assert_close(output.refined_boxes, output.stock_boxes, rtol=0, atol=0)
    torch.testing.assert_close(
        output.effective_correction,
        torch.zeros_like(output.effective_correction),
        rtol=0,
        atol=0,
    )
    assert output.gates.shape == (2, 5, 4)
    assert output.residuals.shape == (2, 5, 4)


def test_all_probe_modes_have_identical_parameter_count() -> None:
    counts = {
        probe: sum(parameter.numel() for parameter in _refiner(probe).parameters())
        for probe in ("p0", "p1", "p2", "p3")
    }

    assert len(set(counts.values())) == 1


def test_probe_modalities_are_zero_filled_without_removing_parameters() -> None:
    inputs = tuple(value.detach() for value in _inputs())

    p0 = _refiner("p0")(*inputs)
    p3 = _refiner("p3")(*inputs)

    torch.testing.assert_close(p0.trajectory, torch.zeros_like(p0.trajectory))
    torch.testing.assert_close(
        p0.boundary_features, torch.zeros_like(p0.boundary_features)
    )
    assert torch.count_nonzero(p3.trajectory) > 0
    assert torch.count_nonzero(p3.boundary_features) > 0


def test_detector_evidence_is_detached_while_private_parameters_receive_gradients() -> None:
    refiner = _refiner()
    with torch.no_grad():
        refiner.residual_head.bias.fill_(0.2)
    inputs = _inputs()

    refiner(*inputs).refined_boxes.sum().backward()

    assert all(value.grad is None for value in inputs)
    assert refiner.residual_head.bias.grad is not None
    assert refiner.f3_projection.weight.grad is not None


def test_private_initialization_does_not_advance_global_rng() -> None:
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()

    _refiner()

    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)


def test_extreme_scores_and_tiny_boxes_stay_finite() -> None:
    refiner = _refiner()
    hidden, box_l2, box_l1, stock, scores, f3 = _inputs()
    tiny = torch.tensor([0.0, 1.0, 1e-12, 1e-12]).view(1, 1, 4).expand_as(stock)
    scores = torch.empty_like(scores).fill_(1000)

    output = refiner(hidden, tiny, tiny, tiny, scores, f3)

    for value in (
        output.refined_boxes,
        output.gates,
        output.residuals,
        output.quality,
        output.entropy,
        output.trajectory,
        output.boundary_features,
    ):
        assert torch.isfinite(value).all()
