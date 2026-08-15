from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.itber_geometry import apply_edge_update
from src.itber_loss import itber_private_loss


def _output(query_count: int, *, requires_grad: bool = True) -> SimpleNamespace:
    stock = torch.tensor([0.30, 0.30, 0.50, 0.50]).repeat(1, query_count, 1)
    gate_logits = torch.zeros(1, query_count, 4, requires_grad=requires_grad)
    residual_raw = torch.zeros(1, query_count, 4, requires_grad=requires_grad)
    gates = gate_logits.sigmoid()
    residuals = residual_raw.tanh()
    refined = apply_edge_update(stock, gates, residuals, rho=0.05)
    return SimpleNamespace(
        stock_edges=stock,
        refined_edges=refined,
        gate_logits=gate_logits,
        gates=gates,
        residual_raw=residual_raw,
        residuals=residuals,
        quality=torch.ones(1, query_count, 1),
    )


def _one_match() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(torch.tensor([0]), torch.tensor([0]))]


def test_private_loss_reuses_match_and_never_touches_detector() -> None:
    detector_tensor = torch.tensor(
        [[[0.30, 0.30, 0.50, 0.50]] * 5], requires_grad=True
    )
    output = _output(query_count=5)
    output.stock_edges = detector_tensor.detach()
    output.refined_edges = apply_edge_update(
        output.stock_edges, output.gates, output.residuals, rho=0.05
    )

    losses = itber_private_loss(
        output,
        target_edges=torch.tensor([[0.29, 0.31, 0.52, 0.48]]),
        match_indices=_one_match(),
        rho=0.05,
    )
    losses.total.backward()

    assert detector_tensor.grad is None
    assert output.gate_logits.grad is not None
    assert output.residual_raw.grad is not None
    assert output.gate_logits.grad[0, 1:].abs().sum() > 0


def test_positive_and_negative_gate_means_are_separately_normalized() -> None:
    target = torch.tensor([[0.29, 0.31, 0.52, 0.48]])
    small = itber_private_loss(
        _output(3), target_edges=target, match_indices=_one_match(), rho=0.05
    )
    large = itber_private_loss(
        _output(300), target_edges=target, match_indices=_one_match(), rho=0.05
    )

    torch.testing.assert_close(small.gate_positive, large.gate_positive)
    torch.testing.assert_close(small.gate_negative, large.gate_negative)
    torch.testing.assert_close(small.gate, large.gate)


def test_direction_uses_magnitude_weighted_sign_target() -> None:
    output = _output(2)
    target = torch.tensor([[0.29, 0.30, 0.52, 0.50]])

    losses = itber_private_loss(
        output, target_edges=target, match_indices=_one_match(), rho=0.05
    )
    losses.direction.backward()

    gradient = output.residual_raw.grad[0, 0]
    assert gradient[0] > 0  # Gradient descent moves the left edge residual negative.
    assert gradient[2] < 0  # Gradient descent moves the right edge residual positive.
    assert gradient[1].item() == 0
    assert gradient[3].item() == 0


@pytest.mark.parametrize("mode", ["no_matches", "no_unmatched"])
def test_empty_sets_return_finite_graph_connected_zero(mode: str) -> None:
    if mode == "no_matches":
        output = _output(3)
        matches = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
        targets = torch.empty(0, 4)
    else:
        output = _output(1)
        matches = _one_match()
        targets = torch.tensor([[0.30, 0.30, 0.50, 0.50]])

    losses = itber_private_loss(
        output, target_edges=targets, match_indices=matches, rho=0.05
    )

    assert torch.isfinite(losses.total)
    if mode == "no_matches":
        assert losses.box.item() == 0
        assert losses.direction.item() == 0
        assert losses.gate_positive.item() == 0
    else:
        assert losses.gate_negative.item() == 0
        assert losses.noop.item() == 0
    losses.total.backward()
    assert output.gate_logits.grad is not None
    assert output.residual_raw.grad is not None


def test_loss_rejects_batch_or_match_index_mismatch() -> None:
    output = _output(3)
    with pytest.raises(ValueError, match="batch"):
        itber_private_loss(
            output,
            target_edges=torch.tensor([[0.30, 0.30, 0.50, 0.50]]),
            match_indices=[],
            rho=0.05,
        )
    with pytest.raises(IndexError, match="query"):
        itber_private_loss(
            output,
            target_edges=torch.tensor([[0.30, 0.30, 0.50, 0.50]]),
            match_indices=[(torch.tensor([3]), torch.tensor([0]))],
            rho=0.05,
        )
