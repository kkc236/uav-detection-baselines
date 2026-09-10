from __future__ import annotations

import torch

import src.bpdd_capacity as capacity
from src.bpdd_capacity import LocalBoundaryExpert


def _inputs(*, queries: int = 4, hidden: int = 32):
    return (
        torch.randn(1, queries, hidden),
        torch.randn(1, queries, 132),
        torch.randn(1, queries, 3),
        torch.tensor([0.5, 0.5, 0.25, 0.2]).expand(1, queries, 4),
        [
            torch.randn(1, 8, 16, 16),
            torch.randn(1, 12, 8, 8),
            torch.randn(1, 16, 4, 4),
        ],
    )


def test_expert_samples_before_linear_projection_and_starts_as_identity() -> None:
    module = LocalBoundaryExpert(
        channels=(8, 12, 16), hidden=32, nc=3, heads=4, ff=64
    )
    assert all(isinstance(layer, torch.nn.Linear) for layer in module.projections)
    h, z, c, boxes, features = _inputs()
    refined_z, refined_c = module(h, z, c, boxes, features)
    torch.testing.assert_close(refined_z, z, rtol=0, atol=0)
    torch.testing.assert_close(refined_c, c, rtol=0, atol=0)


def test_expert_constructor_does_not_call_global_manual_seed(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("private construction must not call torch.manual_seed")

    monkeypatch.setattr(torch, "manual_seed", forbidden)
    LocalBoundaryExpert(channels=(8, 12, 16), hidden=32, nc=3, heads=4, ff=64)


def test_expert_training_uses_non_reentrant_checkpoint(monkeypatch) -> None:
    calls: list[bool] = []
    actual = capacity.checkpoint

    def record(function, *args, **kwargs):
        calls.append(kwargs.get("use_reentrant"))
        return actual(function, *args, **kwargs)

    monkeypatch.setattr(capacity, "checkpoint", record)
    module = LocalBoundaryExpert(
        channels=(8, 12, 16), hidden=32, nc=3, heads=4, ff=64, chunk_size=2
    ).train()
    with torch.no_grad():
        module.box_out.weight[0, 0] = 0.1
    h, z, c, boxes, features = _inputs()
    h.requires_grad_()
    for feature in features:
        feature.requires_grad_()
    refined_z, _ = module(h, z, c, boxes, features)
    refined_z.sum().backward()
    assert calls == [False, False]
    assert h.grad is not None and h.grad.abs().sum() > 0
    assert all(feature.grad is not None for feature in features)


def test_zero_outputs_unlock_inner_expert_gradients_after_one_update() -> None:
    module = LocalBoundaryExpert(
        channels=(8, 12, 16), hidden=32, nc=3, heads=4, ff=64
    )
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    h, z, c, boxes, features = _inputs(queries=2)
    inner_sums: list[float] = []
    for _step in range(2):
        optimizer.zero_grad(set_to_none=True)
        refined_z, refined_c = module(h, z, c, boxes, features)
        (refined_z.square().mean() + refined_c.square().mean()).backward()
        inner_sums.append(float(module.projections[0].weight.grad.abs().sum()))
        optimizer.step()
    assert inner_sums[0] == 0.0
    assert inner_sums[1] > 0.0
