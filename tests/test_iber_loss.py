from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

import src.iber_probe as iber_probe
from src.iber_loss import (
    IBERBucketCounts,
    balanced_boundary_direction_loss,
    boundary_edge_margin_loss,
    iber_private_loss,
)
from src.iber_head import IBERRefiner
from src.itber_geometry import apply_edge_update


def _target_edges_for_buckets() -> torch.Tensor:
    # At 640 px these are tiny (<16^2), small (<32^2), and other.
    return torch.tensor(
        [
            [0.1000, 0.1000, 0.1200, 0.1200],
            [0.2000, 0.2000, 0.2400, 0.2400],
            [0.3000, 0.3000, 0.4000, 0.4000],
        ],
        dtype=torch.float32,
    )


def _loss_output(
    *,
    base_gate_raw: torch.Tensor,
    boundary_gate_raw: torch.Tensor,
    base_residual_raw: torch.Tensor,
    boundary_residual_raw: torch.Tensor,
    stock_edges: torch.Tensor,
    rho: float = 0.05,
) -> SimpleNamespace:
    gate_logits = base_gate_raw + boundary_gate_raw
    residual_raw = base_residual_raw + boundary_residual_raw
    gates = gate_logits.sigmoid()
    residuals = residual_raw.tanh()
    refined_edges = apply_edge_update(stock_edges, gates, residuals, rho=rho)
    boundary_off_edges = apply_edge_update(
        stock_edges,
        base_gate_raw.sigmoid(),
        base_residual_raw.tanh(),
        rho=rho,
    )
    return SimpleNamespace(
        stock_edges=stock_edges,
        refined_edges=refined_edges,
        boundary_off_edges=boundary_off_edges,
        gate_logits=gate_logits,
        gates=gates,
        residual_raw=residual_raw,
        residuals=residuals,
        effective_correction=gates * residuals,
        quality=torch.ones(stock_edges.shape[:2]),
        base_gate_raw=base_gate_raw,
        boundary_gate_raw=boundary_gate_raw,
        base_residual_raw=base_residual_raw,
        boundary_residual_raw=boundary_residual_raw,
        boundary_aux_gate_raw=boundary_gate_raw,
        boundary_aux_residual_raw=boundary_residual_raw,
    )


def test_iber_probe_uses_independent_iber_loss_identity() -> None:
    source = inspect.getsource(iber_probe)
    assert "from src.iber_loss import" in source
    assert "iber_private_loss" in source
    assert "src.itber_loss" not in source


def test_boundary_direction_weights_small_and_large_corrections_equally() -> None:
    predicted = torch.tensor([[0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    normalized_target = torch.tensor([[0.1, 0.9, -0.1, -0.9]])
    target_edges = torch.tensor([[0.10, 0.10, 0.12, 0.12]])

    loss = balanced_boundary_direction_loss(
        predicted,
        normalized_target,
        target_edges,
        image_size=640,
        global_bucket_counts=(4, 0, 0),
        batches_per_epoch=1,
    )
    loss.backward()

    torch.testing.assert_close(
        predicted.grad[0, 0].abs(), predicted.grad[0, 1].abs(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        predicted.grad[0, 2].abs(), predicted.grad[0, 3].abs(), rtol=0, atol=0
    )


def test_boundary_direction_balances_tiny_small_and_other_buckets() -> None:
    target_edges = _target_edges_for_buckets()
    normalized_target = torch.ones((3, 4))
    predicted = torch.tensor(
        [
            [-1.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        requires_grad=True,
    )

    original = balanced_boundary_direction_loss(
        predicted,
        normalized_target,
        target_edges,
        image_size=640,
        global_bucket_counts=(4, 4, 4),
        batches_per_epoch=1,
    )
    repeated = balanced_boundary_direction_loss(
        torch.cat((predicted[:2], predicted[2:].repeat(20, 1))),
        torch.cat((normalized_target[:2], normalized_target[2:].repeat(20, 1))),
        torch.cat((target_edges[:2], target_edges[2:].repeat(20, 1))),
        image_size=640,
        global_bucket_counts=(4, 4, 80),
        batches_per_epoch=1,
    )
    expected = balanced_boundary_direction_loss(
        torch.cat((predicted[:2], predicted[2:])),
        torch.cat((normalized_target[:2], normalized_target[2:])),
        torch.cat((target_edges[:2], target_edges[2:])),
        image_size=640,
        global_bucket_counts=(4, 4, 4),
        batches_per_epoch=1,
    )

    torch.testing.assert_close(original, expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(repeated, expected, rtol=0, atol=1e-7)


def test_boundary_direction_is_invariant_to_minibatch_partition() -> None:
    target_edges = _target_edges_for_buckets()
    normalized_target = torch.ones((3, 4))
    predicted = torch.tensor(
        [[-0.10] * 4, [0.00] * 4, [0.10] * 4], dtype=torch.float32
    )
    counts = (4, 4, 4)
    whole = balanced_boundary_direction_loss(
        predicted,
        normalized_target,
        target_edges,
        image_size=640,
        global_bucket_counts=counts,
        batches_per_epoch=1,
    )
    split = [
        balanced_boundary_direction_loss(
            predicted[index : index + 1],
            normalized_target[index : index + 1],
            target_edges[index : index + 1],
            image_size=640,
            global_bucket_counts=counts,
            batches_per_epoch=3,
        )
        for index in range(3)
    ]

    torch.testing.assert_close(torch.stack(split).mean(), whole, rtol=0, atol=1e-7)


def test_boundary_direction_masks_exact_zero_corrections() -> None:
    predicted = torch.tensor([[0.25, -0.25, 0.5, -0.5]], requires_grad=True)
    normalized_target = torch.zeros_like(predicted)
    target_edges = torch.tensor([[0.10, 0.10, 0.12, 0.12]])

    loss = balanced_boundary_direction_loss(
        predicted,
        normalized_target,
        target_edges,
        image_size=640,
        global_bucket_counts=(0, 0, 0),
        batches_per_epoch=1,
    )
    loss.backward()

    assert loss.item() == 0.0
    torch.testing.assert_close(predicted.grad, torch.zeros_like(predicted), rtol=0, atol=0)


def test_boundary_direction_stops_after_safe_sign_margin() -> None:
    predicted = torch.tensor([[0.06, -0.06, 0.06, -0.06]], requires_grad=True)
    normalized_target = torch.tensor([[0.1, -0.1, 0.9, -0.9]])
    target_edges = torch.tensor([[0.10, 0.10, 0.12, 0.12]])

    loss = balanced_boundary_direction_loss(
        predicted,
        normalized_target,
        target_edges,
        image_size=640,
        global_bucket_counts=(4, 0, 0),
        batches_per_epoch=1,
    )
    loss.backward()

    assert loss.item() == 0.0
    torch.testing.assert_close(predicted.grad, torch.zeros_like(predicted), rtol=0, atol=0)


def test_edge_margin_detaches_boundary_off_and_has_exact_hinge() -> None:
    target = torch.tensor([[0.20, 0.20, 0.40, 0.40]])
    stock = torch.tensor([[0.18, 0.18, 0.42, 0.42]])
    boundary_off = torch.tensor(
        [[0.19, 0.19, 0.41, 0.41]], requires_grad=True
    )
    full_bad = boundary_off.detach().clone().requires_grad_(True)
    failed = boundary_edge_margin_loss(
        full_bad,
        boundary_off,
        stock,
        target,
        image_size=640,
        global_bucket_counts=(0, 0, 4),
        batches_per_epoch=1,
    )
    failed.backward()

    assert failed.item() > 0.0
    assert boundary_off.grad is None
    assert full_bad.grad is not None and full_bad.grad.abs().sum() > 0

    full_good = target + 0.89 * (boundary_off.detach() - target)
    passed = boundary_edge_margin_loss(
        full_good,
        boundary_off,
        stock,
        target,
        image_size=640,
        global_bucket_counts=(0, 0, 4),
        batches_per_epoch=1,
    )
    assert passed.item() == 0.0


def test_edge_margin_cannot_pass_by_worsening_boundary_off() -> None:
    target = torch.tensor([[0.20, 0.20, 0.40, 0.40]])
    stock = torch.tensor([[0.19, 0.19, 0.41, 0.41]])
    full = stock.clone()
    normal_off = stock.clone()
    worse_off = torch.tensor([[0.18, 0.18, 0.42, 0.42]])
    kwargs = {
        "image_size": 640,
        "global_bucket_counts": (0, 0, 4),
        "batches_per_epoch": 1,
    }

    normal = boundary_edge_margin_loss(full, normal_off, stock, target, **kwargs)
    worsened = boundary_edge_margin_loss(full, worse_off, stock, target, **kwargs)

    torch.testing.assert_close(normal, worsened, rtol=0, atol=0)


def test_edge_margin_ignores_subpixel_reference_and_bounds_gradient() -> None:
    target = torch.tensor([[0.20, 0.20, 0.40, 0.40]])
    subpixel = 0.5 / 640
    stock = target + torch.tensor([[-subpixel, -subpixel, subpixel, subpixel]])
    boundary_off = stock.clone().requires_grad_(True)
    full = stock.clone().requires_grad_(True)

    loss = boundary_edge_margin_loss(
        full,
        boundary_off,
        stock,
        target,
        image_size=640,
        global_bucket_counts=(0, 0, 0),
        batches_per_epoch=1,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert boundary_off.grad is None
    torch.testing.assert_close(full.grad, torch.zeros_like(full), rtol=0, atol=0)


def test_boundary_auxiliary_gradients_reach_only_boundary_logits() -> None:
    stock = torch.tensor(
        [[[0.10, 0.10, 0.12, 0.12]]], requires_grad=True
    )
    base_gate = torch.zeros((1, 1, 4), requires_grad=True)
    boundary_gate = torch.zeros((1, 1, 4), requires_grad=True)
    base_residual = torch.zeros((1, 1, 4), requires_grad=True)
    boundary_residual = torch.zeros((1, 1, 4), requires_grad=True)
    output = _loss_output(
        base_gate_raw=base_gate,
        boundary_gate_raw=boundary_gate,
        base_residual_raw=base_residual,
        boundary_residual_raw=boundary_residual,
        stock_edges=stock,
    )
    target = torch.tensor([[0.099, 0.101, 0.121, 0.119]])

    losses = iber_private_loss(
        output,
        target_edges=target,
        match_indices=[(torch.tensor([0]), torch.tensor([0]))],
        rho=0.05,
        image_size=640,
        boundary_supervision=True,
        bucket_counts=IBERBucketCounts(direction=(4, 0, 0), margin=(4, 0, 0)),
        batches_per_epoch=1,
    )
    (losses.boundary_direction + losses.boundary_margin).backward()

    assert stock.grad is None
    assert base_gate.grad is None
    assert base_residual.grad is None
    assert boundary_residual.grad is not None
    assert boundary_residual.grad.abs().sum() > 0
    assert boundary_gate.grad is not None
    assert torch.isfinite(boundary_gate.grad).all()


def test_b0_boundary_auxiliary_is_exact_graph_zero() -> None:
    stock = torch.tensor([[[0.10, 0.10, 0.12, 0.12]]])
    base_gate = torch.randn((1, 1, 4), requires_grad=True)
    boundary_gate = torch.randn((1, 1, 4), requires_grad=True)
    base_residual = torch.randn((1, 1, 4), requires_grad=True)
    boundary_residual = torch.randn((1, 1, 4), requires_grad=True)
    output = _loss_output(
        base_gate_raw=base_gate,
        boundary_gate_raw=boundary_gate,
        base_residual_raw=base_residual,
        boundary_residual_raw=boundary_residual,
        stock_edges=stock,
    )

    losses = iber_private_loss(
        output,
        target_edges=torch.tensor([[0.099, 0.101, 0.121, 0.119]]),
        match_indices=[(torch.tensor([0]), torch.tensor([0]))],
        rho=0.05,
        image_size=640,
        boundary_supervision=False,
        bucket_counts=IBERBucketCounts(direction=(4, 0, 0), margin=(4, 0, 0)),
        batches_per_epoch=1,
    )
    auxiliary = losses.boundary_direction + losses.boundary_margin
    auxiliary.backward()

    assert auxiliary.item() == 0.0
    torch.testing.assert_close(boundary_gate.grad, torch.zeros_like(boundary_gate), rtol=0, atol=0)
    torch.testing.assert_close(
        boundary_residual.grad, torch.zeros_like(boundary_residual), rtol=0, atol=0
    )


def test_zero_error_and_tiny_border_boxes_remain_finite() -> None:
    target = torch.tensor([[0.0, 0.0, 1e-12, 1e-12]])
    full = target.clone().requires_grad_(True)
    boundary_off = target.clone().requires_grad_(True)

    direction = balanced_boundary_direction_loss(
        torch.zeros_like(target, requires_grad=True),
        torch.zeros_like(target),
        target,
        image_size=640,
        global_bucket_counts=(0, 0, 0),
        batches_per_epoch=1,
    )
    margin = boundary_edge_margin_loss(
        full,
        boundary_off,
        boundary_off,
        target,
        image_size=640,
        global_bucket_counts=(0, 0, 0),
        batches_per_epoch=1,
    )

    assert torch.isfinite(direction)
    assert torch.isfinite(margin)
    assert direction.item() == 0.0
    assert margin.item() == 0.0


def test_real_head_boundary_auxiliary_does_not_update_shared_base_path() -> None:
    model = IBERRefiner(
        hidden_dim=8,
        f3_channels=4,
        private_seed=17,
        probe="b3",
        image_size=640,
        rho=0.05,
    )
    with torch.no_grad():
        for head in (
            model.boundary_gate_head,
            model.boundary_residual_head,
            *model.scale_gate_heads,
            *model.scale_residual_heads,
            *model.boundary_edge_gate_heads,
            *model.boundary_edge_residual_heads,
        ):
            head.weight.fill_(0.02)
    generator = torch.Generator().manual_seed(44)
    output = model(
        torch.randn(1, 1, 8, generator=generator),
        torch.tensor([[[0.11, 0.11, 0.02, 0.02]]]),
        torch.randn(1, 1, 3, generator=generator),
        torch.randn(1, 4, 8, 8, generator=generator),
        torch.randn(1, 3, 32, 32, generator=generator),
    )
    losses = iber_private_loss(
        output,
        target_edges=torch.tensor([[0.099, 0.101, 0.121, 0.119]]),
        match_indices=[(torch.tensor([0]), torch.tensor([0]))],
        rho=0.05,
        image_size=640,
        boundary_supervision=True,
        bucket_counts=IBERBucketCounts(direction=(4, 0, 0), margin=(4, 0, 0)),
        batches_per_epoch=1,
    )
    (losses.boundary_direction + losses.boundary_margin).backward()

    for module in (
        model.area_calibration,
        model.context_path,
        model.base_gate_head,
        model.base_residual_head,
    ):
        assert all(parameter.grad is None for parameter in module.parameters())
    boundary_gradients = [
        parameter.grad
        for module in (model.f3_encoder, model.rgb_encoder)
        for parameter in module.parameters()
    ]
    assert all(gradient is not None for gradient in boundary_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in boundary_gradients) > 0
