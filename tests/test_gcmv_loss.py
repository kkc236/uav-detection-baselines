from __future__ import annotations

import torch

from src.gcmv_loss import (
    build_gcmv_scale_targets,
    gcmv_auxiliary_loss,
)


def test_scale_targets_separate_tiny_gaussian_and_non_tiny_region():
    bboxes = torch.tensor(
        [
            [0.25, 0.25, 8 / 640, 8 / 640],
            [0.75, 0.75, 64 / 640, 48 / 640],
        ],
        dtype=torch.float32,
    )
    batch_idx = torch.tensor([0, 0])

    tiny, non_tiny = build_gcmv_scale_targets(
        bboxes=bboxes,
        batch_idx=batch_idx,
        batch_size=1,
        feature_shape=(80, 80),
        image_shape=(640, 640),
        tiny_max_size=16.0,
    )

    assert tiny.shape == (1, 1, 80, 80)
    assert non_tiny.shape == tiny.shape
    assert tiny[0, 0, 20, 20] == 1.0
    assert non_tiny[0, 0, 60, 60] == 1.0
    assert non_tiny[0, 0, 20, 20] == 0.0


def test_auxiliary_loss_is_finite_and_backpropagates_to_maps():
    tiny_map = torch.full(
        (1, 1, 8, 8),
        0.4,
        requires_grad=True,
    )
    gate_hat = torch.full(
        (1, 1, 8, 8),
        0.5,
        requires_grad=True,
    )
    gate = torch.full(
        (1, 1, 8, 8),
        0.2,
        requires_grad=True,
    )
    coverage = torch.ones_like(tiny_map)
    tiny_target = torch.zeros_like(tiny_map)
    tiny_target[:, :, 3, 4] = 1.0
    non_tiny = torch.zeros_like(tiny_map)
    non_tiny[:, :, 5:7, 5:7] = 1.0

    output = gcmv_auxiliary_loss(
        tiny_map=tiny_map,
        gate_hat=gate_hat,
        gate=gate,
        coverage=coverage,
        tiny_target=tiny_target,
        non_tiny_mask=non_tiny,
    )
    output.total.backward()

    assert torch.isfinite(output.total)
    assert output.tiny.item() > 0
    assert output.gate.item() > 0
    assert output.protect.item() > 0
    assert tiny_map.grad is not None and tiny_map.grad.abs().sum() > 0
    assert gate_hat.grad is not None and gate_hat.grad.abs().sum() > 0
    assert gate.grad is not None and gate.grad.abs().sum() > 0
