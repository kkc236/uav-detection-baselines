from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.vsf_rmr_loss import (
    compute_vsf_rmr_loss,
    ensure_finite_vsf_losses,
    sample_scale_field,
    scale_targets_from_boxes,
)


def test_scale_targets_use_actual_augmented_image_size():
    boxes = torch.tensor([[0.5, 0.5, 0.10, 0.40]])

    targets = scale_targets_from_boxes(boxes, image_size=(100, 200))

    expected_radius = math.sqrt(20.0 * 40.0)
    expected = math.log2(expected_radius / 8.0)
    torch.testing.assert_close(targets, torch.tensor([expected]))


def test_scale_targets_are_clipped_to_open_routing_interval():
    boxes = torch.tensor(
        [[0.5, 0.5, 1e-6, 1e-6], [0.5, 0.5, 1.0, 1.0]],
        dtype=torch.float16,
    )

    targets = scale_targets_from_boxes(boxes, image_size=(640, 640))

    assert targets.dtype == torch.float32
    torch.testing.assert_close(targets, torch.tensor([0.05, 1.95]))


def test_center_sampling_uses_align_corners_false_coordinates():
    field = torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]])
    centers = torch.tensor([[0.25, 0.25], [0.75, 0.75]])
    batch_indices = torch.tensor([0, 0])

    sampled = sample_scale_field(field, centers, batch_indices)

    torch.testing.assert_close(sampled, torch.tensor([0.2, 0.8]))


def test_local_loss_balances_images_instead_of_individual_targets():
    field = torch.zeros((2, 1, 4, 4), requires_grad=True)
    field.data[1].fill_(1.0)
    global_scale = torch.ones((2, 1, 1, 1), requires_grad=True)
    boxes = torch.tensor(
        [
            [0.5, 0.5, 0.025, 0.025],
            [0.2, 0.2, 0.025, 0.025],
            [0.5, 0.5, 0.025, 0.025],
            [0.8, 0.8, 0.025, 0.025],
        ]
    )
    batch_indices = torch.tensor([0, 1, 1, 1])

    result = compute_vsf_rmr_loss(
        scale_field=field,
        global_scale=global_scale,
        bboxes=boxes,
        batch_indices=batch_indices,
        image_size=(640, 640),
    )

    targets = scale_targets_from_boxes(boxes, image_size=(640, 640))
    image0 = F.smooth_l1_loss(torch.tensor([0.0]), targets[:1], beta=1.0)
    image1 = F.smooth_l1_loss(torch.ones(3), targets[1:], beta=1.0)
    torch.testing.assert_close(result.local, (image0 + image1) / 2.0)


def test_half_inputs_produce_fp32_losses_and_finite_gradients():
    field = torch.full((1, 1, 4, 4), 0.9, dtype=torch.float16, requires_grad=True)
    global_scale = torch.full((1, 1, 1, 1), 0.9, dtype=torch.float16, requires_grad=True)
    boxes = torch.tensor([[0.5, 0.5, 0.025, 0.025]], dtype=torch.float16)

    result = compute_vsf_rmr_loss(
        scale_field=field,
        global_scale=global_scale,
        bboxes=boxes,
        batch_indices=torch.tensor([0]),
        image_size=(640, 640),
    )
    (result.local + result.global_).backward()

    assert result.local.dtype == torch.float32
    assert result.global_.dtype == torch.float32
    assert torch.isfinite(result.local)
    assert torch.isfinite(result.global_)
    assert torch.isfinite(field.grad).all()
    assert torch.isfinite(global_scale.grad).all()


def test_empty_targets_return_exact_graph_connected_fp32_zero():
    field = torch.randn(2, 1, 4, 4, dtype=torch.float16, requires_grad=True)
    global_scale = torch.randn(2, 1, 1, 1, dtype=torch.float16, requires_grad=True)

    result = compute_vsf_rmr_loss(
        scale_field=field,
        global_scale=global_scale,
        bboxes=torch.empty((0, 4), dtype=torch.float16),
        batch_indices=torch.empty((0,), dtype=torch.long),
        image_size=(640, 640),
    )
    total = result.local + result.global_
    total.backward()

    assert total.dtype == torch.float32
    assert total.item() == 0.0
    assert field.grad is not None
    assert global_scale.grad is not None


def test_nonfinite_guard_raises_restart_marker():
    with pytest.raises(FloatingPointError, match="NONFINITE_LOSS"):
        ensure_finite_vsf_losses(local=torch.tensor(float("nan")), global_=torch.tensor(0.0))

