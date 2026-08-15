from __future__ import annotations

import torch

from src.itber_sampling import boundary_sample_grid, sample_boundary_evidence


def test_grid_has_four_edges_three_positions_and_three_normal_offsets() -> None:
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.1]]])

    grid = boundary_sample_grid(boxes, image_size=640)

    assert grid.shape == (1, 1, 4, 3, 3, 2)
    assert torch.isfinite(grid).all()
    assert grid.min() >= -1
    assert grid.max() <= 1


def test_constant_feature_produces_zero_inside_outside_difference() -> None:
    f3 = torch.ones(1, 32, 80, 80)
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])

    evidence = sample_boundary_evidence(f3, boxes, image_size=640)

    assert evidence.shape == (1, 1, 4, 96)
    torch.testing.assert_close(
        evidence[..., 32:], torch.zeros_like(evidence[..., 32:]), rtol=0, atol=0
    )


def test_edge_normals_point_inside_for_horizontal_coordinate_feature() -> None:
    x_coordinate = torch.linspace(0, 1, 80).view(1, 1, 1, 80).expand(1, 1, 80, 80)
    boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]])

    evidence = sample_boundary_evidence(x_coordinate, boxes, image_size=640)
    inside_minus_outside = evidence[..., 1]

    assert inside_minus_outside[0, 0, 0] > 0
    assert inside_minus_outside[0, 0, 2] < 0
    assert inside_minus_outside[0, 0, 1].abs() < 1e-6
    assert inside_minus_outside[0, 0, 3].abs() < 1e-6


def test_sampling_detaches_boxes_but_keeps_feature_gradient() -> None:
    f3 = torch.randn(1, 4, 20, 20, requires_grad=True)
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]], requires_grad=True)

    sample_boundary_evidence(f3, boxes, image_size=640).sum().backward()

    assert f3.grad is not None
    assert torch.isfinite(f3.grad).all()
    assert boxes.grad is None


def test_tiny_and_out_of_bounds_boxes_stay_finite() -> None:
    f3 = torch.randn(1, 2, 16, 16)
    boxes = torch.tensor(
        [[[0.0, 0.0, 1e-12, 1e-12], [1.2, -0.2, 0.5, 0.5]]],
        dtype=torch.float32,
    )

    grid = boundary_sample_grid(boxes, image_size=640)
    evidence = sample_boundary_evidence(f3, boxes, image_size=640)

    assert torch.isfinite(grid).all()
    assert torch.isfinite(evidence).all()
