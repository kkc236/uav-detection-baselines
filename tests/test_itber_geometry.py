from __future__ import annotations

import torch

from src.itber_geometry import (
    apply_edge_update,
    correction_targets,
    cxcywh_to_xyxy,
    trajectory_state,
    xyxy_to_cxcywh,
)


def test_gate_magnitude_times_direction_reconstructs_clipped_correction() -> None:
    stock = torch.tensor([[[0.40, 0.40, 0.60, 0.60]]])
    target = torch.tensor([[[0.39, 0.42, 0.61, 0.58]]])

    magnitude, direction, normalized = correction_targets(stock, target, rho=0.05)

    torch.testing.assert_close(magnitude * direction, normalized)
    assert torch.all((magnitude >= 0) & (magnitude <= 1))
    assert set(torch.unique(direction).tolist()) <= {-1.0, 0.0, 1.0}


def test_zero_correction_has_zero_direction_and_identity_update() -> None:
    box = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
    edges = cxcywh_to_xyxy(box)

    magnitude, direction, normalized = correction_targets(edges, edges, rho=0.05)
    refined = apply_edge_update(edges, magnitude, direction, rho=0.05)

    torch.testing.assert_close(magnitude, torch.zeros_like(magnitude), rtol=0, atol=0)
    torch.testing.assert_close(direction, torch.zeros_like(direction), rtol=0, atol=0)
    torch.testing.assert_close(normalized, torch.zeros_like(normalized), rtol=0, atol=0)
    torch.testing.assert_close(refined, edges, rtol=0, atol=0)


def test_box_conversion_round_trip_is_exact_for_binary_fractions() -> None:
    boxes = torch.tensor([[[0.5, 0.5, 0.25, 0.125]]])

    restored = xyxy_to_cxcywh(cxcywh_to_xyxy(boxes))

    torch.testing.assert_close(restored, boxes, rtol=0, atol=0)


def test_trajectory_state_keeps_per_edge_direction() -> None:
    edge_l2 = torch.tensor([[[0.20, 0.20, 0.40, 0.40]]])
    edge_l1 = torch.tensor([[[0.21, 0.19, 0.41, 0.39]]])
    edge_l = torch.tensor([[[0.22, 0.18, 0.42, 0.38]]])

    state = trajectory_state(edge_l2, edge_l1, edge_l)

    assert state.shape == (1, 1, 4, 6)
    assert state[0, 0, 0, 0] > 0
    assert state[0, 0, 1, 0] < 0
    assert torch.isfinite(state).all()


def test_update_clamps_extreme_corrections_to_valid_ordered_edges() -> None:
    stock = torch.tensor([[[0.9999999, 0.9999999, 1.0, 1.0]]])
    gate = torch.ones_like(stock)
    residual = torch.tensor([[[1.0, 1.0, -1.0, -1.0]]])

    refined = apply_edge_update(stock, gate, residual, rho=10.0)

    assert torch.isfinite(refined).all()
    assert torch.all((refined >= 0) & (refined <= 1))
    assert torch.all(refined[..., 2] > refined[..., 0])
    assert torch.all(refined[..., 3] > refined[..., 1])
