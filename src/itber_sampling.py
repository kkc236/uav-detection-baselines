"""Sparse four-edge evidence sampling for I-TBER."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from src.itber_geometry import cxcywh_to_xyxy


def _require_boxes(boxes: torch.Tensor) -> None:
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [batch, queries, 4]")
    if not torch.is_floating_point(boxes):
        raise TypeError("boxes must be a floating-point tensor")


def boundary_sample_grid(
    boxes: torch.Tensor,
    *,
    image_size: int,
) -> torch.Tensor:
    """Build an outside/edge/inside grid for left, top, right, and bottom."""
    _require_boxes(boxes)
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    boxes = boxes.detach()
    edges = cxcywh_to_xyxy(boxes)
    left, top, right, bottom = edges.unbind(dim=-1)
    width = (right - left).clamp_min(0)
    height = (bottom - top).clamp_min(0)
    distance = (0.08 * torch.minimum(width, height)).clamp(
        min=1.0 / image_size,
        max=4.0 / image_size,
    )
    along = torch.tensor(
        (0.25, 0.50, 0.75),
        device=boxes.device,
        dtype=boxes.dtype,
    ).view(1, 1, 3)

    vertical = top.unsqueeze(-1) + height.unsqueeze(-1) * along
    horizontal = left.unsqueeze(-1) + width.unsqueeze(-1) * along
    offsets = torch.stack((-distance, torch.zeros_like(distance), distance), dim=-1)

    left_x = (left.unsqueeze(-1).unsqueeze(-1) + offsets.unsqueeze(-2)).expand(
        -1, -1, 3, -1
    )
    left_y = vertical.unsqueeze(-1).expand(-1, -1, -1, 3)
    top_x = horizontal.unsqueeze(-1).expand(-1, -1, -1, 3)
    top_y = (top.unsqueeze(-1).unsqueeze(-1) + offsets.unsqueeze(-2)).expand(
        -1, -1, 3, -1
    )
    right_x = (right.unsqueeze(-1).unsqueeze(-1) - offsets.unsqueeze(-2)).expand(
        -1, -1, 3, -1
    )
    right_y = vertical.unsqueeze(-1).expand(-1, -1, -1, 3)
    bottom_x = horizontal.unsqueeze(-1).expand(-1, -1, -1, 3)
    bottom_y = (bottom.unsqueeze(-1).unsqueeze(-1) - offsets.unsqueeze(-2)).expand(
        -1, -1, 3, -1
    )

    grid = torch.stack(
        (
            torch.stack((left_x, left_y), dim=-1),
            torch.stack((top_x, top_y), dim=-1),
            torch.stack((right_x, right_y), dim=-1),
            torch.stack((bottom_x, bottom_y), dim=-1),
        ),
        dim=2,
    )
    return grid.clamp(0, 1).mul(2).sub(1)


def sample_boundary_evidence(
    features: torch.Tensor,
    boxes: torch.Tensor,
    *,
    image_size: int,
) -> torch.Tensor:
    """Sample F3 and return edge/contrast/absolute-contrast evidence per edge."""
    if features.ndim != 4:
        raise ValueError("features must have shape [batch, channels, height, width]")
    _require_boxes(boxes)
    if features.shape[0] != boxes.shape[0]:
        raise ValueError("features and boxes must have the same batch size")
    if not torch.is_floating_point(features):
        raise TypeError("features must be a floating-point tensor")

    batch, channels = features.shape[:2]
    queries = boxes.shape[1]
    grid = boundary_sample_grid(boxes, image_size=image_size).to(dtype=features.dtype)
    flat_grid = grid.reshape(batch, queries * 4 * 3, 3, 2)
    sampled = F.grid_sample(
        features,
        flat_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    sampled = sampled.reshape(batch, channels, queries, 4, 3, 3)
    sampled = sampled.permute(0, 2, 3, 1, 4, 5).mean(dim=-2)
    outside = sampled[..., 0]
    edge = sampled[..., 1]
    inside = sampled[..., 2]
    contrast = inside - outside
    return torch.cat((edge, contrast, contrast.abs()), dim=-1)
