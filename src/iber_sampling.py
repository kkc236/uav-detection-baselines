"""Exact dual-resolution boundary evidence sampling for IBER-BE v1.0."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from src.itber_geometry import cxcywh_to_xyxy


def _require_boxes(boxes: torch.Tensor) -> None:
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [batch, queries, 4]")
    if not torch.is_floating_point(boxes):
        raise TypeError("boxes must be a floating-point tensor")


def _require_image_size(image_size: int) -> None:
    if image_size <= 0:
        raise ValueError("image_size must be positive")


def _require_values(value: torch.Tensor, name: str, channels: int) -> None:
    if (
        value.ndim != 4
        or value.shape[1] != channels
        or value.shape[2] <= 0
        or value.shape[3] <= 0
    ):
        raise ValueError(
            f"{name} must have shape [batch, {channels}, height, width] "
            "with positive spatial dimensions"
        )
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating-point tensor")


def _boxes_fp32(boxes: torch.Tensor) -> torch.Tensor:
    return boxes.detach().to(dtype=torch.float32)


def rgb_normal_radii(
    boxes: torch.Tensor, image_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return clipped near and far RGB normal radii in normalized units."""
    _require_boxes(boxes)
    _require_image_size(image_size)

    boxes_fp32 = _boxes_fp32(boxes)
    width, height = boxes_fp32[..., 2:].unbind(-1)
    minimum = torch.minimum(width, height)
    near = (0.08 * minimum).clamp(1 / image_size, 4 / image_size)
    far = (0.20 * minimum).clamp(2 / image_size, 8 / image_size)
    return near, far


def _f3_normal_radius(boxes: torch.Tensor) -> torch.Tensor:
    boxes_fp32 = _boxes_fp32(boxes)
    width, height = boxes_fp32[..., 2:].unbind(-1)
    minimum = torch.minimum(width, height)
    return (0.08 * minimum).clamp(1 / 640, 4 / 640)


def _boundary_grid_fp32(
    boxes: torch.Tensor, normal_positions: torch.Tensor
) -> torch.Tensor:
    """Build left/top/right/bottom grids with semantic outside-to-inside positions."""
    boxes_fp32 = _boxes_fp32(boxes)
    positions_fp32 = normal_positions.detach().to(
        device=boxes.device, dtype=torch.float32
    )
    left, top, right, bottom = cxcywh_to_xyxy(boxes_fp32).unbind(-1)
    width = right - left
    height = bottom - top
    along = torch.tensor(
        (0.25, 0.50, 0.75), device=boxes.device, dtype=torch.float32
    ).view(1, 1, 3)

    vertical = top.unsqueeze(-1) + height.unsqueeze(-1) * along
    horizontal = left.unsqueeze(-1) + width.unsqueeze(-1) * along
    offsets = positions_fp32.unsqueeze(-2)
    normal_count = positions_fp32.shape[-1]

    left_x = (left.unsqueeze(-1).unsqueeze(-1) + offsets).expand(
        -1, -1, 3, -1
    )
    left_y = vertical.unsqueeze(-1).expand(-1, -1, -1, normal_count)
    top_x = horizontal.unsqueeze(-1).expand(-1, -1, -1, normal_count)
    top_y = (top.unsqueeze(-1).unsqueeze(-1) + offsets).expand(
        -1, -1, 3, -1
    )
    right_x = (right.unsqueeze(-1).unsqueeze(-1) - offsets).expand(
        -1, -1, 3, -1
    )
    right_y = vertical.unsqueeze(-1).expand(-1, -1, -1, normal_count)
    bottom_x = horizontal.unsqueeze(-1).expand(-1, -1, -1, normal_count)
    bottom_y = (bottom.unsqueeze(-1).unsqueeze(-1) - offsets).expand(
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
    return grid.mul(2).sub(1)


def _sample_grid(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    batch, channels = values.shape[:2]
    queries = grid.shape[1]
    normal_count = grid.shape[-2]
    flat_grid = grid.reshape(batch, queries * 4 * 3, normal_count, 2)
    sampled = F.grid_sample(
        values.to(dtype=torch.float32),
        flat_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    sampled = sampled.reshape(batch, channels, queries, 4, 3, normal_count)
    sampled = sampled.permute(0, 2, 3, 1, 4, 5).mean(dim=-2)
    return sampled


def _require_sampling_inputs(
    values: torch.Tensor,
    boxes: torch.Tensor,
    *,
    name: str,
    channels: int,
    image_size: int,
) -> None:
    _require_values(values, name, channels)
    _require_boxes(boxes)
    _require_image_size(image_size)
    if values.shape[0] != boxes.shape[0]:
        raise ValueError(f"{name} and boxes must have the same batch size")
    if values.device != boxes.device:
        raise ValueError(f"{name} and boxes must be on the same device")


def sample_rgb_boundary_evidence(
    images: torch.Tensor, boxes: torch.Tensor, *, image_size: int
) -> torch.Tensor:
    """Return edge and dual-radius RGB contrasts with shape [B, Q, 4, 15]."""
    _require_sampling_inputs(
        images, boxes, name="images", channels=3, image_size=image_size
    )
    near, far = rgb_normal_radii(boxes, image_size)
    zero = torch.zeros_like(near)
    normal_positions = torch.stack((-far, -near, zero, near, far), dim=-1)
    sampled = _sample_grid(images, _boundary_grid_fp32(boxes, normal_positions))

    far_outside = sampled[..., 0]
    near_outside = sampled[..., 1]
    edge = sampled[..., 2]
    near_inside = sampled[..., 3]
    far_inside = sampled[..., 4]
    near_contrast = near_inside - near_outside
    far_contrast = far_inside - far_outside
    evidence = torch.cat(
        (
            edge,
            near_contrast,
            near_contrast.abs(),
            far_contrast,
            far_contrast.abs(),
        ),
        dim=-1,
    )
    return evidence.to(dtype=images.dtype)


def sample_f3_boundary_evidence(
    features: torch.Tensor, boxes: torch.Tensor, *, image_size: int
) -> torch.Tensor:
    """Return edge and one-radius F3 contrasts with shape [B, Q, 4, 96]."""
    _require_sampling_inputs(
        features, boxes, name="features", channels=32, image_size=image_size
    )
    distance = _f3_normal_radius(boxes)
    normal_positions = torch.stack(
        (-distance, torch.zeros_like(distance), distance), dim=-1
    )
    sampled = _sample_grid(features, _boundary_grid_fp32(boxes, normal_positions))

    outside = sampled[..., 0]
    edge = sampled[..., 1]
    inside = sampled[..., 2]
    contrast = inside - outside
    evidence = torch.cat((edge, contrast, contrast.abs()), dim=-1)
    return evidence.to(dtype=features.dtype)
