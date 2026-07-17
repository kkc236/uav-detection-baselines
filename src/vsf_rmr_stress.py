from __future__ import annotations

from collections.abc import Callable

import torch


def _box_corners(boxes: torch.Tensor, *, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    boxes = boxes.float()
    centers = boxes[:, :2] * boxes.new_tensor((width, height))
    half_sizes = boxes[:, 2:4].clamp_min(0.0) * boxes.new_tensor((width, height)) * 0.5
    minimum = centers - half_sizes
    maximum = centers + half_sizes
    return torch.stack(
        (
            minimum,
            torch.stack((maximum[:, 0], minimum[:, 1]), dim=1),
            maximum,
            torch.stack((minimum[:, 0], maximum[:, 1]), dim=1),
        ),
        dim=1,
    )


def _transform_boxes(
    boxes: torch.Tensor,
    *,
    image_size: tuple[int, int],
    transform: Callable[[torch.Tensor], torch.Tensor],
    minimum_area_retained: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [N, 4]")
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image_size must be positive")
    if not 0 <= minimum_area_retained <= 1:
        raise ValueError("minimum_area_retained must be in [0, 1]")

    transformed_corners = transform(_box_corners(boxes, image_size=image_size))
    minimum = transformed_corners.min(dim=1).values
    maximum = transformed_corners.max(dim=1).values
    original_size = (maximum - minimum).clamp_min(0.0)
    original_area = original_size.prod(dim=1)

    clipped_minimum = torch.maximum(minimum, minimum.new_tensor((0.0, 0.0)))
    clipped_maximum = torch.minimum(maximum, maximum.new_tensor((float(width), float(height))))
    clipped_size = (clipped_maximum - clipped_minimum).clamp_min(0.0)
    clipped_area = clipped_size.prod(dim=1)
    keep = (
        (clipped_size[:, 0] >= 1.0)
        & (clipped_size[:, 1] >= 1.0)
        & (clipped_area >= minimum_area_retained * original_area.clamp_min(1e-12))
    )

    center = (clipped_minimum + clipped_maximum) * 0.5
    normalized = torch.cat((center, clipped_size), dim=1) / boxes.new_tensor(
        (float(width), float(height), float(width), float(height)), dtype=torch.float32
    )
    return normalized.clamp(0.0, 1.0), keep


def centered_scale_boxes(
    boxes: torch.Tensor,
    *,
    image_size: tuple[int, int],
    factor: float,
    minimum_area_retained: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if factor <= 0:
        raise ValueError("factor must be positive")
    height, width = image_size
    center = torch.tensor((width * 0.5, height * 0.5), device=boxes.device, dtype=torch.float32)
    return _transform_boxes(
        boxes,
        image_size=image_size,
        transform=lambda points: center + (points - center) * float(factor),
        minimum_area_retained=minimum_area_retained,
    )


def vertical_perspective_boxes(
    boxes: torch.Tensor,
    *,
    image_size: tuple[int, int],
    coefficient: float,
    minimum_area_retained: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = image_size
    center = torch.tensor((width * 0.5, height * 0.5), device=boxes.device, dtype=torch.float32)

    def transform(points: torch.Tensor) -> torch.Tensor:
        centered = points - center
        denominator = 1.0 + float(coefficient) * centered[..., 1:2]
        if bool((denominator <= 1e-6).any()):
            raise ValueError("perspective coefficient produces a non-positive homogeneous denominator")
        return centered / denominator + center

    return _transform_boxes(
        boxes,
        image_size=image_size,
        transform=transform,
        minimum_area_retained=minimum_area_retained,
    )

