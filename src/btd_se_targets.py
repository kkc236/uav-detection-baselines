from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AuxiliaryTargets:
    background: torch.Tensor
    saliency: torch.Tensor
    saliency_valid: torch.Tensor


def _box_slices(
    box: torch.Tensor,
    *,
    height: int,
    width: int,
) -> tuple[slice, slice]:
    cx, cy, box_width, box_height = (float(value) for value in box)
    x0 = max(0, min(width - 1, math.floor((cx - box_width / 2) * width)))
    y0 = max(0, min(height - 1, math.floor((cy - box_height / 2) * height)))
    x1 = max(x0 + 1, min(width, math.ceil((cx + box_width / 2) * width)))
    y1 = max(y0 + 1, min(height, math.ceil((cy + box_height / 2) * height)))
    return slice(y0, y1), slice(x0, x1)


def build_auxiliary_targets(
    *,
    bboxes: torch.Tensor,
    classes: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
    minimum_sigma: float = 1.0,
) -> AuxiliaryTargets:
    """Rasterize augmented normalized boxes into BTD-SE background and saliency supervision maps."""
    if batch_size <= 0 or height <= 0 or width <= 0:
        raise ValueError("batch_size, height, and width must be positive")
    if minimum_sigma <= 0:
        raise ValueError("minimum_sigma must be positive")

    device = bboxes.device
    dtype = bboxes.dtype
    background = torch.ones((batch_size, 1, height, width), device=device, dtype=dtype)
    saliency = torch.zeros_like(background)
    saliency_valid = torch.ones_like(background, dtype=torch.bool)
    grid_y = torch.arange(height, device=device, dtype=dtype).view(height, 1)
    grid_x = torch.arange(width, device=device, dtype=dtype).view(1, width)

    flat_classes = classes.reshape(-1)
    flat_batch_idx = batch_idx.reshape(-1)
    if not (len(bboxes) == len(flat_classes) == len(flat_batch_idx)):
        raise ValueError("bboxes, classes, and batch_idx must have matching lengths")

    for box, class_id, image_index in zip(bboxes, flat_classes, flat_batch_idx):
        image = int(image_index.item())
        if image < 0 or image >= batch_size:
            raise ValueError(f"batch index {image} is outside batch size {batch_size}")

        y_slice, x_slice = _box_slices(box, height=height, width=width)
        background[image, 0, y_slice, x_slice] = 0

        if class_id.item() < 0:
            saliency_valid[image, 0, y_slice, x_slice] = False
            continue

        cx = torch.clamp(box[0] * width, min=0, max=width - 1)
        cy = torch.clamp(box[1] * height, min=0, max=height - 1)
        sigma_x = max(minimum_sigma, float(box[2]) * width / 8.0)
        sigma_y = max(minimum_sigma, float(box[3]) * height / 8.0)
        gaussian = torch.exp(
            -((grid_x - cx).square() / (2.0 * sigma_x**2) + (grid_y - cy).square() / (2.0 * sigma_y**2))
        )
        saliency[image, 0] = torch.maximum(saliency[image, 0], gaussian)

    saliency = torch.where(saliency_valid, saliency, torch.zeros_like(saliency))
    return AuxiliaryTargets(
        background=background,
        saliency=saliency,
        saliency_valid=saliency_valid,
    )
