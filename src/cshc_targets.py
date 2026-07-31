"""Target construction for CSHC's class-agnostic tiny-location supervision."""

from __future__ import annotations

import torch
from torch import Tensor


def build_tiny_center_targets(
    bboxes: Tensor,
    batch_idx: Tensor,
    batch_size: int,
    height: int,
    width: int,
    tiny_area_threshold: float = 0.0025,
) -> Tensor:
    """Return a BCHW center map for normalized ``xywh`` boxes below the area threshold."""
    if batch_size <= 0 or height <= 0 or width <= 0:
        raise ValueError("batch_size, height and width must all be positive")
    if bboxes.ndim != 2 or bboxes.shape[-1] != 4:
        raise ValueError(f"bboxes must have shape (N, 4), got {tuple(bboxes.shape)}")
    if batch_idx.ndim != 1 or batch_idx.numel() != bboxes.shape[0]:
        raise ValueError("batch_idx must have one entry per bbox")
    if not 0.0 < tiny_area_threshold <= 1.0:
        raise ValueError("tiny_area_threshold must be in (0, 1]")
    if not torch.isfinite(bboxes).all():
        raise ValueError("bboxes must be finite")
    if bboxes.numel() and not ((bboxes >= 0.0) & (bboxes <= 1.0)).all():
        raise ValueError("normalized bboxes must lie in [0, 1]")
    if batch_idx.numel() and ((batch_idx < 0) | (batch_idx >= batch_size)).any():
        raise ValueError("batch_idx contains an out-of-range image index")

    target = bboxes.new_zeros((batch_size, 1, height, width))
    if not bboxes.numel():
        return target

    tiny = bboxes[:, 2].mul(bboxes[:, 3]) <= tiny_area_threshold
    if not tiny.any():
        return target

    boxes = bboxes[tiny]
    images = batch_idx[tiny].to(dtype=torch.long, device=bboxes.device)
    col = torch.floor(boxes[:, 0] * width).to(dtype=torch.long).clamp_(0, width - 1)
    row = torch.floor(boxes[:, 1] * height).to(dtype=torch.long).clamp_(0, height - 1)
    target[images, 0, row, col] = 1.0
    return target
