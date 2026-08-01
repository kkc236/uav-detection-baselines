"""Exact localization and activity metrics for I-TBER evaluations."""

from __future__ import annotations

import torch

from src.itber_geometry import xyxy_to_cxcywh


def aligned_iou(first: torch.Tensor, second: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Return aligned IoU for two equally shaped xyxy tensors."""
    if first.shape != second.shape or first.shape[-1] != 4:
        raise ValueError("aligned IoU inputs must have equal [..., 4] shapes")
    first_lower, first_upper = first.split(2, -1)
    second_lower, second_upper = second.split(2, -1)
    intersection = (
        torch.minimum(first_upper, second_upper)
        - torch.maximum(first_lower, second_lower)
    ).clamp_min(0).prod(-1)
    first_area = (first_upper - first_lower).clamp_min(0).prod(-1)
    second_area = (second_upper - second_lower).clamp_min(0).prod(-1)
    return intersection / (first_area + second_area - intersection).clamp_min(eps)


def area_bucket(boxes: torch.Tensor, *, image_size: int) -> torch.Tensor:
    """Map normalized cxcywh boxes to tiny(0), small(1), or other(2)."""
    if boxes.ndim != 2 or boxes.shape[-1] != 4 or image_size < 1:
        raise ValueError("boxes must have shape [N,4] and image size must be positive")
    area_pixels = boxes[:, 2:].clamp_min(0).prod(-1) * float(image_size**2)
    tiny_limit = float(16**2)
    small_limit = float(32**2)
    return torch.where(
        area_pixels < tiny_limit,
        torch.zeros_like(area_pixels, dtype=torch.long),
        torch.where(
            area_pixels < small_limit,
            torch.ones_like(area_pixels, dtype=torch.long),
            torch.full_like(area_pixels, 2, dtype=torch.long),
        ),
    )


def edge_area_bucket(target_edges: torch.Tensor, *, image_size: int) -> torch.Tensor:
    """Repeat each target area bucket for its four edge coordinates."""
    return area_bucket(xyxy_to_cxcywh(target_edges), image_size=image_size).unsqueeze(-1).expand(-1, 4)


def direction_accuracy(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Measure sign accuracy only where target direction is defined."""
    if predicted.shape != target.shape:
        raise ValueError("predicted and target directions must have equal shapes")
    valid = target.abs() > eps
    if mask is not None:
        if mask.shape != target.shape:
            raise ValueError("direction mask must match target shape")
        valid &= mask
    if not bool(valid.any()):
        return predicted.sum() * 0.0
    return (predicted[valid].sign() == target[valid].sign()).float().mean()


def correction_rms(correction: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return root-mean-square effective normalized edge correction."""
    values = correction if mask is None else correction[mask]
    if values.numel() == 0:
        return correction.sum() * 0.0
    return values.float().square().mean().sqrt()
