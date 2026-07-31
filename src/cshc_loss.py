"""Numerically stable loss for the CSHC candidate-location map."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def focal_binary_logits(logits: Tensor, target: Tensor, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    """Mean binary focal loss, evaluated directly from logits for numerical stability."""
    if logits.shape != target.shape:
        raise ValueError(f"logits and target must share a shape, got {tuple(logits.shape)} and {tuple(target.shape)}")
    if not 0.0 <= alpha <= 1.0 or gamma < 0.0:
        raise ValueError("alpha must be in [0, 1] and gamma must be non-negative")
    target = target.to(device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()
