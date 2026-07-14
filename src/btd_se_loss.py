from __future__ import annotations

import torch


def binary_focal_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.25,
    exponent: float = 2.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return mean class-balanced focal BCE for probabilities and hard or soft binary targets."""
    if probabilities.shape != target.shape:
        raise ValueError("probabilities and target must have identical shapes")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if exponent < 0:
        raise ValueError("exponent must be non-negative")

    eps = 1e-4 if probabilities.dtype == torch.float16 else 1e-7
    probability = probabilities.clamp(min=eps, max=1.0 - eps)
    positive = -alpha * (1.0 - probability).pow(exponent) * target * probability.log()
    negative = -(1.0 - alpha) * probability.pow(exponent) * (1.0 - target) * (1.0 - probability).log()
    loss = positive + negative

    if valid_mask is None:
        return loss.mean()
    if valid_mask.shape != loss.shape:
        raise ValueError("valid_mask must have the same shape as probabilities")
    valid = valid_mask.to(dtype=torch.bool)
    if not torch.any(valid):
        return probabilities.sum() * 0.0
    return loss[valid].mean()
