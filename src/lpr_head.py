"""Localization-prior refinement components for RT-DETR.

The refiner is identity-initialized so adding it to a stock decoder does not
change any predicted box before training. Geometry is intentionally detached:
it conditions refinement without creating a second gradient path through the
stock box regression head.
"""

from __future__ import annotations

import torch
from torch import nn


def box_geometry_prior(boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Encode normalized ``(cx, cy, width, height)`` boxes into six priors."""
    detached = boxes.detach()
    center = detached[..., :2].mul(2).sub(1)
    width, height = detached[..., 2:].clamp_min(eps).unbind(-1)
    scale = torch.stack(
        (width.log(), height.log(), (width * height).log(), (width / height).log()),
        dim=-1,
    ).clamp(-12, 12)
    return torch.cat((center, scale), dim=-1)


class LocalizationPriorRefiner(nn.Module):
    """Predict a bounded geometry-aware residual around a stock decoder box."""

    def __init__(self, hidden_dim: int, seed: int, max_logit_delta: float = 0.5) -> None:
        super().__init__()
        # Deterministic private initialization must not perturb the parent
        # model's global parameter-initialization stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.query_path = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 64),
                nn.SiLU(),
            )
            self.geometry_path = nn.Sequential(nn.Linear(6, 16), nn.SiLU())
            self.residual_head = nn.Linear(80, 4)

        self.alpha = nn.Parameter(torch.zeros(()))
        self.max_logit_delta = float(max_logit_delta)

    def forward(self, hidden: torch.Tensor, stock_boxes: torch.Tensor) -> torch.Tensor:
        geometry = self.geometry_path(box_geometry_prior(stock_boxes).to(hidden.dtype))
        features = torch.cat((self.query_path(hidden), geometry), dim=-1)
        residual = torch.tanh(self.residual_head(features))
        stock_logits = torch.logit(stock_boxes.clamp(1e-6, 1 - 1e-6))
        candidate = torch.sigmoid(stock_logits + self.max_logit_delta * residual)
        gate = 0.5 * torch.tanh(self.alpha)
        return (stock_boxes + gate * (candidate - stock_boxes)).clamp(1e-6, 1 - 1e-6)
