"""Detached quality-gated last-layer box refinement for RT-DETR."""

from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.modules.utils import inverse_sigmoid


def box_geometry_prior(boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Encode detached normalized boxes as stable center, scale, and aspect priors."""
    detached = boxes.detach()
    center = detached[..., :2].mul(2).sub(1)
    width, height = detached[..., 2:].clamp_min(eps).unbind(-1)
    scale = torch.stack(
        (width.log(), height.log(), (width * height).log(), (width / height).log()),
        dim=-1,
    ).clamp(-12, 12)
    return torch.cat((center, scale), dim=-1)


def detached_vfl_quality(stock_scores: torch.Tensor) -> torch.Tensor:
    """Return one detached Varifocal-score quality prior per query."""
    return stock_scores.detach().sigmoid().amax(dim=-1, keepdim=True)


class QualityGatedRefiner(nn.Module):
    """Predict a bounded per-query box residual without gradients to stock paths."""

    def __init__(self, hidden_dim: int, private_seed: int, max_logit_delta: float = 0.5) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(private_seed)
            self.query_path = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 64),
                nn.SiLU(),
            )
            self.geometry_path = nn.Sequential(nn.Linear(6, 16), nn.SiLU())
            self.gate_head = nn.Linear(80, 1)
            self.residual_head = nn.Linear(80, 4)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.max_logit_delta = float(max_logit_delta)
        self.last_quality: torch.Tensor | None = None
        self.last_gate: torch.Tensor | None = None
        self.last_residual: torch.Tensor | None = None

    def forward(
        self,
        hidden: torch.Tensor,
        stock_boxes: torch.Tensor,
        stock_scores: torch.Tensor,
    ) -> torch.Tensor:
        hidden = hidden.detach()
        stock_boxes = stock_boxes.detach()
        quality = detached_vfl_quality(stock_scores)
        geometry = box_geometry_prior(stock_boxes).to(dtype=hidden.dtype)
        features = torch.cat((self.query_path(hidden), self.geometry_path(geometry)), dim=-1)
        learned_gate = torch.sigmoid(self.gate_head(features))
        gate = (1 - quality.to(dtype=learned_gate.dtype)) * learned_gate
        residual = self.max_logit_delta * torch.tanh(self.residual_head(features))
        stock_logits = inverse_sigmoid(stock_boxes.clamp(1e-6, 1 - 1e-6))
        reconstructed_stock = torch.sigmoid(stock_logits)
        candidate = torch.sigmoid(stock_logits + gate * residual)
        refined = (stock_boxes + (candidate - reconstructed_stock)).clamp(1e-6, 1 - 1e-6)
        self.last_quality = quality.detach()
        self.last_gate = gate.detach()
        self.last_residual = residual.detach()
        return refined
