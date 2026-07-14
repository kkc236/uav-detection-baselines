from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def ring_sum(x: torch.Tensor) -> torch.Tensor:
    """Return the zero-padded 9x9 neighborhood sum excluding its central 5x5 region."""
    outer = 81.0 * F.avg_pool2d(x, kernel_size=9, stride=1, padding=4, count_include_pad=True)
    inner = 25.0 * F.avg_pool2d(x, kernel_size=5, stride=1, padding=2, count_include_pad=True)
    return outer - inner


class BTDSE(nn.Module):
    """Background-reference-guided target-background decoupling and saliency enhancement."""

    def __init__(self, channels: int = 256, embedding_channels: int = 32, tau: float = 1.0) -> None:
        super().__init__()
        if channels <= 0 or embedding_channels <= 0:
            raise ValueError("channels and embedding_channels must be positive")
        if tau <= 0:
            raise ValueError("tau must be positive")

        self.channels = channels
        self.tau = float(tau)
        self.background_head = nn.Conv2d(channels * 2, 1, kernel_size=1)
        self.residual_projection = nn.Conv2d(channels, embedding_channels, kernel_size=1)
        self.semantic_projection = nn.Conv2d(channels, embedding_channels, kernel_size=1)
        self.saliency_head = nn.Conv2d(channels + 1, 1, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.last_background_reliability: torch.Tensor | None = None
        self.last_saliency: torch.Tensor | None = None
        self.last_residual: torch.Tensor | None = None
        self.last_normalizer: torch.Tensor | None = None

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.ndim != 4 or pair.shape[1] != self.channels * 2:
            raise ValueError(
                f"BTDSE expects Bx{self.channels * 2}xHxW input, received shape {tuple(pair.shape)}"
            )

        semantic, projected = pair.split(self.channels, dim=1)
        reliability = torch.sigmoid(self.background_head(torch.cat((projected, semantic), dim=1)))
        numerator = ring_sum(reliability * projected)
        normalizer = ring_sum(reliability).clamp_min(0.0)
        background = numerator / (normalizer + self.tau)
        residual = projected - background

        residual_embedding = self.residual_projection(residual)
        semantic_embedding = self.semantic_projection(semantic)
        consistency = F.cosine_similarity(
            residual_embedding,
            semantic_embedding,
            dim=1,
            eps=1e-6,
        ).unsqueeze(1)
        saliency = torch.sigmoid(self.saliency_head(torch.cat((residual, consistency), dim=1)))
        enhanced = projected + self.gamma * saliency * residual

        self.last_background_reliability = reliability
        self.last_saliency = saliency
        self.last_residual = residual
        self.last_normalizer = normalizer

        return torch.cat((semantic, enhanced), dim=1)
