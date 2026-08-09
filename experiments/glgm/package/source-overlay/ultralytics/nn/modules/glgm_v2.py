# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Lightweight scale-routed receptive-field module for the GLGM-v2 experiment."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv


class _ReceptiveFieldScaleGate(nn.Module):
    """Route channel groups between local and dilated receptive-field branches."""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        if channels % groups:
            raise ValueError(
                f"channels ({channels}) must be divisible by groups ({groups})"
            )
        hidden = max(16, channels // 4)
        self.channels = channels
        self.groups = groups
        self.channels_per_group = channels // groups
        self.reduce = nn.Conv2d(channels * 4, hidden, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.expand = nn.Conv2d(hidden, groups * 2, kernel_size=1, bias=True)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)
        self.last_weights: torch.Tensor | None = None

    def forward(self, local: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        descriptors = torch.cat(
            (
                F.adaptive_avg_pool2d(local, 1),
                F.adaptive_max_pool2d(local, 1),
                F.adaptive_avg_pool2d(context, 1),
                F.adaptive_max_pool2d(context, 1),
            ),
            dim=1,
        )
        logits = self.expand(self.act(self.reduce(descriptors)))
        weights = logits.view(logits.shape[0], 2, self.groups, 1, 1).softmax(dim=1)
        self.last_weights = weights.detach()
        batch, _, height, width = local.shape
        local_groups = local.view(
            batch, self.groups, self.channels_per_group, height, width
        )
        context_groups = context.view(
            batch, self.groups, self.channels_per_group, height, width
        )
        fused = (
            weights[:, 0].unsqueeze(2) * local_groups
            + weights[:, 1].unsqueeze(2) * context_groups
        )
        return fused.reshape(batch, self.channels, height, width)


class GLGMLite(nn.Module):
    """Lightweight GLGM with optional content-conditioned receptive-field routing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 96,
        gate_groups: int = 16,
        gated: bool = True,
        layer_scale_init: float = 0.01,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0 or gate_groups <= 0:
            raise ValueError("hidden_channels and gate_groups must be positive")
        if hidden_channels % gate_groups:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by gate_groups ({gate_groups})"
            )
        self.gated = bool(gated)
        self.reduce = Conv(in_channels, hidden_channels, k=1, act=nn.SiLU())
        self.local = Conv(
            hidden_channels,
            hidden_channels,
            k=3,
            g=hidden_channels,
            d=1,
            act=nn.SiLU(),
        )
        self.context = Conv(
            hidden_channels,
            hidden_channels,
            k=3,
            g=hidden_channels,
            d=3,
            act=nn.SiLU(),
        )
        self.gate = (
            _ReceptiveFieldScaleGate(hidden_channels, gate_groups)
            if self.gated
            else None
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            Conv(in_channels, out_channels, k=1, act=False)
            if in_channels != out_channels
            else None
        )
        self.layer_scale = nn.Parameter(
            torch.full((1, out_channels, 1, 1), float(layer_scale_init))
        )

    @property
    def gate_weights(self) -> torch.Tensor | None:
        return None if self.gate is None else self.gate.last_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x) if self.shortcut is not None else x
        reduced = self.reduce(x)
        local = self.local(reduced)
        context = self.context(reduced)
        fused = (
            self.gate(local, context)
            if self.gate is not None
            else 0.5 * (local + context)
        )
        return residual + self.layer_scale * self.project(fused)
