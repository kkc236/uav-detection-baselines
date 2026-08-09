"""Residual-difficulty-aware global-local refinement for FDR P3 features."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _ConvBNAct(nn.Sequential):
    """Bias-free convolution, BatchNorm, and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class RAGLGM(nn.Module):
    """P3-only global/local competition behind an exact identity gate.

    The private experts consume ``x.detach()``.  Consequently, detection and
    auxiliary gradients cannot alter the public FDR path through the private
    input branch.  A zero per-channel ``alpha`` makes both the initial output
    and the initial public gradient byte-identical to the baseline while the
    non-zero output projection gives ``alpha`` a first-step gradient.
    """

    def __init__(
        self,
        channels: int = 256,
        hidden_channels: int = 192,
        route_groups: int = 8,
        max_residual_scale: float = 0.5,
        private_seed: int = 20_000,
    ) -> None:
        super().__init__()
        if channels <= 0 or hidden_channels <= 0:
            raise ValueError("channels must be positive")
        if route_groups <= 0 or hidden_channels % route_groups:
            raise ValueError("hidden_channels must be divisible by route_groups")
        if not 0.0 < max_residual_scale <= 1.0:
            raise ValueError("max_residual_scale must be in (0, 1]")

        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.route_groups = int(route_groups)
        self.max_residual_scale = float(max_residual_scale)
        self.private_seed = int(private_seed)

        # All private initialization is deterministic and cannot advance the
        # public CPU RNG used by the surrounding FDR model construction.
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(self.private_seed)
            self.reduce = _ConvBNAct(self.channels, self.hidden_channels, 1)
            self.local_one = _ConvBNAct(
                self.hidden_channels,
                self.hidden_channels,
                3,
                padding=1,
            )
            self.local_two = _ConvBNAct(
                self.hidden_channels,
                self.hidden_channels,
                3,
                padding=1,
            )
            # These remain pure depthwise spatial operators: no pointwise
            # channel mixing is hidden behind either large receptive field.
            self.global_large = _ConvBNAct(
                self.hidden_channels,
                self.hidden_channels,
                7,
                padding=3,
                groups=self.hidden_channels,
            )
            self.global_dilated = _ConvBNAct(
                self.hidden_channels,
                self.hidden_channels,
                3,
                padding=3,
                dilation=3,
                groups=self.hidden_channels,
            )
            self.global_pool_projection = nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels,
                1,
                bias=True,
            )
            self.router = nn.Conv2d(
                self.hidden_channels,
                2 * self.route_groups,
                1,
                groups=self.route_groups,
                bias=True,
            )
            self.support_head = nn.Conv2d(
                self.hidden_channels,
                1,
                1,
                bias=True,
            )
            self.output_projection = nn.Conv2d(
                self.hidden_channels,
                self.channels,
                1,
                bias=False,
            )

        # Equal local/global routing is a scientific invariant, not a random
        # initialization side effect.
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.alpha = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
        self.last_support_map: Tensor | None = None
        self.last_route_weights: Tensor | None = None

    @property
    def private_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _route(
        self,
        routing_feature: Tensor,
        local: Tensor,
        global_feature: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, channels, height, width = local.shape
        if (
            routing_feature.shape != local.shape
            or global_feature.shape != local.shape
            or channels != self.hidden_channels
        ):
            raise ValueError("RA-GLGM expert feature shapes do not match")
        # Grouped-convolution channels are ordered [g0-local, g0-global,
        # g1-local, g1-global, ...], so expose expert before group explicitly.
        logits = self.router(routing_feature).view(
            batch,
            self.route_groups,
            2,
            height,
            width,
        ).transpose(1, 2)
        weights = logits.softmax(dim=1)
        channels_per_group = channels // self.route_groups
        channel_weights = weights.repeat_interleave(channels_per_group, dim=2)
        fused = (
            channel_weights[:, 0] * local
            + channel_weights[:, 1] * global_feature
        )
        return fused, weights

    def forward_with_diagnostics(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return refined P3, two-expert route weights, and support probability."""

        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"RAGLGM expects [B,{self.channels},H,W], got {tuple(x.shape)}"
            )
        private_x = x.detach()
        reduced = self.reduce(private_x)
        local = reduced + self.local_two(self.local_one(reduced))
        global_feature = self.global_dilated(self.global_large(reduced))
        global_feature = global_feature + self.global_pool_projection(
            F.adaptive_avg_pool2d(reduced, 1)
        )
        fused, weights = self._route(reduced, local, global_feature)
        support = self.support_head(fused).sigmoid()
        residual = (
            self.max_residual_scale
            * self.alpha.tanh()
            * support
            * self.output_projection(fused).tanh()
        )
        return x + residual, weights, support

    def forward(self, x: Tensor) -> Tensor:
        output, weights, support = self.forward_with_diagnostics(x)
        self.last_support_map = support
        self.last_route_weights = weights.detach()
        return output


__all__ = ["RAGLGM"]
