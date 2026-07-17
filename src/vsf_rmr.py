from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VSFRMRAuxiliaryState:
    scale_field: torch.Tensor
    global_scale: torch.Tensor


def ordered_scale_weights(scale_field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map a scale field in (0, 2) to adjacent P3/P4/P5 routing weights."""
    alpha3 = F.relu(1.0 - scale_field)
    alpha5 = F.relu(scale_field - 1.0)
    alpha4 = 1.0 - alpha3 - alpha5
    return alpha3, alpha4, alpha5


class VSFRMR(nn.Module):
    """View-scale-field-guided residual multi-scale routing."""

    def __init__(
        self,
        channels: int = 256,
        route_channels: int = 32,
    ) -> None:
        super().__init__()
        if channels <= 0 or route_channels <= 0:
            raise ValueError("channels and route_channels must be positive")

        self.channels = int(channels)
        self.route_channels = int(route_channels)
        self.shared_projection = nn.Conv2d(channels, route_channels, kernel_size=1)

        fused_channels = route_channels * 3
        self.global_bias = nn.Parameter(torch.full((1, 1, 1, 1), -0.1))

        self.local_head = nn.Conv2d(
            fused_channels,
            1,
            kernel_size=3,
            padding=1,
        )
        nn.init.normal_(self.local_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.local_head.bias)

        self.shared_restore = nn.Conv2d(route_channels, channels, kernel_size=1)
        self.gamma = nn.ParameterList(
            nn.Parameter(torch.zeros(1, channels, 1, 1)) for _ in range(3)
        )
        self._auxiliary_state: VSFRMRAuxiliaryState | None = None

    def _validate(self, features: Sequence[torch.Tensor]) -> None:
        if len(features) != 3:
            raise ValueError(f"VSFRMR expects three feature levels, received {len(features)}")
        for index, feature in enumerate(features):
            if feature.ndim != 4:
                raise ValueError(f"feature level {index} must have shape BxCxHxW")
            if feature.shape[1] != self.channels:
                raise ValueError(
                    f"feature level {index} must have {self.channels} channels, received {feature.shape[1]}"
                )
        f3, f4, f5 = features
        if f3.shape[0] != f4.shape[0] or f3.shape[0] != f5.shape[0]:
            raise ValueError("all feature levels must have the same batch size")
        if (
            f3.shape[-2] != 2 * f4.shape[-2]
            or f3.shape[-1] != 2 * f4.shape[-1]
            or f3.shape[-2] != 4 * f5.shape[-2]
            or f3.shape[-1] != 4 * f5.shape[-1]
        ):
            raise ValueError("feature spatial sizes must follow exact 2x and 4x pyramid ratios")

    def peek_auxiliary_state(self) -> VSFRMRAuxiliaryState | None:
        return self._auxiliary_state

    def pop_auxiliary_state(self) -> VSFRMRAuxiliaryState | None:
        state = self._auxiliary_state
        self._auxiliary_state = None
        return state

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        self._auxiliary_state = None
        self._validate(features)
        f3, f4, f5 = features

        routed = [self.shared_projection(feature) for feature in (f3, f4, f5)]
        u3, u4, u5 = routed
        u4_high = F.interpolate(u4, size=u3.shape[-2:], mode="nearest")
        u5_high = F.interpolate(u5, size=u3.shape[-2:], mode="nearest")
        fused = torch.cat((u3, u4_high, u5_high), dim=1)

        local_logit = self.local_head(fused)
        global_logit = self.global_bias + local_logit.mean(dim=(2, 3), keepdim=True)
        global_scale = 2.0 * torch.sigmoid(global_logit)
        scale_field = 2.0 * torch.sigmoid(self.global_bias + local_logit)

        alpha3, alpha4, alpha5 = ordered_scale_weights(scale_field)
        mixed = alpha3 * u3 + alpha4 * u4_high + alpha5 * u5_high
        high_correction = self.shared_restore(mixed - u3)
        corrections = (
            high_correction,
            F.avg_pool2d(high_correction, kernel_size=2, stride=2),
            F.avg_pool2d(high_correction, kernel_size=4, stride=4),
        )
        if self.training and not bool(torch.isfinite(scale_field).all()):
            raise FloatingPointError("NONFINITE_VSF_RMR: scale_field")

        outputs = [
            feature + gamma * correction
            for feature, gamma, correction in zip((f3, f4, f5), self.gamma, corrections)
        ]
        if self.training:
            self._auxiliary_state = VSFRMRAuxiliaryState(
                scale_field=scale_field,
                global_scale=global_scale,
            )
        return outputs
