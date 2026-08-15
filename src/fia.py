from __future__ import annotations

import torch
from torch import Tensor, nn


def _validate_channels(channels: int) -> int:
    if isinstance(channels, bool) or not isinstance(channels, int):
        raise TypeError("channels must be a positive integer")
    if channels <= 0:
        raise ValueError("channels must be greater than zero")
    return channels


def _validate_feature(x: Tensor, channels: int) -> None:
    if not isinstance(x, Tensor):
        raise TypeError("FIA input must be a torch.Tensor")
    if x.ndim != 4:
        raise ValueError("FIA input must use BCHW layout")
    if x.shape[1] != channels:
        raise ValueError(f"FIA input must have {channels} channels")


def _fia_construction_cuda_devices() -> list[int]:
    """Return CUDA RNG devices that FIA construction could consume."""

    if not torch.cuda.is_available():
        return []
    default_device = torch.get_default_device()
    if default_device.type != "cuda":
        return []
    return [
        int(default_device.index)
        if default_device.index is not None
        else int(torch.cuda.current_device())
    ]


class FIABaseBlock(nn.Module):
    """Depth-wise feature refinement with two internal residual paths."""

    internal_residuals = 2

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = _validate_channels(channels)
        self.project_in = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.depthwise = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            groups=self.channels,
        )
        self.project_out = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        _validate_feature(x, self.channels)
        local = x + self.depthwise(self.activation(self.project_in(x)))
        return local + self.project_out(self.activation(local))


class FIAAttention(nn.Module):
    """Joint channel-mean and spatial mean/max feature attention."""

    uses_spatial_attention = True
    uses_channel_attention = True

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = _validate_channels(channels)
        hidden_channels = max(1, self.channels // 4)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, self.channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        _validate_feature(x, self.channels)
        channel_gate = self.channel_attention(x)
        spatial_summary = torch.cat(
            (x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)),
            dim=1,
        )
        spatial_gate = self.spatial_attention(spatial_summary)
        return x * channel_gate * spatial_gate


class FIA(nn.Module):
    """Identity-safe image representation enhancement for BCHW features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = _validate_channels(channels)
        # Do not perturb the initialization trajectory of modules constructed
        # after this private YAML layer (stock P4/P5 and the FDR decoder).
        with torch.random.fork_rng(
            devices=_fia_construction_cuda_devices(),
            enabled=True,
        ):
            self.refine = nn.Sequential(
                FIABaseBlock(self.channels),
                FIABaseBlock(self.channels),
                FIAAttention(self.channels),
            )
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor) -> Tensor:
        _validate_feature(x, self.channels)
        refined = self.refine(x)
        return x + self.residual_scale * (refined - x)


__all__ = ["FIA", "FIAAttention", "FIABaseBlock"]
