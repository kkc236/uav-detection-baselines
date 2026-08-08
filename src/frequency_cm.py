"""Identity-initialized frequency/spatial modulation for the FDR P5 feature."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from ultralytics.nn.modules.transformer import LayerNorm2d


class _FrequencyMagnitudeTransform(nn.Module):
    """Transform the full FFT magnitude while preserving the observed phase."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.process = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, value: Tensor) -> Tensor:
        output_dtype = value.dtype
        device_type = value.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            working = value.float()
            spectrum = torch.fft.rfft2(working, norm="backward")
            magnitude = self.process(torch.abs(spectrum))
            phase = torch.angle(spectrum)
            reconstructed = torch.complex(
                magnitude * torch.cos(phase),
                magnitude * torch.sin(phase),
            )
            output = torch.fft.irfft2(
                reconstructed,
                s=working.shape[-2:],
                norm="backward",
            )
        return output.to(dtype=output_dtype)


class FrequencyCM(nn.Module):
    """Cooperatively modulate one feature through frequency and spatial residuals.

    The two learned residual scales are initialized to zero, so a same-channel
    module is an exact identity before optimization. Frequency-domain arithmetic
    runs in FP32 while the returned tensor preserves the detector input dtype.
    """

    def __init__(self, channels: int, *, private_seed: int = 20_000) -> None:
        super().__init__()
        channels = int(channels)
        if channels <= 0:
            raise ValueError("FrequencyCM channels must be positive")
        expanded = 2 * channels
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(int(private_seed))
            self.norm1 = LayerNorm2d(channels)
            self.frequency = _FrequencyMagnitudeTransform(channels)
            self.norm2 = LayerNorm2d(channels)
            self.spatial_extra = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            )
            self.spatial_expand = nn.Conv2d(
                channels,
                expanded,
                kernel_size=3,
                padding=1,
            )
            self.spatial_depthwise = nn.Conv2d(
                expanded,
                expanded,
                kernel_size=3,
                padding=1,
                groups=expanded,
            )
            self.spatial_scale = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels, kernel_size=1),
            )
            self.spatial_project = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.channels = channels
        self.private_seed = int(private_seed)

    def _spatial_residual(self, value: Tensor) -> Tensor:
        expanded = self.spatial_expand(self.spatial_extra(value))
        expanded = self.spatial_depthwise(expanded)
        first, second = expanded.chunk(2, dim=1)
        gated = first * second
        return self.spatial_project(self.spatial_scale(gated) * gated)

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 4 or int(value.shape[1]) != self.channels:
            raise ValueError(
                f"FrequencyCM expected [B,{self.channels},H,W], got {tuple(value.shape)}"
            )
        frequency = self.frequency(self.norm1(value))
        low = value + frequency * self.gamma
        high = self._spatial_residual(self.norm2(low))
        return low + high * self.beta


__all__ = ["FrequencyCM"]
