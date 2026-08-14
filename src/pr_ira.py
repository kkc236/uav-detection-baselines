from __future__ import annotations

import math
from numbers import Real
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor, nn


def _validate_channels(channels: int) -> int:
    if isinstance(channels, bool) or not isinstance(channels, int):
        raise TypeError("channels must be a positive integer")
    if channels <= 0:
        raise ValueError("channels must be greater than zero")
    return channels


def _validate_positive_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number")
    return result


def _validate_progress_value(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer epoch count")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer epoch count")
    return value


def relative_open_ratio(epoch: int, epochs: int) -> float:
    """Return the frozen relative-progress opening ratio for a 1-based epoch."""

    current = _validate_progress_value(epoch, "epoch")
    total = _validate_progress_value(epochs, "epochs")
    if current > total:
        raise ValueError("epoch must not be greater than epochs")

    identity_end = (total + 9) // 10
    fully_open_start = (3 * total) // 10 + 1
    if current <= identity_end:
        return 0.0
    if current >= fully_open_start:
        return 1.0
    return (current - identity_end) / (fully_open_start - identity_end)


def _validate_feature(x: Tensor, channels: int) -> None:
    if not isinstance(x, Tensor):
        raise TypeError("PRIRA input must be a torch.Tensor")
    if x.ndim != 4:
        raise ValueError("PRIRA input must use BCHW layout")
    if x.shape[1] != channels:
        raise ValueError(f"PRIRA input must have {channels} channels")
    if not torch.is_floating_point(x):
        raise TypeError("PRIRA input must use a floating dtype")
    if any(size <= 0 for size in (x.shape[0], x.shape[2], x.shape[3])):
        raise ValueError("PRIRA input must have non-empty batch and spatial dimensions")


def _construction_cuda_devices() -> list[int]:
    """Return CUDA RNG devices that module construction could consume."""

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


class _LocalDepthwiseResidualBlock(nn.Module):
    """A local depthwise-separable transform with an identity main path."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pointwise(self.activation(self.depthwise(x)))


class _IdentityForwardResidualBackward(torch.autograd.Function):
    """Return the identity bit-for-bit while retaining residual-path gradients."""

    @staticmethod
    def forward(x: Tensor, increment: Tensor) -> Tensor:
        return x.clone(memory_format=torch.preserve_format)

    @staticmethod
    def setup_context(ctx: object, inputs: tuple[Tensor, Tensor], output: Tensor) -> None:
        del ctx, inputs, output

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        del ctx
        return grad_output, grad_output


class PRIRA(nn.Module):
    """Protected local residual adapter for floating BCHW feature maps."""

    _DIAGNOSTIC_NAMES = (
        "effective_amplitude",
        "gate_mean",
        "gate_max",
        "residual_rms_ratio",
    )

    def __init__(
        self,
        channels: int,
        alpha_max: float = 0.20,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.channels = _validate_channels(channels)
        self.alpha_max = _validate_positive_scalar(alpha_max, "alpha_max")
        self.epsilon = _validate_positive_scalar(epsilon, "epsilon")
        hidden_channels = max(1, self.channels // 4)

        # Keep all random initialization inside a private RNG domain so adding
        # this YAML layer cannot perturb modules constructed after it.
        with torch.random.fork_rng(
            devices=_construction_cuda_devices(),
            enabled=True,
        ):
            self.local_blocks = nn.Sequential(
                _LocalDepthwiseResidualBlock(self.channels),
                _LocalDepthwiseResidualBlock(self.channels),
            )
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.channels, hidden_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_channels, self.channels, kernel_size=1),
            )
            self.spatial_gate = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        self.amplitude = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "_open_ratio",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self._zero_gate_outputs()
        self._diagnostics = {
            name: torch.zeros((), dtype=torch.float32)
            for name in self._DIAGNOSTIC_NAMES
        }

    def _zero_gate_outputs(self) -> None:
        channel_final = self.channel_gate[-1]
        if not isinstance(channel_final, nn.Conv2d):
            raise TypeError("channel gate must end with Conv2d")
        with torch.no_grad():
            channel_final.weight.zero_()
            if channel_final.bias is not None:
                channel_final.bias.zero_()
            self.spatial_gate.weight.zero_()
            if self.spatial_gate.bias is not None:
                self.spatial_gate.bias.zero_()

    @property
    def open_ratio(self) -> float:
        """Current non-persistent runtime opening ratio."""

        return float(self._open_ratio.detach().cpu())

    @property
    def diagnostics(self) -> Mapping[str, Tensor]:
        """Detached scalar diagnostics from the most recent forward pass."""

        return MappingProxyType(self._diagnostics)

    @property
    def effective_amplitude(self) -> Tensor:
        return self._diagnostics["effective_amplitude"]

    @property
    def gate_mean(self) -> Tensor:
        return self._diagnostics["gate_mean"]

    @property
    def gate_max(self) -> Tensor:
        return self._diagnostics["gate_max"]

    @property
    def residual_rms_ratio(self) -> Tensor:
        return self._diagnostics["residual_rms_ratio"]

    def set_training_progress(self, epoch: int, epochs: int) -> None:
        """Set the runtime schedule from a validated 1-based epoch pair."""

        ratio = relative_open_ratio(epoch, epochs)
        self._open_ratio.fill_(ratio)

    def _set_inactive_diagnostics(self, x: Tensor) -> None:
        zero = x.new_zeros(())
        self._diagnostics = {
            name: zero.clone() for name in self._DIAGNOSTIC_NAMES
        }

    def forward(self, x: Tensor) -> Tensor:
        _validate_feature(x, self.channels)
        if self._open_ratio.item() == 0.0:
            self._set_inactive_diagnostics(x)
            return x

        d_raw = self.local_blocks(x) - x
        d_raw_rms = d_raw.square().mean(dim=(-2, -1), keepdim=True).sqrt()
        stock_rms = (
            x.square().mean(dim=(-2, -1), keepdim=True).sqrt().detach()
        )
        residual = d_raw / (d_raw_rms + self.epsilon) * stock_rms

        magnitude = d_raw.abs()
        channel_gate = torch.sigmoid(self.channel_gate(magnitude))
        spatial_summary = torch.cat(
            (
                magnitude.mean(dim=1, keepdim=True),
                magnitude.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_gate = torch.sigmoid(self.spatial_gate(spatial_summary))
        gate = channel_gate * spatial_gate

        open_ratio = self._open_ratio.to(dtype=x.dtype, device=x.device)
        effective_amplitude = (
            open_ratio * self.alpha_max * torch.tanh(self.amplitude)
        )
        increment = (
            effective_amplitude * channel_gate * spatial_gate * residual
        )
        if effective_amplitude.detach().item() == 0.0:
            output = _IdentityForwardResidualBackward.apply(x, increment)
            finite_gate = torch.nan_to_num(
                gate.detach(), nan=0.0, posinf=1.0, neginf=0.0
            )
            self._diagnostics = {
                "effective_amplitude": effective_amplitude.detach(),
                "gate_mean": finite_gate.mean(),
                "gate_max": finite_gate.amax(),
                "residual_rms_ratio": x.new_zeros(()),
            }
            return output

        output = x + increment
        stock_global_rms = x.square().mean().sqrt().detach()
        increment_rms = increment.square().mean().sqrt()
        self._diagnostics = {
            "effective_amplitude": effective_amplitude.detach(),
            "gate_mean": gate.mean().detach(),
            "gate_max": gate.amax().detach(),
            "residual_rms_ratio": (
                increment_rms / (stock_global_rms + self.epsilon)
            ).detach(),
        }
        return output


__all__ = ["PRIRA", "relative_open_ratio"]
