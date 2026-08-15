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


def _validate_feature(x: Tensor, channels: int, reference: Tensor) -> None:
    if not isinstance(x, Tensor):
        raise TypeError("PRFIA input must be a torch.Tensor")
    if x.ndim != 4:
        raise ValueError("PRFIA input must use BCHW layout")
    if x.shape[1] != channels:
        raise ValueError(f"PRFIA input must have {channels} channels")
    if not torch.is_floating_point(x):
        raise TypeError("PRFIA input must use a floating dtype")
    if any(size <= 0 for size in (x.shape[0], x.shape[2], x.shape[3])):
        raise ValueError("PRFIA input must have non-empty batch and spatial dimensions")
    if x.device != reference.device:
        raise ValueError("PRFIA input and module parameters must use the same device")
    if (
        x.dtype != reference.dtype
        and not torch.is_autocast_enabled(x.device.type)
    ):
        raise ValueError(
            "PRFIA input dtype must match module floating parameter dtype "
            "outside autocast"
        )


def _promote_for_rms(x: Tensor) -> Tensor:
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    return x


def _stable_rms(
    x: Tensor,
    dim: tuple[int, ...] | None = None,
    *,
    keepdim: bool = False,
) -> Tensor:
    promoted = _promote_for_rms(x)
    if dim is None:
        element_count = promoted.numel()
        scale = promoted.abs().amax()
        safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        normalized = promoted / safe_scale
    else:
        normalized_dims = tuple(index % promoted.ndim for index in dim)
        element_count = 1
        for index in normalized_dims:
            element_count *= promoted.shape[index]
        scale_for_division = promoted.abs().amax(
            dim=normalized_dims,
            keepdim=True,
        )
        safe_scale = torch.where(
            scale_for_division == 0,
            torch.ones_like(scale_for_division),
            scale_for_division,
        )
        normalized = promoted / safe_scale
        scale = scale_for_division
        if not keepdim:
            for index in sorted(normalized_dims, reverse=True):
                scale = scale.squeeze(index)
    normalized_rms = torch.linalg.vector_norm(
        normalized,
        ord=2,
        dim=dim,
        keepdim=keepdim,
    ) / math.sqrt(element_count)
    return normalized_rms * scale


def _low_precision_amplitude_guard(dtype: torch.dtype) -> float:
    """Leave one quantization margin below the configured residual cap."""

    if dtype not in (torch.float16, torch.bfloat16):
        return 1.0
    return 1.0 - 2.0 * torch.finfo(dtype).eps


def _assert_finite_when_active(x: Tensor, active: Tensor) -> Tensor:
    condition = torch.logical_or(torch.logical_not(active), x.isfinite().all())
    token = torch._functional_assert_async(
        condition,
        "PRFIA produced non-finite values",
        x.new_zeros(()),
    )
    return x + token


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


class _ExactAdd(torch.autograd.Function):
    """Add elementwise while preserving base bits for exact-zero increments."""

    generate_vmap_rule = True

    @staticmethod
    def forward(x: Tensor, increment: Tensor) -> Tensor:
        return torch.where(increment == 0, x, x + increment)

    @staticmethod
    def setup_context(ctx: object, inputs: tuple[Tensor, Tensor], output: Tensor) -> None:
        del ctx, inputs, output

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        del ctx
        return grad_output, grad_output

    @staticmethod
    def jvp(
        ctx: object,
        grad_x: Tensor | None,
        grad_increment: Tensor | None,
    ) -> Tensor | None:
        del ctx
        if grad_x is None:
            return grad_increment
        if grad_increment is None:
            return grad_x
        return grad_x + grad_increment


@torch.compiler.allow_in_graph
def _exact_add(x: Tensor, increment: Tensor) -> Tensor:
    return _ExactAdd.apply(x, increment)


class PRFIA(nn.Module):
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
        *,
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

    def forward(self, x: Tensor) -> Tensor:
        _validate_feature(x, self.channels, self.amplitude)
        open_ratio = self._open_ratio.to(dtype=x.dtype, device=x.device)
        active = open_ratio != 0

        d_raw = self.local_blocks(x) - x
        d_raw = _assert_finite_when_active(d_raw, active)
        active_d_raw = torch.where(active, d_raw, torch.zeros_like(d_raw))
        d_raw_for_rms = _promote_for_rms(active_d_raw)
        d_raw_rms = _stable_rms(
            active_d_raw,
            dim=(-2, -1),
            keepdim=True,
        )
        stock_rms = _stable_rms(
            x,
            dim=(-2, -1),
            keepdim=True,
        ).detach()
        raw_residual = (
            d_raw_for_rms / (d_raw_rms + self.epsilon) * stock_rms
        ).to(
            dtype=x.dtype,
            device=x.device,
        )
        raw_residual = _assert_finite_when_active(raw_residual, active)
        residual = torch.where(
            active,
            raw_residual,
            torch.zeros_like(raw_residual),
        )

        magnitude = active_d_raw.abs()
        channel_gate = torch.sigmoid(self.channel_gate(magnitude))
        spatial_summary = torch.cat(
            (
                magnitude.mean(dim=1, keepdim=True),
                magnitude.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_gate = torch.sigmoid(self.spatial_gate(spatial_summary))
        channel_gate = _assert_finite_when_active(channel_gate, active)
        spatial_gate = _assert_finite_when_active(spatial_gate, active)
        channel_gate = torch.where(
            active,
            channel_gate,
            torch.zeros_like(channel_gate),
        )
        spatial_gate = torch.where(
            active,
            spatial_gate,
            torch.zeros_like(spatial_gate),
        )
        gate = channel_gate * spatial_gate

        raw_effective_amplitude = (
            open_ratio
            * self.alpha_max
            * _low_precision_amplitude_guard(x.dtype)
            * torch.tanh(self.amplitude).to(dtype=x.dtype, device=x.device)
        )
        raw_effective_amplitude = _assert_finite_when_active(
            raw_effective_amplitude,
            active,
        )
        effective_amplitude = torch.where(
            active,
            raw_effective_amplitude,
            x.new_zeros(()),
        )
        increment = (
            effective_amplitude * channel_gate * spatial_gate * residual
        ).to(dtype=x.dtype, device=x.device)
        increment = _assert_finite_when_active(increment, active)

        output = _exact_add(x, increment)
        stock_global_rms = _stable_rms(x).detach()
        increment_rms = _stable_rms(increment)
        raw_residual_rms_ratio = increment_rms / (
            stock_global_rms + self.epsilon
        )
        residual_rms_ratio = torch.where(
            active,
            raw_residual_rms_ratio,
            raw_residual_rms_ratio.new_zeros(()),
        )
        self._diagnostics = {
            "effective_amplitude": effective_amplitude.detach(),
            "gate_mean": gate.mean().detach(),
            "gate_max": gate.amax().detach(),
            "residual_rms_ratio": residual_rms_ratio.detach(),
        }
        return output


__all__ = ["PRFIA", "relative_open_ratio"]
