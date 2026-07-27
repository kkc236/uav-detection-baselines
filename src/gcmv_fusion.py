"""GGLF and PEG stages for the integrated GCMV-EI network module."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.gcmv_plec import ChannelLayerNorm, PLECOutput


@dataclass(frozen=True)
class GGLFOutput:
    correction: torch.Tensor
    tiny_map: torch.Tensor
    confidence: torch.Tensor
    attention: torch.Tensor
    attention_entropy: torch.Tensor


@dataclass(frozen=True)
class PEGOutput:
    enhanced: torch.Tensor
    gate_hat: torch.Tensor
    gate: torch.Tensor
    gamma: torch.Tensor


@dataclass(frozen=True)
class GCMVEvidenceOutput:
    enhanced: torch.Tensor
    correction: torch.Tensor
    tiny_map: torch.Tensor
    confidence: torch.Tensor
    attention: torch.Tensor
    attention_entropy: torch.Tensor
    gate_hat: torch.Tensor
    gate: torch.Tensor
    gamma: torch.Tensor


def _feature_shape(
    global_p3: torch.Tensor,
    other: torch.Tensor,
    *,
    channels: int,
) -> tuple[int, int, int]:
    if (
        not isinstance(global_p3, torch.Tensor)
        or not isinstance(other, torch.Tensor)
        or global_p3.ndim != 4
        or other.shape != global_p3.shape
    ):
        raise ValueError("GCMV feature tensors must share BxCxHxW shape")
    if global_p3.shape[1] != channels:
        raise ValueError(f"GCMV expected {channels} feature channels")
    if not global_p3.is_floating_point() or not other.is_floating_point():
        raise TypeError("GCMV features must be floating point")
    return (
        int(global_p3.shape[0]),
        int(global_p3.shape[2]),
        int(global_p3.shape[3]),
    )


def _reliability_tensor(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, int, int, int],
) -> None:
    if not isinstance(value, torch.Tensor) or value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite floating point")


class GeometryConstrainedGlobalLocalFusion(nn.Module):
    """Four-head attention in a deterministic canonical P3 neighborhood."""

    def __init__(
        self,
        channels: int = 256,
        interaction_channels: int = 64,
        num_heads: int = 4,
        window_size: int = 3,
        residual_channels: int = 128,
    ) -> None:
        super().__init__()
        if min(channels, interaction_channels, num_heads, residual_channels) <= 0:
            raise ValueError("GGLF dimensions must be positive")
        if interaction_channels % num_heads:
            raise ValueError("interaction_channels must be divisible by num_heads")
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        self.channels = int(channels)
        self.interaction_channels = int(interaction_channels)
        self.num_heads = int(num_heads)
        self.head_channels = self.interaction_channels // self.num_heads
        self.window_size = int(window_size)
        self.window_area = self.window_size**2

        self.global_norm = ChannelLayerNorm(self.channels)
        self.local_norm = ChannelLayerNorm(self.channels)
        self.query = nn.Conv2d(
            self.channels,
            self.interaction_channels,
            kernel_size=1,
            bias=False,
        )
        self.key = nn.Conv2d(
            self.channels,
            self.interaction_channels,
            kernel_size=1,
            bias=False,
        )
        self.value = nn.Conv2d(
            self.channels,
            self.interaction_channels,
            kernel_size=1,
            bias=False,
        )
        self.global_projection = nn.Conv2d(
            self.channels,
            self.interaction_channels,
            kernel_size=1,
            bias=False,
        )
        self.relative_position_bias = nn.Parameter(
            torch.zeros(self.num_heads, self.window_area)
        )
        difference_channels = 4 * self.interaction_channels
        self.residual_reduce = nn.Sequential(
            nn.Conv2d(
                difference_channels,
                residual_channels,
                kernel_size=1,
                bias=False,
            ),
            ChannelLayerNorm(residual_channels),
            nn.SiLU(),
        )
        self.detail_mixer = nn.Sequential(
            nn.Conv2d(
                residual_channels,
                residual_channels,
                kernel_size=3,
                padding=1,
                groups=residual_channels,
                bias=False,
            ),
            ChannelLayerNorm(residual_channels),
            nn.SiLU(),
        )
        self.evidence_project = nn.Conv2d(
            residual_channels,
            self.channels,
            kernel_size=1,
            bias=False,
        )
        self.tiny_head = nn.Conv2d(
            residual_channels,
            1,
            kernel_size=1,
        )

    def _unfold(
        self,
        feature: torch.Tensor,
        *,
        channels: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        return F.unfold(
            feature,
            kernel_size=self.window_size,
            padding=self.window_size // 2,
        ).reshape(
            feature.shape[0],
            channels,
            self.window_area,
            height,
            width,
        )

    def forward(
        self,
        global_p3: torch.Tensor,
        local_canonical: torch.Tensor,
        valid_count: torch.Tensor,
        edge_prior: torch.Tensor,
    ) -> GGLFOutput:
        batch, height, width = _feature_shape(
            global_p3,
            local_canonical,
            channels=self.channels,
        )
        map_shape = (batch, 1, height, width)
        _reliability_tensor(valid_count, name="valid_count", shape=map_shape)
        _reliability_tensor(edge_prior, name="edge_prior", shape=map_shape)

        coverage = valid_count > 0
        coverage_numeric = coverage.to(global_p3.dtype)
        valid_fraction = (valid_count / 4.0).clamp(0.0, 1.0)
        boundary = edge_prior.clamp(0.0, 1.0)
        plec_confidence = (
            valid_fraction * (0.5 + 0.5 * boundary)
        ).detach()

        global_normalized = self.global_norm(global_p3)
        local_normalized = self.local_norm(local_canonical)
        query = self.query(global_normalized).reshape(
            batch,
            self.num_heads,
            self.head_channels,
            height,
            width,
        )
        key = self.key(local_normalized)
        value = self.value(local_normalized)
        key_windows = self._unfold(
            key,
            channels=self.interaction_channels,
            height=height,
            width=width,
        ).reshape(
            batch,
            self.num_heads,
            self.head_channels,
            self.window_area,
            height,
            width,
        )
        value_windows = self._unfold(
            value,
            channels=self.interaction_channels,
            height=height,
            width=width,
        ).reshape(
            batch,
            self.num_heads,
            self.head_channels,
            self.window_area,
            height,
            width,
        )
        valid_windows = self._unfold(
            coverage_numeric,
            channels=1,
            height=height,
            width=width,
        ).squeeze(1) > 0
        confidence_windows = self._unfold(
            plec_confidence,
            channels=1,
            height=height,
            width=width,
        ).squeeze(1)

        scores = (
            query.unsqueeze(3) * key_windows
        ).sum(dim=2) / math.sqrt(self.head_channels)
        scores = scores + self.relative_position_bias.view(
            1,
            self.num_heads,
            self.window_area,
            1,
            1,
        )
        eps = torch.finfo(scores.dtype).eps
        scores = scores + torch.log(
            confidence_windows[:, None].clamp_min(eps)
        )
        head_valid = valid_windows[:, None].expand(
            -1,
            self.num_heads,
            -1,
            -1,
            -1,
        )
        scores = scores.masked_fill(
            ~head_valid,
            torch.finfo(scores.dtype).min,
        )
        attention = torch.softmax(scores, dim=2)
        attention = attention * head_valid.to(attention.dtype)
        attention = attention / attention.sum(
            dim=2,
            keepdim=True,
        ).clamp_min(eps)
        attention = attention * coverage_numeric[:, None]

        aggregated = (
            value_windows * attention.unsqueeze(2)
        ).sum(dim=3).reshape(
            batch,
            self.interaction_channels,
            height,
            width,
        )
        global_projected = self.global_projection(global_normalized)
        difference = torch.cat(
            (
                aggregated,
                global_projected,
                aggregated - global_projected,
                (aggregated - global_projected).abs(),
            ),
            dim=1,
        )
        hidden = self.detail_mixer(self.residual_reduce(difference))
        correction = (
            self.evidence_project(hidden) * coverage_numeric
        )
        tiny_map = torch.sigmoid(self.tiny_head(hidden)) * coverage_numeric

        head_entropy = -(
            attention * torch.log(attention.clamp_min(eps))
        ).sum(dim=2) / math.log(self.window_area)
        entropy = head_entropy.mean(dim=1, keepdim=True) * coverage_numeric
        concentration = (1.0 - entropy).clamp(0.0, 1.0)
        semantic = (
            0.5
            + 0.5
            * F.cosine_similarity(
                global_projected,
                aggregated,
                dim=1,
                eps=1e-6,
            ).unsqueeze(1)
        ).clamp(0.0, 1.0)
        confidence = (
            coverage_numeric
            * plec_confidence
            * concentration
            * semantic
        ).detach()
        return GGLFOutput(
            correction=correction,
            tiny_map=tiny_map,
            confidence=confidence,
            attention=attention,
            attention_entropy=entropy,
        )


class ProtectedEvidenceGate(nn.Module):
    """Reliability-constrained PEG with an exact scalar identity guard."""

    def __init__(
        self,
        channels: int = 256,
        reduced_channels: int = 32,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        if min(channels, reduced_channels, hidden_channels) <= 0:
            raise ValueError("PEG dimensions must be positive")
        self.channels = int(channels)
        self.global_norm = ChannelLayerNorm(self.channels)
        self.evidence_norm = ChannelLayerNorm(self.channels)
        self.global_reduce = nn.Conv2d(
            self.channels,
            reduced_channels,
            kernel_size=1,
            bias=False,
        )
        self.evidence_reduce = nn.Conv2d(
            self.channels,
            reduced_channels,
            kernel_size=1,
            bias=False,
        )
        gate_inputs = 3 * reduced_channels + 4
        self.gate_head = nn.Sequential(
            nn.Conv2d(
                gate_inputs,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)
        self.evidence_projection = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=1,
            bias=False,
        )
        self.rho = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        global_p3: torch.Tensor,
        correction: torch.Tensor,
        tiny_map: torch.Tensor,
        correspondence: torch.Tensor,
        plec_confidence: torch.Tensor,
        edge_prior: torch.Tensor,
        valid_count: torch.Tensor,
    ) -> PEGOutput:
        batch, height, width = _feature_shape(
            global_p3,
            correction,
            channels=self.channels,
        )
        map_shape = (batch, 1, height, width)
        for name, value in (
            ("tiny_map", tiny_map),
            ("correspondence", correspondence),
            ("plec_confidence", plec_confidence),
            ("edge_prior", edge_prior),
            ("valid_count", valid_count),
        ):
            _reliability_tensor(value, name=name, shape=map_shape)

        coverage = (valid_count > 0).to(global_p3.dtype)
        global_reduced = self.global_reduce(self.global_norm(global_p3))
        evidence_reduced = self.evidence_reduce(
            self.evidence_norm(correction)
        )
        gate_features = torch.cat(
            (
                global_reduced,
                evidence_reduced,
                (global_reduced - evidence_reduced).abs(),
                tiny_map,
                correspondence.detach(),
                plec_confidence.detach(),
                edge_prior.detach().clamp(0.0, 1.0),
            ),
            dim=1,
        )
        gate_hat = torch.sigmoid(self.gate_head(gate_features))
        reliability = coverage * torch.pow(
            (
                plec_confidence.detach().clamp(0.0, 1.0)
                * correspondence.detach().clamp(0.0, 1.0)
                * edge_prior.detach().clamp(0.0, 1.0)
            ).clamp_min(0.0),
            1.0 / 3.0,
        )
        gate = reliability * gate_hat
        gamma = torch.tanh(self.rho)
        enhanced = (
            global_p3
            + gamma * gate * self.evidence_projection(correction)
        )
        return PEGOutput(
            enhanced=enhanced,
            gate_hat=gate_hat,
            gate=gate,
            gamma=gamma,
        )


class GCMVEvidenceInjectionModule(nn.Module):
    """Integrated post-encoder, pre-decoder GGLF-plus-PEG stages."""

    def __init__(
        self,
        channels: int = 256,
        interaction_channels: int = 64,
        num_heads: int = 4,
        window_size: int = 3,
    ) -> None:
        super().__init__()
        self.gglf = GeometryConstrainedGlobalLocalFusion(
            channels=channels,
            interaction_channels=interaction_channels,
            num_heads=num_heads,
            window_size=window_size,
            residual_channels=128,
        )
        self.peg = ProtectedEvidenceGate(channels=channels)

    def forward(
        self,
        global_p3: torch.Tensor,
        plec_output: PLECOutput,
    ) -> GCMVEvidenceOutput:
        if not isinstance(plec_output, PLECOutput):
            raise TypeError("plec_output must be a PLECOutput")
        fused = self.gglf(
            global_p3,
            plec_output.canonical,
            plec_output.valid_count,
            plec_output.edge_prior,
        )
        plec_confidence = (
            (plec_output.valid_count / 4.0).clamp(0.0, 1.0)
            * (0.5 + 0.5 * plec_output.edge_prior.clamp(0.0, 1.0))
        ).detach()
        gated = self.peg(
            global_p3,
            fused.correction,
            fused.tiny_map,
            fused.confidence,
            plec_confidence,
            plec_output.edge_prior,
            plec_output.valid_count,
        )
        return GCMVEvidenceOutput(
            enhanced=gated.enhanced,
            correction=fused.correction,
            tiny_map=fused.tiny_map,
            confidence=fused.confidence,
            attention=fused.attention,
            attention_entropy=fused.attention_entropy,
            gate_hat=gated.gate_hat,
            gate=gated.gate,
            gamma=gated.gamma,
        )
