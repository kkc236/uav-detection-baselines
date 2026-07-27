"""Phase-preserving local evidence canonicalization for GCMV-RTDETR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.gcmv_geometry import PLECGeometry


@dataclass(frozen=True)
class PLECOutput:
    canonical: torch.Tensor
    valid_count: torch.Tensor
    edge_prior: torch.Tensor
    overlap_weights: torch.Tensor


class ChannelLayerNorm(nn.Module):
    """Apply LayerNorm over channels independently at each spatial position."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.norm = nn.LayerNorm(channels)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.norm.normalized_shape[0]:
            raise ValueError(
                "ChannelLayerNorm expects BxCxHxW with configured channels"
            )
        return self.norm(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class PhasePreservingLocalEvidenceCanonicalizer(nn.Module):
    """Trainable phase-preserving canonicalization of four local P3 maps."""

    def __init__(
        self,
        channels: int = 256,
        embedding_hidden: int = 64,
        overlap_hidden: int = 64,
        *,
        use_phase_embedding: bool = True,
        use_view_embedding: bool = True,
        use_metadata_embedding: bool = True,
        learned_overlap: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0 or embedding_hidden <= 0 or overlap_hidden <= 0:
            raise ValueError("channels and hidden dimensions must be positive")
        self.channels = int(channels)
        self.embedding_hidden = int(embedding_hidden)
        self.overlap_hidden = int(overlap_hidden)
        self.use_phase_embedding = bool(use_phase_embedding)
        self.use_view_embedding = bool(use_view_embedding)
        self.use_metadata_embedding = bool(use_metadata_embedding)
        self.learned_overlap = bool(learned_overlap)

        self.view_embedding = (
            nn.Embedding(4, self.channels) if self.use_view_embedding else None
        )
        self.phase_mlp = (
            nn.Sequential(
                nn.Linear(2, self.embedding_hidden),
                nn.SiLU(),
                nn.Linear(self.embedding_hidden, self.channels),
            )
            if self.use_phase_embedding
            else None
        )
        self.metadata_mlp = (
            nn.Sequential(
                nn.Linear(3, self.embedding_hidden),
                nn.SiLU(),
                nn.Linear(self.embedding_hidden, self.channels),
            )
            if self.use_metadata_embedding
            else None
        )
        self.phase_reducer = nn.Conv2d(
            9 * self.channels,
            self.channels,
            kernel_size=1,
            groups=self.channels,
            bias=False,
        )
        self.spatial_mixer = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            groups=self.channels,
        )
        self.pointwise = nn.Conv2d(
            self.channels, self.channels, kernel_size=1
        )
        self.overlap_head = (
            nn.Sequential(
                nn.Conv2d(
                    self.channels + 1, self.overlap_hidden, kernel_size=1
                ),
                nn.SiLU(),
                nn.Conv2d(self.overlap_hidden, 1, kernel_size=1),
            )
            if self.learned_overlap
            else None
        )
        self.output_norm = ChannelLayerNorm(self.channels)

    def forward(
        self,
        local_features: Sequence[torch.Tensor],
        geometry: PLECGeometry,
    ) -> PLECOutput:
        sampled = sample_local_phases(local_features, geometry)
        if sampled.shape[3] != self.channels:
            raise ValueError(
                f"PLEC was configured for {self.channels} channels, "
                f"received {sampled.shape[3]}"
            )

        batch_size, _, _, _, global_height, global_width = sampled.shape
        numeric_geometry = geometry.to(
            device=sampled.device, dtype=sampled.dtype
        )
        sample_mask = numeric_geometry.sample_valid.unsqueeze(3).to(
            sampled.dtype
        )
        enriched = sampled

        if self.view_embedding is not None:
            view_ids = torch.arange(4, device=sampled.device)
            view = self.view_embedding(view_ids).view(
                1, 4, 1, self.channels, 1, 1
            )
            enriched = enriched + view

        if self.phase_mlp is not None:
            phase = numeric_geometry.subcell_offset.permute(
                0, 1, 2, 4, 5, 3
            )
            phase = self.phase_mlp(phase).permute(0, 1, 2, 5, 3, 4)
            enriched = enriched + phase

        if self.metadata_mlp is not None:
            eps = torch.finfo(numeric_geometry.magnification.dtype).eps
            metadata = torch.cat(
                (
                    torch.log2(
                        numeric_geometry.magnification.clamp_min(eps)
                    ),
                    numeric_geometry.edge_distance,
                ),
                dim=2,
            ).permute(0, 1, 3, 4, 2)
            metadata = self.metadata_mlp(metadata).permute(0, 1, 4, 2, 3)
            enriched = enriched + metadata.unsqueeze(2)

        enriched = enriched * sample_mask
        phase_input = enriched.permute(0, 1, 3, 2, 4, 5).reshape(
            batch_size * 4,
            self.channels * 9,
            global_height,
            global_width,
        )
        center_mask = numeric_geometry.center_valid.reshape(
            batch_size * 4, 1, global_height, global_width
        ).to(phase_input.dtype)
        encoded = F.silu(self.phase_reducer(phase_input)) * center_mask
        encoded = F.silu(self.spatial_mixer(encoded)) * center_mask
        encoded = self.pointwise(encoded) * center_mask
        encoded = encoded.reshape(
            batch_size,
            4,
            self.channels,
            global_height,
            global_width,
        )

        valid = numeric_geometry.center_valid
        edge = numeric_geometry.edge_distance
        if self.overlap_head is None:
            numeric_valid = valid.to(encoded.dtype)
            weights = numeric_valid / numeric_valid.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
        else:
            logits_input = torch.cat((encoded, edge), dim=2).reshape(
                batch_size * 4,
                self.channels + 1,
                global_height,
                global_width,
            )
            logits = self.overlap_head(logits_input).reshape(
                batch_size, 4, 1, global_height, global_width
            )
            masked_logits = logits.masked_fill(
                ~valid, torch.finfo(logits.dtype).min
            )
            weights = torch.softmax(masked_logits, dim=1) * valid.to(
                logits.dtype
            )
            weights = weights / weights.sum(
                dim=1, keepdim=True
            ).clamp_min(torch.finfo(logits.dtype).eps)

        valid_count = valid.sum(dim=1).to(encoded.dtype)
        any_valid = (valid_count > 0).to(encoded.dtype)
        canonical = (weights * encoded).sum(dim=1)
        canonical = self.output_norm(canonical) * any_valid
        edge_prior = (weights * edge).sum(dim=1) * any_valid
        return PLECOutput(
            canonical=canonical,
            valid_count=valid_count,
            edge_prior=edge_prior,
            overlap_weights=weights,
        )


def _validate_local_features(
    local_features: Sequence[torch.Tensor],
    geometry: PLECGeometry,
) -> tuple[int, int, int, int]:
    if len(local_features) != 4:
        raise ValueError(f"PLEC expects exactly four local features, received {len(local_features)}")
    first = local_features[0]
    if not isinstance(first, torch.Tensor) or first.ndim != 4:
        raise ValueError("each local feature must have shape BxCxHxW")
    if not first.is_floating_point():
        raise TypeError("local features must use a floating dtype")
    batch_size, channels, local_height, local_width = first.shape
    expected_shape = (batch_size, channels, local_height, local_width)
    for index, feature in enumerate(local_features):
        if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
            raise ValueError(f"local feature {index} must have shape BxCxHxW")
        if tuple(feature.shape) != expected_shape:
            raise ValueError("all local features must share batch, channel, and spatial shape")
        if feature.device != first.device or feature.dtype != first.dtype:
            raise ValueError("all local features must share device and dtype")
    if (local_height, local_width) != geometry.local_feature_shape:
        raise ValueError("local feature spatial shape does not match geometry")
    _validate_geometry_contract(
        geometry, batch_size=batch_size, device=first.device
    )
    return batch_size, channels, local_height, local_width


def _validate_geometry_contract(
    geometry: PLECGeometry,
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    if not isinstance(geometry, PLECGeometry):
        raise TypeError("geometry must be a PLECGeometry")
    try:
        global_height, global_width = geometry.global_feature_shape
    except (TypeError, ValueError) as error:
        raise ValueError("global_feature_shape must contain height and width") from error
    if global_height <= 0 or global_width <= 0:
        raise ValueError("global_feature_shape dimensions must be positive")

    expected_shapes = {
        "sample_grid": (batch_size, 4, 9, global_height, global_width, 2),
        "sample_valid": (batch_size, 4, 9, global_height, global_width),
        "center_valid": (batch_size, 4, 1, global_height, global_width),
        "subcell_offset": (
            batch_size,
            4,
            9,
            2,
            global_height,
            global_width,
        ),
        "magnification": (
            batch_size,
            4,
            2,
            global_height,
            global_width,
        ),
        "edge_distance": (
            batch_size,
            4,
            1,
            global_height,
            global_width,
        ),
    }
    for name, expected in expected_shapes.items():
        value = getattr(geometry, name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if value.device != device:
            raise ValueError(f"{name} and local features must share a device")

    for name in ("sample_grid", "subcell_offset", "magnification", "edge_distance"):
        if not getattr(geometry, name).is_floating_point():
            raise ValueError(f"{name} must use a floating dtype")
    for name in ("sample_valid", "center_valid"):
        if getattr(geometry, name).dtype != torch.bool:
            raise ValueError(f"{name} must use boolean dtype")


def sample_local_phases(
    local_features: Sequence[torch.Tensor],
    geometry: PLECGeometry,
) -> torch.Tensor:
    batch_size, channels, local_height, local_width = _validate_local_features(
        local_features, geometry
    )
    global_height, global_width = geometry.global_feature_shape
    stacked = torch.stack(tuple(local_features), dim=1).reshape(
        batch_size * 4, channels, local_height, local_width
    )
    grid = geometry.sample_grid
    if grid.dtype != stacked.dtype and not torch.is_autocast_enabled(
        stacked.device.type
    ):
        grid = grid.to(dtype=stacked.dtype)
    flattened_grid = grid.reshape(
        batch_size * 4, 9 * global_height, global_width, 2
    )
    sampled = F.grid_sample(
        stacked,
        flattened_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled = sampled.reshape(
        batch_size, 4, channels, 9, global_height, global_width
    ).permute(0, 1, 3, 2, 4, 5)
    return sampled * geometry.sample_valid.unsqueeze(3).to(sampled.dtype)


def uniform_bilinear_canonicalize(
    local_features: Sequence[torch.Tensor],
    geometry: PLECGeometry,
) -> PLECOutput:
    sampled = sample_local_phases(local_features, geometry)
    center = sampled[:, :, 4]
    valid = geometry.center_valid.to(device=center.device, dtype=center.dtype)
    weights = valid / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    canonical = (weights * center).sum(dim=1)
    valid_count = valid.sum(dim=1)
    edge_distance = geometry.edge_distance.to(
        device=center.device, dtype=center.dtype
    )
    edge_prior = (weights * edge_distance).sum(dim=1)
    any_valid = (valid_count > 0).to(center.dtype)
    return PLECOutput(
        canonical=canonical * any_valid,
        valid_count=valid_count,
        edge_prior=edge_prior * any_valid,
        overlap_weights=weights,
    )
