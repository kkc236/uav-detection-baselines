"""Geometry-canonicalized constrained query fusion.

The public ``GCQF`` class is the single trainable network module.  Its three
registered stages project crop queries into a global frame, read frozen global
query context, and learn a bounded score residual around the fixed SADED
anchor.  The fixed discrete SADED router remains authoritative for candidate
admission and protected-global ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.gcte_types import QueryEvidence, ViewGeometry
from src.gcte_views import transform_xywh_homography
from src.sr_peg import ScaleRiskProtectedEvidenceGate


@dataclass(frozen=True)
class GeometryProjectionOutput:
    canonical_local: QueryEvidence
    geometry_embedding: torch.Tensor


@dataclass(frozen=True)
class GCQFOutput:
    global_evidence: QueryEvidence
    canonical_local: QueryEvidence
    geometry_embedding: torch.Tensor
    global_context: torch.Tensor
    tiny_utility_logits: torch.Tensor
    non_tiny_risk_logits: torch.Tensor
    global_retain_logits: torch.Tensor
    score_residual: torch.Tensor
    adjusted_local_scores: torch.Tensor


def _zero_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class GeometryQueryProjector(nn.Module):
    """Map local boxes exactly and adapt local queries with geometry context."""

    def __init__(
        self,
        *,
        query_dim: int,
        num_views: int,
        residual_cap: float,
    ) -> None:
        super().__init__()
        if query_dim <= 0 or num_views <= 0:
            raise ValueError("query_dim and num_views must be positive")
        if not 0.0 < residual_cap <= 1.0:
            raise ValueError("residual_cap must be in (0,1]")
        self.query_dim = int(query_dim)
        self.num_views = int(num_views)
        self.residual_cap = float(residual_cap)
        # 4 global-box + 6 crop + 4 local-boundary + 1 base-score values,
        # concatenated with one fixed sine and cosine frequency.
        geometry_input_dim = 15 * 3
        self.geometry_mlp = nn.Sequential(
            nn.Linear(geometry_input_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
            nn.LayerNorm(query_dim),
        )
        self.query_adapter = nn.Sequential(
            nn.Linear(query_dim * 2, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )
        self.output_norm = nn.LayerNorm(query_dim)
        _zero_linear(self.query_adapter[-1])

    @staticmethod
    def _boundary_distances(local_xywh: torch.Tensor) -> torch.Tensor:
        x, y, width, height = local_xywh.unbind(dim=-1)
        return torch.stack(
            (
                x - width * 0.5,
                y - height * 0.5,
                1.0 - x - width * 0.5,
                1.0 - y - height * 0.5,
            ),
            dim=-1,
        ).clamp(-1.0, 1.0)

    def forward(
        self,
        local: QueryEvidence,
        geometry: ViewGeometry,
    ) -> GeometryProjectionOutput:
        if local.query_dim != self.query_dim:
            raise ValueError("local query_dim does not match geometry projector")
        if (local.batch_size, local.query_count) != (
            geometry.batch_size,
            geometry.query_count,
        ):
            raise ValueError("local evidence and view geometry shape mismatch")
        if bool(
            (
                geometry.view_index[geometry.valid_mask]
                >= self.num_views
            ).any()
        ):
            raise ValueError("view_index exceeds configured num_views")

        global_boxes = transform_xywh_homography(
            local.boxes,
            geometry.homography,
            clip=True,
        )
        raw_geometry = torch.cat(
            (
                global_boxes.detach(),
                geometry.crop_metadata.detach(),
                self._boundary_distances(local.boxes).detach(),
                local.quality.detach(),
            ),
            dim=-1,
        )
        positional = torch.cat(
            (
                raw_geometry,
                torch.sin(2.0 * torch.pi * raw_geometry),
                torch.cos(2.0 * torch.pi * raw_geometry),
            ),
            dim=-1,
        )
        embedding = self.geometry_mlp(positional)
        residual = self.residual_cap * torch.tanh(
            self.query_adapter(
                torch.cat((local.queries.detach(), embedding), dim=-1)
            )
        )
        canonical_queries = self.output_norm(
            local.queries.detach() + residual
        )
        valid = geometry.valid_mask.unsqueeze(-1)
        canonical_queries = torch.where(
            valid,
            canonical_queries,
            local.queries.detach(),
        )
        canonical = QueryEvidence(
            queries=canonical_queries,
            logits=local.logits.detach(),
            boxes=global_boxes,
            quality=local.quality.detach(),
        )
        return GeometryProjectionOutput(
            canonical_local=canonical,
            geometry_embedding=embedding,
        )


class GlobalLocalQueryInteraction(nn.Module):
    """Use one cross-attention layer to read frozen global query context."""

    def __init__(self, *, query_dim: int, num_heads: int) -> None:
        super().__init__()
        if query_dim <= 0 or num_heads <= 0 or query_dim % num_heads:
            raise ValueError("query_dim must be divisible by positive num_heads")
        self.query_dim = int(query_dim)
        self.global_box_position = nn.Sequential(
            nn.Linear(4, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(query_dim)

    def forward(
        self,
        canonical_local: QueryEvidence,
        global_evidence: QueryEvidence,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            canonical_local.query_dim != self.query_dim
            or global_evidence.query_dim != self.query_dim
        ):
            raise ValueError("query_dim does not match interaction module")
        if canonical_local.batch_size != global_evidence.batch_size:
            raise ValueError("global and local batch sizes must match")
        global_queries = global_evidence.queries.detach()
        keys = global_queries + self.global_box_position(
            global_evidence.boxes.detach()
        )
        attended, _ = self.attention(
            canonical_local.queries,
            keys,
            global_queries,
            need_weights=False,
        )
        context = self.output_norm(canonical_local.queries + attended)
        return context * valid_mask.unsqueeze(-1).to(context.dtype)


class AnchorPreservedResidualFusion(nn.Module):
    """Predict only a bounded local-score residual inside the anchor domain."""

    def __init__(
        self,
        *,
        query_dim: int,
        residual_eta: float,
    ) -> None:
        super().__init__()
        if query_dim <= 0:
            raise ValueError("query_dim must be positive")
        if not 0.0 < residual_eta <= 1.0:
            raise ValueError("residual_eta must be in (0,1]")
        self.residual_eta = float(residual_eta)
        self.head = nn.Sequential(
            nn.Linear(query_dim * 3 + 1, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, 1),
        )
        _zero_linear(self.head[-1])

    def forward(
        self,
        *,
        canonical_queries: torch.Tensor,
        global_context: torch.Tensor,
        geometry_embedding: torch.Tensor,
        base_scores: torch.Tensor,
        anchor_mask: torch.Tensor,
        residual_enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not residual_enabled:
            # The explicit branch is required for bitwise SADED fallback.
            return torch.zeros_like(base_scores), base_scores
        if anchor_mask.shape != base_scores.shape or anchor_mask.dtype != torch.bool:
            raise ValueError("anchor_mask must be bool and match base_scores")
        if bool(((base_scores < 0.0) | (base_scores > 1.0)).any()):
            raise ValueError("base_scores must be probabilities in [0,1]")
        features = torch.cat(
            (
                canonical_queries,
                global_context,
                geometry_embedding,
                base_scores,
            ),
            dim=-1,
        )
        residual = torch.tanh(self.head(features))
        eligible_residual = torch.where(
            anchor_mask,
            residual,
            torch.zeros_like(residual),
        )
        rescored = base_scores * torch.exp(
            self.residual_eta * eligible_residual
        )
        rescored = torch.minimum(torch.ones_like(rescored), rescored)
        adjusted = torch.where(anchor_mask, rescored, base_scores)
        return eligible_residual, adjusted


class GCQF(nn.Module):
    """The single GCTE trainable module exposed to configuration and diagrams."""

    def __init__(
        self,
        *,
        query_dim: int = 256,
        num_classes: int = 10,
        num_heads: int = 8,
        num_views: int = 4,
        residual_cap: float = 0.2,
        residual_eta: float = 0.2,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.query_dim = int(query_dim)
        self.num_classes = int(num_classes)
        self.num_views = int(num_views)
        self.geometry_projector = GeometryQueryProjector(
            query_dim=query_dim,
            num_views=num_views,
            residual_cap=residual_cap,
        )
        self.query_interaction = GlobalLocalQueryInteraction(
            query_dim=query_dim,
            num_heads=num_heads,
        )
        self.sr_peg = ScaleRiskProtectedEvidenceGate(
            query_dim=query_dim,
            num_heads=num_heads,
            residual_eta=residual_eta,
        )

    def forward(
        self,
        global_evidence: QueryEvidence,
        local_evidence: QueryEvidence,
        geometry: ViewGeometry,
        *,
        anchor_mask: torch.Tensor,
        residual_enabled: bool = True,
    ) -> GCQFOutput:
        for name, evidence in (
            ("global", global_evidence),
            ("local", local_evidence),
        ):
            if evidence.query_dim != self.query_dim:
                raise ValueError(f"{name} query_dim mismatch")
            if evidence.num_classes != self.num_classes:
                raise ValueError(f"{name} num_classes mismatch")
        projected = self.geometry_projector(local_evidence, geometry)
        context = self.query_interaction(
            projected.canonical_local,
            global_evidence,
            geometry.valid_mask,
        )
        gated = self.sr_peg(
            canonical_queries=projected.canonical_local.queries,
            global_context=context,
            geometry_embedding=projected.geometry_embedding,
            local_scores=local_evidence.quality,
            global_queries=global_evidence.queries,
            global_boxes=global_evidence.boxes,
            global_scores=global_evidence.quality,
            local_valid_mask=geometry.valid_mask,
            residual_enabled=residual_enabled,
            residual_eligible_mask=anchor_mask,
        )
        return GCQFOutput(
            global_evidence=global_evidence,
            canonical_local=projected.canonical_local,
            geometry_embedding=projected.geometry_embedding,
            global_context=context,
            tiny_utility_logits=gated.tiny_utility_logits,
            non_tiny_risk_logits=gated.non_tiny_risk_logits,
            global_retain_logits=gated.global_retain_logits,
            score_residual=gated.score_residual,
            adjusted_local_scores=gated.adjusted_local_scores,
        )


__all__ = ["GCQF", "GCQFOutput"]
