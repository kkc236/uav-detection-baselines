"""Trainable scale-risk protected evidence gate for GCQF."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SRPEGOutput:
    """Query-level predictions emitted by the third GCQF stage."""

    tiny_utility_logits: torch.Tensor
    non_tiny_risk_logits: torch.Tensor
    global_retain_logits: torch.Tensor
    score_residual: torch.Tensor
    adjusted_local_scores: torch.Tensor


def _zero_linear(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class ScaleRiskProtectedEvidenceGate(nn.Module):
    """Learn tiny utility, non-tiny risk, global retention, and score residuals."""

    def __init__(
        self,
        *,
        query_dim: int,
        num_heads: int,
        residual_eta: float = 0.2,
    ) -> None:
        super().__init__()
        if query_dim <= 0 or num_heads <= 0 or query_dim % num_heads:
            raise ValueError("query_dim must be divisible by positive num_heads")
        if not 0.0 < residual_eta <= 1.0:
            raise ValueError("residual_eta must be in (0,1]")
        self.query_dim = int(query_dim)
        self.residual_eta = float(residual_eta)
        self.local_trunk = nn.Sequential(
            nn.Linear(query_dim * 3 + 1, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
            nn.LayerNorm(query_dim),
        )
        self.tiny_utility_head = nn.Linear(query_dim, 1)
        self.non_tiny_risk_head = nn.Linear(query_dim, 1)
        self.score_residual_head = nn.Linear(query_dim, 1)
        self.global_attention = nn.MultiheadAttention(
            query_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.global_box_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.global_retain_head = nn.Sequential(
            nn.Linear(query_dim * 2 + 64 + 1, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, 1),
        )
        _zero_linear(self.tiny_utility_head)
        _zero_linear(self.non_tiny_risk_head)
        _zero_linear(self.score_residual_head)
        _zero_linear(self.global_retain_head[-1])

    def forward(
        self,
        *,
        canonical_queries: torch.Tensor,
        global_context: torch.Tensor,
        geometry_embedding: torch.Tensor,
        local_scores: torch.Tensor,
        global_queries: torch.Tensor,
        global_boxes: torch.Tensor,
        global_scores: torch.Tensor,
        local_valid_mask: torch.Tensor,
        residual_enabled: bool,
        residual_eligible_mask: torch.Tensor | None = None,
    ) -> SRPEGOutput:
        local_shape = canonical_queries.shape
        if canonical_queries.ndim != 3 or local_shape[-1] != self.query_dim:
            raise ValueError("canonical_queries must have shape [B,L,query_dim]")
        if global_context.shape != local_shape:
            raise ValueError("global_context must match canonical_queries")
        if geometry_embedding.shape != local_shape:
            raise ValueError("geometry_embedding must match canonical_queries")
        if local_scores.shape != (*local_shape[:2], 1):
            raise ValueError("local_scores must have shape [B,L,1]")
        if local_valid_mask.shape != local_shape[:2] or local_valid_mask.dtype != torch.bool:
            raise ValueError("local_valid_mask must be bool with shape [B,L]")
        if bool(((local_scores < 0.0) | (local_scores > 1.0)).any()):
            raise ValueError("local_scores must be probabilities in [0,1]")
        if (
            global_queries.ndim != 3
            or global_queries.shape[0] != local_shape[0]
            or global_queries.shape[-1] != self.query_dim
        ):
            raise ValueError("global_queries must have shape [B,G,query_dim]")
        global_prefix = global_queries.shape[:2]
        if global_boxes.shape != (*global_prefix, 4):
            raise ValueError("global_boxes must have shape [B,G,4]")
        if global_scores.shape != (*global_prefix, 1):
            raise ValueError("global_scores must have shape [B,G,1]")
        if bool(((global_scores < 0.0) | (global_scores > 1.0)).any()):
            raise ValueError("global_scores must be probabilities in [0,1]")
        if not bool(local_valid_mask.any(dim=1).all()):
            raise ValueError("each batch item must contain a valid local query")
        if residual_eligible_mask is None:
            residual_eligible_mask = local_valid_mask.unsqueeze(-1)
        if (
            residual_eligible_mask.shape != local_scores.shape
            or residual_eligible_mask.dtype != torch.bool
        ):
            raise ValueError(
                "residual_eligible_mask must be bool and match local_scores"
            )

        local_features = torch.cat(
            (
                canonical_queries,
                global_context,
                geometry_embedding,
                local_scores.detach(),
            ),
            dim=-1,
        )
        local_hidden = self.local_trunk(local_features)
        valid = local_valid_mask.unsqueeze(-1)
        tiny_logits = torch.where(
            valid,
            self.tiny_utility_head(local_hidden),
            torch.zeros_like(local_scores),
        )
        risk_logits = torch.where(
            valid,
            self.non_tiny_risk_head(local_hidden),
            torch.zeros_like(local_scores),
        )
        residual = torch.where(
            valid & residual_eligible_mask,
            torch.tanh(self.score_residual_head(local_hidden)),
            torch.zeros_like(local_scores),
        )

        frozen_global = global_queries.detach()
        attended_local, _ = self.global_attention(
            frozen_global,
            canonical_queries,
            canonical_queries,
            key_padding_mask=~local_valid_mask,
            need_weights=False,
        )
        retain_features = torch.cat(
            (
                frozen_global,
                attended_local,
                self.global_box_mlp(global_boxes.detach()),
                global_scores.detach(),
            ),
            dim=-1,
        )
        retain_logits = self.global_retain_head(retain_features)

        if residual_enabled:
            adjusted = local_scores * torch.exp(self.residual_eta * residual)
            adjusted = adjusted.clamp(max=1.0)
        else:
            residual = torch.zeros_like(local_scores)
            adjusted = local_scores
        return SRPEGOutput(
            tiny_utility_logits=tiny_logits,
            non_tiny_risk_logits=risk_logits,
            global_retain_logits=retain_logits,
            score_residual=residual,
            adjusted_local_scores=adjusted,
        )


__all__ = [
    "SRPEGOutput",
    "ScaleRiskProtectedEvidenceGate",
]
