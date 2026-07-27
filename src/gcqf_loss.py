"""Frozen supervision for the GCQF screening stage."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


EQUIVARIANCE_WEIGHT = 0.1
RESIDUAL_WEIGHT = 0.01


@dataclass(frozen=True)
class GCQFLoss:
    total: torch.Tensor
    quality: torch.Tensor
    equivariance: torch.Tensor
    residual_regularization: torch.Tensor
    equivariance_weight: float = EQUIVARIANCE_WEIGHT
    residual_weight: float = RESIDUAL_WEIGHT


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.mean()


def compute_gcqf_loss(
    *,
    adjusted_scores: torch.Tensor,
    quality_targets: torch.Tensor,
    canonical_queries: torch.Tensor,
    equivariance_pairs: torch.Tensor,
    score_residual: torch.Tensor,
    valid_mask: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> GCQFLoss:
    """Compute quality, cross-view equivariance, and anchor residual losses."""

    if adjusted_scores.ndim != 3 or adjusted_scores.shape[-1] != 1:
        raise ValueError("adjusted_scores must be [B,Q,1]")
    if quality_targets.shape != adjusted_scores.shape:
        raise ValueError("quality_targets must match adjusted_scores")
    if score_residual.shape != adjusted_scores.shape:
        raise ValueError("score_residual must match adjusted_scores")
    batch, query_count, _ = adjusted_scores.shape
    if canonical_queries.shape[:2] != (batch, query_count):
        raise ValueError("canonical_queries must share [B,Q]")
    if valid_mask.shape != (batch, query_count) or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool [B,Q]")
    if anchor_mask.shape != adjusted_scores.shape or anchor_mask.dtype != torch.bool:
        raise ValueError("anchor_mask must be bool [B,Q,1]")
    if equivariance_pairs.ndim != 2 or equivariance_pairs.shape[1:] != (3,):
        raise ValueError("equivariance_pairs must be [P,3]")
    if equivariance_pairs.dtype != torch.long:
        raise ValueError("equivariance_pairs must use torch.long")
    if not bool(torch.isfinite(adjusted_scores).all()):
        raise FloatingPointError("adjusted_scores must be finite")
    detached_targets = quality_targets.detach().to(adjusted_scores.dtype)
    if bool(((detached_targets < 0.0) | (detached_targets > 1.0)).any()):
        raise ValueError("quality_targets must be in [0,1]")

    eligible = valid_mask.unsqueeze(-1) & anchor_mask
    probabilities = adjusted_scores.float().clamp(1e-6, 1.0 - 1e-6)
    quality_terms = F.binary_cross_entropy(
        probabilities,
        detached_targets.float(),
        reduction="none",
    )
    quality = _masked_mean(quality_terms, eligible)
    residual_regularization = _masked_mean(
        score_residual.float().square(),
        eligible,
    )

    if equivariance_pairs.numel() == 0:
        equivariance = canonical_queries.float().sum() * 0.0
    else:
        if bool((equivariance_pairs < 0).any()):
            raise ValueError("equivariance pair index must be nonnegative")
        pair_batch, left_index, right_index = equivariance_pairs.unbind(dim=1)
        if bool((pair_batch >= batch).any()) or bool(
            (left_index >= query_count).any()
        ) or bool((right_index >= query_count).any()):
            raise ValueError("equivariance pair index is out of range")
        if not bool(
            (
                valid_mask[pair_batch, left_index]
                & valid_mask[pair_batch, right_index]
            ).all()
        ):
            raise ValueError("equivariance pair index references invalid query")
        left = canonical_queries[pair_batch, left_index].float()
        right = canonical_queries[pair_batch, right_index].float()
        equivariance = (
            1.0 - F.cosine_similarity(left, right, dim=-1, eps=1e-8)
        ).mean()

    total = (
        quality
        + EQUIVARIANCE_WEIGHT * equivariance
        + RESIDUAL_WEIGHT * residual_regularization
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("GCQF loss is nonfinite")
    return GCQFLoss(
        total=total,
        quality=quality,
        equivariance=equivariance,
        residual_regularization=residual_regularization,
    )


__all__ = [
    "EQUIVARIANCE_WEIGHT",
    "GCQFLoss",
    "RESIDUAL_WEIGHT",
    "compute_gcqf_loss",
]
