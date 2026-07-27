"""Frozen supervision for the GCQF screening stage."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


EQUIVARIANCE_WEIGHT = 0.1
RESIDUAL_WEIGHT = 0.01
TINY_UTILITY_WEIGHT = 1.0
NON_TINY_RISK_WEIGHT = 2.0
GLOBAL_RETAIN_WEIGHT = 2.0


@dataclass(frozen=True)
class GCQFLoss:
    total: torch.Tensor
    quality: torch.Tensor
    equivariance: torch.Tensor
    residual_regularization: torch.Tensor
    tiny_utility: torch.Tensor
    non_tiny_risk: torch.Tensor
    global_retain: torch.Tensor
    equivariance_weight: float = EQUIVARIANCE_WEIGHT
    residual_weight: float = RESIDUAL_WEIGHT
    tiny_utility_weight: float = TINY_UTILITY_WEIGHT
    non_tiny_risk_weight: float = NON_TINY_RISK_WEIGHT
    global_retain_weight: float = GLOBAL_RETAIN_WEIGHT


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
    tiny_utility_logits: torch.Tensor | None = None,
    tiny_utility_targets: torch.Tensor | None = None,
    non_tiny_risk_logits: torch.Tensor | None = None,
    non_tiny_risk_targets: torch.Tensor | None = None,
    global_retain_logits: torch.Tensor | None = None,
    global_retain_targets: torch.Tensor | None = None,
    positive_weights: dict[str, float] | None = None,
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
    # Probability-form BCE is deliberately rejected by CUDA autocast. Keep
    # this numerically sensitive legacy term in FP32 while the surrounding
    # query module remains under the frozen FP16 autocast protocol.
    with torch.autocast(
        device_type=adjusted_scores.device.type,
        enabled=False,
    ):
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

    sr_values = (
        tiny_utility_logits,
        tiny_utility_targets,
        non_tiny_risk_logits,
        non_tiny_risk_targets,
        global_retain_logits,
        global_retain_targets,
        positive_weights,
    )
    if all(value is None for value in sr_values):
        tiny_utility = adjusted_scores.sum() * 0.0
        non_tiny_risk = adjusted_scores.sum() * 0.0
        global_retain = adjusted_scores.sum() * 0.0
    elif any(value is None for value in sr_values):
        raise ValueError("all SR-PEG loss inputs must be supplied together")
    else:
        assert tiny_utility_logits is not None
        assert tiny_utility_targets is not None
        assert non_tiny_risk_logits is not None
        assert non_tiny_risk_targets is not None
        assert global_retain_logits is not None
        assert global_retain_targets is not None
        assert positive_weights is not None
        if tiny_utility_logits.shape != adjusted_scores.shape:
            raise ValueError("tiny utility logits must match local scores")
        if non_tiny_risk_logits.shape != adjusted_scores.shape:
            raise ValueError("non-tiny risk logits must match local scores")
        if tiny_utility_targets.shape != adjusted_scores.shape:
            raise ValueError("tiny utility targets must match local scores")
        if non_tiny_risk_targets.shape != adjusted_scores.shape:
            raise ValueError("non-tiny risk targets must match local scores")
        if (
            global_retain_logits.ndim != 3
            or global_retain_logits.shape[0] != batch
            or global_retain_logits.shape[-1] != 1
            or global_retain_targets.shape != global_retain_logits.shape
        ):
            raise ValueError("global retain tensors must share [B,G,1]")
        if set(positive_weights) != {"tiny", "risk", "retain"}:
            raise ValueError("positive_weights schema drift")
        if any(
            not 1.0 <= float(value) <= 20.0
            for value in positive_weights.values()
        ):
            raise ValueError("positive weights must be in [1,20]")

        def binary_head_loss(
            logits: torch.Tensor,
            targets: torch.Tensor,
            *,
            pos_weight: float,
            mask: torch.Tensor | None,
        ) -> torch.Tensor:
            detached = targets.detach().to(
                device=logits.device,
                dtype=logits.dtype,
            )
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("SR-PEG logits must be finite")
            if bool(((detached < 0.0) | (detached > 1.0)).any()):
                raise ValueError("SR-PEG targets must be in [0,1]")
            terms = F.binary_cross_entropy_with_logits(
                logits.float(),
                detached.float(),
                pos_weight=torch.tensor(
                    float(pos_weight),
                    device=logits.device,
                ),
                reduction="none",
            )
            return terms.mean() if mask is None else _masked_mean(terms, mask)

        local_mask = valid_mask.unsqueeze(-1)
        tiny_utility = binary_head_loss(
            tiny_utility_logits,
            tiny_utility_targets,
            pos_weight=positive_weights["tiny"],
            mask=local_mask,
        )
        non_tiny_risk = binary_head_loss(
            non_tiny_risk_logits,
            non_tiny_risk_targets,
            pos_weight=positive_weights["risk"],
            mask=local_mask,
        )
        global_retain = binary_head_loss(
            global_retain_logits,
            global_retain_targets,
            pos_weight=positive_weights["retain"],
            mask=None,
        )

    total = (
        quality
        + EQUIVARIANCE_WEIGHT * equivariance
        + RESIDUAL_WEIGHT * residual_regularization
        + TINY_UTILITY_WEIGHT * tiny_utility
        + NON_TINY_RISK_WEIGHT * non_tiny_risk
        + GLOBAL_RETAIN_WEIGHT * global_retain
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("GCQF loss is nonfinite")
    return GCQFLoss(
        total=total,
        quality=quality,
        equivariance=equivariance,
        residual_regularization=residual_regularization,
        tiny_utility=tiny_utility,
        non_tiny_risk=non_tiny_risk,
        global_retain=global_retain,
    )


__all__ = [
    "EQUIVARIANCE_WEIGHT",
    "GLOBAL_RETAIN_WEIGHT",
    "GCQFLoss",
    "RESIDUAL_WEIGHT",
    "NON_TINY_RISK_WEIGHT",
    "TINY_UTILITY_WEIGHT",
    "compute_gcqf_loss",
]
