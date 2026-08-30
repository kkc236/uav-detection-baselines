"""Training-only Best-Progressive Distribution Distillation primitives.

BPDD consumes cumulative FDR corner distributions and existing stock matches.
It owns no model parameters and is never called by the inference path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from src.fdr_loss import FDRDetectionLoss, MatchIndices
from src.fdr_math import REG_MAX, REG_SCALE, UP, bbox2distance, cxcywh_to_xyxy


@dataclass(frozen=True)
class BPDDOptions:
    """Frozen numerical options for the first BPDD research candidate."""

    enabled: bool = True
    weight: float = 0.5
    temperature: float = 0.5
    margin: float = 0.02
    eps: float = 1e-6
    assignment_mode: str = "final"

    def __post_init__(self) -> None:
        numeric = {
            "weight": self.weight,
            "temperature": self.temperature,
            "margin": self.margin,
            "eps": self.eps,
        }
        if not all(math.isfinite(float(value)) for value in numeric.values()):
            raise ValueError("BPDD numerical options must be finite")
        if self.weight < 0:
            raise ValueError("BPDD weight must be non-negative")
        if self.temperature <= 0:
            raise ValueError("BPDD temperature must be positive")
        if self.margin < 0:
            raise ValueError("BPDD margin must be non-negative")
        if self.eps <= 0:
            raise ValueError("BPDD eps must be positive")
        if self.assignment_mode not in {"final", "consistent"}:
            raise ValueError("assignment_mode must be 'final' or 'consistent'")


@dataclass(frozen=True)
class BPDDResult:
    """One scalar loss and detached evidence for the current batch."""

    loss: Tensor
    statistics: dict[str, Tensor]


class BPDDDetectionLoss(FDRDetectionLoss):
    """Unchanged stock+FDR criterion followed by isolated normal-query BPDD."""

    def __init__(
        self,
        *args,
        bpdd_options: BPDDOptions | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bpdd_options = bpdd_options or BPDDOptions()
        self.bpdd_runtime_enabled = True
        self.last_bpdd_statistics: dict[str, Tensor] = {}

    def _matched_bpdd_inputs(
        self,
        corner_logits: Tensor,
        pre_boxes: Tensor,
        gt_bboxes: Tensor,
        matches: MatchIndices,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        predicted_index, target_index = self._get_index(matches)
        if target_index.numel() == 0:
            empty_logits = corner_logits[:, :0].reshape(
                corner_logits.shape[0], 0, 4, REG_MAX + 1
            )
            empty_target = torch.empty(
                (0, 4), dtype=torch.long, device=corner_logits.device
            )
            empty_weight = torch.empty(
                (0, 4), dtype=torch.float32, device=corner_logits.device
            )
            return empty_logits, empty_target, empty_weight, empty_weight

        batch_index, query_index = predicted_index
        matched_logits = corner_logits[:, batch_index, query_index].reshape(
            corner_logits.shape[0], -1, 4, REG_MAX + 1
        )
        matched_reference = pre_boxes[predicted_index].detach()
        matched_targets = gt_bboxes[target_index]
        target_indices, weight_right, weight_left = bbox2distance(
            matched_reference,
            cxcywh_to_xyxy(matched_targets),
            REG_MAX,
            REG_SCALE,
            UP,
        )
        matches_count = int(target_index.numel())
        return (
            matched_logits,
            target_indices.reshape(matches_count, 4).long(),
            weight_right.reshape(matches_count, 4),
            weight_left.reshape(matches_count, 4),
        )

    def forward(self, *args, **kwargs) -> dict[str, Tensor]:
        """Add BPDD only after the parent criterion records stock assignments."""

        losses = super().forward(*args, **kwargs)
        self.last_bpdd_statistics = {}
        if (
            not self.bpdd_runtime_enabled
            or not self.bpdd_options.enabled
            or self.bpdd_options.weight == 0
        ):
            return losses

        corner_logits = kwargs.get("corner_logits")
        pre_boxes = kwargs.get("pre_boxes")
        if corner_logits is None or pre_boxes is None:
            raise ValueError("enabled BPDD requires corner_logits and pre_boxes")
        if not isinstance(corner_logits, Tensor) or not isinstance(pre_boxes, Tensor):
            raise TypeError("BPDD corner_logits and pre_boxes must be tensors")
        if corner_logits.ndim != 4 or corner_logits.shape[-1] != 4 * (REG_MAX + 1):
            raise ValueError("corner_logits must have shape [layers,batch,queries,132]")

        assignments = self.normal_assignment_snapshot()
        if len(assignments) != corner_logits.shape[0] + 1:
            raise ValueError("BPDD requires encoder plus one stock assignment per decoder layer")
        batch = args[1] if len(args) >= 2 else kwargs.get("batch")
        if not isinstance(batch, dict) or "bboxes" not in batch:
            raise TypeError("BPDD requires the stock target batch")
        if self.bpdd_options.assignment_mode == "consistent":
            decoder_assignments = assignments[1:]
            layer_matches: list[LayerMatchTriples] = []
            for matches in decoder_assignments:
                predicted_index, target_index = self._get_index(matches)
                batch_index, query_index = predicted_index
                layer_matches.append(
                    (batch_index, query_index, target_index)
                )
            result = assignment_consistent_bpdd_loss(
                corner_logits.reshape(
                    *corner_logits.shape[:-1], 4, REG_MAX + 1
                ),
                pre_boxes,
                batch["bboxes"],
                layer_matches,
                options=self.bpdd_options,
            )
        else:
            matched = self._matched_bpdd_inputs(
                corner_logits,
                pre_boxes,
                batch["bboxes"],
                assignments[-1],
            )
            result = bpdd_distribution_loss(
                *matched,
                options=self.bpdd_options,
            )
        losses["loss_bpdd"] = result.loss
        self.last_bpdd_statistics = result.statistics
        return losses


def _expand_edge_tensor(value: Tensor, target_shape: torch.Size, name: str) -> Tensor:
    if value.shape != target_shape[-2:]:
        raise ValueError(f"{name} must have shape {tuple(target_shape[-2:])}")
    expanded = value
    for _ in range(len(target_shape) - 2):
        expanded = expanded.unsqueeze(0)
    return expanded.expand(target_shape)


def interpolated_edge_nll(
    log_probabilities: Tensor,
    target_indices: Tensor,
    weight_right: Tensor,
    weight_left: Tensor,
) -> Tensor:
    """Evaluate the exact adjacent-bin FGL target as a proper score."""

    if log_probabilities.ndim < 3:
        raise ValueError("log_probabilities must end in [matches,edges,bins]")
    matches, edges, bins = log_probabilities.shape[-3:]
    expected = torch.Size((matches, edges))
    if target_indices.shape != expected:
        raise ValueError(f"target_indices must have shape {tuple(expected)}")
    if weight_right.shape != expected or weight_left.shape != expected:
        raise ValueError(f"target weights must have shape {tuple(expected)}")
    if target_indices.dtype != torch.long:
        raise TypeError("target_indices must use torch.long")
    if not (
        log_probabilities.device
        == target_indices.device
        == weight_right.device
        == weight_left.device
    ):
        raise ValueError("BPDD targets and distributions must share a device")
    if target_indices.numel() and (
        int(target_indices.min()) < 0 or int(target_indices.max()) + 1 >= bins
    ):
        raise ValueError("target_indices must identify an adjacent in-range bin pair")

    edge_shape = log_probabilities.shape[:-1]
    indices = _expand_edge_tensor(target_indices, edge_shape, "target_indices")
    right_weights = _expand_edge_tensor(weight_right, edge_shape, "weight_right")
    left_weights = _expand_edge_tensor(weight_left, edge_shape, "weight_left")
    left_log = log_probabilities.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    right_log = log_probabilities.gather(
        -1, (indices + 1).unsqueeze(-1)
    ).squeeze(-1)
    return -(left_weights * left_log + right_weights * right_log)


def build_progressive_teachers(
    probabilities: Tensor,
    target_errors: Tensor,
    *,
    temperature: float,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Build detached softmin mixtures from future layers only."""

    if probabilities.ndim < 3:
        raise ValueError("probabilities must have layer and distribution axes")
    if probabilities.shape[:-1] != target_errors.shape:
        raise ValueError("target_errors must match probabilities without bins")
    if probabilities.shape[0] < 2:
        raise ValueError("BPDD requires at least two decoder layers")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    detached_probabilities = probabilities.detach().float()
    detached_errors = target_errors.detach().float()
    teachers: list[Tensor] = []
    weights: list[Tensor] = []
    for layer in range(probabilities.shape[0] - 1):
        future_errors = detached_errors[layer + 1 :]
        future_weights = torch.softmax(-future_errors / temperature, dim=0)
        teacher = (
            future_weights.unsqueeze(-1)
            * detached_probabilities[layer + 1 :]
        ).sum(dim=0)
        teachers.append(teacher.detach())
        weights.append(future_weights.detach())
    return tuple(teachers), tuple(weights)


def _zero_result(corner_logits: Tensor, matched_queries: int) -> BPDDResult:
    zero = corner_logits.float().sum() * 0.0
    scalar_zero = zero.detach()
    return BPDDResult(
        loss=zero,
        statistics={
            "active_edge_ratio": scalar_zero,
            "mean_reliability": scalar_zero,
            "mean_teacher_improvement": scalar_zero,
            "mixture_beats_final_ratio": scalar_zero,
            "mean_mixture_advantage_over_final": scalar_zero,
            "matched_queries": torch.tensor(
                matched_queries, dtype=torch.long, device=corner_logits.device
            ),
            "eligible_edges": torch.tensor(
                0, dtype=torch.long, device=corner_logits.device
            ),
        },
    )


LayerMatchTriples = tuple[Tensor, Tensor, Tensor]


def assignment_consistent_bpdd_loss(
    corner_logits: Tensor,
    pre_boxes: Tensor,
    gt_bboxes: Tensor,
    layer_matches: Sequence[LayerMatchTriples],
    *,
    options: BPDDOptions,
) -> BPDDResult:
    """Distill only across future layers that preserve a query-target match."""

    if corner_logits.ndim != 5:
        raise ValueError(
            "corner_logits must have shape [layers,batch,queries,4,bins]"
        )
    layers, batch_size, queries, edges, bins = corner_logits.shape
    if layers < 2 or edges != 4 or bins != REG_MAX + 1:
        raise ValueError(
            "corner_logits must have shape [layers>=2,batch,queries,4,33]"
        )
    if pre_boxes.shape != (batch_size, queries, 4):
        raise ValueError("pre_boxes must have shape [batch,queries,4]")
    if gt_bboxes.ndim != 2 or gt_bboxes.shape[-1] != 4:
        raise ValueError("gt_bboxes must have shape [targets,4]")
    if len(layer_matches) != layers:
        raise ValueError("expected one assignment triple per decoder layer")
    if options.assignment_mode != "consistent":
        raise ValueError("assignment-consistent BPDD requires consistent mode")

    normalized_matches: list[LayerMatchTriples] = []
    for triple in layer_matches:
        if len(triple) != 3:
            raise ValueError("each layer match must contain batch, query, target")
        batch_index, query_index, target_index = triple
        if not (
            batch_index.ndim == query_index.ndim == target_index.ndim == 1
            and batch_index.shape == query_index.shape == target_index.shape
        ):
            raise ValueError("layer match indices must be same-shaped vectors")
        batch_index = batch_index.to(device=corner_logits.device, dtype=torch.long)
        query_index = query_index.to(device=corner_logits.device, dtype=torch.long)
        target_index = target_index.to(device=corner_logits.device, dtype=torch.long)
        if batch_index.numel() and (
            int(batch_index.min()) < 0
            or int(batch_index.max()) >= batch_size
            or int(query_index.min()) < 0
            or int(query_index.max()) >= queries
            or int(target_index.min()) < 0
            or int(target_index.max()) >= gt_bboxes.shape[0]
        ):
            raise ValueError("layer match index is out of range")
        normalized_matches.append((batch_index, query_index, target_index))

    graph_zero = corner_logits.float().sum() * 0.0
    candidate_matches = sum(
        int(batch_index.numel())
        for batch_index, _, _ in normalized_matches[:-1]
    )
    candidate_edges = candidate_matches * edges
    if (
        not options.enabled
        or options.weight == 0
        or candidate_matches == 0
    ):
        scalar_zero = graph_zero.detach()
        return BPDDResult(
            loss=graph_zero,
            statistics={
                "active_edge_ratio": scalar_zero,
                "mean_reliability": scalar_zero,
                "mean_teacher_improvement": scalar_zero,
                "mixture_beats_final_ratio": scalar_zero,
                "mean_mixture_advantage_over_final": scalar_zero,
                "stable_match_ratio": scalar_zero,
                "matched_queries": torch.tensor(
                    candidate_matches, dtype=torch.long, device=corner_logits.device
                ),
                "candidate_source_matches": torch.tensor(
                    candidate_matches, dtype=torch.long, device=corner_logits.device
                ),
                "stable_source_matches": torch.tensor(
                    0, dtype=torch.long, device=corner_logits.device
                ),
                "eligible_edges": torch.tensor(
                    0, dtype=torch.long, device=corner_logits.device
                ),
            },
        )

    logits = corner_logits.float()
    pre_boxes = pre_boxes.detach().to(device=corner_logits.device)
    gt_bboxes = gt_bboxes.to(device=corner_logits.device)
    terms: list[Tensor] = []
    reliabilities: list[Tensor] = []
    improvements: list[Tensor] = []
    active_masks: list[Tensor] = []
    stable_matches = torch.zeros((), dtype=torch.long, device=corner_logits.device)

    for source_layer in range(layers - 1):
        batch_index, query_index, target_index = normalized_matches[source_layer]
        matches = int(batch_index.numel())
        if matches == 0:
            continue

        matched_reference = pre_boxes[batch_index, query_index]
        matched_targets = gt_bboxes[target_index]
        target_indices, weight_right, weight_left = bbox2distance(
            matched_reference,
            cxcywh_to_xyxy(matched_targets),
            REG_MAX,
            REG_SCALE,
            UP,
        )
        target_indices = target_indices.reshape(matches, edges).long()
        weight_right = weight_right.reshape(matches, edges).float()
        weight_left = weight_left.reshape(matches, edges).float()

        future_stability: list[Tensor] = []
        for future_layer in range(source_layer + 1, layers):
            future_batch, future_query, future_target = normalized_matches[
                future_layer
            ]
            if future_batch.numel() == 0:
                stable = torch.zeros(
                    matches, dtype=torch.bool, device=corner_logits.device
                )
            else:
                stable = (
                    (future_batch[:, None] == batch_index[None, :])
                    & (future_query[:, None] == query_index[None, :])
                    & (future_target[:, None] == target_index[None, :])
                ).any(dim=0)
            future_stability.append(stable)
        stable_tensor = torch.stack(future_stability)
        stable_source = stable_tensor.any(dim=0)
        stable_matches = stable_matches + stable_source.sum()

        future_logits = logits[source_layer + 1 :, batch_index, query_index]
        future_probabilities = future_logits.detach().softmax(dim=-1)
        future_log_probabilities = future_probabilities.clamp_min(
            options.eps
        ).log()
        future_errors = interpolated_edge_nll(
            future_log_probabilities,
            target_indices,
            weight_right,
            weight_left,
        )
        stable_edges = stable_tensor.unsqueeze(-1).expand_as(future_errors)
        scores = -future_errors / options.temperature
        scores = scores.masked_fill(~stable_edges, float("-inf"))
        has_teacher = stable_edges.any(dim=0)
        max_score = scores.amax(dim=0)
        safe_max = torch.where(has_teacher, max_score, torch.zeros_like(max_score))
        unnormalized = torch.where(
            stable_edges,
            torch.exp(scores - safe_max.unsqueeze(0)),
            torch.zeros_like(scores),
        )
        mixture_weights = unnormalized / unnormalized.sum(dim=0).clamp_min(
            options.eps
        )
        teacher = (
            mixture_weights.unsqueeze(-1) * future_probabilities
        ).sum(dim=0).detach()
        teacher_log = teacher.clamp_min(options.eps).log()
        teacher_error = interpolated_edge_nll(
            teacher_log,
            target_indices,
            weight_right,
            weight_left,
        )

        source_log = torch.log_softmax(
            logits[source_layer, batch_index, query_index], dim=-1
        )
        source_error = interpolated_edge_nll(
            source_log,
            target_indices,
            weight_right,
            weight_left,
        )
        improvement = source_error.detach() - teacher_error.detach()
        reliability = (
            (improvement - options.margin)
            / source_error.detach().clamp_min(options.eps)
        ).clamp(0.0, 1.0)
        reliability = torch.where(
            has_teacher, reliability, torch.zeros_like(reliability)
        )
        divergence = (teacher * (teacher_log - source_log)).sum(dim=-1)
        terms.append(reliability * divergence)
        reliabilities.append(reliability)
        improvements.append(
            torch.where(has_teacher, improvement, torch.zeros_like(improvement))
        )
        active_masks.append(reliability > 0)

    if not terms:
        raise RuntimeError("candidate BPDD matches produced no source terms")
    term_tensor = torch.cat([value.reshape(-1) for value in terms])
    reliability_tensor = torch.cat(
        [value.reshape(-1) for value in reliabilities]
    )
    improvement_tensor = torch.cat(
        [value.reshape(-1) for value in improvements]
    )
    active_tensor = torch.cat([value.reshape(-1) for value in active_masks])
    loss = term_tensor.sum() * options.weight / max(candidate_edges, 1)
    scalar_zero = graph_zero.detach()
    return BPDDResult(
        loss=loss,
        statistics={
            "active_edge_ratio": active_tensor.float().mean().detach(),
            "mean_reliability": reliability_tensor.mean().detach(),
            "mean_teacher_improvement": improvement_tensor.clamp_min(0).mean().detach(),
            "mixture_beats_final_ratio": scalar_zero,
            "mean_mixture_advantage_over_final": scalar_zero,
            "stable_match_ratio": (
                stable_matches.float() / max(candidate_matches, 1)
            ).detach(),
            "matched_queries": torch.tensor(
                candidate_matches, dtype=torch.long, device=corner_logits.device
            ),
            "candidate_source_matches": torch.tensor(
                candidate_matches, dtype=torch.long, device=corner_logits.device
            ),
            "stable_source_matches": stable_matches.detach(),
            "eligible_edges": stable_matches.detach() * edges,
        },
    )


def bpdd_distribution_loss(
    corner_logits: Tensor,
    target_indices: Tensor,
    weight_right: Tensor,
    weight_left: Tensor,
    *,
    options: BPDDOptions,
) -> BPDDResult:
    """Distill a GT-better detached future mixture into earlier FDR layers."""

    if corner_logits.ndim != 4:
        raise ValueError("corner_logits must have shape [layers,matches,4,bins]")
    layers, matches, edges, bins = corner_logits.shape
    if layers < 2 or edges != 4 or bins < 2:
        raise ValueError("corner_logits must have shape [layers>=2,matches,4,bins>=2]")
    expected = (matches, edges)
    if target_indices.shape != expected:
        raise ValueError(f"target_indices must have shape {expected}")
    if weight_right.shape != expected or weight_left.shape != expected:
        raise ValueError(f"target weights must have shape {expected}")
    if not options.enabled or options.weight == 0 or matches == 0:
        return _zero_result(corner_logits, matches)

    logits = corner_logits.float()
    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    target_errors = interpolated_edge_nll(
        log_probabilities,
        target_indices,
        weight_right.float(),
        weight_left.float(),
    )
    teachers, _mixture_weights = build_progressive_teachers(
        probabilities,
        target_errors,
        temperature=options.temperature,
    )

    weighted_terms: list[Tensor] = []
    reliabilities: list[Tensor] = []
    improvements: list[Tensor] = []
    mixture_advantages: list[Tensor] = []
    final_error = target_errors[-1].detach()
    for layer, teacher in enumerate(teachers):
        teacher_log = teacher.clamp_min(options.eps).log()
        teacher_error = interpolated_edge_nll(
            teacher_log,
            target_indices,
            weight_right.float(),
            weight_left.float(),
        )
        improvement = target_errors[layer].detach() - teacher_error.detach()
        reliability = (
            (improvement - options.margin)
            / target_errors[layer].detach().clamp_min(options.eps)
        ).clamp(0.0, 1.0)
        divergence = (
            teacher * (teacher_log - log_probabilities[layer])
        ).sum(dim=-1)
        weighted_terms.append(reliability * divergence)
        reliabilities.append(reliability)
        improvements.append(improvement)
        mixture_advantages.append(final_error - teacher_error.detach())

    reliability_tensor = torch.stack(reliabilities)
    improvement_tensor = torch.stack(improvements)
    mixture_advantage_tensor = torch.stack(mixture_advantages)
    term_tensor = torch.stack(weighted_terms)
    eligible_edges = int(term_tensor.numel())
    loss = term_tensor.sum() * options.weight / max(eligible_edges, 1)
    active = reliability_tensor > 0
    return BPDDResult(
        loss=loss,
        statistics={
            "active_edge_ratio": active.float().mean().detach(),
            "mean_reliability": reliability_tensor.mean().detach(),
            "mean_teacher_improvement": improvement_tensor.clamp_min(0).mean().detach(),
            "mixture_beats_final_ratio": (
                mixture_advantage_tensor > 0
            ).float().mean().detach(),
            "mean_mixture_advantage_over_final": mixture_advantage_tensor.mean().detach(),
            "matched_queries": torch.tensor(
                matches, dtype=torch.long, device=corner_logits.device
            ),
            "eligible_edges": torch.tensor(
                eligible_edges, dtype=torch.long, device=corner_logits.device
            ),
        },
    )


__all__ = [
    "BPDDDetectionLoss",
    "BPDDOptions",
    "BPDDResult",
    "assignment_consistent_bpdd_loss",
    "bpdd_distribution_loss",
    "build_progressive_teachers",
    "interpolated_edge_nll",
]
