"""Quality-gated BPDD from an independently supervised local expert."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor
import torch.nn.functional as F

from src.bpdd_loss import decoded_teacher_iou_gate, interpolated_edge_nll
from src.fdr_math import REG_MAX, REG_SCALE, UP, translate_gt, weighting_function


LayerMatchTriples = tuple[Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class CapacityBPDDOptions:
    """Frozen weights and gates for the capacity-v2 experiment."""

    enabled: bool = True
    loc_weight: float = 0.15
    cls_weight: float = 0.10
    loc_margin: float = 0.02
    cls_margin: float = 0.02
    loc_tau: float = 0.10
    cls_tau: float = 0.10
    warmup_start: int = 10
    warmup_end: int = 20

    def __post_init__(self) -> None:
        numeric = (
            self.loc_weight,
            self.cls_weight,
            self.loc_margin,
            self.cls_margin,
            self.loc_tau,
            self.cls_tau,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("capacity BPDD options must be finite")
        if self.loc_weight < 0 or self.cls_weight < 0:
            raise ValueError("capacity BPDD weights must be non-negative")
        if self.loc_margin < 0 or self.cls_margin < 0:
            raise ValueError("capacity BPDD margins must be non-negative")
        if self.loc_tau <= 0 or self.cls_tau <= 0:
            raise ValueError("capacity BPDD taus must be positive")
        if self.warmup_start < 0 or self.warmup_end <= self.warmup_start:
            raise ValueError("capacity BPDD warmup must have an increasing non-negative range")


@dataclass(frozen=True)
class CapacityBPDDResult:
    """Connected training losses and detached audit statistics."""

    loss: Tensor
    localization_loss: Tensor
    classification_loss: Tensor
    statistics: dict[str, Tensor]


def capacity_bpdd_warmup(completed_epoch: int, start: int = 10, end: int = 20) -> float:
    """Return the exact zero/linear/full distillation multiplier."""

    if start < 0 or end <= start:
        raise ValueError("warmup end must be greater than its non-negative start")
    epoch = int(completed_epoch)
    if epoch <= start:
        return 0.0
    if epoch >= end:
        return 1.0
    return float(epoch - start) / float(end - start)


def bounded_improvement(
    student_error: Tensor,
    teacher_error: Tensor,
    *,
    margin: float,
    tau: float,
) -> Tensor:
    """Map a detached teacher advantage to a bounded reliability in [0, 1]."""

    if student_error.shape != teacher_error.shape:
        raise ValueError("student and teacher errors must have the same shape")
    if margin < 0 or tau <= 0 or not math.isfinite(float(margin + tau)):
        raise ValueError("margin must be non-negative and tau must be positive")
    advantage = (student_error.detach() - teacher_error.detach() - margin).clamp_min(0)
    return advantage / (advantage + tau)


def raw_fdr_targets(reference: Tensor, gt_boxes: Tensor) -> tuple[Tensor, Tensor]:
    """Return unclipped FDR edge targets and their exact support mask."""

    if reference.ndim != 2 or reference.shape[-1] != 4 or gt_boxes.shape != reference.shape:
        raise ValueError("reference and gt_boxes must have shape [matches,4]")
    reference = reference.detach().float()
    gt_boxes = gt_boxes.detach().to(device=reference.device, dtype=torch.float32)
    if not torch.isfinite(reference).all() or not torch.isfinite(gt_boxes).all():
        raise ValueError("FDR targets require finite boxes")
    if (reference[:, 2:] <= 0).any() or (gt_boxes[:, 2:] <= 0).any():
        raise ValueError("FDR targets require positive box sizes")
    delta = reference[:, :2] - gt_boxes[:, :2]
    lengths = torch.cat(
        (gt_boxes[:, 2:] / 2 + delta, gt_boxes[:, 2:] / 2 - delta), dim=-1
    )
    scale = reference[:, 2:].repeat(1, 2) / REG_SCALE
    raw = lengths / scale - REG_SCALE / 2
    support = weighting_function(REG_MAX, UP, REG_SCALE).to(raw)
    representable = (
        torch.isfinite(raw) & (raw >= support[0]) & (raw <= support[-1])
    )
    return raw, representable


def _normalize_match(
    match: LayerMatchTriples,
    *,
    device: torch.device,
    batch_size: int,
    queries: int,
    targets: int,
) -> LayerMatchTriples:
    if len(match) != 3:
        raise ValueError("each assignment must contain batch, query and target indices")
    values = tuple(value.to(device=device, dtype=torch.long) for value in match)
    if not (values[0].ndim == values[1].ndim == values[2].ndim == 1):
        raise ValueError("assignment indices must be vectors")
    if not (values[0].shape == values[1].shape == values[2].shape):
        raise ValueError("assignment index vectors must have the same shape")
    if values[0].numel() and (
        int(values[0].min()) < 0
        or int(values[0].max()) >= batch_size
        or int(values[1].min()) < 0
        or int(values[1].max()) >= queries
        or int(values[2].min()) < 0
        or int(values[2].max()) >= targets
    ):
        raise ValueError("assignment index is out of range")
    return values  # type: ignore[return-value]


def _identity_mask(source: LayerMatchTriples, final: LayerMatchTriples) -> Tensor:
    batch, query, target = source
    final_batch, final_query, final_target = final
    if final_batch.numel() == 0:
        return torch.zeros_like(batch, dtype=torch.bool)
    return (
        (final_batch[:, None] == batch[None, :])
        & (final_query[:, None] == query[None, :])
        & (final_target[:, None] == target[None, :])
    ).any(dim=0)


def _bernoulli_kl(teacher_logits: Tensor, student_logits: Tensor) -> Tensor:
    eps = torch.finfo(torch.float32).eps
    teacher = teacher_logits.detach().float().sigmoid().clamp(eps, 1 - eps)
    student = student_logits.float().sigmoid().clamp(eps, 1 - eps)
    return (
        teacher * (teacher.log() - student.log())
        + (1 - teacher) * ((1 - teacher).log() - (1 - student).log())
    ).mean(dim=-1)


def quality_gated_capacity_distillation(
    student_corners: Tensor,
    student_classes: Tensor,
    expert_corners: Tensor,
    expert_classes: Tensor,
    reference: Tensor,
    gt_boxes: Tensor,
    gt_classes: Tensor,
    layer_matches: Sequence[LayerMatchTriples],
    final_matches: LayerMatchTriples,
    options: CapacityBPDDOptions,
    *,
    epoch: int,
) -> CapacityBPDDResult:
    """Distill a detached expert only where target quality and identity improve."""

    if student_corners.ndim == 4 and student_corners.shape[-1] == 4 * (REG_MAX + 1):
        student_corners = student_corners.reshape(*student_corners.shape[:-1], 4, REG_MAX + 1)
    if expert_corners.ndim == 3 and expert_corners.shape[-1] == 4 * (REG_MAX + 1):
        expert_corners = expert_corners.reshape(*expert_corners.shape[:-1], 4, REG_MAX + 1)
    if student_corners.ndim != 5 or student_corners.shape[-2:] != (4, REG_MAX + 1):
        raise ValueError("student_corners must have shape [layers,batch,queries,4,33]")
    layers, batch_size, queries = student_corners.shape[:3]
    if student_classes.ndim != 4 or student_classes.shape[:3] != (layers, batch_size, queries):
        raise ValueError("student_classes must have shape [layers,batch,queries,classes]")
    classes = student_classes.shape[-1]
    if expert_corners.shape != (batch_size, queries, 4, REG_MAX + 1):
        raise ValueError("expert_corners must contain every normal query")
    if expert_classes.shape != (batch_size, queries, classes):
        raise ValueError("expert_classes must contain every normal query")
    if reference.shape != (batch_size, queries, 4):
        raise ValueError("reference must have shape [batch,queries,4]")
    if gt_boxes.ndim != 2 or gt_boxes.shape[-1] != 4:
        raise ValueError("gt_boxes must have shape [targets,4]")
    if gt_classes.shape != (gt_boxes.shape[0],):
        raise ValueError("gt_classes must contain one class per target")
    if len(layer_matches) != layers:
        raise ValueError("one assignment triple is required per student layer")

    device = student_corners.device
    gt_boxes = gt_boxes.to(device=device, dtype=torch.float32)
    gt_classes = gt_classes.to(device=device, dtype=torch.long)
    reference = reference.detach().to(device=device, dtype=torch.float32)
    normalized = [
        _normalize_match(
            match,
            device=device,
            batch_size=batch_size,
            queries=queries,
            targets=gt_boxes.shape[0],
        )
        for match in layer_matches
    ]
    final = _normalize_match(
        final_matches,
        device=device,
        batch_size=batch_size,
        queries=queries,
        targets=gt_boxes.shape[0],
    )
    graph_zero = student_corners.float().sum() * 0 + student_classes.float().sum() * 0
    warmup = capacity_bpdd_warmup(epoch, options.warmup_start, options.warmup_end)

    loc_terms: list[Tensor] = []
    cls_terms: list[Tensor] = []
    candidate_matches = 0
    consistent_matches = 0
    representable_edges = 0
    active_edges = 0
    active_classes = 0
    loc_advantages: list[Tensor] = []
    cls_advantages: list[Tensor] = []
    loc_active_layers = 0
    cls_active_layers = 0

    teacher_corner_all = expert_corners.detach().to(device=device, dtype=torch.float32)
    teacher_class_all = expert_classes.detach().to(device=device, dtype=torch.float32)
    for layer, match in enumerate(normalized):
        batch_index, query_index, target_index = match
        count = int(batch_index.numel())
        candidate_matches += count
        if count == 0:
            continue
        consistent = _identity_mask(match, final)
        consistent_matches += int(consistent.sum())
        if not consistent.any():
            continue
        batch_index = batch_index[consistent]
        query_index = query_index[consistent]
        target_index = target_index[consistent]
        matched_reference = reference[batch_index, query_index]
        matched_targets = gt_boxes[target_index]

        source_log = student_corners[layer, batch_index, query_index].float().log_softmax(-1)
        teacher_log = teacher_corner_all[batch_index, query_index].log_softmax(-1)
        raw, representable = raw_fdr_targets(matched_reference, matched_targets)
        representable_edges += int(representable.sum())
        target_indices, weight_right, weight_left = translate_gt(raw.reshape(-1))
        target_indices = target_indices.reshape(-1, 4).long()
        weight_right = weight_right.reshape(-1, 4).float()
        weight_left = weight_left.reshape(-1, 4).float()
        source_error = interpolated_edge_nll(
            source_log, target_indices, weight_right, weight_left
        )
        teacher_error = interpolated_edge_nll(
            teacher_log, target_indices, weight_right, weight_left
        )
        loc_reliability = bounded_improvement(
            source_error,
            teacher_error,
            margin=options.loc_margin,
            tau=options.loc_tau,
        )
        loc_reliability = torch.where(representable, loc_reliability, torch.zeros_like(loc_reliability))
        loc_active = loc_reliability > 0
        keep, _iou_gain = decoded_teacher_iou_gate(
            source_log,
            teacher_log,
            matched_reference,
            matched_targets,
            loc_active,
            margin=0.0,
        )
        loc_reliability = loc_reliability * keep.unsqueeze(-1)
        loc_active = loc_reliability > 0
        active_edges += int(loc_active.sum())
        loc_advantages.append((source_error.detach() - teacher_error.detach()).clamp_min(0))
        loc_kl = teacher_log.exp() * (teacher_log - source_log)
        loc_kl = loc_kl.sum(dim=-1)
        active_localization_edges = int(loc_active.sum())
        if active_localization_edges:
            loc_active_layers += 1
            loc_terms.append((loc_reliability * loc_kl).sum() / active_localization_edges)

        source_class = student_classes[layer, batch_index, query_index]
        teacher_class = teacher_class_all[batch_index, query_index]
        one_hot = F.one_hot(gt_classes[target_index], num_classes=classes).float()
        source_bce = F.binary_cross_entropy_with_logits(
            source_class.float(), one_hot, reduction="none"
        ).mean(-1)
        teacher_bce = F.binary_cross_entropy_with_logits(
            teacher_class, one_hot, reduction="none"
        ).mean(-1)
        cls_reliability = bounded_improvement(
            source_bce,
            teacher_bce,
            margin=options.cls_margin,
            tau=options.cls_tau,
        )
        # Classification logits are consumed by VFL, whose positive target is
        # IoU-weighted.  Require the expert's decoded box to improve the same
        # matched query before its class confidence can be distilled; this
        # prevents an overconfident but geometrically inferior teacher from
        # pushing against the base classification objective.
        cls_quality_keep, _ = decoded_teacher_iou_gate(
            source_log,
            teacher_log,
            matched_reference,
            matched_targets,
            torch.ones_like(cls_reliability, dtype=torch.bool),
            margin=0.0,
        )
        cls_reliability = cls_reliability * cls_quality_keep.to(cls_reliability.dtype)
        cls_active = cls_reliability > 0
        active_classes += int(cls_active.sum())
        cls_advantages.append((source_bce.detach() - teacher_bce.detach()).clamp_min(0))
        active_classifications = int(cls_active.sum())
        if active_classifications:
            cls_active_layers += 1
            cls_terms.append(
                (cls_reliability * _bernoulli_kl(teacher_class, source_class)).sum()
                / active_classifications
            )

    loc_mean = torch.stack(loc_terms).mean() if loc_terms else graph_zero
    cls_mean = torch.stack(cls_terms).mean() if cls_terms else graph_zero
    enabled_scale = warmup if options.enabled else 0.0
    localization_loss = loc_mean * options.loc_weight * enabled_scale
    classification_loss = cls_mean * options.cls_weight * enabled_scale
    loss = localization_loss + classification_loss
    scalar_zero = graph_zero.detach()
    candidate_edges = candidate_matches * 4
    statistics = {
        "warmup": scalar_zero.new_tensor(enabled_scale),
        "identity_consistent_ratio": scalar_zero.new_tensor(
            consistent_matches / max(candidate_matches, 1)
        ),
        "representable_edge_ratio": scalar_zero.new_tensor(
            representable_edges / max(consistent_matches * 4, 1)
        ),
        "active_edge_ratio": scalar_zero.new_tensor(active_edges / max(candidate_edges, 1)),
        "active_class_ratio": scalar_zero.new_tensor(active_classes / max(candidate_matches, 1)),
        "mean_localization_advantage": (
            torch.cat([value.reshape(-1) for value in loc_advantages]).mean().detach()
            if loc_advantages
            else scalar_zero
        ),
        "mean_classification_advantage": (
            torch.cat([value.reshape(-1) for value in cls_advantages]).mean().detach()
            if cls_advantages
            else scalar_zero
        ),
        "candidate_source_matches": torch.tensor(candidate_matches, device=device, dtype=torch.long),
        "identity_consistent_matches": torch.tensor(consistent_matches, device=device, dtype=torch.long),
        "active_edges": torch.tensor(active_edges, device=device, dtype=torch.long),
        "active_classes": torch.tensor(active_classes, device=device, dtype=torch.long),
        "localization_active_layers": scalar_zero.new_tensor(loc_active_layers),
        "classification_active_layers": scalar_zero.new_tensor(cls_active_layers),
        "localization_loss": localization_loss.detach(),
        "classification_loss": classification_loss.detach(),
    }
    return CapacityBPDDResult(loss, localization_loss, classification_loss, statistics)


__all__ = [
    "CapacityBPDDOptions",
    "CapacityBPDDResult",
    "bounded_improvement",
    "capacity_bpdd_warmup",
    "quality_gated_capacity_distillation",
    "raw_fdr_targets",
]
