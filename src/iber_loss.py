"""Isolated boundary-evidence losses for IBER-BE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F

from src.itber_geometry import (
    apply_edge_update,
    correction_targets,
    xyxy_to_cxcywh,
)
from src.iber_protocol import BOUNDARY_LOSS_CONTRACT


class IBERLossInput(Protocol):
    stock_edges: torch.Tensor
    refined_edges: torch.Tensor
    boundary_off_edges: torch.Tensor
    gate_logits: torch.Tensor
    gates: torch.Tensor
    residual_raw: torch.Tensor
    residuals: torch.Tensor
    quality: torch.Tensor
    base_gate_raw: torch.Tensor
    boundary_gate_raw: torch.Tensor
    base_residual_raw: torch.Tensor
    boundary_residual_raw: torch.Tensor
    boundary_aux_gate_raw: torch.Tensor
    boundary_aux_residual_raw: torch.Tensor


@dataclass(frozen=True)
class IBERBucketCounts:
    """Fixed cache-global valid-edge counts for the two boundary objectives."""

    direction: tuple[int, int, int]
    margin: tuple[int, int, int]

    def __post_init__(self) -> None:
        for name, values in (("direction", self.direction), ("margin", self.margin)):
            if len(values) != 3 or any(type(value) is not int or value < 0 for value in values):
                raise ValueError(f"{name} bucket counts must be three non-negative integers")

    def __add__(self, other: object) -> "IBERBucketCounts":
        if not isinstance(other, IBERBucketCounts):
            return NotImplemented
        return IBERBucketCounts(
            direction=tuple(a + b for a, b in zip(self.direction, other.direction)),
            margin=tuple(a + b for a, b in zip(self.margin, other.margin)),
        )

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "direction": list(self.direction),
            "margin": list(self.margin),
        }


@dataclass(frozen=True)
class IBERLosses:
    """Named private losses and matched-query diagnostics."""

    box_l1: torch.Tensor
    box_giou: torch.Tensor
    box: torch.Tensor
    direction: torch.Tensor
    gate_positive: torch.Tensor
    gate_negative: torch.Tensor
    gate: torch.Tensor
    noop: torch.Tensor
    boundary_direction: torch.Tensor
    boundary_margin: torch.Tensor
    total: torch.Tensor
    matched_queries: int
    unmatched_queries: int


def _paired_generalized_iou(
    first: torch.Tensor, second: torch.Tensor, eps: float
) -> torch.Tensor:
    first_lower, first_upper = first.split(2, dim=-1)
    second_lower, second_upper = second.split(2, dim=-1)
    intersection_size = (
        torch.minimum(first_upper, second_upper)
        - torch.maximum(first_lower, second_lower)
    ).clamp_min(0)
    intersection = intersection_size.prod(dim=-1)
    first_area = (first_upper - first_lower).clamp_min(0).prod(dim=-1)
    second_area = (second_upper - second_lower).clamp_min(0).prod(dim=-1)
    union = first_area + second_area - intersection
    iou = intersection / union.clamp_min(eps)
    enclosing_size = (
        torch.maximum(first_upper, second_upper)
        - torch.minimum(first_lower, second_lower)
    ).clamp_min(0)
    enclosing = enclosing_size.prod(dim=-1)
    return iou - (enclosing - union) / enclosing.clamp_min(eps)


def _edge_area_buckets(target_edges: torch.Tensor, image_size: int) -> torch.Tensor:
    if target_edges.ndim != 2 or target_edges.shape[-1] != 4:
        raise ValueError("target edges must have shape [targets, 4]")
    if image_size < 1:
        raise ValueError("image_size must be positive")
    width_height = (target_edges[:, 2:] - target_edges[:, :2]).clamp_min(0)
    area_pixels = width_height.prod(dim=-1) * float(image_size**2)
    buckets = torch.where(
        area_pixels < float(16**2),
        torch.zeros_like(area_pixels, dtype=torch.long),
        torch.where(
            area_pixels < float(32**2),
            torch.ones_like(area_pixels, dtype=torch.long),
            torch.full_like(area_pixels, 2, dtype=torch.long),
        ),
    )
    return buckets.unsqueeze(-1).expand(-1, 4)


def _bucket_balanced_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
    buckets: torch.Tensor,
    *,
    global_bucket_counts: tuple[int, int, int],
    batches_per_epoch: int,
) -> torch.Tensor:
    if values.shape != valid.shape or values.shape != buckets.shape:
        raise ValueError("values, valid, and buckets must have identical shapes")
    if (
        len(global_bucket_counts) != 3
        or any(type(value) is not int or value < 0 for value in global_bucket_counts)
    ):
        raise ValueError("global bucket counts must be three non-negative integers")
    if type(batches_per_epoch) is not int or batches_per_epoch < 1:
        raise ValueError("batches_per_epoch must be a positive integer")
    graph_zero = values.sum() * 0.0
    active_buckets = sum(count > 0 for count in global_bucket_counts)
    if active_buckets == 0:
        if bool(valid.any()):
            raise ValueError("valid edges exist but global bucket counts are zero")
        return graph_zero
    total = graph_zero
    for bucket in range(3):
        selected = valid & buckets.eq(bucket)
        if bool(selected.any()):
            count = global_bucket_counts[bucket]
            if count <= 0 or int(selected.sum()) > count:
                raise ValueError("batch valid edges exceed the fixed global bucket count")
            total = total + values[selected].sum() / float(count)
    return total * (float(batches_per_epoch) / float(active_buckets))


def balanced_boundary_direction_loss(
    predicted_correction: torch.Tensor,
    normalized_target: torch.Tensor,
    target_edges: torch.Tensor,
    *,
    image_size: int,
    global_bucket_counts: tuple[int, int, int],
    batches_per_epoch: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Supervise correction signs with equal weight per object-size bucket."""
    if predicted_correction.shape != normalized_target.shape:
        raise ValueError("predicted and target corrections must have equal shapes")
    if predicted_correction.ndim != 2 or predicted_correction.shape[-1] != 4:
        raise ValueError("corrections must have shape [matched, 4]")
    if target_edges.shape != predicted_correction.shape:
        raise ValueError("target edges must match correction shape")
    target = normalized_target.detach().to(
        device=predicted_correction.device, dtype=predicted_correction.dtype
    )
    buckets = _edge_area_buckets(
        target_edges.detach().to(device=predicted_correction.device), image_size
    )
    valid = target.abs() > eps
    direction_margin = float(BOUNDARY_LOSS_CONTRACT["direction_margin"])
    values = F.relu(direction_margin - target.sign() * predicted_correction)
    return _bucket_balanced_mean(
        values,
        valid,
        buckets,
        global_bucket_counts=global_bucket_counts,
        batches_per_epoch=batches_per_epoch,
    )


def boundary_edge_margin_loss(
    full_edges: torch.Tensor,
    boundary_off_edges: torch.Tensor,
    stock_edges: torch.Tensor,
    target_edges: torch.Tensor,
    *,
    image_size: int,
    global_bucket_counts: tuple[int, int, int],
    batches_per_epoch: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Require evidence to improve upon the detached area-only edge error."""
    if (
        full_edges.shape != boundary_off_edges.shape
        or full_edges.shape != stock_edges.shape
        or full_edges.shape != target_edges.shape
    ):
        raise ValueError("full, boundary-off, stock, and target edges must have equal shapes")
    if full_edges.ndim != 2 or full_edges.shape[-1] != 4:
        raise ValueError("edge tensors must have shape [matched, 4]")
    target = target_edges.detach().to(device=full_edges.device, dtype=full_edges.dtype)
    boundary_reference = boundary_off_edges.detach().to(
        device=full_edges.device, dtype=full_edges.dtype
    )
    stock_reference = stock_edges.detach().to(
        device=full_edges.device, dtype=full_edges.dtype
    )
    stock_error = (stock_reference - target).abs()
    boundary_error = (boundary_reference - target).abs()
    reference_error = torch.minimum(stock_error, boundary_error)
    full_error = (full_edges - target).abs()
    pixel_floor = float(BOUNDARY_LOSS_CONTRACT["reference_floor_pixels"]) / float(
        image_size
    )
    valid = stock_error >= pixel_floor
    relative_error = full_error / reference_error.clamp_min(pixel_floor)
    relative_margin = float(BOUNDARY_LOSS_CONTRACT["edge_relative_margin"])
    values = F.relu(relative_error - (1.0 - relative_margin))
    buckets = _edge_area_buckets(target, image_size)
    return _bucket_balanced_mean(
        values,
        valid,
        buckets,
        global_bucket_counts=global_bucket_counts,
        batches_per_epoch=batches_per_epoch,
    )


def boundary_bucket_counts(
    stock_edges: torch.Tensor,
    target_edges: torch.Tensor,
    *,
    rho: float,
    image_size: int,
    eps: float = 1e-6,
) -> IBERBucketCounts:
    """Count fixed valid edges per bucket from immutable stock assignments."""
    if stock_edges.shape != target_edges.shape:
        raise ValueError("stock and target edges must have equal shapes")
    if stock_edges.ndim != 2 or stock_edges.shape[-1] != 4:
        raise ValueError("stock and target edges must have shape [matched, 4]")
    _, _, normalized = correction_targets(
        stock_edges.detach(), target_edges.detach(), rho=rho, eps=eps
    )
    buckets = _edge_area_buckets(target_edges.detach(), image_size)
    direction_valid = normalized.abs() > eps
    pixel_floor = float(BOUNDARY_LOSS_CONTRACT["reference_floor_pixels"]) / float(
        image_size
    )
    margin_valid = (stock_edges.detach() - target_edges.detach()).abs() >= pixel_floor

    def counts(mask: torch.Tensor) -> tuple[int, int, int]:
        return tuple(int((mask & buckets.eq(bucket)).sum()) for bucket in range(3))

    return IBERBucketCounts(
        direction=counts(direction_valid),
        margin=counts(margin_valid),
    )


def _validate_loss_inputs(
    output: IBERLossInput,
    target_edges: torch.Tensor,
    match_indices: list[tuple[torch.Tensor, torch.Tensor]],
    rho: float,
    image_size: int,
) -> tuple[int, int]:
    if rho <= 0:
        raise ValueError("rho must be positive")
    if image_size < 1:
        raise ValueError("image_size must be positive")
    if output.stock_edges.ndim != 3 or output.stock_edges.shape[-1] != 4:
        raise ValueError("stock edges must have shape [batch, queries, 4]")
    batch, queries = output.stock_edges.shape[:2]
    expected_edges = (batch, queries, 4)
    names = (
        "refined_edges",
        "boundary_off_edges",
        "gate_logits",
        "gates",
        "residual_raw",
        "residuals",
        "base_gate_raw",
        "boundary_gate_raw",
        "base_residual_raw",
        "boundary_residual_raw",
        "boundary_aux_gate_raw",
        "boundary_aux_residual_raw",
    )
    for name in names:
        if getattr(output, name).shape != expected_edges:
            raise ValueError(f"{name} must have shape {expected_edges}")
    if output.quality.shape not in {(batch, queries), (batch, queries, 1)}:
        raise ValueError("quality must have shape [batch, queries] or [batch, queries, 1]")
    if target_edges.ndim != 2 or target_edges.shape[-1] != 4:
        raise ValueError("target edges must have shape [targets, 4]")
    if len(match_indices) != batch:
        raise ValueError("match index batch count does not match IBER output batch")
    target_count = target_edges.shape[0]
    for source, destination in match_indices:
        if source.ndim != 1 or destination.ndim != 1 or len(source) != len(destination):
            raise ValueError("each match must contain equal-length one-dimensional indices")
        if len(source) and (int(source.min()) < 0 or int(source.max()) >= queries):
            raise IndexError("matched query index is out of range")
        if len(destination) and (
            int(destination.min()) < 0 or int(destination.max()) >= target_count
        ):
            raise IndexError("matched target index is out of range")
    return batch, queries


def iber_private_loss(
    output: IBERLossInput,
    *,
    target_edges: torch.Tensor,
    match_indices: list[tuple[torch.Tensor, torch.Tensor]],
    rho: float,
    image_size: int,
    boundary_supervision: bool,
    bucket_counts: IBERBucketCounts,
    batches_per_epoch: int,
    eps: float = 1e-6,
) -> IBERLosses:
    """Compute isolated losses from detached stock assignments and targets."""
    if type(boundary_supervision) is not bool:
        raise TypeError("boundary_supervision must be bool")
    if not isinstance(bucket_counts, IBERBucketCounts):
        raise TypeError("bucket_counts must be IBERBucketCounts")
    if type(batches_per_epoch) is not int or batches_per_epoch < 1:
        raise ValueError("batches_per_epoch must be a positive integer")
    batch, queries = _validate_loss_inputs(
        output, target_edges, match_indices, rho, image_size
    )
    device = output.gate_logits.device
    with torch.autocast(device_type=device.type, enabled=False):
        stock = output.stock_edges.detach().to(device=device, dtype=torch.float32)
        refined = output.refined_edges.to(dtype=torch.float32)
        targets = target_edges.detach().to(device=device, dtype=torch.float32)
        gate_logits = output.gate_logits.to(dtype=torch.float32)
        gates = output.gates.to(dtype=torch.float32)
        residual_raw = output.residual_raw.to(dtype=torch.float32)
        residuals = output.residuals.to(dtype=torch.float32)
        base_gate_raw = output.base_gate_raw.detach().to(dtype=torch.float32)
        boundary_gate_raw = output.boundary_aux_gate_raw.to(dtype=torch.float32)
        base_residual_raw = output.base_residual_raw.detach().to(dtype=torch.float32)
        boundary_residual_raw = output.boundary_aux_residual_raw.to(dtype=torch.float32)
        boundary_off_edges = output.boundary_off_edges.detach().to(dtype=torch.float32)
        quality = output.quality.detach().to(device=device, dtype=torch.float32)
        if quality.ndim == 3:
            quality = quality.squeeze(-1)
        quality = quality.clamp(0, 1)

        graph_zero = (refined.sum() + gate_logits.sum() + residual_raw.sum()) * 0.0
        boundary_graph_zero = (
            boundary_gate_raw.sum() + boundary_residual_raw.sum()
        ) * 0.0
        matched_mask = torch.zeros((batch, queries), device=device, dtype=torch.bool)
        batch_parts: list[torch.Tensor] = []
        source_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
        for image_index, (source, destination) in enumerate(match_indices):
            source = source.to(device=device, dtype=torch.long)
            destination = destination.to(device=device, dtype=torch.long)
            if len(source):
                if bool(matched_mask[image_index, source].any()):
                    raise ValueError("a query cannot be matched more than once")
                matched_mask[image_index, source] = True
                batch_parts.append(torch.full_like(source, image_index))
                source_parts.append(source)
                target_parts.append(destination)

        if source_parts:
            batch_index = torch.cat(batch_parts)
            source_index = torch.cat(source_parts)
            target_index = torch.cat(target_parts)
            matched_stock = stock[batch_index, source_index]
            matched_refined = refined[batch_index, source_index]
            matched_targets = targets[target_index]

            box_l1 = F.l1_loss(
                xyxy_to_cxcywh(matched_refined),
                xyxy_to_cxcywh(matched_targets),
                reduction="sum",
            ) / len(source_index)
            box_giou = (
                1.0 - _paired_generalized_iou(matched_refined, matched_targets, eps)
            ).sum() / len(source_index)
            magnitude, direction_target, normalized_target = correction_targets(
                matched_stock, matched_targets, rho=rho, eps=eps
            )
            direction_error = F.smooth_l1_loss(
                residuals[batch_index, source_index],
                direction_target,
                reduction="none",
                beta=1.0,
            )
            direction = (direction_error * magnitude).sum() / magnitude.sum().clamp_min(eps)
            gate_positive = F.binary_cross_entropy_with_logits(
                gate_logits[batch_index, source_index], magnitude, reduction="mean"
            )

            if boundary_supervision:
                boundary_gates = (
                    base_gate_raw[batch_index, source_index]
                    + boundary_gate_raw[batch_index, source_index]
                ).sigmoid()
                boundary_residuals = (
                    base_residual_raw[batch_index, source_index]
                    + boundary_residual_raw[batch_index, source_index]
                ).tanh()
                boundary_correction = boundary_gates * boundary_residuals
                boundary_full_edges = apply_edge_update(
                    matched_stock,
                    boundary_gates,
                    boundary_residuals,
                    rho=rho,
                    eps=eps,
                )
                boundary_direction = balanced_boundary_direction_loss(
                    boundary_correction,
                    normalized_target,
                    matched_targets,
                    image_size=image_size,
                    global_bucket_counts=bucket_counts.direction,
                    batches_per_epoch=batches_per_epoch,
                    eps=eps,
                )
                boundary_margin = boundary_edge_margin_loss(
                    boundary_full_edges,
                    boundary_off_edges[batch_index, source_index],
                    matched_stock,
                    matched_targets,
                    image_size=image_size,
                    global_bucket_counts=bucket_counts.margin,
                    batches_per_epoch=batches_per_epoch,
                    eps=eps,
                )
            else:
                boundary_direction = boundary_graph_zero
                boundary_margin = boundary_graph_zero
        else:
            box_l1 = graph_zero
            box_giou = graph_zero
            direction = graph_zero
            gate_positive = graph_zero
            boundary_direction = boundary_graph_zero
            boundary_margin = boundary_graph_zero

        unmatched_mask = ~matched_mask
        if bool(unmatched_mask.any()):
            unmatched_logits = gate_logits[unmatched_mask]
            gate_negative = F.binary_cross_entropy_with_logits(
                unmatched_logits, torch.zeros_like(unmatched_logits), reduction="mean"
            )
            effective = (gates[unmatched_mask] * residuals[unmatched_mask]).abs()
            noop = (effective * quality[unmatched_mask].unsqueeze(-1)).mean()
        else:
            gate_negative = graph_zero
            noop = graph_zero

        box = box_l1 + box_giou
        gate = 0.5 * gate_positive + 0.5 * gate_negative
        total = (
            box
            + direction
            + 0.25 * gate
            + 0.05 * noop
            + float(BOUNDARY_LOSS_CONTRACT["direction_weight"]) * boundary_direction
            + float(BOUNDARY_LOSS_CONTRACT["edge_margin_weight"]) * boundary_margin
        )
        named = {
            "box_l1": box_l1,
            "box_giou": box_giou,
            "box": box,
            "direction": direction,
            "gate_positive": gate_positive,
            "gate_negative": gate_negative,
            "gate": gate,
            "noop": noop,
            "boundary_direction": boundary_direction,
            "boundary_margin": boundary_margin,
            "total": total,
        }
        invalid = [name for name, value in named.items() if not bool(torch.isfinite(value))]
        if invalid:
            raise FloatingPointError("NONFINITE_IBER_LOSS: " + ", ".join(invalid))
        matched_queries = int(matched_mask.sum())
        return IBERLosses(
            **named,
            matched_queries=matched_queries,
            unmatched_queries=batch * queries - matched_queries,
        )


__all__ = [
    "IBERBucketCounts",
    "IBERLosses",
    "balanced_boundary_direction_loss",
    "boundary_bucket_counts",
    "boundary_edge_margin_loss",
    "iber_private_loss",
]
