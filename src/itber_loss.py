"""Isolated supervised losses for the private I-TBER v1.1 head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F

from src.itber_geometry import correction_targets, xyxy_to_cxcywh


class ITBERLossInput(Protocol):
    stock_edges: torch.Tensor
    refined_edges: torch.Tensor
    gate_logits: torch.Tensor
    gates: torch.Tensor
    residual_raw: torch.Tensor
    residuals: torch.Tensor
    quality: torch.Tensor


@dataclass(frozen=True)
class ITBERLosses:
    """Named private loss terms and matched-query diagnostics."""

    box_l1: torch.Tensor
    box_giou: torch.Tensor
    box: torch.Tensor
    direction: torch.Tensor
    gate_positive: torch.Tensor
    gate_negative: torch.Tensor
    gate: torch.Tensor
    noop: torch.Tensor
    total: torch.Tensor
    matched_queries: int
    unmatched_queries: int


def _paired_generalized_iou(first: torch.Tensor, second: torch.Tensor, eps: float) -> torch.Tensor:
    """Return aligned GIoU values for normalized xyxy edge tensors."""
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


def _validate_loss_inputs(
    output: ITBERLossInput,
    target_edges: torch.Tensor,
    match_indices: list[tuple[torch.Tensor, torch.Tensor]],
    rho: float,
) -> tuple[int, int]:
    if rho <= 0:
        raise ValueError("rho must be positive")
    if output.stock_edges.ndim != 3 or output.stock_edges.shape[-1] != 4:
        raise ValueError("stock edges must have shape [batch, queries, 4]")
    batch, queries = output.stock_edges.shape[:2]
    expected_edges = (batch, queries, 4)
    for name in ("refined_edges", "gate_logits", "gates", "residual_raw", "residuals"):
        value = getattr(output, name)
        if value.shape != expected_edges:
            raise ValueError(f"{name} must have shape {expected_edges}")
    if output.quality.shape not in {(batch, queries), (batch, queries, 1)}:
        raise ValueError("quality must have shape [batch, queries] or [batch, queries, 1]")
    if target_edges.ndim != 2 or target_edges.shape[-1] != 4:
        raise ValueError("target edges must have shape [targets, 4]")
    if len(match_indices) != batch:
        raise ValueError("match index batch count does not match I-TBER output batch")

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


def itber_private_loss(
    output: ITBERLossInput,
    *,
    target_edges: torch.Tensor,
    match_indices: list[tuple[torch.Tensor, torch.Tensor]],
    rho: float,
    eps: float = 1e-6,
) -> ITBERLosses:
    """Compute private losses using only the detached stock assignment.

    Positive and negative gate losses are normalized independently so the
    number of normal queries cannot drown out the matched supervision.
    """
    batch, queries = _validate_loss_inputs(output, target_edges, match_indices, rho)
    device = output.gate_logits.device
    with torch.autocast(device_type=device.type, enabled=False):
        stock = output.stock_edges.detach().to(device=device, dtype=torch.float32)
        refined = output.refined_edges.to(dtype=torch.float32)
        targets = target_edges.detach().to(device=device, dtype=torch.float32)
        gate_logits = output.gate_logits.to(dtype=torch.float32)
        gates = output.gates.to(dtype=torch.float32)
        residual_raw = output.residual_raw.to(dtype=torch.float32)
        residuals = output.residuals.to(dtype=torch.float32)
        quality = output.quality.detach().to(device=device, dtype=torch.float32)
        if quality.ndim == 3:
            quality = quality.squeeze(-1)
        quality = quality.clamp(0, 1)

        graph_zero = (
            refined.sum() + gate_logits.sum() + residual_raw.sum()
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
                    raise ValueError("a normal query cannot be matched more than once")
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

            magnitude, direction_target, _ = correction_targets(
                matched_stock,
                matched_targets,
                rho=rho,
                eps=eps,
            )
            direction_error = F.smooth_l1_loss(
                residuals[batch_index, source_index],
                direction_target,
                reduction="none",
                beta=1.0,
            )
            direction = (direction_error * magnitude).sum() / magnitude.sum().clamp_min(eps)
            gate_positive = F.binary_cross_entropy_with_logits(
                gate_logits[batch_index, source_index],
                magnitude,
                reduction="mean",
            )
        else:
            box_l1 = graph_zero
            box_giou = graph_zero
            direction = graph_zero
            gate_positive = graph_zero

        unmatched_mask = ~matched_mask
        if bool(unmatched_mask.any()):
            unmatched_logits = gate_logits[unmatched_mask]
            gate_negative = F.binary_cross_entropy_with_logits(
                unmatched_logits,
                torch.zeros_like(unmatched_logits),
                reduction="mean",
            )
            effective = (gates[unmatched_mask] * residuals[unmatched_mask]).abs()
            noop = (effective * quality[unmatched_mask].unsqueeze(-1)).mean()
        else:
            gate_negative = graph_zero
            noop = graph_zero

        box = box_l1 + box_giou
        gate = 0.5 * gate_positive + 0.5 * gate_negative
        total = box + direction + 0.25 * gate + 0.05 * noop
        named = {
            "box_l1": box_l1,
            "box_giou": box_giou,
            "box": box,
            "direction": direction,
            "gate_positive": gate_positive,
            "gate_negative": gate_negative,
            "gate": gate,
            "noop": noop,
            "total": total,
        }
        invalid = [name for name, value in named.items() if not bool(torch.isfinite(value))]
        if invalid:
            raise FloatingPointError(f"NONFINITE_ITBER_LOSS: {', '.join(invalid)}")

        matched_queries = int(matched_mask.sum())
        return ITBERLosses(
            **named,
            matched_queries=matched_queries,
            unmatched_queries=batch * queries - matched_queries,
        )
