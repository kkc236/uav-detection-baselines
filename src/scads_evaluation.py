"""Pure SCADS representation summaries and frozen screen-gate decisions."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from src.scads import smallest_covering_support, translate_with_project


BUCKET_NAMES = ("tiny", "small", "other")
METRIC_NAMES = ("map", "ap50", "ap75", "ap_tiny", "ap_small", "precision", "recall")


def area_buckets(boxes: Tensor, *, image_size: int = 640) -> Tensor:
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("target boxes must have shape [N,4]")
    area = boxes[:, 2].clamp_min(0) * boxes[:, 3].clamp_min(0) * image_size**2
    return torch.where(
        area < 16**2,
        torch.zeros_like(area, dtype=torch.long),
        torch.where(
            area < 32**2,
            torch.ones_like(area, dtype=torch.long),
            torch.full_like(area, 2, dtype=torch.long),
        ),
    )


def _expanded_projects(projects: Tensor, rows: int) -> Tensor:
    if projects.ndim == 1:
        projects = projects.unsqueeze(0).expand(rows, -1)
    if projects.ndim != 2 or projects.shape[0] != rows or projects.shape[1] < 2:
        raise ValueError("projects must have shape [bins] or [N,bins]")
    if not torch.isfinite(projects).all():
        raise ValueError("projects contain non-finite values")
    return projects


def saturation_mask(offsets: Tensor, projects: Tensor) -> Tensor:
    if offsets.ndim != 2 or offsets.shape[-1] != 4:
        raise ValueError("offsets must have shape [N,4]")
    expanded = _expanded_projects(projects, offsets.shape[0]).to(offsets)
    return (offsets < expanded[:, :1]) | (offsets > expanded[:, -1:])


def reconstruction_summary(offsets: Tensor, projects: Tensor) -> dict[str, float | int]:
    expanded = _expanded_projects(projects, offsets.shape[0]).to(offsets)
    indices, right, left = translate_with_project(
        offsets, expanded, reg_max=expanded.shape[1] - 1
    )
    indices = indices.reshape_as(offsets)
    right = right.reshape_as(offsets)
    left = left.reshape_as(offsets)
    left_index = indices.floor().long().clamp(0, expanded.shape[1] - 2)
    rows = torch.arange(offsets.shape[0], device=offsets.device)[:, None].expand_as(left_index)
    left_value = expanded[rows, left_index]
    right_value = expanded[rows, left_index + 1]
    decoded = left_value * left + right_value * right
    unsaturated = ~saturation_mask(offsets, expanded)
    error = (decoded - offsets).abs()[unsaturated]
    return {
        "unsaturated_edges": int(error.numel()),
        "l1_mean": float(error.mean().item()) if error.numel() else 0.0,
        "max_error": float(error.max().item()) if error.numel() else 0.0,
    }


def _rate(mask: Tensor) -> float:
    return float(mask.float().mean().item()) if mask.numel() else 0.0


def saturation_summary(
    offsets: Tensor,
    projects: Tensor,
    buckets: Tensor,
) -> dict[str, Any]:
    saturated = saturation_mask(offsets, projects)

    def subset(mask: Tensor) -> dict[str, Any]:
        values = saturated[mask]
        return {
            "objects": int(values.shape[0]),
            "edge_saturation_rate": _rate(values),
            "object_saturation_rate": _rate(values.any(dim=-1)),
            "saturated_edges": int(values.sum().item()),
            "total_edges": int(values.numel()),
        }

    return {
        "overall": subset(torch.ones(offsets.shape[0], dtype=torch.bool, device=offsets.device)),
        "by_scale": {
            name: subset(buckets == index) for index, name in enumerate(BUCKET_NAMES)
        },
        "reconstruction": reconstruction_summary(offsets, projects),
    }


def _shares(indices: Tensor, classes: int) -> list[float]:
    counts = torch.bincount(indices.long().cpu(), minlength=classes).float()
    total = counts.sum().item()
    return (counts / total).tolist() if total else [0.0] * classes


def _balanced_accuracy(predicted: Tensor, target: Tensor, classes: int) -> float:
    recalls = []
    for index in range(classes):
        mask = target == index
        if mask.any():
            recalls.append(_rate(predicted[mask] == index))
    return statistics.fmean(recalls) if recalls else 0.0


def summarize_representation(
    *,
    fdr_offsets: Tensor,
    scads_offsets: Tensor,
    target_boxes: Tensor,
    base_project: Tensor,
    scads_projects: Tensor,
    route_weights: Tensor,
    project_bank: Tensor,
    support_ups: Sequence[float],
    margin_ratio: float,
) -> dict[str, Any]:
    rows = target_boxes.shape[0]
    expected = {
        "fdr_offsets": (rows, 4),
        "scads_offsets": (rows, 4),
        "scads_projects": (rows, project_bank.shape[1]),
        "route_weights": (rows, project_bank.shape[0]),
    }
    actual = {
        "fdr_offsets": tuple(fdr_offsets.shape),
        "scads_offsets": tuple(scads_offsets.shape),
        "scads_projects": tuple(scads_projects.shape),
        "route_weights": tuple(route_weights.shape),
    }
    if actual != expected:
        raise ValueError(f"representation tensor shapes differ: expected={expected}, actual={actual}")
    if rows == 0 or not all(torch.isfinite(value).all() for value in (
        fdr_offsets,
        scads_offsets,
        target_boxes,
        scads_projects,
        route_weights,
    )):
        raise ValueError("representation tensors are empty or non-finite")
    buckets = area_buckets(target_boxes)
    base = saturation_summary(fdr_offsets, base_project, buckets)
    adaptive = saturation_summary(scads_offsets, scads_projects, buckets)
    counterfactual_base = saturation_summary(scads_offsets, base_project, buckets)
    oracle_target, overflow = smallest_covering_support(
        scads_offsets,
        project_bank,
        margin_ratio=margin_ratio,
    )
    oracle_projects = project_bank.to(scads_offsets)[oracle_target]
    oracle = saturation_summary(scads_offsets, oracle_projects, buckets)
    predicted = route_weights.argmax(dim=-1)
    entropy = -(route_weights.clamp_min(1e-12) * route_weights.clamp_min(1e-12).log()).sum(-1)
    ups = torch.tensor(support_ups, device=route_weights.device, dtype=route_weights.dtype)
    effective_up = route_weights @ ups
    usage_by_scale = {}
    for index, name in enumerate(BUCKET_NAMES):
        mask = buckets == index
        usage_by_scale[name] = {
            "objects": int(mask.sum().item()),
            "predicted_hard_share": _shares(predicted[mask], len(support_ups)),
            "oracle_share": _shares(oracle_target[mask], len(support_ups)),
            "mean_effective_up": float(effective_up[mask].mean().item()) if mask.any() else None,
            "route_accuracy": _rate(predicted[mask] == oracle_target[mask]),
        }
    fdr_tiny = float(base["by_scale"]["tiny"]["edge_saturation_rate"])
    scads_tiny = float(adaptive["by_scale"]["tiny"]["edge_saturation_rate"])
    relative_reduction = (fdr_tiny - scads_tiny) / fdr_tiny if fdr_tiny > 0 else 0.0
    scale_means = [
        value["mean_effective_up"]
        for value in usage_by_scale.values()
        if value["mean_effective_up"] is not None
    ]
    predicted_shares = _shares(predicted, len(support_ups))
    oracle_shares = _shares(oracle_target, len(support_ups))
    return {
        "matched_objects": rows,
        "fdr_fixed_base": base,
        "scads_adaptive": adaptive,
        "scads_counterfactual_fixed_base": counterfactual_base,
        "scads_oracle": oracle,
        "tiny_saturation_relative_reduction": relative_reduction,
        "route": {
            "predicted_hard_share": predicted_shares,
            "oracle_share": oracle_shares,
            "accuracy": _rate(predicted == oracle_target),
            "balanced_accuracy": _balanced_accuracy(
                predicted, oracle_target, len(support_ups)
            ),
            "entropy_mean": float(entropy.mean().item()),
            "wide_overflow_objects": int(overflow.sum().item()),
            "wide_overflow_rate": _rate(overflow),
            "active_predicted_routes_ge_5pct": sum(value >= 0.05 for value in predicted_shares),
            "active_oracle_routes_ge_5pct": sum(value >= 0.05 for value in oracle_shares),
            "effective_up_scale_range": max(scale_means) - min(scale_means) if scale_means else 0.0,
            "by_scale": usage_by_scale,
        },
    }


def metric_delta(fdr: Mapping[str, Any], scads: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    for name in METRIC_NAMES:
        left, right = float(fdr[name]), float(scads[name])
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError(f"non-finite metric: {name}")
        result[name] = right - left
    return result


def gate_decision(
    *,
    final_delta: Mapping[str, float],
    tail3_delta: Mapping[str, float],
    exact_delta: Mapping[str, float],
    representation: Mapping[str, Any],
    engineering_complete: bool,
) -> dict[str, Any]:
    route = representation["route"]
    reconstruction = representation["scads_adaptive"]["reconstruction"]
    checks = {
        "engineering_complete": bool(engineering_complete),
        "final_map_positive": float(final_delta["map"]) > 0.0,
        "tail3_map_positive": float(tail3_delta["map"]) > 0.0,
        "final_ap75_positive": float(exact_delta["ap75"]) > 0.0,
        "final_ap_tiny_positive": float(exact_delta["ap_tiny"]) > 0.0,
        "tiny_saturation_reduction_ge_50pct": float(
            representation["tiny_saturation_relative_reduction"]
        ) >= 0.50,
        "unsaturated_reconstruction_l1_le_1e_6": float(
            reconstruction["l1_mean"]
        ) <= 1e-6,
        "oracle_routes_not_degenerate": int(
            route["active_oracle_routes_ge_5pct"]
        ) >= 2,
        "predicted_routes_not_constant": (
            int(route["active_predicted_routes_ge_5pct"]) >= 2
            and float(route["effective_up_scale_range"]) > 0.01
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


__all__ = [
    "BUCKET_NAMES",
    "METRIC_NAMES",
    "area_buckets",
    "gate_decision",
    "metric_delta",
    "reconstruction_summary",
    "saturation_mask",
    "saturation_summary",
    "summarize_representation",
]
