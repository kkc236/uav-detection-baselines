"""Pure D0 oracle and candidate-pool math for the frozen OAR protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.oar_protocol import (
    OAR_GAIN_RECOVERY,
    OAR_K_GRID,
    OAR_NUM_CLASSES,
    OAR_NUM_QUERIES,
    OAR_PAIR_CAP,
)
from src.rtdetr_quality_oracle import same_class_iou_quality


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


class OARRanker(nn.Module):
    """Zero-initialized all-pair objective-aligned residual ranker."""

    def __init__(self, feature_dim: int = 276, width: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.network(features.detach()).squeeze(-1)
        return 2.0 * torch.tanh(raw / 2.0)


def apply_oar_r2(
    model: OARRanker,
    features: torch.Tensor,
    stock_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjust every detached stock Query-by-class logit with an OAR residual."""
    residual = model(features)
    if residual.shape != stock_logits.shape:
        raise ValueError("residual and stock_logits must have the same shape")
    adjusted_logits = stock_logits.detach() + residual
    return adjusted_logits.sigmoid(), residual


@dataclass(frozen=True)
class RankPairs:
    """Immutable teacher-oriented pair indices and detached utility-gap weights."""

    preferred: torch.Tensor
    other: torch.Tensor
    weight: torch.Tensor


def _require_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    return value


def _require_floating_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = _require_tensor(value, name)
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be a floating-point tensor")
    return tensor


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def teacher_utility(
    stock_logits: torch.Tensor,
    quality: torch.Tensor,
) -> torch.Tensor:
    """Return the detached objective-aligned teacher utility for every pair."""
    stock_logits = _require_floating_tensor(stock_logits, "stock_logits")
    quality = _require_floating_tensor(quality, "quality")
    if stock_logits.shape != quality.shape:
        raise ValueError("stock_logits and quality must have the same shape")
    if stock_logits.device != quality.device:
        raise ValueError("stock_logits and quality must share a device")
    _require_finite(stock_logits, "stock_logits")
    _require_finite(quality, "quality")
    return stock_logits.detach().sigmoid() * quality.detach().square()


def _oriented_non_tied_pairs(
    teacher: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Orient candidate pairs by teacher utility and remove exact ties."""
    first_utility = teacher[first]
    second_utility = teacher[second]
    non_tied = first_utility != second_utility
    first = first[non_tied]
    second = second[non_tied]
    first_preferred = first_utility[non_tied] > second_utility[non_tied]
    preferred = torch.where(first_preferred, first, second)
    other = torch.where(first_preferred, second, first)
    return preferred, other


def build_boundary_pairs(
    teacher: torch.Tensor,
    stock: torch.Tensor,
) -> RankPairs:
    """Build the frozen all-3,000-pair Top-300 boundary supervision."""
    teacher = _require_floating_tensor(teacher, "teacher")
    stock = _require_floating_tensor(stock, "stock")
    expected_pairs = OAR_NUM_QUERIES * OAR_NUM_CLASSES
    if teacher.ndim != 1 or teacher.numel() != expected_pairs:
        raise ValueError(f"teacher must have shape [{expected_pairs}]")
    if stock.shape != teacher.shape:
        raise ValueError("stock and teacher must have the same shape")
    if stock.device != teacher.device:
        raise ValueError("stock and teacher must share a device")
    _require_finite(teacher, "teacher")
    _require_finite(stock, "stock")

    teacher = teacher.detach()
    stock = stock.detach()
    teacher_order = torch.argsort(teacher, descending=True, stable=True)
    stock_order = torch.argsort(stock, descending=True, stable=True)
    teacher_top = teacher_order[:OAR_NUM_QUERIES]
    stock_top = stock_order[:OAR_NUM_QUERIES]

    teacher_top_mask = torch.zeros(expected_pairs, dtype=torch.bool, device=teacher.device)
    stock_top_mask = torch.zeros_like(teacher_top_mask)
    teacher_top_mask[teacher_top] = True
    stock_top_mask[stock_top] = True
    teacher_only = teacher_top[~stock_top_mask[teacher_top]]
    stock_only = stock_top[~teacher_top_mask[stock_top]]

    pair_groups: list[tuple[torch.Tensor, torch.Tensor]] = []
    if teacher_only.numel() and stock_only.numel():
        first = teacher_only.repeat_interleave(stock_only.numel())
        second = stock_only.repeat(teacher_only.numel())
        preferred, other = _oriented_non_tied_pairs(teacher, first, second)
        pair_groups.append((preferred[:2048], other[:2048]))

    adjacent_preferred, adjacent_other = _oriented_non_tied_pairs(
        teacher,
        teacher_order[: OAR_NUM_QUERIES - 1],
        teacher_order[1:OAR_NUM_QUERIES],
    )
    pair_groups.append((adjacent_preferred, adjacent_other))

    offset_preferred, offset_other = _oriented_non_tied_pairs(
        teacher,
        teacher_order[:OAR_NUM_QUERIES],
        teacher_order[OAR_NUM_QUERIES : 2 * OAR_NUM_QUERIES],
    )
    pair_groups.append((offset_preferred, offset_other))

    seen: set[tuple[int, int]] = set()
    unique_pairs: list[tuple[int, int]] = []
    for preferred_group, other_group in pair_groups:
        for preferred_index, other_index in zip(
            preferred_group.tolist(), other_group.tolist()
        ):
            pair = (preferred_index, other_index)
            if pair not in seen:
                seen.add(pair)
                unique_pairs.append(pair)

    if len(unique_pairs) > OAR_PAIR_CAP:
        raise RuntimeError("boundary pair construction exceeded the frozen cap")
    if unique_pairs:
        pair_tensor = torch.tensor(
            unique_pairs,
            dtype=torch.long,
            device=teacher.device,
        )
        preferred = pair_tensor[:, 0]
        other = pair_tensor[:, 1]
        weight = (teacher[preferred] - teacher[other]).detach()
    else:
        preferred = torch.empty(0, dtype=torch.long, device=teacher.device)
        other = torch.empty(0, dtype=torch.long, device=teacher.device)
        weight = teacher.new_empty(0)

    if weight.numel() and (
        not bool(torch.isfinite(weight).all()) or not bool((weight > 0).all())
    ):
        raise RuntimeError("boundary pair weights must be finite and positive")
    return RankPairs(preferred=preferred, other=other, weight=weight)


def boundary_rank_loss(
    adjusted_logits: torch.Tensor,
    pairs: RankPairs,
) -> torch.Tensor:
    """Compute teacher-gap-weighted RankNet loss over flattened adjusted logits."""
    adjusted_logits = _require_floating_tensor(adjusted_logits, "adjusted_logits")
    if not isinstance(pairs, RankPairs):
        raise TypeError("pairs must be RankPairs")
    _require_finite(adjusted_logits, "adjusted_logits")
    if pairs.preferred.dtype != torch.long or pairs.other.dtype != torch.long:
        raise TypeError("pair indices must have dtype torch.long")
    if not torch.is_floating_point(pairs.weight):
        raise TypeError("pair weights must be floating-point")
    if (
        pairs.preferred.ndim != 1
        or pairs.other.shape != pairs.preferred.shape
        or pairs.weight.shape != pairs.preferred.shape
    ):
        raise ValueError("pair tensors must be one-dimensional and have the same shape")
    if any(
        value.device != adjusted_logits.device
        for value in (pairs.preferred, pairs.other, pairs.weight)
    ):
        raise ValueError("adjusted_logits and pair tensors must share a device")
    if pairs.weight.numel() and (
        not bool(torch.isfinite(pairs.weight).all())
        or not bool((pairs.weight > 0).all())
    ):
        raise ValueError("pair weights must be finite and positive")

    flat = adjusted_logits.flatten()
    if pairs.preferred.numel() and (
        int(pairs.preferred.min()) < 0
        or int(pairs.other.min()) < 0
        or int(pairs.preferred.max()) >= flat.numel()
        or int(pairs.other.max()) >= flat.numel()
    ):
        raise IndexError("pair index is outside adjusted_logits")
    difference = flat[pairs.preferred] - flat[pairs.other]
    weight = pairs.weight.detach()
    element = F.softplus(-difference) * weight
    denominator = weight.sum().clamp_min(torch.finfo(element.dtype).eps)
    return element.sum() / denominator


def topk_per_class_mask(probabilities: torch.Tensor, k: int) -> torch.Tensor:
    """Select exactly the stock top-K query candidates independently per class."""
    probabilities = _require_floating_tensor(probabilities, "probabilities")
    if probabilities.ndim != 2 or 0 in probabilities.shape:
        raise ValueError("probabilities must have shape [Q, C] with nonzero dimensions")
    if type(k) is not int or k not in OAR_K_GRID:
        raise ValueError(f"k must be in OAR_K_GRID={OAR_K_GRID}")
    if probabilities.shape[0] < k:
        raise ValueError("probabilities must contain at least k queries")
    _require_finite(probabilities, "probabilities")

    indices = torch.argsort(
        probabilities, dim=0, descending=True, stable=True
    )[:k]
    mask = torch.zeros_like(probabilities, dtype=torch.bool)
    mask.scatter_(0, indices, True)
    return mask


def oracle_score_families(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    *,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    """Compute the frozen stock, presence, query-IoU, and same-class scores."""
    boxes = _require_floating_tensor(boxes, "boxes")
    logits = _require_floating_tensor(logits, "logits")
    target_boxes = _require_floating_tensor(target_boxes, "target_boxes")
    target_classes = _require_tensor(target_classes, "target_classes")

    if type(num_classes) is not int or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [Q, 4]")
    if logits.ndim != 2 or logits.shape != (boxes.shape[0], num_classes):
        raise ValueError("logits must have shape [Q, num_classes]")
    if target_boxes.ndim != 2 or target_boxes.shape[1] != 4:
        raise ValueError("target_boxes must have shape [N, 4]")
    if target_classes.ndim != 1:
        raise ValueError("target_classes must have shape [N]")
    if target_boxes.shape[0] != target_classes.shape[0]:
        raise ValueError(
            "target_boxes and target_classes must contain the same number of targets"
        )
    if target_classes.dtype not in _INTEGER_DTYPES:
        raise TypeError("target_classes must contain integer class indices")
    tensors = (boxes, logits, target_boxes, target_classes)
    if any(value.device != boxes.device for value in tensors[1:]):
        raise ValueError(
            "boxes, logits, target_boxes, and target_classes must share a device"
        )
    _require_finite(boxes, "boxes")
    _require_finite(logits, "logits")
    _require_finite(target_boxes, "target_boxes")

    detached_boxes = boxes.detach().float()
    detached_targets = target_boxes.detach().float()
    detached_classes = target_classes.detach().long()
    probabilities = logits.detach().float().sigmoid()
    same_class = same_class_iou_quality(
        detached_boxes,
        detached_targets,
        detached_classes,
        num_classes,
    )
    query_iou = same_class.amax(dim=1, keepdim=True)
    presence = torch.zeros(
        num_classes,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    if detached_classes.numel():
        presence[detached_classes.unique()] = 1

    return {
        "stock": probabilities,
        "presence": probabilities * presence,
        "query_iou": probabilities * query_iou.square(),
        "same_class": probabilities * same_class.square(),
    }


def restrict_oracle(
    stock: torch.Tensor,
    oracle: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply oracle scores only inside a pool, retaining exact stock scores outside."""
    stock = _require_floating_tensor(stock, "stock")
    oracle = _require_floating_tensor(oracle, "oracle")
    mask = _require_tensor(mask, "mask")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be a boolean tensor")
    if stock.ndim != 2:
        raise ValueError("stock, oracle, and mask must have shape [Q, C]")
    if oracle.shape != stock.shape or mask.shape != stock.shape:
        raise ValueError("stock, oracle, and mask must have the same shape")
    if oracle.dtype != stock.dtype:
        raise TypeError("stock and oracle must have the same dtype")
    if oracle.device != stock.device or mask.device != stock.device:
        raise ValueError("stock, oracle, and mask must share a device")
    _require_finite(stock, "stock")
    _require_finite(oracle, "oracle")
    return torch.where(mask, oracle, stock)


def _decimal_metric(value: Any, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not decimal_value.is_finite():
        raise ValueError(f"{name} must be finite")
    return decimal_value


def select_candidate_k(
    *,
    stock_map: Any,
    full_map: Any,
    restricted_map: Mapping[int, Any],
) -> dict[str, Any]:
    """Choose the smallest frozen K recovering at least 90% of positive full gain."""
    if not isinstance(restricted_map, Mapping):
        raise TypeError("restricted_map must be a mapping")
    keys = tuple(restricted_map)
    if (
        len(keys) != len(OAR_K_GRID)
        or any(type(k) is not int for k in keys)
        or set(keys) != set(OAR_K_GRID)
    ):
        raise ValueError(f"restricted_map must contain exactly OAR_K_GRID={OAR_K_GRID}")

    stock = _decimal_metric(stock_map, "stock_map")
    full = _decimal_metric(full_map, "full_map")
    restricted = {
        k: _decimal_metric(restricted_map[k], f"restricted_map[{k}]")
        for k in OAR_K_GRID
    }
    total_gain = full - stock
    if total_gain <= 0:
        return {"status": "scientific_failed", "selected_k": None}

    for k in OAR_K_GRID:
        recovered = (restricted[k] - stock) / total_gain
        if recovered >= OAR_GAIN_RECOVERY:
            return {
                "status": "passed",
                "selected_k": k,
                "recovered": str(recovered),
            }
    return {"status": "scientific_failed", "selected_k": None}
