"""Protected Frequency Candidate Rescue representation and pointwise gate."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.rtdetr_complementarity_oracle import (
    candidate_iou_matrix,
    one_to_one_same_class_assignment,
)
from src.rtdetr_quality_oracle import flattened_topk


NUM_QUERIES = 300
NUM_CLASSES = 10
PFCR_FEATURE_DIM = 35
_LOG_GUARD = 1e-6
RESCUE_SLOT_GRID = (0, 15, 30, 60)


def pfcr_split(image_id: str) -> str:
    """Assign an image basename to the deterministic four-to-one train/dev split."""

    if not isinstance(image_id, str):
        raise TypeError("image_id must be a string")
    normalized = Path(image_id.replace("\\", "/")).name
    value = int(hashlib.sha256(normalized.encode("utf-8")).hexdigest(), 16)
    return "dev" if value % 5 == 0 else "train"


def _validate_detector_inputs(
    fdr_boxes: object,
    fdr_logits: object,
    cm_boxes: object,
    cm_logits: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    named = (
        ("fdr_boxes", fdr_boxes, (NUM_QUERIES, 4)),
        ("fdr_logits", fdr_logits, (NUM_QUERIES, NUM_CLASSES)),
        ("cm_boxes", cm_boxes, (NUM_QUERIES, 4)),
        ("cm_logits", cm_logits, (NUM_QUERIES, NUM_CLASSES)),
    )
    for name, value, shape in named:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {list(shape)}")
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} must be a floating-point tensor")

    tensors = tuple(value for _, value, _ in named)
    if any(value.device != tensors[0].device for value in tensors[1:]):
        raise ValueError("detector boxes and logits must share one device")
    for (name, _, _), value in zip(named, tensors):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")

    compute_dtype = tensors[0].dtype
    for value in tensors[1:]:
        compute_dtype = torch.promote_types(compute_dtype, value.dtype)
    if compute_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    return tuple(value.detach().to(dtype=compute_dtype) for value in tensors)  # type: ignore[return-value]


def _pairwise_valid_box_iou(boxes: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    box_centers, box_sizes = boxes.split(2, dim=-1)
    other_centers, other_sizes = other.split(2, dim=-1)
    box_lower = box_centers - box_sizes / 2
    box_upper = box_centers + box_sizes / 2
    other_lower = other_centers - other_sizes / 2
    other_upper = other_centers + other_sizes / 2
    valid_boxes = (box_sizes > 0).all(dim=-1)
    valid_other = (other_sizes > 0).all(dim=-1)
    intersection = (
        torch.minimum(box_upper[:, None], other_upper[None])
        - torch.maximum(box_lower[:, None], other_lower[None])
    ).clamp_min(0).prod(dim=-1)
    union = (
        box_sizes.clamp_min(0).prod(dim=-1)[:, None]
        + other_sizes.clamp_min(0).prod(dim=-1)[None]
        - intersection
    )
    valid = valid_boxes[:, None] & valid_other[None] & (union > 0)
    return torch.where(
        valid,
        intersection / union.clamp_min(torch.finfo(union.dtype).tiny),
        torch.zeros_like(union),
    ).clamp(0, 1)


def _normalized_flattened_rank(probabilities: torch.Tensor) -> torch.Tensor:
    flat = probabilities.flatten()
    order = torch.argsort(flat, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(0, order, torch.arange(flat.numel(), device=flat.device))
    denominator = max(flat.numel() - 1, 1)
    return ranks.reshape_as(probabilities).to(dtype=probabilities.dtype) / denominator


def _query_class_statistics(logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    sorted_probabilities = torch.sort(
        probabilities, dim=-1, descending=True, stable=True
    ).values
    query_max = sorted_probabilities[:, 0]
    top_two_margin = sorted_probabilities[:, 0] - sorted_probabilities[:, 1]
    normalizer = probabilities.sum(dim=-1, keepdim=True).clamp_min(_LOG_GUARD)
    distribution = probabilities / normalizer
    entropy = -(
        distribution * distribution.clamp_min(_LOG_GUARD).log()
    ).sum(dim=-1) / math.log(NUM_CLASSES)
    rank = _normalized_flattened_rank(probabilities)
    expanded_query_max = query_max[:, None].expand(-1, NUM_CLASSES)
    expanded_margin = top_two_margin[:, None].expand(-1, NUM_CLASSES)
    expanded_entropy = entropy[:, None].expand(-1, NUM_CLASSES)
    return torch.stack(
        (
            logits,
            probabilities,
            expanded_query_max,
            expanded_margin,
            expanded_entropy,
            rank,
        ),
        dim=-1,
    )


def _expanded_box_geometry(boxes: torch.Tensor) -> torch.Tensor:
    width = boxes[:, 2]
    height = boxes[:, 3]
    safe_width = width.clamp_min(_LOG_GUARD)
    safe_height = height.clamp_min(_LOG_GUARD)
    geometry = torch.stack(
        (
            boxes[:, 0],
            boxes[:, 1],
            width,
            height,
            (safe_width * safe_height).log(),
            (safe_width / safe_height).log(),
        ),
        dim=-1,
    )
    return geometry[:, None, :].expand(-1, NUM_CLASSES, -1)


def pfcr_features(
    fdr_boxes: torch.Tensor,
    fdr_logits: torch.Tensor,
    cm_boxes: torch.Tensor,
    cm_logits: torch.Tensor,
) -> torch.Tensor:
    """Build one detached 35-value feature vector per FrequencyCM query/class."""

    fdr_boxes, fdr_logits, cm_boxes, cm_logits = _validate_detector_inputs(
        fdr_boxes, fdr_logits, cm_boxes, cm_logits
    )
    fdr_probabilities = fdr_logits.sigmoid()
    cm_probabilities = cm_logits.sigmoid()
    overlap = _pairwise_valid_box_iou(cm_boxes, fdr_boxes)
    match_quality = overlap[:, :, None] * fdr_probabilities[None, :, :]
    match_index = match_quality.argmax(dim=1)
    class_index = torch.arange(NUM_CLASSES, device=fdr_boxes.device)[None, :].expand(
        NUM_QUERIES, -1
    )

    cm_statistics = _query_class_statistics(cm_logits)
    all_fdr_statistics = _query_class_statistics(fdr_logits)
    matched_fdr_statistics = all_fdr_statistics[match_index, class_index]
    matched_boxes = fdr_boxes[match_index]
    matched_overlap = overlap.gather(1, match_index)

    cm_boxes_expanded = cm_boxes[:, None, :].expand(-1, NUM_CLASSES, -1)
    cm_width = cm_boxes_expanded[..., 2].clamp_min(_LOG_GUARD)
    cm_height = cm_boxes_expanded[..., 3].clamp_min(_LOG_GUARD)
    fdr_width = matched_boxes[..., 2].clamp_min(_LOG_GUARD)
    fdr_height = matched_boxes[..., 3].clamp_min(_LOG_GUARD)
    cross_model = torch.stack(
        (
            matched_overlap,
            cm_boxes_expanded[..., 0] - matched_boxes[..., 0],
            cm_boxes_expanded[..., 1] - matched_boxes[..., 1],
            (cm_width / fdr_width).log(),
            (cm_height / fdr_height).log(),
            cm_probabilities - matched_fdr_statistics[..., 1],
            cm_statistics[..., 2] - matched_fdr_statistics[..., 2],
        ),
        dim=-1,
    )
    one_hot_class = F.one_hot(class_index, num_classes=NUM_CLASSES).to(
        dtype=cm_logits.dtype
    )
    result = torch.cat(
        (
            cm_statistics,
            _expanded_box_geometry(cm_boxes),
            matched_fdr_statistics,
            cross_model,
            one_hot_class,
        ),
        dim=-1,
    )
    if result.shape != (NUM_QUERIES, NUM_CLASSES, PFCR_FEATURE_DIM):
        raise RuntimeError("PFCR feature schema drift")
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("PFCR features must contain only finite values")
    return result.contiguous().detach()


class PFCRGate(nn.Module):
    """Zero-initialized bounded residual gate over detached PFCR features."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(PFCR_FEATURE_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a tensor")
        if features.ndim == 0 or features.shape[-1] != PFCR_FEATURE_DIM:
            raise ValueError(f"features must have shape [..., {PFCR_FEATURE_DIM}]")
        if not torch.is_floating_point(features):
            raise TypeError("features must be a floating-point tensor")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features must contain only finite values")
        raw = self.network(features.detach()).squeeze(-1)
        return 2.0 * torch.tanh(raw / 2.0)


def _validate_single_detector(
    boxes: torch.Tensor, logits: torch.Tensor, *, label: str
) -> tuple[torch.Tensor, torch.Tensor]:
    checked = _validate_detector_inputs(boxes, logits, boxes, logits)
    return checked[0], checked[1]


def stock_predictions(boxes: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Return the exact Ultralytics flattened Top-300 detector predictions."""

    boxes, logits = _validate_single_detector(boxes, logits, label="stock")
    return flattened_topk(
        boxes.unsqueeze(0), logits.sigmoid().unsqueeze(0), NUM_CLASSES, NUM_QUERIES
    )[0].detach()


def _all_prediction_pairs(boxes: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    expanded_boxes = boxes.repeat_interleave(NUM_CLASSES, dim=0)
    classes = torch.arange(NUM_CLASSES, device=boxes.device).repeat(NUM_QUERIES)
    return torch.cat(
        (
            expanded_boxes,
            probabilities.reshape(-1, 1),
            classes.to(dtype=boxes.dtype).unsqueeze(1),
        ),
        dim=1,
    )


def protected_merge(
    fdr_boxes: torch.Tensor,
    fdr_logits: torch.Tensor,
    cm_boxes: torch.Tensor,
    cm_logits: torch.Tensor,
    *,
    rescue_slots: int,
) -> torch.Tensor:
    """Preserve the high-rank FDR prefix and contest only registered tail slots."""

    if type(rescue_slots) is not int or rescue_slots not in RESCUE_SLOT_GRID:
        raise ValueError(f"rescue_slots must be one of {RESCUE_SLOT_GRID}")
    fdr_boxes, fdr_logits, cm_boxes, cm_logits = _validate_detector_inputs(
        fdr_boxes, fdr_logits, cm_boxes, cm_logits
    )
    stock = stock_predictions(fdr_boxes, fdr_logits)
    if rescue_slots == 0:
        return stock.clone()
    protected_count = NUM_QUERIES - rescue_slots
    protected = stock[:protected_count]
    pool = torch.cat((stock[protected_count:], _all_prediction_pairs(cm_boxes, cm_logits)))
    order = torch.argsort(pool[:, 4], descending=True, stable=True)
    rescued = pool[order[:rescue_slots]]
    return torch.cat((protected, rescued), dim=0).contiguous().detach()


@dataclass(frozen=True)
class PFCRTeacher:
    """Detached one-to-one objective utility for both detector candidate sets."""

    fdr: torch.Tensor
    frequencycm: torch.Tensor


def one_to_one_union_teacher(
    fdr_boxes: torch.Tensor,
    fdr_logits: torch.Tensor,
    cm_boxes: torch.Tensor,
    cm_logits: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
) -> PFCRTeacher:
    """Assign at most one same-class union candidate to every target."""

    fdr_boxes, fdr_logits, cm_boxes, cm_logits = _validate_detector_inputs(
        fdr_boxes, fdr_logits, cm_boxes, cm_logits
    )
    if not isinstance(target_boxes, torch.Tensor) or target_boxes.ndim != 2 or target_boxes.shape[1] != 4:
        raise ValueError("target_boxes must have shape [N, 4]")
    if not torch.is_floating_point(target_boxes):
        raise TypeError("target_boxes must be a floating-point tensor")
    if not isinstance(target_classes, torch.Tensor) or target_classes.ndim != 1:
        raise ValueError("target_classes must have shape [N]")
    if target_classes.shape[0] != target_boxes.shape[0]:
        raise ValueError("target boxes and classes must have the same length")
    if target_classes.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("target_classes must be an integer tensor")
    if any(value.device != fdr_boxes.device for value in (target_boxes, target_classes)):
        raise ValueError("targets and detector evidence must share one device")
    if not bool(torch.isfinite(target_boxes).all()):
        raise ValueError("target_boxes must contain only finite values")
    if target_classes.numel() and not bool(
        ((target_classes >= 0) & (target_classes < NUM_CLASSES)).all()
    ):
        raise ValueError("target classes are outside the registered range")

    union_boxes = torch.cat((fdr_boxes, cm_boxes), dim=0)
    expanded_boxes = union_boxes.repeat_interleave(NUM_CLASSES, dim=0)
    expanded_classes = torch.arange(NUM_CLASSES, device=fdr_boxes.device).repeat(
        union_boxes.shape[0]
    )
    iou = candidate_iou_matrix(expanded_boxes, target_boxes.detach().to(fdr_boxes.dtype))
    assignment = one_to_one_same_class_assignment(
        iou, expanded_classes, target_classes.detach().long()
    )
    probabilities = torch.cat((fdr_logits.sigmoid(), cm_logits.sigmoid()), dim=0).reshape(-1)
    utility = torch.zeros_like(probabilities)
    if assignment.prediction_indices.numel():
        selected = assignment.prediction_indices
        utility[selected] = probabilities[selected] * assignment.ious.square()
    split = NUM_QUERIES * NUM_CLASSES
    return PFCRTeacher(
        fdr=utility[:split].reshape(NUM_QUERIES, NUM_CLASSES).detach(),
        frequencycm=utility[split:].reshape(NUM_QUERIES, NUM_CLASSES).detach(),
    )


def _require_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != (NUM_QUERIES, NUM_CLASSES):
        raise ValueError(f"{name} must have shape [{NUM_QUERIES}, {NUM_CLASSES}]")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _stable_unique_indices(values: torch.Tensor) -> torch.Tensor:
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise ValueError("values must have shape [N]")
    if values.dtype != torch.long:
        raise TypeError("values must have dtype torch.long")
    seen: set[int] = set()
    selected: list[int] = []
    for value in values.detach().cpu().tolist():
        if value not in seen:
            seen.add(value)
            selected.append(value)
    return torch.tensor(selected, dtype=torch.long, device=values.device)


def pfcr_boundary_loss(
    adjusted_cm_logits: torch.Tensor,
    cm_teacher: torch.Tensor,
    fdr_logits: torch.Tensor,
    fdr_teacher: torch.Tensor,
    *,
    rescue_slots: int,
) -> torch.Tensor:
    """Optimize CM candidates against the protected FDR Top-300 boundary."""

    if type(rescue_slots) is not int or rescue_slots not in RESCUE_SLOT_GRID[1:]:
        raise ValueError(f"rescue_slots must be one of {RESCUE_SLOT_GRID[1:]}")
    adjusted_cm_logits = _require_matrix(adjusted_cm_logits, "adjusted_cm_logits")
    cm_teacher = _require_matrix(cm_teacher, "cm_teacher")
    fdr_logits = _require_matrix(fdr_logits, "fdr_logits")
    fdr_teacher = _require_matrix(fdr_teacher, "fdr_teacher")
    if any(
        value.device != adjusted_cm_logits.device
        for value in (cm_teacher, fdr_logits, fdr_teacher)
    ):
        raise ValueError("loss tensors must share one device")

    cm_flat = adjusted_cm_logits.reshape(-1)
    cm_target = cm_teacher.detach().reshape(-1)
    fdr_flat = fdr_logits.detach().reshape(-1)
    fdr_target = fdr_teacher.detach().reshape(-1)
    fdr_order = torch.argsort(fdr_flat, descending=True, stable=True)
    fdr_tail = fdr_order[NUM_QUERIES - rescue_slots : NUM_QUERIES]
    candidate_count = min(cm_flat.numel(), rescue_slots * 4)
    score_pool = torch.argsort(cm_flat.detach(), descending=True, stable=True)[:candidate_count]
    teacher_pool = torch.argsort(cm_target, descending=True, stable=True)[:candidate_count]
    cm_pool = _stable_unique_indices(torch.cat((teacher_pool, score_pool)))

    cm_index = cm_pool.repeat_interleave(fdr_tail.numel())
    fdr_index = fdr_tail.repeat(cm_pool.numel())
    cm_utility = cm_target[cm_index]
    fdr_utility = fdr_target[fdr_index]
    gap = (cm_utility - fdr_utility).detach()
    non_tied = gap != 0
    if bool(non_tied.any()):
        difference = cm_flat[cm_index[non_tied]] - fdr_flat[fdr_index[non_tied]]
        oriented = difference * gap[non_tied].sign()
        weight = gap[non_tied].abs()
        rank_loss = (F.softplus(-oriented) * weight).sum() / weight.sum().clamp_min(
            torch.finfo(weight.dtype).eps
        )
    else:
        rank_loss = cm_flat.sum() * 0.0
    quality_loss = F.binary_cross_entropy_with_logits(
        cm_flat[cm_pool], cm_target[cm_pool].clamp(0, 1)
    )
    return rank_loss + 0.25 * quality_loss
