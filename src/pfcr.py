"""Protected Frequency Candidate Rescue representation and pointwise gate."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


NUM_QUERIES = 300
NUM_CLASSES = 10
PFCR_FEATURE_DIM = 35
_LOG_GUARD = 1e-6


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
