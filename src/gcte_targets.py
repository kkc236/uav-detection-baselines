"""Deterministic training targets for the frozen-detector GCQF stage."""

from __future__ import annotations

import torch


TINY_EFFECTIVE_SIZE = 16.0
EFFECTIVE_FRAME = 640.0


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center = boxes[..., :2]
    half = boxes[..., 2:] * 0.5
    return torch.cat((center - half, center + half), dim=-1)


def _pairwise_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_xyxy = _xywh_to_xyxy(left)
    right_xyxy = _xywh_to_xyxy(right)
    intersection_min = torch.maximum(
        left_xyxy[:, None, :2],
        right_xyxy[None, :, :2],
    )
    intersection_max = torch.minimum(
        left_xyxy[:, None, 2:],
        right_xyxy[None, :, 2:],
    )
    intersection_size = (intersection_max - intersection_min).clamp_min(0.0)
    intersection = intersection_size.prod(dim=-1)
    left_area = (left_xyxy[:, 2:] - left_xyxy[:, :2]).prod(dim=-1)
    right_area = (
        right_xyxy[:, 2:] - right_xyxy[:, :2]
    ).prod(dim=-1)
    union = left_area[:, None] + right_area[None, :] - intersection
    return intersection / union.clamp_min(1e-12)


def build_quality_targets(
    canonical_local_boxes: torch.Tensor,
    local_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
) -> torch.Tensor:
    """Return best same-class global IoU for every local decoder query."""

    if (
        canonical_local_boxes.ndim != 3
        or canonical_local_boxes.shape[0] != 1
        or canonical_local_boxes.shape[-1] != 4
    ):
        raise ValueError("canonical_local_boxes must be [1,Q,4]")
    if (
        local_logits.ndim != 3
        or local_logits.shape[:2] != canonical_local_boxes.shape[:2]
    ):
        raise ValueError("local_logits must share [1,Q]")
    if gt_boxes.ndim != 2 or gt_boxes.shape[-1] != 4:
        raise ValueError("gt_boxes must be [N,4]")
    if gt_classes.ndim != 1 or gt_classes.shape[0] != gt_boxes.shape[0]:
        raise ValueError("gt_classes must be [N]")
    if gt_classes.dtype != torch.long:
        raise ValueError("gt_classes must use torch.long")
    query_count = canonical_local_boxes.shape[1]
    if gt_boxes.shape[0] == 0:
        return canonical_local_boxes.new_zeros((1, query_count, 1))
    predicted_classes = local_logits[0].argmax(dim=-1)
    iou = _pairwise_iou(canonical_local_boxes[0], gt_boxes)
    same_class = predicted_classes[:, None] == gt_classes[None, :]
    target = torch.where(
        same_class,
        iou,
        torch.zeros_like(iou),
    ).amax(dim=1)
    return target.reshape(1, query_count, 1).detach()


def build_equivariance_pairs(
    matched_gt_indices: torch.Tensor,
    view_indices: torch.Tensor,
) -> torch.Tensor:
    """Join matcher-assigned queries for one GT across distinct local views."""

    if (
        matched_gt_indices.ndim != 1
        or view_indices.shape != matched_gt_indices.shape
        or matched_gt_indices.dtype != torch.long
        or view_indices.dtype != torch.long
    ):
        raise ValueError("matched GT and view indices must be long [Q]")
    pairs: list[tuple[int, int]] = []
    for left in range(int(matched_gt_indices.numel())):
        gt_index = int(matched_gt_indices[left])
        if gt_index < 0:
            continue
        for right in range(left + 1, int(matched_gt_indices.numel())):
            if (
                int(matched_gt_indices[right]) == gt_index
                and int(view_indices[right]) != int(view_indices[left])
            ):
                pairs.append((left, right))
    if not pairs:
        return torch.empty(
            (0, 2),
            dtype=torch.long,
            device=matched_gt_indices.device,
        )
    return torch.tensor(
        pairs,
        dtype=torch.long,
        device=matched_gt_indices.device,
    )


def build_tiny_anchor_mask(
    canonical_local_boxes: torch.Tensor,
) -> torch.Tensor:
    """Apply the fixed SADED 16-pixel effective-size admission boundary."""

    if canonical_local_boxes.ndim != 3 or canonical_local_boxes.shape[-1] != 4:
        raise ValueError("canonical_local_boxes must be [B,Q,4]")
    if bool((canonical_local_boxes[..., 2:] < 0.0).any()):
        raise ValueError("box width and height must be nonnegative")
    effective_size = (
        canonical_local_boxes[..., 2:].prod(dim=-1).clamp_min(0.0).sqrt()
        * EFFECTIVE_FRAME
    )
    return (effective_size <= TINY_EFFECTIVE_SIZE).unsqueeze(-1)


__all__ = [
    "EFFECTIVE_FRAME",
    "TINY_EFFECTIVE_SIZE",
    "build_equivariance_pairs",
    "build_quality_targets",
    "build_tiny_anchor_mask",
]
