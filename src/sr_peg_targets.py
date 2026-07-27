"""Ground-truth supervision for the SR-PEG query heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.gcte_targets import TINY_EFFECTIVE_SIZE


@dataclass(frozen=True)
class SRPEGTargets:
    """Per-query targets for local utility/risk and global retention."""

    local_tiny_utility: torch.Tensor
    local_non_tiny_risk: torch.Tensor
    global_retain: torch.Tensor


def _validate_predictions(
    name: str,
    boxes: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    if (
        not isinstance(boxes, torch.Tensor)
        or boxes.ndim != 3
        or boxes.shape[0] != 1
        or boxes.shape[-1] != 4
        or not boxes.is_floating_point()
    ):
        raise ValueError(f"{name}_boxes must be floating [1,Q,4]")
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or logits.shape[:2] != boxes.shape[:2]
        or not logits.is_floating_point()
    ):
        raise ValueError(f"{name}_logits must be floating and share [1,Q]")
    if not bool(torch.isfinite(boxes).all() and torch.isfinite(logits).all()):
        raise ValueError(f"{name} predictions must be finite")
    if bool((boxes[..., 2:] < 0).any()):
        raise ValueError(f"{name} box sizes must be nonnegative")
    if boxes.device != logits.device:
        raise ValueError(f"{name} boxes and logits must share one device")


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    half = boxes[..., 2:] * 0.5
    return torch.cat((boxes[..., :2] - half, boxes[..., :2] + half), dim=-1)


def _pairwise_overlap(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_xyxy = _xywh_to_xyxy(predictions)
    gt_xyxy = _xywh_to_xyxy(ground_truth)
    intersection_min = torch.maximum(
        predicted_xyxy[:, None, :2],
        gt_xyxy[None, :, :2],
    )
    intersection_max = torch.minimum(
        predicted_xyxy[:, None, 2:],
        gt_xyxy[None, :, 2:],
    )
    intersection = (
        (intersection_max - intersection_min)
        .clamp_min(0.0)
        .prod(dim=-1)
    )
    predicted_area = predictions[:, 2:].prod(dim=-1)
    gt_area = ground_truth[:, 2:].prod(dim=-1)
    union = predicted_area[:, None] + gt_area[None, :] - intersection
    iou = intersection / union.clamp_min(1e-12)
    smaller = torch.minimum(predicted_area[:, None], gt_area[None, :])
    ios = intersection / smaller.clamp_min(1e-12)
    return iou, ios


def _effective_size(
    boxes: torch.Tensor,
    *,
    source_shape: tuple[int, int],
) -> torch.Tensor:
    height, width = source_shape
    return (
        boxes[..., 2].clamp_min(0.0)
        * float(width)
        * boxes[..., 3].clamp_min(0.0)
        * float(height)
    ).sqrt()


def build_sr_peg_targets(
    *,
    global_boxes: torch.Tensor,
    global_logits: torch.Tensor,
    local_boxes: torch.Tensor,
    local_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    source_shape: tuple[int, int],
) -> SRPEGTargets:
    """Build source-frame tiny, fragment-risk, and global-retain targets."""

    _validate_predictions("global", global_boxes, global_logits)
    _validate_predictions("local", local_boxes, local_logits)
    if global_boxes.device != local_boxes.device:
        raise ValueError("global and local predictions must share one device")
    if global_logits.shape[-1] != local_logits.shape[-1]:
        raise ValueError("global and local class counts must match")
    if (
        not isinstance(gt_boxes, torch.Tensor)
        or gt_boxes.ndim != 2
        or gt_boxes.shape[-1] != 4
        or not gt_boxes.is_floating_point()
    ):
        raise ValueError("gt_boxes must be floating [N,4]")
    if (
        not isinstance(gt_classes, torch.Tensor)
        or gt_classes.ndim != 1
        or gt_classes.shape[0] != gt_boxes.shape[0]
        or gt_classes.dtype != torch.long
    ):
        raise ValueError("gt_classes must be torch.long [N]")
    if gt_boxes.device != global_boxes.device or gt_classes.device != global_boxes.device:
        raise ValueError("targets and predictions must share one device")
    if not bool(torch.isfinite(gt_boxes).all()):
        raise ValueError("gt_boxes must be finite")
    if bool((gt_boxes[:, 2:] < 0).any()):
        raise ValueError("GT box sizes must be nonnegative")
    if (
        not isinstance(source_shape, tuple)
        or len(source_shape) != 2
        or any(not isinstance(value, int) or value <= 0 for value in source_shape)
    ):
        raise ValueError("source_shape must be a positive integer (height,width)")

    local_count = local_boxes.shape[1]
    global_count = global_boxes.shape[1]
    if gt_boxes.shape[0] == 0:
        return SRPEGTargets(
            local_tiny_utility=local_boxes.new_zeros((1, local_count, 1)),
            local_non_tiny_risk=local_boxes.new_zeros((1, local_count, 1)),
            global_retain=global_boxes.new_zeros((1, global_count, 1)),
        )

    gt_sizes = _effective_size(gt_boxes, source_shape=source_shape)
    tiny_gt = gt_sizes <= TINY_EFFECTIVE_SIZE
    non_tiny_gt = ~tiny_gt

    local_classes = local_logits[0].argmax(dim=-1)
    local_iou, local_ios = _pairwise_overlap(local_boxes[0], gt_boxes)
    same_local_class = local_classes[:, None] == gt_classes[None, :]
    utility_candidates = torch.where(
        same_local_class & tiny_gt[None, :] & (local_iou >= 0.5),
        local_iou,
        torch.zeros_like(local_iou),
    )
    local_utility = utility_candidates.amax(dim=1)
    local_risk = (
        (local_ios >= 0.5) & non_tiny_gt[None, :]
    ).any(dim=1).to(local_boxes.dtype)

    global_classes = global_logits[0].argmax(dim=-1)
    _, global_ios = _pairwise_overlap(global_boxes[0], gt_boxes)
    global_is_predicted_tiny = (
        _effective_size(global_boxes[0], source_shape=source_shape)
        <= TINY_EFFECTIVE_SIZE
    )
    same_global_class = global_classes[:, None] == gt_classes[None, :]
    global_retain = (
        same_global_class
        & non_tiny_gt[None, :]
        & (global_ios >= 0.5)
    ).any(dim=1)
    global_retain = (
        global_retain & global_is_predicted_tiny
    ).to(global_boxes.dtype)

    return SRPEGTargets(
        local_tiny_utility=local_utility.reshape(1, local_count, 1).detach(),
        local_non_tiny_risk=local_risk.reshape(1, local_count, 1).detach(),
        global_retain=global_retain.reshape(1, global_count, 1).detach(),
    )


__all__ = ["SRPEGTargets", "build_sr_peg_targets"]
