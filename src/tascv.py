"""Tiny-only asymmetric cross-view localization primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.ascv_loc import (
    ASCV_CROP_SIZE,
    ASCV_TINY_BOUNDARY_PX,
    local_to_full_xywh,
)


TASCV_TINY_BOUNDARY_PX = ASCV_TINY_BOUNDARY_PX


@dataclass(frozen=True)
class TASCVLossResult:
    loss: torch.Tensor
    matched_pair_count: int
    auxiliary_tiny_pair_count: int
    excluded_non_tiny_pair_count: int
    tiny_teacher_advantage_sum: torch.Tensor
    tiny_teacher_win_count: int


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center = boxes[..., :2]
    half_extent = boxes[..., 2:] / 2
    return torch.cat((center - half_extent, center + half_extent), dim=-1)


def _aligned_giou_xywh(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    first_xyxy = _xywh_to_xyxy(first)
    second_xyxy = _xywh_to_xyxy(second)
    intersection_low = torch.maximum(
        first_xyxy[:, :2],
        second_xyxy[:, :2],
    )
    intersection_high = torch.minimum(
        first_xyxy[:, 2:],
        second_xyxy[:, 2:],
    )
    intersection = (
        intersection_high - intersection_low
    ).clamp(min=0).prod(dim=-1)
    first_area = (
        first_xyxy[:, 2:] - first_xyxy[:, :2]
    ).clamp(min=0).prod(dim=-1)
    second_area = (
        second_xyxy[:, 2:] - second_xyxy[:, :2]
    ).clamp(min=0).prod(dim=-1)
    union = first_area + second_area - intersection
    epsilon = torch.finfo(first.dtype).eps
    iou = intersection / union.clamp(min=epsilon)
    enclosure_low = torch.minimum(
        first_xyxy[:, :2],
        second_xyxy[:, :2],
    )
    enclosure_high = torch.maximum(
        first_xyxy[:, 2:],
        second_xyxy[:, 2:],
    )
    enclosure = (
        enclosure_high - enclosure_low
    ).clamp(min=0).prod(dim=-1)
    return iou - (enclosure - union) / enclosure.clamp(min=epsilon)


def _box_loss_per_pair(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    l1 = (student - teacher).abs().sum(dim=-1)
    return l1 + 1.0 - _aligned_giou_xywh(student, teacher)


def _validate_geometry(
    name: str,
    tensor: torch.Tensor,
) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f"T-ASCV received non-finite {name}")
    if bool((tensor[:, 2:] <= 0).any()):
        raise RuntimeError(f"T-ASCV received degenerate {name}")


def compute_tascv_loss(
    *,
    full_pred_boxes: torch.Tensor,
    local_pred_boxes: torch.Tensor,
    full_gt_boxes: torch.Tensor,
    pair_crops: torch.Tensor,
    image_hw: tuple[int, int],
) -> TASCVLossResult:
    """Apply local-teacher consistency only to matched tiny targets."""

    pair_count = int(full_pred_boxes.shape[0])
    if (
        full_pred_boxes.ndim != 2
        or full_pred_boxes.shape[1:] != (4,)
        or local_pred_boxes.shape != full_pred_boxes.shape
        or full_gt_boxes.shape != full_pred_boxes.shape
        or pair_crops.shape != full_pred_boxes.shape
    ):
        raise ValueError(
            "paired prediction, target, and crop tensors must all have "
            "shape [pairs, 4]"
        )
    height, width = (int(image_hw[0]), int(image_hw[1]))
    if (height, width) != (640, 640):
        raise ValueError("T-ASCV requires the frozen 640x640 frame")
    if not (
        full_pred_boxes.device
        == local_pred_boxes.device
        == full_gt_boxes.device
    ):
        raise ValueError(
            "full predictions, local predictions, and targets must use "
            "the same device"
        )

    full_fp32 = full_pred_boxes.float()
    local_fp32 = local_pred_boxes.float()
    targets_fp32 = full_gt_boxes.float()
    zero = (full_fp32.sum() + local_fp32.sum()) * 0.0
    if pair_count == 0:
        return TASCVLossResult(
            loss=zero,
            matched_pair_count=0,
            auxiliary_tiny_pair_count=0,
            excluded_non_tiny_pair_count=0,
            tiny_teacher_advantage_sum=zero.detach(),
            tiny_teacher_win_count=0,
        )

    _validate_geometry("full predictions", full_fp32)
    _validate_geometry("local predictions", local_fp32)
    _validate_geometry("full targets", targets_fp32)
    crops_fp32 = pair_crops.to(
        device=full_fp32.device,
        dtype=torch.float32,
    )
    if (
        not bool(torch.isfinite(crops_fp32).all())
        or bool((crops_fp32[:, 2:] <= crops_fp32[:, :2]).any())
    ):
        raise RuntimeError("T-ASCV received invalid pair crops")
    if bool(
        (crops_fp32 != crops_fp32.round()).any()
        or (
            crops_fp32[:, 2:] - crops_fp32[:, :2]
            != ASCV_CROP_SIZE
        ).any()
    ):
        raise RuntimeError(
            "T-ASCV requires integer 384x384 crop-v2 geometry"
        )
    image_extent = crops_fp32.new_tensor([width, height])
    if bool(
        (crops_fp32[:, :2] < 0).any()
        or (crops_fp32[:, 2:] > image_extent).any()
    ):
        raise RuntimeError("T-ASCV received a crop outside the full image")

    target_xyxy = _xywh_to_xyxy(targets_fp32)
    if bool(
        (target_xyxy[:, :2] < 0).any()
        or (target_xyxy[:, 2:] > 1).any()
    ):
        raise RuntimeError("T-ASCV received a target outside the full image")
    target_absolute = target_xyxy * targets_fp32.new_tensor(
        [width, height, width, height]
    )
    epsilon = targets_fp32.new_tensor(1e-6)
    fully_contained = (
        (target_absolute[:, :2] + epsilon >= crops_fp32[:, :2]).all(dim=1)
        & (
            target_absolute[:, 2:] - epsilon
            <= crops_fp32[:, 2:]
        ).all(dim=1)
    )
    if not bool(fully_contained.all()):
        raise RuntimeError(
            "T-ASCV received a paired target not fully contained in its crop"
        )

    target_effective_size = torch.sqrt(
        targets_fp32[:, 2] * float(width)
        * targets_fp32[:, 3] * float(height)
    )
    tiny = target_effective_size <= TASCV_TINY_BOUNDARY_PX
    tiny_count = int(tiny.sum().item())
    if tiny_count == 0:
        return TASCVLossResult(
            loss=zero,
            matched_pair_count=pair_count,
            auxiliary_tiny_pair_count=0,
            excluded_non_tiny_pair_count=pair_count,
            tiny_teacher_advantage_sum=zero.detach(),
            tiny_teacher_win_count=0,
        )

    mapped_local = local_to_full_xywh(
        local_fp32[tiny].detach(),
        crops_fp32[tiny],
        image_hw=image_hw,
    )
    tiny_students = full_fp32[tiny]
    loss = _box_loss_per_pair(tiny_students, mapped_local).mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("T-ASCV produced a non-finite auxiliary loss")

    tiny_targets = targets_fp32[tiny]
    full_error = _box_loss_per_pair(
        tiny_students.detach(),
        tiny_targets,
    )
    local_error = _box_loss_per_pair(
        mapped_local,
        tiny_targets,
    )
    advantage = full_error - local_error
    return TASCVLossResult(
        loss=loss,
        matched_pair_count=pair_count,
        auxiliary_tiny_pair_count=tiny_count,
        excluded_non_tiny_pair_count=pair_count - tiny_count,
        tiny_teacher_advantage_sum=advantage.sum().detach(),
        tiny_teacher_win_count=int((advantage > 0).sum().item()),
    )


__all__ = [
    "TASCV_TINY_BOUNDARY_PX",
    "TASCVLossResult",
    "compute_tascv_loss",
]
