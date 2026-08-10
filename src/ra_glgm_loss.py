"""Detached residual-difficulty targets and support supervision for RA-GLGM."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from src.btd_se_loss import binary_focal_loss
from src.fdr_loss import MatchIndices


@dataclass(frozen=True)
class ResidualDifficultyTargets:
    heatmap: Tensor
    valid_mask: Tensor
    difficulty: Tensor
    scale_distribution: Tensor | None = None
    scale_mask: Tensor | None = None


def _paired_iou_xywh(first: Tensor, second: Tensor, eps: float = 1e-7) -> Tensor:
    first_half = first[:, 2:] / 2
    second_half = second[:, 2:] / 2
    first_lo, first_hi = first[:, :2] - first_half, first[:, :2] + first_half
    second_lo, second_hi = second[:, :2] - second_half, second[:, :2] + second_half
    intersection = (
        torch.minimum(first_hi, second_hi) - torch.maximum(first_lo, second_lo)
    ).clamp_min(0).prod(dim=1)
    first_area = (first_hi - first_lo).clamp_min(0).prod(dim=1)
    second_area = (second_hi - second_lo).clamp_min(0).prod(dim=1)
    return intersection / (first_area + second_area - intersection).clamp_min(eps)


def _box_slice(box: Tensor, *, height: int, width: int) -> tuple[slice, slice]:
    cx, cy, box_width, box_height = [float(value) for value in box]
    x0 = max(0, min(width - 1, math.floor((cx - box_width / 2) * width)))
    y0 = max(0, min(height - 1, math.floor((cy - box_height / 2) * height)))
    x1 = max(x0 + 1, min(width, math.ceil((cx + box_width / 2) * width)))
    y1 = max(y0 + 1, min(height, math.ceil((cy + box_height / 2) * height)))
    return slice(y0, y1), slice(x0, x1)


@torch.no_grad()
def build_residual_difficulty_targets(
    *,
    pred_bboxes: Tensor,
    pred_scores: Tensor,
    detection_bboxes: Tensor,
    detection_classes: Tensor,
    detection_batch_idx: Tensor,
    match_indices: MatchIndices,
    all_bboxes: Tensor,
    all_classes: Tensor,
    all_batch_idx: Tensor,
    height: int,
    width: int,
    chunk_size: int = 64,
) -> ResidualDifficultyTargets:
    """Rasterize FDR's detached final-layer residual difficulty on P3.

    The supplied assignment must be the already-recorded final normal decoder
    assignment.  No matcher is accepted or called here.  Ground truths absent
    from that assignment retain difficulty one.
    """

    if pred_bboxes.ndim != 3 or pred_bboxes.shape[-1] != 4:
        raise ValueError("pred_bboxes must have shape [batch,queries,4]")
    if pred_scores.ndim != 3 or pred_scores.shape[:2] != pred_bboxes.shape[:2]:
        raise ValueError("pred_scores must match prediction batch/query axes")
    if height <= 0 or width <= 0 or chunk_size <= 0:
        raise ValueError("target dimensions and chunk_size must be positive")
    batch_size = int(pred_bboxes.shape[0])
    if len(match_indices) != batch_size:
        raise ValueError("assignment batch count does not match predictions")

    device = pred_bboxes.device
    with torch.autocast(device_type=device.type, enabled=False):
        finite_inputs = (
            pred_bboxes,
            pred_scores,
            detection_bboxes,
            detection_classes,
            detection_batch_idx,
            all_bboxes,
            all_classes,
            all_batch_idx,
        )
        if any(not bool(torch.isfinite(value).all()) for value in finite_inputs):
            raise FloatingPointError("NONFINITE_RA_GLGM_TARGET_INPUT")
        boxes = detection_bboxes.detach().to(device=device, dtype=torch.float32)
        classes = detection_classes.detach().to(device=device, dtype=torch.long).view(-1)
        batch_idx = detection_batch_idx.detach().to(device=device, dtype=torch.long).view(-1)
        if not (len(boxes) == len(classes) == len(batch_idx)):
            raise ValueError("detection target fields must have matching lengths")
        if bool((classes < 0).any()):
            raise ValueError("ignored classes must not enter residual targets")

        predictions = pred_bboxes.detach().to(dtype=torch.float32)
        scores = pred_scores.detach().to(dtype=torch.float32)
        difficulty = torch.ones(len(boxes), device=device, dtype=torch.float32)
        assigned = torch.zeros(len(boxes), device=device, dtype=torch.bool)

        for image, (source, destination) in enumerate(match_indices):
            source = source.to(device=device, dtype=torch.long)
            destination = destination.to(device=device, dtype=torch.long)
            if source.ndim != 1 or destination.ndim != 1 or len(source) != len(destination):
                raise ValueError("each assignment must contain equal one-dimensional indices")
            if not len(source):
                continue
            if int(source.min()) < 0 or int(source.max()) >= predictions.shape[1]:
                raise IndexError("assigned query index is out of range")
            if int(destination.min()) < 0 or int(destination.max()) >= len(boxes):
                raise IndexError("assigned target index is out of range")
            if bool(assigned[destination].any()):
                raise ValueError("a target cannot appear twice in the final assignment")
            if bool((batch_idx[destination] != image).any()):
                raise ValueError("assignment target belongs to a different image")
            target_classes = classes[destination]
            if int(target_classes.max()) >= scores.shape[-1]:
                raise IndexError("assigned target class is out of range")
            probability = scores[image, source, target_classes].sigmoid()
            iou = _paired_iou_xywh(predictions[image, source], boxes[destination])
            difficulty[destination] = (
                0.7 * (1.0 - probability) + 0.3 * (1.0 - iou)
            ).clamp(0.25, 1.0)
            assigned[destination] = True

        heatmap = torch.zeros(
            (batch_size, 1, height, width),
            device=device,
            dtype=torch.float32,
        )
        scale_heatmap = torch.zeros(
            (batch_size, 3, height, width),
            device=device,
            dtype=torch.float32,
        )
        area_640 = boxes[:, 2].clamp_min(0) * boxes[:, 3].clamp_min(0) * (640.0**2)
        scale_indices = torch.where(
            area_640 < 16.0**2,
            torch.zeros_like(area_640, dtype=torch.long),
            torch.where(
                area_640 < 32.0**2,
                torch.ones_like(area_640, dtype=torch.long),
                torch.full_like(area_640, 2, dtype=torch.long),
            ),
        )
        grid_y = torch.arange(height, device=device, dtype=torch.float32).view(1, height, 1)
        grid_x = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, width)
        for image in range(batch_size):
            image_indices = torch.nonzero(batch_idx == image, as_tuple=False).flatten()
            for start in range(0, len(image_indices), chunk_size):
                indices = image_indices[start : start + chunk_size]
                if not len(indices):
                    continue
                chunk = boxes[indices]
                center_x = (chunk[:, 0] * width).view(-1, 1, 1)
                center_y = (chunk[:, 1] * height).view(-1, 1, 1)
                sigma_x = (chunk[:, 2] * width / 8.0).clamp_min(1.0).view(-1, 1, 1)
                sigma_y = (chunk[:, 3] * height / 8.0).clamp_min(1.0).view(-1, 1, 1)
                dx = (grid_x - center_x) / sigma_x
                dy = (grid_y - center_y) / sigma_y
                gaussian = torch.exp(-0.5 * (dx.square() + dy.square()))
                gaussian = gaussian * ((dx.abs() <= 3) & (dy.abs() <= 3))
                weighted = gaussian * difficulty[indices].view(-1, 1, 1)
                heatmap[image, 0] = torch.maximum(
                    heatmap[image, 0],
                    weighted.amax(dim=0),
                )
                chunk_scales = scale_indices[indices]
                for scale_index in range(3):
                    selected = chunk_scales == scale_index
                    if bool(selected.any()):
                        scale_heatmap[image, scale_index] = torch.maximum(
                            scale_heatmap[image, scale_index],
                            weighted[selected].amax(dim=0),
                        )

        valid = torch.ones_like(heatmap, dtype=torch.bool)
        raw_boxes = all_bboxes.detach().to(device=device, dtype=torch.float32)
        raw_classes = all_classes.detach().to(device=device).view(-1)
        raw_batch = all_batch_idx.detach().to(device=device, dtype=torch.long).view(-1)
        if not (len(raw_boxes) == len(raw_classes) == len(raw_batch)):
            raise ValueError("unfiltered target fields must have matching lengths")
        for box, class_id, image in zip(raw_boxes, raw_classes, raw_batch):
            if float(class_id) >= 0:
                continue
            image_index = int(image)
            if image_index < 0 or image_index >= batch_size:
                raise IndexError("ignored target batch index is out of range")
            y_slice, x_slice = _box_slice(box, height=height, width=width)
            valid[image_index, 0, y_slice, x_slice] = False
        # Ignore regions suppress only negative supervision; a real positive
        # target remains valid if annotations overlap an ignored rectangle.
        valid |= heatmap > 0
        scale_sum = scale_heatmap.sum(dim=1, keepdim=True)
        scale_mask = scale_sum > 0
        scale_distribution = torch.where(
            scale_mask,
            scale_heatmap / scale_sum.clamp_min(torch.finfo(torch.float32).tiny),
            torch.zeros_like(scale_heatmap),
        )
        return ResidualDifficultyTargets(
            heatmap,
            valid,
            difficulty,
            scale_distribution,
            scale_mask,
        )


def residual_support_focal_loss(
    support: Tensor,
    targets: ResidualDifficultyTargets,
) -> Tensor:
    """Compute the FP32 soft focal objective on valid P3 pixels."""

    return binary_focal_loss(
        support,
        targets.heatmap,
        alpha=0.25,
        exponent=2.0,
        valid_mask=targets.valid_mask,
    )


def scale_conditioning_loss(
    probabilities: Tensor,
    targets: ResidualDifficultyTargets,
) -> Tensor:
    """Soft three-scale cross entropy on positive residual-support pixels only."""

    distribution = targets.scale_distribution
    mask = targets.scale_mask
    if distribution is None or mask is None:
        raise ValueError("scale targets are unavailable")
    if probabilities.shape != distribution.shape or mask.shape != distribution[:, :1].shape:
        raise ValueError("scale prediction and target shapes differ")
    if not bool(torch.isfinite(probabilities).all()):
        raise FloatingPointError("NONFINITE_RA_GLGM_SCALE_PROBABILITY")
    with torch.autocast(device_type=probabilities.device.type, enabled=False):
        predicted = probabilities.float().clamp_min(torch.finfo(torch.float32).tiny)
        per_pixel = -(distribution.float() * predicted.log()).sum(dim=1, keepdim=True)
        if not bool(mask.any()):
            return probabilities.sum() * 0.0
        return per_pixel.masked_select(mask).mean()


@torch.no_grad()
def scale_prediction_diagnostics(
    probabilities: Tensor,
    targets: ResidualDifficultyTargets,
) -> dict[str, Tensor]:
    """Return finite non-collapse evidence for the frozen Screen gates."""

    distribution = targets.scale_distribution
    mask = targets.scale_mask
    if distribution is None or mask is None:
        raise ValueError("scale targets are unavailable")
    if probabilities.shape != distribution.shape:
        raise ValueError("scale prediction and target shapes differ")
    positive = mask[:, 0]
    device = probabilities.device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if not bool(positive.any()):
        return {
            "scale_entropy": zero,
            "scale_tiny_fraction": zero,
            "scale_small_fraction": zero,
            "scale_regular_fraction": zero,
            "scale_tiny_recall": zero,
            "scale_small_recall": zero,
            "scale_regular_recall": zero,
            "scale_positive_pixels": zero,
        }
    predicted = probabilities.float()
    entropy = -(predicted.clamp_min(torch.finfo(torch.float32).tiny).log() * predicted).sum(dim=1)
    predicted_class = predicted.argmax(dim=1)
    target_class = distribution.argmax(dim=1)
    values: dict[str, Tensor] = {
        "scale_entropy": entropy[positive].mean(),
        "scale_positive_pixels": positive.sum().float(),
    }
    for index, name in enumerate(("tiny", "small", "regular")):
        values[f"scale_{name}_fraction"] = (predicted_class[positive] == index).float().mean()
        target_pixels = positive & (target_class == index)
        values[f"scale_{name}_recall"] = (
            (predicted_class[target_pixels] == index).float().mean()
            if bool(target_pixels.any())
            else zero
        )
    return values


__all__ = [
    "ResidualDifficultyTargets",
    "build_residual_difficulty_targets",
    "residual_support_focal_loss",
    "scale_conditioning_loss",
    "scale_prediction_diagnostics",
]
