from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import VarifocalLoss
from ultralytics.utils.metrics import bbox_iou


@dataclass(frozen=True)
class P2Targets:
    gt_boxes: list[torch.Tensor]
    gt_classes: list[torch.Tensor]
    assigned_pairs: list[torch.Tensor]
    topk_indices: torch.Tensor
    anchor_centers: torch.Tensor

    def image(self, index: int) -> _ImageP2Targets:
        return _ImageP2Targets(
            gt_boxes=self.gt_boxes[index],
            gt_classes=self.gt_classes[index],
            assigned_pairs=self.assigned_pairs[index],
            topk_indices=self.topk_indices[index],
            anchor_centers=self.anchor_centers,
        )


@dataclass(frozen=True)
class P2LossResult:
    total: torch.Tensor
    vfl: torch.Tensor
    l1: torch.Tensor
    giou: torch.Tensor
    positive_count: torch.Tensor
    classification_indices: list[torch.Tensor]
    vfl_target: torch.Tensor


@dataclass(frozen=True)
class QualityLossResult:
    total: torch.Tensor
    positive_count: torch.Tensor
    classification_indices: list[torch.Tensor]
    targets: torch.Tensor


@dataclass(frozen=True)
class _ImageP2Targets:
    gt_boxes: torch.Tensor
    gt_classes: torch.Tensor
    assigned_pairs: torch.Tensor
    topk_indices: torch.Tensor
    anchor_centers: torch.Tensor


@dataclass(frozen=True)
class _ImageP2Loss:
    total: torch.Tensor
    vfl: torch.Tensor
    l1: torch.Tensor
    giou: torch.Tensor
    positive_count: torch.Tensor
    classification_indices: torch.Tensor
    vfl_target: torch.Tensor


_VARIFOCAL_LOSS = VarifocalLoss()


def differentiable_zero(*tensors: torch.Tensor) -> torch.Tensor:
    if not tensors:
        raise ValueError("at least one tensor is required")
    zero = tensors[0].sum() * 0.0
    for tensor in tensors[1:]:
        zero = zero + tensor.sum() * 0.0
    return zero.float()


def compute_sparse_p2_loss(
    logits: torch.Tensor,
    boxes_xywh: torch.Tensor,
    targets: P2Targets,
) -> P2LossResult:
    if logits.shape[:2] != boxes_xywh.shape[:2]:
        raise ValueError("logits and boxes must share batch and candidate dimensions")
    if len(targets.gt_boxes) != logits.shape[0] or len(targets.assigned_pairs) != logits.shape[0]:
        raise ValueError("targets must contain one entry per image")

    images = [
        _compute_image_p2_loss(logits[index], boxes_xywh[index], targets.image(index))
        for index in range(logits.shape[0])
    ]
    return P2LossResult(
        total=torch.stack([image.total for image in images]).mean().float(),
        vfl=torch.stack([image.vfl for image in images]).mean().float(),
        l1=torch.stack([image.l1 for image in images]).mean().float(),
        giou=torch.stack([image.giou for image in images]).mean().float(),
        positive_count=torch.stack([image.positive_count for image in images]).sum(),
        classification_indices=[image.classification_indices for image in images],
        vfl_target=torch.cat([image.vfl_target for image in images], dim=0).detach(),
    )


def compute_sparse_quality_loss(
    quality_logits: torch.Tensor,
    boxes_xywh: torch.Tensor,
    targets: P2Targets,
) -> QualityLossResult:
    if quality_logits.ndim != 2:
        raise ValueError("quality logits must have shape [batch, candidates]")
    if quality_logits.shape != boxes_xywh.shape[:2]:
        raise ValueError("quality logits and boxes must share batch and candidate dimensions")
    if len(targets.gt_boxes) != quality_logits.shape[0] or len(targets.assigned_pairs) != quality_logits.shape[0]:
        raise ValueError("targets must contain one entry per image")

    image_losses = []
    image_positive_counts = []
    image_indices = []
    image_targets = []
    for image_index in range(quality_logits.shape[0]):
        image = targets.image(image_index)
        gt_boxes = image.gt_boxes.to(device=quality_logits.device, dtype=torch.float32)
        pairs = image.assigned_pairs.to(device=quality_logits.device, dtype=torch.long)
        assigned_indices = pairs[:, 1] if pairs.numel() else torch.empty(0, dtype=torch.long, device=quality_logits.device)
        classification_indices = _classification_indices(
            anchor_centers=image.anchor_centers.to(device=quality_logits.device, dtype=torch.float32),
            gt_boxes=gt_boxes,
            topk_indices=image.topk_indices.to(device=quality_logits.device, dtype=torch.long),
            assigned_indices=assigned_indices,
        )
        selected_logits = quality_logits[image_index, classification_indices]
        quality_targets = torch.zeros_like(selected_logits, dtype=torch.float32)
        if pairs.numel():
            gt_indices = pairs[:, 0]
            predicted = boxes_xywh[image_index, assigned_indices].float()
            expected = gt_boxes[gt_indices]
            with torch.autocast(device_type=quality_logits.device.type, enabled=False):
                iou_targets = bbox_iou(predicted, expected, xywh=True).squeeze(-1).clamp(0, 1).detach()
            selected_rows = torch.searchsorted(classification_indices, assigned_indices)
            quality_targets[selected_rows] = iou_targets

        if classification_indices.numel():
            element_loss = F.binary_cross_entropy_with_logits(
                selected_logits.float(),
                quality_targets,
                reduction="none",
            )
            positive_rows = torch.isin(classification_indices, assigned_indices)
            group_losses = []
            if positive_rows.any():
                group_losses.append(element_loss[positive_rows].mean())
            if (~positive_rows).any():
                group_losses.append(element_loss[~positive_rows].mean())
            raw_loss = torch.stack(group_losses).mean()
        else:
            raw_loss = differentiable_zero(quality_logits[image_index])
        positive_count = quality_logits.new_tensor(float(len(pairs)))
        image_losses.append(raw_loss)
        image_positive_counts.append(positive_count)
        image_indices.append(classification_indices)
        image_targets.append(quality_targets.detach())

    return QualityLossResult(
        total=torch.stack(image_losses).mean().float(),
        positive_count=torch.stack(image_positive_counts).sum(),
        classification_indices=image_indices,
        targets=torch.cat(image_targets).detach(),
    )


def compute_ebc_loss(
    p2_logits: torch.Tensor,
    assigned_pairs: list[torch.Tensor],
    gt_classes: list[torch.Tensor],
    uncovered: list[torch.Tensor],
    stock_boundary: torch.Tensor,
    gt_boxes: list[torch.Tensor] | None = None,
    p2_boxes: torch.Tensor | None = None,
    quality_weighted: bool = False,
) -> torch.Tensor:
    if quality_weighted and (gt_boxes is None or p2_boxes is None):
        raise ValueError("quality-weighted EBC requires GT and P2 boxes")
    eligible_scores = []
    for batch_index, pairs in enumerate(assigned_pairs):
        pairs = pairs.to(device=p2_logits.device, dtype=torch.long)
        if pairs.numel() == 0:
            continue
        gt_index = pairs[:, 0]
        candidate_index = pairs[:, 1]
        image_uncovered = uncovered[batch_index].to(device=p2_logits.device, dtype=torch.bool).detach()
        eligible = image_uncovered[gt_index]
        if not eligible.any():
            continue
        classes = gt_classes[batch_index].to(device=p2_logits.device, dtype=torch.long)[gt_index[eligible]]
        scores = p2_logits[batch_index, candidate_index[eligible], classes].float()
        boundary = stock_boundary[batch_index].detach().float()
        hinge = torch.relu(boundary - scores)
        if quality_weighted:
            predicted = p2_boxes[batch_index, candidate_index[eligible]].float()
            target = gt_boxes[batch_index].to(device=p2_logits.device, dtype=torch.float32)[gt_index[eligible]]
            quality = bbox_iou(predicted, target, xywh=True).squeeze(-1).clamp(0, 1).detach()
            hinge = hinge * quality
        eligible_scores.append(hinge)

    if not eligible_scores:
        return differentiable_zero(p2_logits)
    return torch.cat(eligible_scores).mean().float()


def _compute_image_p2_loss(
    logits: torch.Tensor,
    boxes_xywh: torch.Tensor,
    targets: _ImageP2Targets,
) -> _ImageP2Loss:
    device = logits.device
    gt_boxes = targets.gt_boxes.to(device=device, dtype=torch.float32)
    gt_classes = targets.gt_classes.to(device=device, dtype=torch.long)
    pairs = targets.assigned_pairs.to(device=device, dtype=torch.long)
    topk_indices = targets.topk_indices.to(device=device, dtype=torch.long)
    anchor_centers = targets.anchor_centers.to(device=device, dtype=torch.float32)

    assigned_indices = pairs[:, 1] if pairs.numel() else torch.empty(0, dtype=torch.long, device=device)
    classification_indices = _classification_indices(
        anchor_centers=anchor_centers,
        gt_boxes=gt_boxes,
        topk_indices=topk_indices,
        assigned_indices=assigned_indices,
    )

    selected_logits = logits[classification_indices]
    vfl_target = torch.zeros_like(selected_logits, dtype=torch.float32)
    labels = torch.zeros_like(selected_logits, dtype=torch.float32)

    if pairs.numel():
        gt_index = pairs[:, 0]
        candidate_index = pairs[:, 1]
        positive_boxes = boxes_xywh[candidate_index]
        target_boxes = gt_boxes[gt_index]
        with torch.autocast(device_type=device.type, enabled=False):
            positive_boxes_fp32 = positive_boxes.float()
            target_boxes_fp32 = target_boxes.float()
            iou_target = bbox_iou(positive_boxes_fp32, target_boxes_fp32, xywh=True).squeeze(-1).clamp(0, 1)
            l1_raw = F.l1_loss(positive_boxes_fp32, target_boxes_fp32, reduction="sum")
            giou_raw = (
                1.0 - bbox_iou(positive_boxes_fp32, target_boxes_fp32, xywh=True, GIoU=True).squeeze(-1)
            ).sum()

        selected_rows = torch.searchsorted(classification_indices, candidate_index)
        positive_classes = gt_classes[gt_index]
        vfl_target[selected_rows, positive_classes] = iou_target.detach()
        labels[selected_rows, positive_classes] = 1.0
    else:
        l1_raw = differentiable_zero(boxes_xywh)
        giou_raw = differentiable_zero(boxes_xywh)

    if classification_indices.numel():
        vfl_raw = _VARIFOCAL_LOSS(selected_logits, vfl_target, labels)
    else:
        vfl_raw = differentiable_zero(logits)

    positive_count = logits.new_tensor(float(len(pairs)))
    denominator = max(len(pairs), 1)
    vfl = vfl_raw / denominator
    l1 = l1_raw / denominator
    giou = giou_raw / denominator
    total = vfl + 5.0 * l1 + 2.0 * giou
    return _ImageP2Loss(
        total=total.float(),
        vfl=vfl.float(),
        l1=l1.float(),
        giou=giou.float(),
        positive_count=positive_count,
        classification_indices=classification_indices,
        vfl_target=vfl_target.detach(),
    )


def _centers_inside_any_box(centers: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros(len(centers), dtype=torch.bool, device=centers.device)
    half = boxes[:, None, 2:] / 2
    delta = (centers[None] - boxes[:, None, :2]).abs()
    return (delta <= half).all(-1).any(0)


def _classification_indices(
    *,
    anchor_centers: torch.Tensor,
    gt_boxes: torch.Tensor,
    topk_indices: torch.Tensor,
    assigned_indices: torch.Tensor,
) -> torch.Tensor:
    inside_any_gt = _centers_inside_any_box(anchor_centers, gt_boxes)
    topk_assigned = torch.isin(topk_indices, assigned_indices)
    eligible_topk = topk_indices[topk_assigned | ~inside_any_gt[topk_indices]]
    return torch.unique(torch.cat((eligible_topk, assigned_indices)), sorted=True)
