from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


PROPOSAL_IOU_THRESHOLDS = (0.3, 0.5, 0.7)


def pairwise_iou_xywh(first: Tensor, second: Tensor) -> Tensor:
    """Class-agnostic pairwise IoU for normalized or pixel cxcywh boxes."""
    if first.ndim != 2 or second.ndim != 2 or first.shape[-1] != 4 or second.shape[-1] != 4:
        raise ValueError("pairwise IoU expects [N,4] and [M,4] cxcywh tensors")
    first_half = first[:, 2:].clamp_min(0) * 0.5
    second_half = second[:, 2:].clamp_min(0) * 0.5
    first_min, first_max = first[:, :2] - first_half, first[:, :2] + first_half
    second_min, second_max = second[:, :2] - second_half, second[:, :2] + second_half
    intersection_min = torch.maximum(first_min[:, None, :], second_min[None, :, :])
    intersection_max = torch.minimum(first_max[:, None, :], second_max[None, :, :])
    intersection = (intersection_max - intersection_min).clamp_min(0).prod(dim=-1)
    first_area = (first_max - first_min).prod(dim=-1)[:, None]
    second_area = (second_max - second_min).prod(dim=-1)[None, :]
    return intersection / (first_area + second_area - intersection).clamp_min(1e-9)


def size_bin_masks(gt_boxes: Tensor, image_size: int = 640) -> dict[str, Tensor]:
    area = gt_boxes[:, 2].clamp_min(0) * gt_boxes[:, 3].clamp_min(0) * float(image_size**2)
    return {
        "all": torch.ones_like(area, dtype=torch.bool),
        "tiny_lt_16sq": area < 16**2,
        "coco_small_lt_32sq": area < 32**2,
        "coco_medium_32_96sq": (area >= 32**2) & (area < 96**2),
        "coco_large_ge_96sq": area >= 96**2,
    }


@dataclass
class ProposalDiagnosticAccumulator:
    thresholds: tuple[float, ...] = PROPOSAL_IOU_THRESHOLDS
    confidence_threshold: float = 0.25
    image_size: int = 640
    objects: dict[str, int] = field(default_factory=dict)
    recalled: dict[str, dict[str, int]] = field(default_factory=dict)
    final_missed: dict[str, int] = field(default_factory=dict)
    recoverable_missed: dict[str, int] = field(default_factory=dict)
    images: int = 0

    def update(
        self,
        proposal_boxes: Tensor,
        gt_boxes: Tensor,
        gt_classes: Tensor,
        final_predictions: Tensor,
    ) -> None:
        if proposal_boxes.shape != (300, 4):
            raise ValueError(f"expected exactly 300 proposal boxes, got {proposal_boxes.shape}")
        if gt_boxes.shape[0] != gt_classes.numel():
            raise ValueError("ground-truth box and class counts differ")
        self.images += 1
        if not gt_boxes.numel():
            return
        proposal_best = pairwise_iou_xywh(gt_boxes, proposal_boxes).max(dim=1).values

        valid_predictions = final_predictions[
            final_predictions[:, 4] >= self.confidence_threshold
        ]
        final_best = gt_boxes.new_zeros(gt_boxes.shape[0])
        if valid_predictions.numel():
            final_ious = pairwise_iou_xywh(gt_boxes, valid_predictions[:, :4])
            same_class = (
                gt_classes.reshape(-1, 1).to(valid_predictions.device)
                == valid_predictions[:, 5].reshape(1, -1).long()
            )
            final_ious = final_ious * same_class.to(final_ious.dtype)
            final_best = final_ious.max(dim=1).values
        missed = final_best < 0.5
        recoverable = missed & (proposal_best >= 0.5)

        for bin_name, mask in size_bin_masks(gt_boxes, self.image_size).items():
            count = int(mask.sum().item())
            self.objects[bin_name] = self.objects.get(bin_name, 0) + count
            self.recalled.setdefault(bin_name, {})
            for threshold in self.thresholds:
                key = f"{threshold:.1f}"
                hits = int((mask & (proposal_best >= threshold)).sum().item())
                self.recalled[bin_name][key] = self.recalled[bin_name].get(key, 0) + hits
            self.final_missed[bin_name] = self.final_missed.get(bin_name, 0) + int(
                (mask & missed).sum().item()
            )
            self.recoverable_missed[bin_name] = self.recoverable_missed.get(
                bin_name, 0
            ) + int((mask & recoverable).sum().item())

    def report(self) -> dict:
        bins = {}
        for bin_name in sorted(self.objects):
            object_count = self.objects[bin_name]
            missed_count = self.final_missed.get(bin_name, 0)
            bins[bin_name] = {
                "objects": object_count,
                "proposal_recall": {
                    threshold: (
                        self.recalled[bin_name].get(threshold, 0) / object_count
                        if object_count
                        else None
                    )
                    for threshold in (f"{value:.1f}" for value in self.thresholds)
                },
                "final_missed_at_conf_0.25_iou_0.5": missed_count,
                "recoverable_missed_proposal_iou_ge_0.5": self.recoverable_missed.get(
                    bin_name, 0
                ),
                "recoverable_missed_ratio": (
                    self.recoverable_missed.get(bin_name, 0) / missed_count
                    if missed_count
                    else None
                ),
            }
        return {
            "images": self.images,
            "proposal_count": 300,
            "class_agnostic_proposal_recall": True,
            "final_detection_confidence": self.confidence_threshold,
            "bins": bins,
        }
