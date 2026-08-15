from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.ioqc_sa_probe import P3SamplingStatistics


EPSILON = 1e-6


@dataclass
class IOQCSATargets:
    boxes: torch.Tensor
    classes: torch.Tensor
    batch_indices: torch.Tensor
    groups: list[int]


@dataclass
class IOQCSALossResult:
    competition: torch.Tensor
    alignment: torch.Tensor
    dense_count: torch.Tensor
    duplicate_count: torch.Tensor
    valid_query_count: torch.Tensor
    p3_mass_min: torch.Tensor
    p3_mass_max: torch.Tensor


def target_scale(boxes: torch.Tensor, *, p3_shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    boxes_fp32 = boxes.float()
    raw = boxes_fp32[..., 2:4].clamp_min(0.0) / math.sqrt(12.0)
    height, width = p3_shape
    floor = torch.tensor((1.0 / width, 1.0 / height), device=boxes.device, dtype=torch.float32)
    return raw, torch.maximum(raw, floor)


def pairwise_xywh_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = first.float()
    second = second.float()
    first_half = first[:, None, 2:4].clamp_min(0.0) * 0.5
    second_half = second[None, :, 2:4].clamp_min(0.0) * 0.5
    first_min = first[:, None, :2] - first_half
    first_max = first[:, None, :2] + first_half
    second_min = second[None, :, :2] - second_half
    second_max = second[None, :, :2] + second_half
    intersection = (torch.minimum(first_max, second_max) - torch.maximum(first_min, second_min)).clamp_min(0.0)
    intersection_area = intersection.prod(dim=-1)
    first_area = (first[:, 2] * first[:, 3]).clamp_min(0.0)[:, None]
    second_area = (second[:, 2] * second[:, 3]).clamp_min(0.0)[None, :]
    union = first_area + second_area - intersection_area
    return intersection_area / union.clamp_min(EPSILON)


def dense_target_mask(targets: IOQCSATargets, *, threshold: float) -> torch.Tensor:
    boxes = targets.boxes.float()
    dense = torch.zeros(len(boxes), device=boxes.device, dtype=torch.bool)
    for image_index, group_size in enumerate(targets.groups):
        if group_size < 2:
            continue
        indices = torch.nonzero(targets.batch_indices == image_index, as_tuple=False).flatten()
        if len(indices) < 2:
            continue
        image_boxes = boxes[indices]
        centers = image_boxes[:, :2]
        distances = torch.cdist(centers, centers, p=2)
        radii = torch.sqrt((image_boxes[:, 2] * image_boxes[:, 3]).clamp_min(0.0))
        normalized = distances / (radii[:, None] + radii[None, :] + EPSILON)
        normalized.fill_diagonal_(float("inf"))
        dense[indices] = normalized.min(dim=1).values < threshold
    return dense


def ensure_finite_losses(**losses: torch.Tensor) -> None:
    invalid = [name for name, value in losses.items() if not bool(torch.isfinite(value).all())]
    if invalid:
        raise FloatingPointError(f"NONFINITE_LOSS: {', '.join(invalid)}")


def ioqc_ramp(epoch: int, total_epochs: int) -> float:
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    ratio = max(0.0, float(epoch) / float(total_epochs))
    if ratio < 0.10:
        return 0.0
    if ratio < 0.15:
        return (ratio - 0.10) / 0.05
    return 1.0


def compute_ioqc_sa_loss(
    *,
    pred_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    statistics: P3SamplingStatistics,
    targets: IOQCSATargets,
    match_indices: list[tuple[torch.Tensor, torch.Tensor]],
    density_threshold: float = 1.0,
    duplicate_threshold: float = 0.10,
) -> IOQCSALossResult:
    if pred_boxes.ndim != 3 or pred_logits.ndim != 3:
        raise ValueError("pred_boxes and pred_logits must have shapes [B, Q, 4] and [B, Q, C]")

    device = pred_boxes.device
    with torch.autocast(device_type=device.type, enabled=False):
        boxes = pred_boxes.float()
        logits = pred_logits.float()
        centers = statistics.center.float()
        extents = statistics.extent.float()
        masses = statistics.p3_mass.float()
        valid = statistics.valid.to(device=device, dtype=torch.bool)
        valid = (
            valid
            & (masses > EPSILON)
            & torch.isfinite(masses)
            & torch.isfinite(centers).all(dim=-1)
            & torch.isfinite(extents).all(dim=-1)
        )

        target_boxes = targets.boxes.to(device=device, dtype=torch.float32)
        target_classes = targets.classes.to(device=device, dtype=torch.long)
        target_batches = targets.batch_indices.to(device=device, dtype=torch.long)
        normalized_targets = IOQCSATargets(target_boxes, target_classes, target_batches, targets.groups)
        dense = dense_target_mask(normalized_targets, threshold=density_threshold)

        zero = (centers.sum() + extents.sum()) * 0.0
        owner_queries = torch.full((len(target_boxes),), -1, device=device, dtype=torch.long)
        for image_index, (source, destination) in enumerate(match_indices):
            source = source.to(device=device, dtype=torch.long)
            destination = destination.to(device=device, dtype=torch.long)
            if len(source):
                owner_queries[destination] = source

        alignment_terms: list[torch.Tensor] = []
        for ground_truth in torch.nonzero(dense, as_tuple=False).flatten():
            image_index = int(target_batches[ground_truth])
            owner = int(owner_queries[ground_truth])
            if owner < 0 or not bool(valid[image_index, owner]):
                continue
            _, stable_scale = target_scale(target_boxes[ground_truth : ground_truth + 1], p3_shape=statistics.p3_shape)
            scale = stable_scale[0]
            center_residual = (centers[image_index, owner] - target_boxes[ground_truth, :2]) / scale
            extent_residual = (extents[image_index, owner] - scale) / scale
            residual = torch.cat((center_residual, extent_residual))
            alignment_terms.append(F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="mean", beta=1.0))

        competition_terms: list[torch.Tensor] = []
        duplicate_count = torch.zeros((), device=device, dtype=torch.long)
        query_count = boxes.shape[1]
        probabilities = torch.sigmoid(logits).detach()
        detached_boxes = boxes.detach()

        for image_index, matches in enumerate(match_indices):
            image_ground_truths = torch.nonzero(target_batches == image_index, as_tuple=False).flatten()
            if len(image_ground_truths) == 0:
                continue
            matched_queries = matches[0].to(device=device, dtype=torch.long)
            unmatched_mask = torch.ones(query_count, device=device, dtype=torch.bool)
            unmatched_mask[matched_queries] = False
            unmatched_queries = torch.nonzero(unmatched_mask & valid[image_index], as_tuple=False).flatten()
            if len(unmatched_queries) == 0:
                continue

            gt_boxes = target_boxes[image_ground_truths]
            gt_classes = target_classes[image_ground_truths]
            quality = (
                probabilities[image_index, unmatched_queries][:, gt_classes]
                * pairwise_xywh_iou(detached_boxes[image_index, unmatched_queries], gt_boxes)
            )
            quality = torch.where(torch.isfinite(quality), quality, torch.full_like(quality, -float("inf")))
            assigned_local_gt = quality.argmax(dim=1)

            for local_gt, ground_truth in enumerate(image_ground_truths):
                if not bool(dense[ground_truth]):
                    continue
                candidates = torch.nonzero(assigned_local_gt == local_gt, as_tuple=False).flatten()
                if len(candidates) == 0:
                    continue
                candidate_quality = quality[candidates, local_gt]
                best_candidate = candidates[candidate_quality.argmax()]
                best_quality = quality[best_candidate, local_gt]
                if not bool(best_quality > duplicate_threshold):
                    continue

                duplicate = unmatched_queries[best_candidate]
                owner = owner_queries[ground_truth]
                if owner < 0 or not bool(valid[image_index, owner]):
                    continue
                _, stable_scale = target_scale(
                    target_boxes[ground_truth : ground_truth + 1], p3_shape=statistics.p3_shape
                )
                scale = stable_scale[0]
                center_difference = (centers[image_index, duplicate] - centers[image_index, owner].detach()).abs() / scale
                extent_difference = (extents[image_index, duplicate] - extents[image_index, owner].detach()).abs() / scale
                distance = torch.cat((center_difference, extent_difference)).mean()
                competition_terms.append(torch.relu(1.0 - distance))
                duplicate_count = duplicate_count + 1

        competition = torch.stack(competition_terms).mean() if competition_terms else zero
        alignment = torch.stack(alignment_terms).mean() if alignment_terms else zero
        ensure_finite_losses(comp=competition, align=alignment)

        valid_masses = masses[valid]
        mass_min = valid_masses.min() if len(valid_masses) else torch.zeros((), device=device)
        mass_max = valid_masses.max() if len(valid_masses) else torch.zeros((), device=device)
        return IOQCSALossResult(
            competition=competition,
            alignment=alignment,
            dense_count=dense.sum(),
            duplicate_count=duplicate_count,
            valid_query_count=valid.sum(),
            p3_mass_min=mass_min,
            p3_mass_max=mass_max,
        )
