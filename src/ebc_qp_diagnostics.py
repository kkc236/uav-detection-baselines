from __future__ import annotations

from dataclasses import dataclass, field

import torch
from scipy.stats import spearmanr

from src.ebc_qp_matching import assign_local_p2, match_centers_inside_boxes
from src.ebc_qp_queries import p2_diversity_statistics, replacement_statistics


@dataclass
class MechanismDiagnosticsAccumulator:
    tiny_radius: float
    active_images: int = 0
    tiny_gt: int = 0
    stock_tiny_covered: int = 0
    local_assigned_gt: int = 0
    n_gain: int = 0
    n_loss: int = 0
    p2_entry_count: int = 0
    ordinary_query_count: int = 0
    positive_boundary_gaps: int = 0
    boundary_gap_count: int = 0
    boundary_gap_sum: float = 0.0
    p2_candidate_count: int = 0
    p2_foreground_count: int = 0
    p2_unique_gt_count: int = 0
    entered_p2_count: int = 0
    assigned_entry_count: int = 0
    low_quality_entry_count: int = 0
    assigned_entry_iou_sum: float = 0.0
    assigned_entry_nwd_sum: float = 0.0
    quality_scores: list[float] = field(default_factory=list)
    quality_ious: list[float] = field(default_factory=list)
    quality_nwds: list[float] = field(default_factory=list)
    quality_head_logits: list[float] = field(default_factory=list)
    quality_head_ious: list[float] = field(default_factory=list)
    quality_head_nwds: list[float] = field(default_factory=list)
    c2_p3_rms_ratio_sum: float = 0.0
    c2_p3_rms_ratio_count: int = 0

    def update(self, state, batch: dict) -> None:
        if not state.competition_active:
            return

        batch_size = int(batch["img"].shape[0])
        image_height, image_width = batch["img"].shape[-2:]
        batch_index = batch["batch_idx"].reshape(-1).long()
        boxes = batch["bboxes"].reshape(-1, 4).float()
        classes = batch["cls"].reshape(-1).long()
        if state.c2_p3_rms_ratio is not None:
            ratios = state.c2_p3_rms_ratio.float()
            self.c2_p3_rms_ratio_sum += float(ratios.sum())
            self.c2_p3_rms_ratio_count += ratios.numel()

        for image_index in range(batch_size):
            self.active_images += 1
            image_boxes = boxes[batch_index == image_index]
            image_classes = classes[batch_index == image_index]
            radius = (
                (image_boxes[:, 2] * image_width) * (image_boxes[:, 3] * image_height)
            ).clamp_min(0).sqrt()
            tiny_mask = radius <= self.tiny_radius
            tiny_boxes = image_boxes[tiny_mask]
            tiny_classes = image_classes[tiny_mask]
            self.tiny_gt += int(tiny_mask.sum())

            replacement = replacement_statistics(
                state.stock_centers[image_index],
                state.final_centers[image_index],
                image_boxes,
                tiny_mask,
            )
            self.n_gain += replacement.n_gain
            self.n_loss += replacement.n_loss

            stock_match = match_centers_inside_boxes(state.stock_centers[image_index], image_boxes)
            tiny_indices = set(torch.where(tiny_mask)[0].tolist())
            self.stock_tiny_covered += len(set(stock_match.covered_gt.tolist()) & tiny_indices)

            p2_centers = state.p2_top_centers[image_index]
            diversity = p2_diversity_statistics(p2_centers, tiny_boxes)
            self.p2_candidate_count += len(p2_centers)
            self.p2_foreground_count += diversity.foreground_at_50
            self.p2_unique_gt_count += diversity.unique_gt_at_50

            entered = state.final_sources[image_index] == 1
            entered_count = int(entered.sum())
            self.p2_entry_count += entered_count
            self.entered_p2_count += entered_count
            self.ordinary_query_count += int(state.ordinary_query_count)
            if entered_count:
                gaps = state.final_ranking_score[image_index][entered] - state.stock_boundary[image_index]
                self.boundary_gap_sum += float(gaps.float().sum())
                self.boundary_gap_count += entered_count
                self.positive_boundary_gaps += int((gaps > 0).sum())
            self._update_quality_diagnostics(state, image_index, tiny_boxes, tiny_classes, entered)

    def _update_quality_diagnostics(
        self,
        state,
        image_index: int,
        tiny_boxes: torch.Tensor,
        tiny_classes: torch.Tensor,
        entered: torch.Tensor,
    ) -> None:
        if (
            state.p2_all_boxes is None
            or state.p2_all_logits is None
            or state.p2_shape is None
            or state.p2_valid_mask is None
        ):
            return

        height, width = state.p2_shape
        local = assign_local_p2(
            height=height,
            width=width,
            boxes=tiny_boxes,
            valid_mask=state.p2_valid_mask.reshape(-1),
            radius=1,
        )
        pairs = local.pairs
        self.local_assigned_gt += len(pairs)
        if pairs.numel():
            gt_indices = pairs[:, 0]
            candidate_indices = pairs[:, 1]
            predicted_boxes = state.p2_all_boxes[image_index, candidate_indices]
            target_boxes = tiny_boxes[gt_indices]
            ious = _aligned_iou_xywh(predicted_boxes, target_boxes)
            nwds = _aligned_nwd_xywh(predicted_boxes, target_boxes)
            scores = state.p2_all_logits[
                image_index,
                candidate_indices,
                tiny_classes[gt_indices],
            ].float()
            self.quality_scores.extend(scores.tolist())
            self.quality_ious.extend(ious.tolist())
            self.quality_nwds.extend(nwds.tolist())
            if state.p2_all_quality_logits is not None:
                quality_logits = state.p2_all_quality_logits[image_index, candidate_indices].float()
                self.quality_head_logits.extend(quality_logits.tolist())
                self.quality_head_ious.extend(ious.tolist())
                self.quality_head_nwds.extend(nwds.tolist())

        assignment = {int(candidate): int(gt) for gt, candidate in pairs.tolist()}
        entry_indices = state.final_source_indices[image_index][entered].tolist()
        entry_boxes = state.final_boxes[image_index][entered]
        for entry_box, candidate_index in zip(entry_boxes, entry_indices):
            gt_index = assignment.get(int(candidate_index))
            if gt_index is None:
                self.low_quality_entry_count += 1
                continue
            iou = float(_aligned_iou_xywh(entry_box[None], tiny_boxes[gt_index][None])[0])
            nwd = float(_aligned_nwd_xywh(entry_box[None], tiny_boxes[gt_index][None])[0])
            self.assigned_entry_count += 1
            self.assigned_entry_iou_sum += iou
            self.assigned_entry_nwd_sum += nwd
            self.low_quality_entry_count += int(iou < 0.1)

    def compute(self) -> dict[str, float | int]:
        duplicate_rate = (
            0.0
            if self.p2_foreground_count == 0
            else 1.0 - _ratio(self.p2_unique_gt_count, self.p2_foreground_count)
        )
        background_rate = (
            0.0
            if self.p2_candidate_count == 0
            else 1.0 - _ratio(self.p2_foreground_count, self.p2_candidate_count)
        )
        unassigned_entry_rate = (
            0.0
            if self.entered_p2_count == 0
            else 1.0 - _ratio(self.assigned_entry_count, self.entered_p2_count)
        )
        return {
            "active_images": self.active_images,
            "tiny_gt": self.tiny_gt,
            "stock_top300_coverage": _ratio(self.stock_tiny_covered, self.tiny_gt),
            "local_assign_rate": _ratio(self.local_assigned_gt, self.tiny_gt),
            "p2_entry_count": self.p2_entry_count,
            "n_gain": self.n_gain,
            "n_loss": self.n_loss,
            "v_replace": self.n_gain - self.n_loss,
            "effective_p2_entry_rate": _ratio(self.p2_entry_count, self.ordinary_query_count),
            "boundary_gap_mean": _ratio(self.boundary_gap_sum, self.boundary_gap_count),
            "boundary_gap_positive_ratio": _ratio(self.positive_boundary_gaps, self.boundary_gap_count),
            "p2_foreground_at_50": self.p2_foreground_count,
            "p2_unique_gt_at_50": self.p2_unique_gt_count,
            "p2_duplicate_rate_at_50": duplicate_rate,
            "p2_background_rate_at_50": background_rate,
            "score_iou_spearman": _spearman(self.quality_scores, self.quality_ious),
            "score_nwd_spearman": _spearman(self.quality_scores, self.quality_nwds),
            "score_quality_sample_count": len(self.quality_scores),
            "quality_logit_iou_spearman": _spearman(self.quality_head_logits, self.quality_head_ious),
            "quality_logit_nwd_spearman": _spearman(self.quality_head_logits, self.quality_head_nwds),
            "quality_logit_sample_count": len(self.quality_head_logits),
            "assigned_entry_mean_iou": _ratio(self.assigned_entry_iou_sum, self.assigned_entry_count),
            "assigned_entry_mean_nwd": _ratio(self.assigned_entry_nwd_sum, self.assigned_entry_count),
            "unassigned_entry_rate": unassigned_entry_rate,
            "low_quality_entry_rate": _ratio(self.low_quality_entry_count, self.entered_p2_count),
            "c2_p3_rms_ratio": _ratio(self.c2_p3_rms_ratio_sum, self.c2_p3_rms_ratio_count),
        }


def _ratio(numerator: float | int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator) / denominator


def _aligned_iou_xywh(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_half = first[:, 2:] / 2
    second_half = second[:, 2:] / 2
    first_min, first_max = first[:, :2] - first_half, first[:, :2] + first_half
    second_min, second_max = second[:, :2] - second_half, second[:, :2] + second_half
    intersection = (torch.minimum(first_max, second_max) - torch.maximum(first_min, second_min)).clamp_min(0).prod(-1)
    union = first[:, 2:].prod(-1) + second[:, 2:].prod(-1) - intersection
    return intersection / union.clamp_min(1e-12)


def _aligned_nwd_xywh(first: torch.Tensor, second: torch.Tensor, constant: float = 12.8 / 640) -> torch.Tensor:
    center_distance = (first[:, :2] - second[:, :2]).square().sum(-1)
    shape_distance = (first[:, 2:] - second[:, 2:]).square().sum(-1) / 4
    return torch.exp(-(center_distance + shape_distance).clamp_min(0).sqrt() / constant)


def _spearman(first: list[float], second: list[float]) -> float | None:
    if len(first) < 3 or len(set(first)) < 2 or len(set(second)) < 2:
        return None
    result = float(spearmanr(first, second).statistic)
    return result if result == result else None
