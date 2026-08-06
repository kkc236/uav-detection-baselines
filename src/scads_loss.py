"""SCADS extensions to the isolated FDR/FGL training criterion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from src.fdr_loss import FDRDetectionLoss, MatchIndices
from src.fdr_math import REG_MAX, cxcywh_to_xyxy
from src.scads import (
    continuous_edge_offsets,
    smallest_covering_support,
    translate_with_project,
)


class SCADSFDRDetectionLoss(FDRDetectionLoss):
    """FDR loss with adaptive target projects and one route supervision term."""

    def __init__(
        self,
        *args: Any,
        support_project_bank: Tensor,
        scads_route_weight: float = 0.05,
        scads_margin_ratio: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if support_project_bank.ndim != 2 or support_project_bank.shape[1] != REG_MAX + 1:
            raise ValueError("SCADS support project bank must have shape [K,33]")
        if scads_route_weight < 0:
            raise ValueError("SCADS route loss weight must be non-negative")
        if not 0 <= scads_margin_ratio < 0.5:
            raise ValueError("SCADS margin ratio must be in [0,0.5)")
        self.register_buffer(
            "support_project_bank",
            support_project_bank.detach().to(dtype=torch.float32).clone(),
        )
        self.scads_route_weight = float(scads_route_weight)
        self.scads_margin_ratio = float(scads_margin_ratio)
        self.last_route_target_counts = torch.zeros(
            support_project_bank.shape[0], dtype=torch.long
        )
        self.last_route_overflow_count = 0
        self.last_route_positive_count = 0

    def _encode_fgl_targets(
        self,
        references: Tensor,
        targets_xyxy: Tensor,
        support_projects: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if support_projects is None:
            raise ValueError("SCADS FGL requires query-specific support projects")
        offsets = continuous_edge_offsets(references, targets_xyxy)
        return translate_with_project(offsets, support_projects, reg_max=REG_MAX)

    def _route_loss(
        self,
        support_logits: Tensor,
        pre_boxes: Tensor,
        gt_bboxes: Tensor,
        matches: MatchIndices,
    ) -> Tensor:
        if support_logits.shape[:-1] != pre_boxes.shape[:-1]:
            raise ValueError("SCADS support logits must align with preliminary boxes")
        if support_logits.shape[-1] != self.support_project_bank.shape[0]:
            raise ValueError("SCADS support logits have the wrong expert count")
        predicted_index, target_index = self._get_index(matches)
        if target_index.numel() == 0:
            self.last_route_target_counts = torch.zeros(
                support_logits.shape[-1], dtype=torch.long
            )
            self.last_route_overflow_count = 0
            self.last_route_positive_count = 0
            return support_logits.sum() * 0.0

        matched_reference = pre_boxes[predicted_index].detach()
        targets_xyxy = cxcywh_to_xyxy(gt_bboxes[target_index])
        offsets = continuous_edge_offsets(matched_reference, targets_xyxy)
        targets, overflow = smallest_covering_support(
            offsets,
            self.support_project_bank,
            margin_ratio=self.scads_margin_ratio,
        )
        self.last_route_target_counts = torch.bincount(
            targets.detach().cpu(),
            minlength=support_logits.shape[-1],
        )
        self.last_route_overflow_count = int(overflow.sum().item())
        self.last_route_positive_count = int(targets.numel())
        return F.cross_entropy(support_logits[predicted_index], targets)

    def forward(
        self,
        preds: tuple[Tensor, Tensor],
        batch: dict[str, Any],
        dn_bboxes: Tensor | None = None,
        dn_scores: Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
        *,
        corner_logits: Tensor | None = None,
        pre_boxes: Tensor | None = None,
        dn_corner_logits: Tensor | None = None,
        dn_pre_boxes: Tensor | None = None,
        normal_match_indices: Sequence[MatchIndices] | None = None,
        support_logits: Tensor | None = None,
        support_projects: Tensor | None = None,
        dn_support_projects: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if corner_logits is not None and support_projects is None:
            raise ValueError("SCADS corner logits require adaptive support projects")
        losses = super().forward(
            preds,
            batch,
            dn_bboxes=dn_bboxes,
            dn_scores=dn_scores,
            dn_meta=dn_meta,
            corner_logits=corner_logits,
            pre_boxes=pre_boxes,
            dn_corner_logits=dn_corner_logits,
            dn_pre_boxes=dn_pre_boxes,
            normal_match_indices=normal_match_indices,
            support_projects=support_projects,
            dn_support_projects=dn_support_projects,
        )
        if support_logits is None or pre_boxes is None:
            raise ValueError("SCADS route supervision requires logits and pre_boxes")
        normal_assignments = self._to_layer_order(
            self._recorded_assignments.get("", [])
        )
        if len(normal_assignments) < 2:
            raise RuntimeError("SCADS route supervision requires decoder assignments")
        route = self._route_loss(
            support_logits,
            pre_boxes,
            batch["bboxes"],
            normal_assignments[-1],
        )
        losses["loss_scads_route"] = route * self.scads_route_weight
        return losses


__all__ = ["SCADSFDRDetectionLoss"]
