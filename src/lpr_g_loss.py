"""Stock-match recording and isolated localization loss for LPR-G."""

from __future__ import annotations

from typing import Any

import torch
from ultralytics.models.utils.loss import RTDETRDetectionLoss


class MatchRecordingRTDETRDetectionLoss(RTDETRDetectionLoss):
    """Preserve stock losses while exposing the normal main-layer assignment."""

    last_stock_match_indices: list[tuple[torch.Tensor, torch.Tensor]] | None
    normal_match_calls: int
    _capture_normal_match: bool

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        self.last_stock_match_indices = None
        self.normal_match_calls = 0
        self._capture_normal_match = True
        try:
            return super().forward(
                preds,
                batch,
                dn_bboxes=dn_bboxes,
                dn_scores=dn_scores,
                dn_meta=dn_meta,
            )
        finally:
            self._capture_normal_match = False

    def _get_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
        postfix: str = "",
        match_indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes,
                pred_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=masks,
                gt_mask=gt_mask,
            )
            if postfix == "":
                self.normal_match_calls += 1
                if self._capture_normal_match and self.last_stock_match_indices is None:
                    self.last_stock_match_indices = [
                        (source.detach().clone(), target.detach().clone())
                        for source, target in match_indices
                    ]
        return super()._get_loss(
            pred_bboxes,
            pred_scores,
            gt_bboxes,
            gt_cls,
            gt_groups,
            masks=masks,
            gt_mask=gt_mask,
            postfix=postfix,
            match_indices=match_indices,
        )

    def refinement_loss(
        self,
        refined_bboxes: torch.Tensor,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Calculate only L1/GIoU using the recorded stock assignment."""
        if self.last_stock_match_indices is None:
            raise RuntimeError("stock normal-query match is unavailable")
        predicted_index, target_index = self._get_index(self.last_stock_match_indices)
        predicted = refined_bboxes[predicted_index]
        target = batch["bboxes"][target_index]
        return self._get_loss_bbox(predicted, target, postfix="_refine")
