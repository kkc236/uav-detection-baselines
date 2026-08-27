"""Isolated stock RT-DETR loss extension for FDR/FGL supervision.

The stock Ultralytics 8.4.90 VFL, L1, GIoU, auxiliary, and denoising
losses remain delegated to :class:`RTDETRDetectionLoss`.  This module only
records the assignments that stock loss already computed and consumes them for
FGL and optional preliminary-box localization supervision.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou

from src.fdr_math import (
    REG_MAX,
    REG_SCALE,
    UP,
    bbox2distance,
    cxcywh_to_xyxy,
    fine_grained_localization_loss,
)


MatchIndices = list[tuple[Tensor, Tensor]]


def _clone_matches(matches: MatchIndices) -> MatchIndices:
    return [
        (source.detach().clone(), target.detach().clone())
        for source, target in matches
    ]


def stock_loss_subtotal(losses: Mapping[str, Tensor]) -> Tensor:
    """Sum only unchanged stock classification/L1/GIoU loss keys."""

    values = [
        value
        for key, value in losses.items()
        if key.startswith(("loss_class", "loss_bbox", "loss_giou"))
        and "_pre" not in key
    ]
    if not values:
        return torch.tensor(0.0)
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total


def layerwise_reliability_shrinkage(
    matched_iou: Tensor,
    batch_indices: Tensor,
    *,
    layer_index: int,
    num_layers: int,
    alpha0: float,
    eligible_mask: Tensor | None = None,
) -> Tensor:
    """Shrink eligible detached IoUs toward their same-image layer mean."""

    if matched_iou.ndim != 1:
        raise ValueError("matched_iou must be one-dimensional")
    if batch_indices.ndim != 1 or batch_indices.shape != matched_iou.shape:
        raise ValueError("batch_indices must match the one-dimensional IoU shape")
    if num_layers < 2:
        raise ValueError("num_layers must be at least two")
    if layer_index < 0 or layer_index >= num_layers:
        raise ValueError("layer_index must identify an existing decoder layer")
    if not math.isfinite(alpha0) or alpha0 < 0.0 or alpha0 >= 1.0:
        raise ValueError("alpha0 must satisfy 0 <= alpha0 < 1")
    if not matched_iou.is_floating_point():
        raise ValueError("matched_iou must be floating point")
    batch_indices = batch_indices.detach().to(device=matched_iou.device)
    if eligible_mask is not None:
        if (
            eligible_mask.dtype != torch.bool
            or eligible_mask.ndim != 1
            or eligible_mask.shape != matched_iou.shape
        ):
            raise ValueError("eligible_mask must be boolean and match matched_iou")
        eligible_mask = eligible_mask.to(device=matched_iou.device)

    quality = matched_iou.detach()
    if alpha0 == 0.0 or layer_index == num_layers - 1:
        return quality

    quality_fp32 = quality.float()
    if quality_fp32.numel() == 0:
        return quality_fp32
    alpha = float(alpha0) * (1.0 - float(layer_index) / float(num_layers - 1))
    weights = quality_fp32.clone()
    eligible = (
        torch.ones_like(batch_indices, dtype=torch.bool)
        if eligible_mask is None
        else eligible_mask
    )
    for batch_index in torch.unique(batch_indices):
        mask = (batch_indices == batch_index) & eligible
        group_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if int(group_indices.numel()) <= 1:
            continue
        image_quality = quality_fp32[group_indices]
        image_weights = (
            (1.0 - alpha) * image_quality + alpha * image_quality.mean()
        )
        correction_index = int(torch.argmax(image_weights))
        for _ in range(2):
            residual = image_quality.sum() - image_weights.sum()
            image_weights[correction_index] += residual
        weights[group_indices] = image_weights
        image_indices = torch.nonzero(
            batch_indices == batch_index, as_tuple=False
        ).flatten()
        global_correction_index = group_indices[correction_index]
        for _ in range(2):
            residual = (
                quality_fp32[image_indices].double().sum()
                - weights[image_indices].double().sum()
            )
            weights[global_correction_index] += residual.to(dtype=weights.dtype)
    return weights


def representable_fgl_targets(target_indices: Tensor) -> Tensor:
    """Select boxes whose four adjacent-bin targets avoid both boundaries."""

    if target_indices.ndim != 1 or target_indices.numel() % 4:
        raise ValueError("target_indices must contain four flattened box edges")
    box_edges = target_indices.detach().reshape(-1, 4)
    return (box_edges > 0.0).all(dim=1) & (box_edges < REG_MAX - 1).all(dim=1)


def edge_adaptive_fgl_weights(
    corner_logits: Tensor,
    target_indices: Tensor,
    left_weight: Tensor,
    right_weight: Tensor,
    edge_iou: Tensor,
) -> Tensor:
    """Redistribute detached box reliability across its four edge targets."""

    if corner_logits.ndim != 2:
        raise ValueError("corner_logits must have shape [matched_edges, bins]")
    edge_count, bin_count = corner_logits.shape
    expected_shape = (edge_count,)
    for name, tensor in (
        ("target_indices", target_indices),
        ("left_weight", left_weight),
        ("right_weight", right_weight),
        ("edge_iou", edge_iou),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    if edge_count % 4:
        raise ValueError("matched edge count must be divisible by four")
    if edge_count == 0:
        return edge_iou.detach()

    left_index = target_indices.long()
    right_index = left_index + 1
    if torch.any(left_index < 0) or torch.any(right_index >= bin_count):
        raise ValueError("adjacent target-bin indices are outside corner_logits")

    probabilities = corner_logits.detach().softmax(dim=-1)
    target_mass = (
        probabilities.gather(1, left_index.unsqueeze(1)).squeeze(1)
        * left_weight.detach()
        + probabilities.gather(1, right_index.unsqueeze(1)).squeeze(1)
        * right_weight.detach()
    )
    difficulty = (1.0 - target_mass).clamp_min(1e-6).reshape(-1, 4)
    mean_difficulty = difficulty.mean(dim=1, keepdim=True).clamp_min(1e-6)
    modulation = (difficulty / mean_difficulty).clamp(0.5, 2.0)
    return edge_iou.detach() * modulation.reshape(-1)


def adjacent_bin_fgl(
    corner_logits: Tensor,
    target_indices: Tensor,
    left_weight: Tensor,
    right_weight: Tensor,
    matched_iou: Tensor,
    *,
    avg_factor: float | Tensor,
) -> Tensor:
    """Apply the pinned adjacent-bin FGL primitive with detached IoU weight."""

    return fine_grained_localization_loss(
        corner_logits,
        target_indices,
        right_weight,
        left_weight,
        weight=matched_iou.detach(),
        avg_factor=avg_factor,
    )


class FDRDetectionLoss(RTDETRDetectionLoss):
    """Stock RT-DETR criterion plus isolated FGL/pre-box localization."""

    def __init__(
        self,
        *args: Any,
        fgl_weight: float = 0.15,
        supervise_pre_boxes: bool = True,
        supervise_dn_fdr: bool = True,
        edge_adaptive_fgl: bool = False,
        reliability_shrinkage_alpha: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if fgl_weight < 0:
            raise ValueError("fgl_weight must be non-negative")
        if (
            not math.isfinite(reliability_shrinkage_alpha)
            or reliability_shrinkage_alpha < 0.0
            or reliability_shrinkage_alpha >= 1.0
        ):
            raise ValueError(
                "reliability_shrinkage_alpha must satisfy 0 <= alpha < 1"
            )
        if edge_adaptive_fgl and reliability_shrinkage_alpha > 0.0:
            raise ValueError("edge-adaptive FGL and LRS are mutually exclusive")
        self.fgl_weight = float(fgl_weight)
        self.supervise_pre_boxes = bool(supervise_pre_boxes)
        self.supervise_dn_fdr = bool(supervise_dn_fdr)
        self.edge_adaptive_fgl = bool(edge_adaptive_fgl)
        self.reliability_shrinkage_alpha = float(reliability_shrinkage_alpha)
        self.stock_match_calls = 0
        self.fgl_extra_match_calls = 0
        self._normal_assignment_queue: list[MatchIndices] | None = None
        self._recorded_assignments: dict[str, list[MatchIndices]] = {}

    def _get_loss(
        self,
        pred_bboxes: Tensor,
        pred_scores: Tensor,
        gt_bboxes: Tensor,
        gt_cls: Tensor,
        gt_groups: list[int],
        masks: Tensor | None = None,
        gt_mask: Tensor | None = None,
        postfix: str = "",
        match_indices: MatchIndices | None = None,
    ) -> dict[str, Tensor]:
        if match_indices is None:
            if postfix == "" and self._normal_assignment_queue is not None:
                if not self._normal_assignment_queue:
                    raise RuntimeError("normal stock assignment queue was exhausted")
                match_indices = self._normal_assignment_queue.pop(0)
            else:
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
                    self.stock_match_calls += 1

        self._recorded_assignments.setdefault(postfix, []).append(
            _clone_matches(match_indices)
        )
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

    @staticmethod
    def _to_layer_order(call_order: Sequence[MatchIndices]) -> list[MatchIndices]:
        """Convert stock main-then-aux call order to decoder layer order."""

        if len(call_order) <= 1:
            return list(call_order)
        return [*call_order[1:], call_order[0]]

    @staticmethod
    def _validate_layer_tensor(
        name: str,
        tensor: Tensor,
        pred_bboxes: Tensor,
        *,
        last_dimension: int,
    ) -> None:
        expected_prefix = pred_bboxes.shape[:3]
        if tensor.shape[:3] != expected_prefix or tensor.shape[-1] != last_dimension:
            raise ValueError(
                f"{name} must have shape {tuple(expected_prefix)} + ({last_dimension},)"
            )

    def _fgl_for_layer(
        self,
        corner_logits: Tensor,
        pred_bboxes: Tensor,
        pre_boxes: Tensor,
        gt_bboxes: Tensor,
        matches: MatchIndices,
        *,
        layer_index: int,
        num_layers: int,
        apply_reliability_shrinkage: bool,
    ) -> Tensor:
        predicted_index, target_index = self._get_index(matches)
        if target_index.numel() == 0:
            return corner_logits.sum() * 0.0

        matched_logits = corner_logits[predicted_index].reshape(
            -1, REG_MAX + 1
        )
        matched_reference = pre_boxes[predicted_index].detach()
        matched_targets_cxcywh = gt_bboxes[target_index]
        target_indices, right_weight, left_weight = bbox2distance(
            matched_reference,
            cxcywh_to_xyxy(matched_targets_cxcywh),
            REG_MAX,
            REG_SCALE,
            UP,
        )
        matched_boxes = pred_bboxes[predicted_index]
        matched_iou = bbox_iou(
            matched_boxes.detach(), matched_targets_cxcywh, xywh=True
        ).squeeze(-1)
        if (
            apply_reliability_shrinkage
            and self.reliability_shrinkage_alpha > 0.0
            and num_layers > 1
        ):
            matched_iou = layerwise_reliability_shrinkage(
                matched_iou,
                predicted_index[0],
                layer_index=layer_index,
                num_layers=num_layers,
                alpha0=self.reliability_shrinkage_alpha,
                eligible_mask=representable_fgl_targets(target_indices),
            )
        edge_iou = matched_iou.repeat_interleave(4)
        if self.edge_adaptive_fgl:
            edge_iou = edge_adaptive_fgl_weights(
                matched_logits,
                target_indices,
                left_weight,
                right_weight,
                edge_iou,
            )
        return adjacent_bin_fgl(
            matched_logits,
            target_indices,
            left_weight,
            right_weight,
            edge_iou,
            avg_factor=max(int(target_index.numel()), 1),
        )

    def _fgl_group(
        self,
        corner_logits: Tensor,
        pred_bboxes: Tensor,
        pre_boxes: Tensor,
        gt_bboxes: Tensor,
        assignments: Sequence[MatchIndices],
        *,
        postfix: str,
    ) -> dict[str, Tensor]:
        self._validate_layer_tensor(
            "corner_logits",
            corner_logits,
            pred_bboxes,
            last_dimension=4 * (REG_MAX + 1),
        )
        if pre_boxes.shape != pred_bboxes.shape[1:]:
            raise ValueError(
                f"pre_boxes must have shape {tuple(pred_bboxes.shape[1:])}"
            )
        if len(assignments) != pred_bboxes.shape[0]:
            raise ValueError("FGL requires one stock assignment per prediction layer")

        per_layer = [
            self._fgl_for_layer(
                corner_logits[layer],
                pred_bboxes[layer],
                pre_boxes,
                gt_bboxes,
                assignments[layer],
                layer_index=layer,
                num_layers=int(pred_bboxes.shape[0]),
                apply_reliability_shrinkage=postfix == "",
            )
            for layer in range(pred_bboxes.shape[0])
        ]
        main = per_layer[-1] * self.fgl_weight
        if len(per_layer) > 1:
            auxiliary = torch.stack(per_layer[:-1]).sum() * self.fgl_weight
        else:
            auxiliary = corner_logits[:0].sum() * self.fgl_weight
        return {
            f"loss_fgl{postfix}": main,
            f"loss_fgl_aux{postfix}": auxiliary,
        }

    def pre_box_localization_loss(
        self,
        pre_boxes: Tensor,
        batch: dict[str, Any],
        match_indices: MatchIndices,
        *,
        postfix: str = "_pre",
    ) -> dict[str, Tensor]:
        """Apply stock-weighted L1/GIoU to pre-boxes without matching again."""

        predicted_index, target_index = self._get_index(match_indices)
        if target_index.numel() == 0:
            zero = pre_boxes.sum() * 0.0
            return {
                f"loss_bbox{postfix}": zero,
                f"loss_giou{postfix}": zero,
            }
        return self._get_loss_bbox(
            pre_boxes[predicted_index],
            batch["bboxes"][target_index],
            postfix=postfix,
        )

    def normal_assignment_snapshot(self) -> list[MatchIndices]:
        """Return detached stock assignments in encoder-then-decoder order."""

        return [
            _clone_matches(matches)
            for matches in self._to_layer_order(
                self._recorded_assignments.get("", [])
            )
        ]

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
    ) -> dict[str, Tensor]:
        """Compute full stock loss, then decoder-only FGL/pre-box losses."""

        pred_bboxes, _ = preds
        layer_count = int(pred_bboxes.shape[0])
        if normal_match_indices is not None:
            if len(normal_match_indices) != layer_count:
                raise ValueError("expected one normal assignment per prediction layer")
            layer_order = [_clone_matches(item) for item in normal_match_indices]
            self._normal_assignment_queue = [
                layer_order[-1],
                *layer_order[:-1],
            ]
        else:
            self._normal_assignment_queue = None

        self.stock_match_calls = 0
        self.fgl_extra_match_calls = 0
        self._recorded_assignments = {}
        try:
            losses = super().forward(
                preds,
                batch,
                dn_bboxes=dn_bboxes,
                dn_scores=dn_scores,
                dn_meta=dn_meta,
            )
        finally:
            unconsumed_assignments = self._normal_assignment_queue
            self._normal_assignment_queue = None
        if unconsumed_assignments:
            raise RuntimeError("not all supplied normal stock assignments were consumed")

        normal_assignments = self._to_layer_order(
            self._recorded_assignments.get("", [])
        )
        if len(normal_assignments) < 2:
            if corner_logits is not None or (
                self.supervise_pre_boxes and pre_boxes is not None
            ):
                raise ValueError(
                    "normal FDR supervision requires encoder plus decoder stock layers"
                )
            decoder_bboxes = pred_bboxes[:0]
            decoder_assignments: list[MatchIndices] = []
        else:
            # Ultralytics prepends encoder predictions to six decoder layers.
            # FDR corners and preliminary boxes belong to decoder layers only.
            decoder_bboxes = pred_bboxes[1:]
            decoder_assignments = normal_assignments[1:]
        if corner_logits is not None:
            if pre_boxes is None:
                raise ValueError("pre_boxes are required when corner_logits are supplied")
            losses.update(
                self._fgl_group(
                    corner_logits,
                    decoder_bboxes,
                    pre_boxes,
                    batch["bboxes"],
                    decoder_assignments,
                    postfix="",
                )
            )

        if self.supervise_pre_boxes and pre_boxes is not None:
            losses.update(
                self.pre_box_localization_loss(
                    pre_boxes,
                    batch,
                    decoder_assignments[0],
                    postfix="_pre",
                )
            )

        if dn_meta is not None and self.supervise_dn_fdr:
            denoising_assignments = self._to_layer_order(
                self._recorded_assignments.get("_dn", [])
            )
            if dn_corner_logits is not None:
                if dn_bboxes is None or dn_pre_boxes is None:
                    raise ValueError(
                        "dn_bboxes and dn_pre_boxes are required with dn_corner_logits"
                    )
                losses.update(
                    self._fgl_group(
                        dn_corner_logits,
                        dn_bboxes,
                        dn_pre_boxes,
                        batch["bboxes"],
                        denoising_assignments,
                        postfix="_dn",
                    )
                )
            if self.supervise_pre_boxes and dn_pre_boxes is not None:
                if not denoising_assignments:
                    raise RuntimeError("fixed denoising assignments are unavailable")
                losses.update(
                    self.pre_box_localization_loss(
                        dn_pre_boxes,
                        batch,
                        denoising_assignments[0],
                        postfix="_pre_dn",
                    )
                )
        return losses

    def stock_plus_fgl(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        """Explicit entry point for the isolated stock-plus-FDR extension."""

        return self.forward(*args, **kwargs)


__all__ = [
    "FDRDetectionLoss",
    "MatchIndices",
    "adjacent_bin_fgl",
    "edge_adaptive_fgl_weights",
    "layerwise_reliability_shrinkage",
    "representable_fgl_targets",
    "stock_loss_subtotal",
]
