"""Independent local-expert and quality-gated BPDD capacity-v2 integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from ultralytics.utils import RANK

from src.bpdd_capacity_loss import CapacityBPDDOptions, quality_gated_capacity_distillation
from src.fdr_loss import FDRDetectionLoss, MatchIndices
from src.rtdetr_fdr import (
    FDRRTDETRDetectionModel,
    FDRTrainer,
    FDRTrainingEvidence,
    _load_initial_state,
)


ROOT = Path(__file__).resolve().parents[1]
CAPACITY_EXPERT_CFG = ROOT / "configs" / "rtdetr-l-lrs-gfdr-capacity-v2-expert.yaml"
CAPACITY_BPDD_CFG = ROOT / "configs" / "rtdetr-l-lrs-gfdr-capacity-v2-bpdd.yaml"
_OPTION_KEYS = {
    "enabled", "loc_weight", "cls_weight", "loc_margin", "cls_margin",
    "loc_tau", "cls_tau", "warmup_start", "warmup_end",
}


def _parse_capacity_options(payload: dict[str, Any]) -> CapacityBPDDOptions:
    unknown = set(payload) - _OPTION_KEYS
    if unknown:
        raise ValueError(f"unknown capacity BPDD options: {sorted(unknown)}")
    return CapacityBPDDOptions(**payload)


def _assignment_triple(criterion: FDRDetectionLoss, matches: MatchIndices):
    predicted, target = criterion._get_index(matches)
    return predicted[0], predicted[1], target


class CapacityBPDDDetectionLoss(FDRDetectionLoss):
    """Stock/FGL plus fixed-assignment expert loss and optional capacity BPDD."""

    def __init__(self, *args, capacity_options: CapacityBPDDOptions, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.capacity_options = capacity_options
        self.capacity_bpdd_runtime_enabled = True
        self.last_capacity_statistics: dict[str, Tensor] = {}

    def forward(
        self,
        *args,
        expert_boxes: Tensor | None = None,
        expert_scores: Tensor | None = None,
        expert_corners: Tensor | None = None,
        student_classes: Tensor | None = None,
        completed_epoch: int = 0,
        **kwargs,
    ) -> dict[str, Tensor]:
        losses = super().forward(*args, **kwargs)
        self.last_capacity_statistics = {}
        if expert_boxes is None:
            return losses
        if expert_scores is None or expert_corners is None or student_classes is None:
            raise ValueError("capacity expert evidence is incomplete")
        assignments = self.normal_assignment_snapshot()
        if len(assignments) != 7:
            raise ValueError("capacity v2 requires encoder plus six decoder assignments")
        decoder_assignments = assignments[1:]
        batch = args[1] if len(args) > 1 else kwargs.get("batch")
        pre_boxes = kwargs.get("pre_boxes")
        if not isinstance(batch, dict) or not isinstance(pre_boxes, Tensor):
            raise TypeError("capacity v2 requires target batch and FDR reference")
        losses.update(
            self.fixed_assignment_expert_loss(
                expert_boxes,
                expert_scores,
                expert_corners,
                pre_boxes,
                batch,
                decoder_assignments[-1],
            )
        )
        layer_matches = [_assignment_triple(self, match) for match in decoder_assignments]
        enabled = self.capacity_options.enabled and self.capacity_bpdd_runtime_enabled
        result = quality_gated_capacity_distillation(
            kwargs["corner_logits"],
            student_classes,
            expert_corners.detach(),
            expert_scores.detach(),
            pre_boxes,
            batch["bboxes"],
            batch["cls"],
            layer_matches,
            layer_matches[-1],
            CapacityBPDDOptions(**{**self.capacity_options.__dict__, "enabled": enabled}),
            epoch=completed_epoch,
        )
        losses["loss_bpdd_capacity_loc"] = result.localization_loss
        losses["loss_bpdd_capacity_cls"] = result.classification_loss
        self.last_capacity_statistics = {
            name: value.detach() for name, value in result.statistics.items()
        }
        return losses


class CapacityBPDDDetectionModel(FDRRTDETRDetectionModel):
    """LRS-GFDR with a separately supervised local expert."""

    def __init__(self, cfg=CAPACITY_BPDD_CFG, *args, **kwargs) -> None:
        super().__init__(cfg=cfg, *args, **kwargs)
        payload = self.yaml.get("capacity_bpdd_loss")
        if not isinstance(payload, dict):
            raise TypeError("capacity v2 YAML requires capacity_bpdd_loss")
        self.capacity_options = _parse_capacity_options(dict(payload))
        self.completed_epoch = 0
        self.last_capacity_statistics: dict[str, Tensor] = {}
        if not self.fdr.local_expert_preserve_base or self.fdr.local_expert is None:
            raise ValueError("capacity v2 requires the preserved local expert")

    def set_completed_epoch(self, value: int) -> None:
        self.completed_epoch = int(value)

    def init_criterion(self) -> CapacityBPDDDetectionLoss:
        return CapacityBPDDDetectionLoss(
            nc=self.nc,
            use_vfl=True,
            fgl_weight=float(self.fdr_loss_options.get("fgl_weight", 0.15)),
            supervise_pre_boxes=False,
            supervise_dn_fdr=False,
            edge_adaptive_fgl=False,
            reliability_shrinkage_alpha=float(
                self.fdr_loss_options.get("reliability_shrinkage_alpha", 0.25)
            ),
            capacity_options=self.capacity_options,
        )

    def _criterion_extra_kwargs(
        self,
        dec_bboxes: Tensor,
        dec_scores: Tensor,
        evidence: FDRTrainingEvidence,
    ) -> dict[str, Any]:
        del dec_bboxes, evidence
        if not self.training:
            return {}
        expert = self.fdr.last_expert_prediction
        if expert is None:
            raise RuntimeError("capacity expert prediction was not retained")
        return {
            "expert_boxes": expert.boxes,
            "expert_scores": expert.classes,
            "expert_corners": expert.corners,
            "student_classes": dec_scores,
            "completed_epoch": self.completed_epoch,
        }

    def loss(self, batch: dict[str, Tensor], preds: tuple | None = None):
        if hasattr(self, "criterion"):
            if not isinstance(self.criterion, CapacityBPDDDetectionLoss):
                raise TypeError("capacity criterion was unexpectedly replaced")
            self.criterion.capacity_bpdd_runtime_enabled = bool(self.training)
        result = super().loss(batch, preds)
        self.last_capacity_statistics = {
            name: value.detach()
            for name, value in self.criterion.last_capacity_statistics.items()
        }
        return result


class CapacityBPDDTrainer(FDRTrainer):
    """Fresh-only trainer with disjoint common/FDR/expert gradient groups."""

    def gradient_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        groups: dict[str, list[torch.nn.Parameter]] = {
            "gradient_norm": [], "fdr_gradient_norm": [], "expert_gradient_norm": []
        }
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".decoder.local_expert." in name:
                groups["expert_gradient_norm"].append(parameter)
            elif any(marker in name for marker in (
                ".dec_bbox_head.", ".decoder.pre_bbox_head.",
                ".decoder.distribution_feedback.",
            )):
                groups["fdr_gradient_norm"].append(parameter)
            else:
                groups["gradient_norm"].append(parameter)
        if any(not values for values in groups.values()):
            raise RuntimeError("capacity v2 gradient partition is incomplete")
        identifiers = [id(value) for values in groups.values() for value in values]
        expected = {id(value) for value in self.model.parameters() if value.requires_grad}
        if len(identifiers) != len(set(identifiers)) or set(identifiers) != expected:
            raise RuntimeError("capacity v2 gradient partition is not exhaustive")
        return groups

    def get_model(self, cfg=None, weights=None, verbose=True) -> CapacityBPDDDetectionModel:
        if weights is not None:
            raise ValueError("capacity v2 arms are fresh-only")
        model = CapacityBPDDDetectionModel(
            cfg or CAPACITY_BPDD_CFG,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000 + self.experiment_seed,
        )
        _load_initial_state(
            model,
            getattr(self, "initial_state_path", None),
            variant="fdr",
        )
        return model


__all__ = [
    "CAPACITY_BPDD_CFG", "CAPACITY_EXPERT_CFG", "CapacityBPDDDetectionLoss",
    "CapacityBPDDDetectionModel", "CapacityBPDDTrainer",
]
