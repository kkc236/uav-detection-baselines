"""Ultralytics RT-DETR integration for FDR with adaptive SCADS support."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from ultralytics.nn import tasks as ultralytics_tasks
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel, yaml_model_load
from ultralytics.utils import RANK

from src.fdr_head import FDR_OUTPUT_DIM
from src.rtdetr_fdr import (
    FDRFixedPairedProtocolMixin,
    FDRRTDETRDetectionModel,
    FDRTrainer,
    _MODEL_PARSE_LOCK,
    _dn_partition,
)
from src.rtdetr_vsf_rmr import apply_resume_runtime_overrides
from src.scads_protocol import load_scads_initial_state
from src.scads_head import (
    SCADSFDRDeformableTransformerDecoder,
    SCADSFDRRTDETRDecoder,
)
from src.scads_loss import SCADSFDRDetectionLoss


SCADS_MODEL_CFG = (
    Path(__file__).resolve().parents[1] / "configs" / "rtdetr-l-fdr-scads.yaml"
)


def register_scads_module() -> None:
    """Expose SCADS to the Ultralytics YAML parser under the parser lock."""

    ultralytics_tasks.SCADSFDRRTDETRDecoder = SCADSFDRRTDETRDecoder


def _cfg_with_scads_seeds(
    cfg: str | Path | dict,
    private_seed: int | None,
    support_private_seed: int | None,
) -> str | Path | dict:
    if private_seed is None and support_private_seed is None:
        return cfg
    payload = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg)
    final_layer = payload.get("head", [None])[-1]
    if len(final_layer) != 4 or final_layer[2] != "SCADSFDRRTDETRDecoder":
        raise TypeError("SCADS YAML must end with SCADSFDRRTDETRDecoder")
    arguments = final_layer[3]
    if len(arguments) != 3 or not isinstance(arguments[2], dict):
        raise TypeError("SCADS decoder arguments must end with an options mapping")
    if private_seed is not None:
        arguments[2]["private_seed"] = int(private_seed)
    if support_private_seed is not None:
        arguments[2]["support_private_seed"] = int(support_private_seed)
    return payload


@dataclass(frozen=True)
class SCADSTrainingEvidence:
    """Normal and optional denoising evidence for FDR plus support routing."""

    corner_logits: Tensor
    references: Tensor
    pre_boxes: Tensor
    support_logits: Tensor
    support_weights: Tensor
    support_projects: Tensor
    dn_corner_logits: Tensor | None = None
    dn_references: Tensor | None = None
    dn_pre_boxes: Tensor | None = None
    dn_support_logits: Tensor | None = None
    dn_support_weights: Tensor | None = None
    dn_support_projects: Tensor | None = None


def split_scads_evidence(
    corner_logits: Tensor,
    references: Tensor,
    pre_boxes: Tensor,
    support_logits: Tensor,
    support_weights: Tensor,
    support_projects: Tensor,
    dn_meta: dict[str, Any] | None,
) -> SCADSTrainingEvidence:
    if corner_logits.ndim != 4 or corner_logits.shape[-1] != FDR_OUTPUT_DIM:
        raise ValueError("SCADS corners must have shape [layers,batch,queries,132]")
    batch, queries = corner_logits.shape[1:3]
    if references.shape != (*corner_logits.shape[:-1], 4):
        raise ValueError("SCADS references do not align with corner logits")
    if pre_boxes.shape != (batch, queries, 4):
        raise ValueError("SCADS preliminary boxes do not align with queries")
    if support_logits.shape[:2] != (batch, queries):
        raise ValueError("SCADS support logits do not align with queries")
    if support_weights.shape != support_logits.shape:
        raise ValueError("SCADS support weights must match support logits")
    if support_projects.shape != (batch, queries, 33):
        raise ValueError("SCADS effective projects must have shape [batch,queries,33]")

    partition = _dn_partition(dn_meta)
    if partition is None:
        return SCADSTrainingEvidence(
            corner_logits,
            references,
            pre_boxes,
            support_logits,
            support_weights,
            support_projects,
        )

    denoising, normal = partition
    if denoising + normal != queries:
        raise ValueError("SCADS denoising partition does not match query count")
    dn_corners, normal_corners = torch.split(
        corner_logits, (denoising, normal), dim=2
    )
    dn_references, normal_references = torch.split(
        references, (denoising, normal), dim=2
    )
    dn_pre, normal_pre = torch.split(pre_boxes, (denoising, normal), dim=1)
    dn_logits, normal_logits = torch.split(
        support_logits, (denoising, normal), dim=1
    )
    dn_weights, normal_weights = torch.split(
        support_weights, (denoising, normal), dim=1
    )
    dn_projects, normal_projects = torch.split(
        support_projects, (denoising, normal), dim=1
    )
    return SCADSTrainingEvidence(
        corner_logits=normal_corners,
        references=normal_references,
        pre_boxes=normal_pre,
        support_logits=normal_logits,
        support_weights=normal_weights,
        support_projects=normal_projects,
        dn_corner_logits=dn_corners,
        dn_references=dn_references,
        dn_pre_boxes=dn_pre,
        dn_support_logits=dn_logits,
        dn_support_weights=dn_weights,
        dn_support_projects=dn_projects,
    )


class SCADSFDRRTDETRDetectionModel(FDRRTDETRDetectionModel):
    """Validated FDR detector with query-conditioned distribution support."""

    def __init__(
        self,
        cfg: str | Path | dict = SCADS_MODEL_CFG,
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        private_seed: int | None = None,
        support_private_seed: int | None = None,
    ) -> None:
        with _MODEL_PARSE_LOCK:
            register_scads_module()
            cfg = _cfg_with_scads_seeds(
                cfg,
                private_seed,
                support_private_seed,
            )
            stock_decoder_type = ultralytics_tasks.RTDETRDecoder
            ultralytics_tasks.RTDETRDecoder = SCADSFDRRTDETRDecoder
            try:
                RTDETRDetectionModel.__init__(
                    self,
                    cfg=cfg,
                    ch=ch,
                    nc=nc,
                    verbose=verbose,
                )
            finally:
                ultralytics_tasks.RTDETRDecoder = stock_decoder_type
        head = self.model[-1]
        if not isinstance(head, SCADSFDRRTDETRDecoder):
            raise TypeError("SCADS YAML must end with SCADSFDRRTDETRDecoder")
        if int(head.num_queries) != 300 or int(head.num_decoder_layers) != 6:
            raise ValueError("SCADS requires 300 queries and six decoder layers")
        self.private_seed = int(head.fdr_options["private_seed"])
        self.support_private_seed = int(head.fdr_options["support_private_seed"])
        self.fdr_loss_options = dict(self.yaml.get("fdr_loss", {}))
        self.nc = int(self.yaml["nc"])
        self.last_fdr_evidence: SCADSTrainingEvidence | None = None
        self.last_fdr_losses: dict[str, Tensor] = {}

    @property
    def fdr(self) -> SCADSFDRDeformableTransformerDecoder:
        decoder = self.model[-1].decoder
        if not isinstance(decoder, SCADSFDRDeformableTransformerDecoder):
            raise RuntimeError("SCADS decoder was unexpectedly replaced")
        return decoder

    def _capture_fdr_evidence(
        self,
        dn_meta: dict[str, Any] | None,
    ) -> SCADSTrainingEvidence:
        decoder = self.fdr
        required = (
            decoder.last_corner_logits,
            decoder.last_references,
            decoder.last_pre_bboxes,
            decoder.last_support_logits,
            decoder.last_support_weights,
        )
        if any(item is None for item in required):
            raise RuntimeError("SCADS decoder did not retain complete training evidence")
        support_projects = decoder.adaptive_integral.effective_project(
            decoder.last_support_weights
        )
        evidence = split_scads_evidence(
            decoder.last_corner_logits,
            decoder.last_references,
            decoder.last_pre_bboxes,
            decoder.last_support_logits,
            decoder.last_support_weights,
            support_projects,
            dn_meta,
        )
        self.last_fdr_evidence = evidence
        return evidence

    def init_criterion(self) -> SCADSFDRDetectionLoss:
        return SCADSFDRDetectionLoss(
            nc=self.nc,
            use_vfl=True,
            fgl_weight=float(self.fdr_loss_options.get("fgl_weight", 0.15)),
            supervise_pre_boxes=bool(
                self.fdr_loss_options.get("supervise_pre_boxes", True)
            ),
            support_project_bank=self.fdr.adaptive_integral.projects,
            scads_route_weight=float(
                self.fdr_loss_options.get("scads_route_weight", 0.05)
            ),
            scads_margin_ratio=float(
                self.fdr_loss_options.get("scads_margin_ratio", 0.02)
            ),
        )

    def loss(
        self,
        batch: dict[str, Tensor],
        preds: tuple | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        image = batch["img"]
        batch_size = int(image.shape[0])
        batch_index = batch["batch_idx"]
        targets: dict[str, Any] = {
            "cls": batch["cls"].to(image.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=image.device),
            "batch_idx": batch_index.to(image.device, dtype=torch.long).view(-1),
            "gt_groups": [
                int((batch_index == index).sum().item())
                for index in range(batch_size)
            ],
        }

        if preds is None:
            preds = self.predict(image, batch=targets)
        if not self.training and isinstance(preds, tuple) and len(preds) == 2:
            preds = preds[1]
        if not isinstance(preds, tuple) or len(preds) != 5:
            raise RuntimeError("stock RT-DETR loss prediction contract changed")
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds

        evidence = self.last_fdr_evidence
        if evidence is None:
            evidence = self._capture_fdr_evidence(dn_meta)

        if dn_meta is None:
            dn_bboxes = None
            dn_scores = None
        else:
            partition = _dn_partition(dn_meta)
            if partition is None:
                raise RuntimeError("SCADS denoising partition disappeared")
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, partition, dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, partition, dim=2)

        stock_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])
        stock_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])
        losses = self.criterion.stock_plus_fgl(
            (stock_bboxes, stock_scores),
            targets,
            dn_bboxes=dn_bboxes,
            dn_scores=dn_scores,
            dn_meta=dn_meta,
            corner_logits=evidence.corner_logits,
            pre_boxes=evidence.pre_boxes,
            dn_corner_logits=evidence.dn_corner_logits,
            dn_pre_boxes=evidence.dn_pre_boxes,
            support_logits=evidence.support_logits,
            support_projects=evidence.support_projects,
            dn_support_projects=evidence.dn_support_projects,
        )
        self.last_fdr_losses = losses
        total = sum(losses.values())
        displayed = torch.stack(
            [
                losses[name].detach()
                for name in ("loss_giou", "loss_class", "loss_bbox")
            ]
        ).to(image.device)
        return total, displayed


def _load_scads_state(
    model: RTDETRDetectionModel,
    path: str | Path | None,
    *,
    variant: str,
) -> None:
    if path is None:
        return
    artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
    load_scads_initial_state(model, artifact, variant=variant)


class SCADSPairedFDRTrainer(FDRTrainer):
    """FDR baseline arm loaded from the new FDR/SCADS paired authority."""

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> FDRRTDETRDetectionModel:
        model = FDRRTDETRDetectionModel(
            cfg or "configs/rtdetr-l-fdr.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000 + self.experiment_seed,
        )
        if weights:
            model.load(weights)
        else:
            _load_scads_state(
                model,
                getattr(self, "initial_state_path", None),
                variant="fdr",
            )
        return model


class SCADSTrainer(FDRFixedPairedProtocolMixin, RTDETRTrainer):
    """SCADS method arm under the strict FDR/SCADS paired protocol."""

    def __init__(
        self,
        *args: Any,
        experiment_seed: int = 0,
        initial_state_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.experiment_seed = int(experiment_seed)
        self.initial_state_path = (
            Path(initial_state_path) if initial_state_path is not None else None
        )
        super().__init__(*args, **kwargs)

    def check_resume(self, overrides: dict[str, Any]) -> None:
        super().check_resume(overrides)
        if self.resume:
            apply_resume_runtime_overrides(self.args, overrides)
            if "epochs" in overrides:
                self.args.epochs = int(overrides["epochs"])

    def gradient_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        common: list[torch.nn.Parameter] = []
        fdr_private: list[torch.nn.Parameter] = []
        scads_private: list[torch.nn.Parameter] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".decoder.support_router." in name:
                destination = scads_private
            elif ".dec_bbox_head." in name or ".decoder.pre_bbox_head." in name:
                destination = fdr_private
            else:
                destination = common
            destination.append(parameter)
        if not common or not fdr_private or not scads_private:
            raise RuntimeError("SCADS gradient parameter partition is incomplete")
        return {
            "gradient_norm": common,
            "fdr_gradient_norm": fdr_private,
            "scads_gradient_norm": scads_private,
        }

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> SCADSFDRRTDETRDetectionModel:
        model = SCADSFDRRTDETRDetectionModel(
            cfg or SCADS_MODEL_CFG,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000 + self.experiment_seed,
            support_private_seed=20_000 + self.experiment_seed,
        )
        if weights:
            model.load(weights)
        else:
            _load_scads_state(
                model,
                getattr(self, "initial_state_path", None),
                variant="scads",
            )
        return model


__all__ = [
    "SCADS_MODEL_CFG",
    "SCADSPairedFDRTrainer",
    "SCADSFDRRTDETRDetectionModel",
    "SCADSTrainer",
    "SCADSTrainingEvidence",
    "register_scads_module",
    "split_scads_evidence",
]
