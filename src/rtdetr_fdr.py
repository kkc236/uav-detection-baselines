"""Repository-owned Ultralytics RT-DETR integration for FDR-only boxes.

Only the decoder box representation changes.  The stock backbone, encoder,
query selection, decoder layers, classification heads, denoising builder and
postprocess implementation remain owned by Ultralytics 8.4.90.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK

from src.fdr_head import (
    FDR_OUTPUT_DIM,
    FDRDeformableTransformerDecoder,
    build_distribution_heads,
)
from src.fdr_loss import FDRDetectionLoss
from src.fdr_math import REG_MAX, REG_SCALE
from src.fdr_protocol import load_fdr_initial_state
from src.rtdetr_lpr import FixedPairedProtocolMixin
from src.rtdetr_vsf_rmr import apply_resume_runtime_overrides


@dataclass(frozen=True)
class FDRTrainingEvidence:
    """Normal-query evidence plus an optional denoising-query partition."""

    corner_logits: Tensor
    references: Tensor
    pre_boxes: Tensor
    dn_corner_logits: Tensor | None = None
    dn_references: Tensor | None = None
    dn_pre_boxes: Tensor | None = None


def _dn_partition(dn_meta: dict[str, Any] | None) -> tuple[int, int] | None:
    if dn_meta is None:
        return None
    split = dn_meta.get("dn_num_split")
    if not isinstance(split, (list, tuple)) or len(split) != 2:
        raise ValueError("dn_meta must contain a two-element dn_num_split partition")
    denoising, normal = (int(split[0]), int(split[1]))
    if denoising < 0 or normal <= 0:
        raise ValueError("dn_num_split partition must be non-negative/positive")
    return denoising, normal


def split_fdr_evidence(
    corner_logits: Tensor,
    references: Tensor,
    pre_boxes: Tensor,
    dn_meta: dict[str, Any] | None,
) -> FDRTrainingEvidence:
    """Validate and split cached FDR tensors into normal and DN queries."""

    if corner_logits.ndim != 4 or corner_logits.shape[-1] != FDR_OUTPUT_DIM:
        raise ValueError("corner_logits must have shape [layers,batch,queries,132]")
    if references.shape != (*corner_logits.shape[:-1], 4):
        raise ValueError("references must match corner logits and end in four coordinates")
    if pre_boxes.shape != (*corner_logits.shape[1:3], 4):
        raise ValueError("pre_boxes must have shape [batch,queries,4]")

    partition = _dn_partition(dn_meta)
    if partition is None:
        return FDRTrainingEvidence(corner_logits, references, pre_boxes)

    denoising, normal = partition
    if denoising + normal != corner_logits.shape[2]:
        raise ValueError("dn_num_split partition does not match FDR query count")
    dn_corners, normal_corners = torch.split(
        corner_logits, (denoising, normal), dim=2
    )
    dn_references, normal_references = torch.split(
        references, (denoising, normal), dim=2
    )
    dn_pre, normal_pre = torch.split(pre_boxes, (denoising, normal), dim=1)
    return FDRTrainingEvidence(
        corner_logits=normal_corners,
        references=normal_references,
        pre_boxes=normal_pre,
        dn_corner_logits=dn_corners,
        dn_references=dn_references,
        dn_pre_boxes=dn_pre,
    )


class FDRRTDETRDetectionModel(RTDETRDetectionModel):
    """Ultralytics RT-DETR-L with only its decoder box path replaced."""

    def __init__(
        self,
        cfg: str | Path = "rtdetr-l.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        private_seed: int = 10_000,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        head = self.model[-1]
        if int(head.num_queries) != 300:
            raise ValueError("the frozen FDR protocol requires exactly 300 queries")
        if int(head.num_decoder_layers) != 6:
            raise ValueError("the frozen FDR protocol requires exactly six decoder layers")

        stock_pre_bbox_head = head.dec_bbox_head[0]
        distribution_heads = build_distribution_heads(
            int(head.hidden_dim),
            int(head.num_decoder_layers),
            private_seed=int(private_seed),
        )
        head.decoder = FDRDeformableTransformerDecoder.from_stock(
            head.decoder,
            pre_bbox_head=stock_pre_bbox_head,
        )
        head.dec_bbox_head = distribution_heads

        # Read-only compatibility view used by protocol and preflight checks.
        head.decoder.reg_max = REG_MAX
        head.decoder.final_layers = [module.layers[-1] for module in distribution_heads]
        self.private_seed = int(private_seed)
        self.nc = int(self.yaml["nc"])
        self.last_fdr_evidence: FDRTrainingEvidence | None = None
        self.last_fdr_losses: dict[str, Tensor] = {}

    @property
    def fdr(self) -> FDRDeformableTransformerDecoder:
        """Expose the repository-owned FDR box path without double-registering it."""

        decoder = self.model[-1].decoder
        if not isinstance(decoder, FDRDeformableTransformerDecoder):
            raise RuntimeError("FDR decoder was unexpectedly replaced")
        return decoder

    def _capture_fdr_evidence(
        self, dn_meta: dict[str, Any] | None
    ) -> FDRTrainingEvidence:
        decoder = self.fdr
        if (
            decoder.last_corner_logits is None
            or decoder.last_references is None
            or decoder.last_pre_bboxes is None
        ):
            raise RuntimeError("FDR decoder did not retain training evidence")
        evidence = split_fdr_evidence(
            decoder.last_corner_logits,
            decoder.last_references,
            decoder.last_pre_bboxes,
            dn_meta,
        )
        self.last_fdr_evidence = evidence
        return evidence

    def init_criterion(self) -> FDRDetectionLoss:
        """Build stock RT-DETR loss extended only by FGL and pre-box losses."""

        return FDRDetectionLoss(
            nc=self.nc,
            use_vfl=True,
            fgl_weight=0.15,
            supervise_pre_boxes=True,
        )

    def loss(
        self,
        batch: dict[str, Tensor],
        preds: tuple | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute seven-layer stock loss plus isolated decoder FDR losses."""

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
        if not isinstance(preds, tuple) or len(preds) != 5:
            raise RuntimeError("stock RT-DETR loss prediction contract changed")
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds

        evidence = self.last_fdr_evidence
        if evidence is None:
            # Supports callers that supplied predictions from this model directly.
            evidence = self._capture_fdr_evidence(dn_meta)

        if dn_meta is None:
            dn_bboxes = None
            dn_scores = None
        else:
            partition = _dn_partition(dn_meta)
            if partition is None:
                raise RuntimeError("denoising partition unexpectedly disappeared")
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, partition, dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, partition, dim=2)

        # Preserve the exact Ultralytics stock contract: encoder first, then
        # all six decoder layers.  FDRDetectionLoss explicitly skips encoder
        # when consuming the six-layer corner/pre evidence.
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

    def predict(
        self,
        x: Tensor,
        profile: bool = False,
        visualize: bool = False,
        batch: dict[str, Any] | None = None,
        augment: bool = False,
        embed: list[int] | None = None,
    ) -> tuple | Tensor:
        """Run stock prediction and retain isolated FDR training evidence."""

        output = super().predict(
            x,
            profile=profile,
            visualize=visualize,
            batch=batch,
            augment=augment,
            embed=embed,
        )
        if self.training:
            if not isinstance(output, tuple) or len(output) != 5:
                raise RuntimeError("stock RT-DETR training output contract changed")
            self._capture_fdr_evidence(output[-1])
        else:
            self.last_fdr_evidence = None
        return output


def _load_initial_state(
    model: RTDETRDetectionModel,
    path: str | Path | None,
    *,
    variant: str,
) -> None:
    if path is None:
        return
    artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
    load_fdr_initial_state(model, artifact, variant=variant)


class FDRTrainer(FixedPairedProtocolMixin, RTDETRTrainer):
    """Strict paired trainer for the isolated FDR-only detector arm."""

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
        private: list[torch.nn.Parameter] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            destination = (
                private
                if ".dec_bbox_head." in name or ".decoder.pre_bbox_head." in name
                else common
            )
            destination.append(parameter)
        if not common or not private:
            raise RuntimeError("FDR stock/private parameter partition is incomplete")
        return {"gradient_norm": common, "fdr_gradient_norm": private}

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> FDRRTDETRDetectionModel:
        model = FDRRTDETRDetectionModel(
            cfg or "rtdetr-l.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            private_seed=10_000 + self.experiment_seed,
        )
        if weights:
            model.load(weights)
        else:
            _load_initial_state(
                model,
                getattr(self, "initial_state_path", None),
                variant="fdr",
            )
        return model


class FDRControlTrainer(FixedPairedProtocolMixin, RTDETRTrainer):
    """Stock RT-DETR control arm under the identical FDR paired protocol."""

    def __init__(
        self,
        *args: Any,
        initial_state_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
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

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> RTDETRDetectionModel:
        model = RTDETRDetectionModel(
            cfg or "rtdetr-l.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        else:
            _load_initial_state(
                model,
                getattr(self, "initial_state_path", None),
                variant="control",
            )
        return model


__all__ = [
    "FDRControlTrainer",
    "FDRRTDETRDetectionModel",
    "FDRTrainer",
    "FDRTrainingEvidence",
    "split_fdr_evidence",
]
