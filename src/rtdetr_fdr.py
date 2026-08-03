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
from ultralytics.nn.tasks import RTDETRDetectionModel

from src.fdr_head import (
    FDR_OUTPUT_DIM,
    FDRDeformableTransformerDecoder,
    build_distribution_heads,
)
from src.fdr_math import REG_MAX, REG_SCALE


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


__all__ = [
    "FDRRTDETRDetectionModel",
    "FDRTrainingEvidence",
    "split_fdr_evidence",
]
