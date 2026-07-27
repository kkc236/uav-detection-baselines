"""Module-only batching and optimizer construction for GCQF G0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn

from src.gcqf_cache import GCQFEvidenceRecord
from src.gcte_types import QueryEvidence, ViewGeometry


GCQF_BATCH_SIZE = 8
GCQF_EPOCHS = 10
GCQF_FIXED_AMP_SCALE = 128.0
GCQF_LR = 0.01
GCQF_LRF = 0.01
GCQF_MOMENTUM = 0.937
GCQF_WARMUP_MOMENTUM = 0.8
GCQF_WEIGHT_DECAY = 0.0005
GCQF_WARMUP_EPOCHS = 3.0


@dataclass(frozen=True)
class GCQFBatch:
    global_evidence: QueryEvidence
    local_evidence: QueryEvidence
    geometry: ViewGeometry
    anchor_mask: torch.Tensor
    quality_targets: torch.Tensor
    equivariance_pairs: torch.Tensor
    image_ids: tuple[str, ...]

    def to(self, device: torch.device | str) -> "GCQFBatch":
        def move_evidence(value: QueryEvidence) -> QueryEvidence:
            return QueryEvidence(
                queries=value.queries.to(device, non_blocking=True),
                logits=value.logits.to(device, non_blocking=True),
                boxes=value.boxes.to(device, non_blocking=True),
                quality=value.quality.to(device, non_blocking=True),
            )

        return GCQFBatch(
            global_evidence=move_evidence(self.global_evidence),
            local_evidence=move_evidence(self.local_evidence),
            geometry=ViewGeometry(
                homography=self.geometry.homography.to(
                    device,
                    non_blocking=True,
                ),
                crop_metadata=self.geometry.crop_metadata.to(
                    device,
                    non_blocking=True,
                ),
                view_index=self.geometry.view_index.to(
                    device,
                    non_blocking=True,
                ),
                valid_mask=self.geometry.valid_mask.to(
                    device,
                    non_blocking=True,
                ),
            ),
            anchor_mask=self.anchor_mask.to(device, non_blocking=True),
            quality_targets=self.quality_targets.to(
                device,
                non_blocking=True,
            ),
            equivariance_pairs=self.equivariance_pairs.to(
                device,
                non_blocking=True,
            ),
            image_ids=self.image_ids,
        )


def _cat_evidence(values: Sequence[QueryEvidence]) -> QueryEvidence:
    return QueryEvidence(
        queries=torch.cat([value.queries for value in values], dim=0),
        logits=torch.cat([value.logits for value in values], dim=0),
        boxes=torch.cat([value.boxes for value in values], dim=0),
        quality=torch.cat([value.quality for value in values], dim=0),
    )


def collate_evidence_records(
    records: Sequence[GCQFEvidenceRecord],
) -> GCQFBatch:
    if not records:
        raise ValueError("GCQF batch must be nonempty")
    pairs: list[torch.Tensor] = []
    for batch_index, record in enumerate(records):
        if record.equivariance_pairs.numel():
            batch_column = torch.full(
                (record.equivariance_pairs.shape[0], 1),
                batch_index,
                dtype=torch.long,
            )
            pairs.append(
                torch.cat(
                    (batch_column, record.equivariance_pairs.cpu()),
                    dim=1,
                )
            )
    equivariance_pairs = (
        torch.cat(pairs, dim=0)
        if pairs
        else torch.empty((0, 3), dtype=torch.long)
    )
    return GCQFBatch(
        global_evidence=_cat_evidence(
            [record.global_evidence for record in records]
        ),
        local_evidence=_cat_evidence(
            [record.local_evidence for record in records]
        ),
        geometry=ViewGeometry(
            homography=torch.cat(
                [record.geometry.homography for record in records],
                dim=0,
            ),
            crop_metadata=torch.cat(
                [record.geometry.crop_metadata for record in records],
                dim=0,
            ),
            view_index=torch.cat(
                [record.geometry.view_index for record in records],
                dim=0,
            ),
            valid_mask=torch.cat(
                [record.geometry.valid_mask for record in records],
                dim=0,
            ),
        ),
        anchor_mask=torch.cat(
            [record.anchor_mask for record in records],
            dim=0,
        ),
        quality_targets=torch.cat(
            [record.quality_targets for record in records],
            dim=0,
        ),
        equivariance_pairs=equivariance_pairs,
        image_ids=tuple(record.image_id for record in records),
    )


def build_module_optimizer(
    module: nn.Module,
    *,
    optimizer_class: Callable | None = None,
):
    """Build Ultralytics-compatible MuSGD groups from GCQF parameters only."""

    if optimizer_class is None:
        try:
            from ultralytics.optim import MuSGD
        except Exception as error:
            raise RuntimeError("Ultralytics MuSGD is required") from error
        optimizer_class = MuSGD
    muon: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    bias: list[nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias"):
            bias.append(parameter)
        elif parameter.ndim >= 2:
            muon.append(parameter)
        else:
            no_decay.append(parameter)
    groups = []
    common = {
        "lr": GCQF_LR,
        "momentum": GCQF_MOMENTUM,
        "nesterov": True,
    }
    if muon:
        groups.append(
            {
                "params": muon,
                **common,
                "weight_decay": GCQF_WEIGHT_DECAY,
                "use_muon": True,
                "param_group": "muon",
            }
        )
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                **common,
                "weight_decay": 0.0,
                "use_muon": False,
                "param_group": "no_decay",
            }
        )
    if bias:
        groups.append(
            {
                "params": bias,
                **common,
                "weight_decay": 0.0,
                "use_muon": False,
                "param_group": "bias",
            }
        )
    expected = {
        id(parameter)
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    observed = {
        id(parameter)
        for group in groups
        for parameter in group["params"]
    }
    if observed != expected:
        raise RuntimeError("GCQF optimizer parameter coverage drift")
    return optimizer_class(groups, muon=0.2, sgd=1.0)


__all__ = [
    "GCQF_BATCH_SIZE",
    "GCQF_EPOCHS",
    "GCQF_FIXED_AMP_SCALE",
    "GCQFBatch",
    "build_module_optimizer",
    "collate_evidence_records",
]
