"""Module-only batching and optimizer construction for GCQF G0."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
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
    local_tiny_utility_targets: torch.Tensor | None
    local_non_tiny_risk_targets: torch.Tensor | None
    global_retain_targets: torch.Tensor | None
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
            local_tiny_utility_targets=(
                None
                if self.local_tiny_utility_targets is None
                else self.local_tiny_utility_targets.to(
                    device,
                    non_blocking=True,
                )
            ),
            local_non_tiny_risk_targets=(
                None
                if self.local_non_tiny_risk_targets is None
                else self.local_non_tiny_risk_targets.to(
                    device,
                    non_blocking=True,
                )
            ),
            global_retain_targets=(
                None
                if self.global_retain_targets is None
                else self.global_retain_targets.to(
                    device,
                    non_blocking=True,
                )
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
    *,
    require_sr_peg_targets: bool = False,
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
    target_presence = [
        record.sr_peg_targets is not None for record in records
    ]
    if any(target_presence) and not all(target_presence):
        raise ValueError("cannot mix SR-PEG supervised and unsupervised records")
    if require_sr_peg_targets and not all(target_presence):
        raise ValueError("SR-PEG targets are required for this batch")
    if all(target_presence):
        targets = [record.sr_peg_targets for record in records]
        local_tiny_targets = torch.cat(
            [
                target.local_tiny_utility
                for target in targets
                if target is not None
            ],
            dim=0,
        )
        local_risk_targets = torch.cat(
            [
                target.local_non_tiny_risk
                for target in targets
                if target is not None
            ],
            dim=0,
        )
        global_retain_targets = torch.cat(
            [
                target.global_retain
                for target in targets
                if target is not None
            ],
            dim=0,
        )
    else:
        local_tiny_targets = None
        local_risk_targets = None
        global_retain_targets = None
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
        local_tiny_utility_targets=local_tiny_targets,
        local_non_tiny_risk_targets=local_risk_targets,
        global_retain_targets=global_retain_targets,
        image_ids=tuple(record.image_id for record in records),
    )


def compute_positive_weights(
    records: Sequence[GCQFEvidenceRecord],
) -> dict[str, float]:
    """Compute sealed per-head Nneg/Npos weights, clipped to [1,20]."""

    if not records:
        raise ValueError("positive-weight computation requires records")
    fields = {
        "tiny": "local_tiny_utility",
        "risk": "local_non_tiny_risk",
        "retain": "global_retain",
    }
    weights: dict[str, float] = {}
    for key, field in fields.items():
        tensors: list[torch.Tensor] = []
        for record in records:
            if record.sr_peg_targets is None:
                raise ValueError("positive weights require SR-PEG targets")
            tensors.append(getattr(record.sr_peg_targets, field))
        values = torch.cat([tensor.reshape(-1) for tensor in tensors])
        positive = int((values > 0).sum())
        negative = int(values.numel()) - positive
        ratio = negative / max(positive, 1)
        weights[key] = float(min(20.0, max(1.0, ratio)))
    return weights


def split_seed0_records(
    records: Sequence[GCQFEvidenceRecord] | Sequence[str],
) -> tuple[tuple[GCQFEvidenceRecord, ...], tuple[GCQFEvidenceRecord, ...]] | tuple[tuple[str, ...], tuple[str, ...]]:
    """Create the sealed 518/129 split using SHA256(seed0:image_id)."""

    if len(records) != 647:
        raise ValueError("seed0 split requires exactly 647 records")
    keyed = []
    identities: set[str] = set()
    for record in records:
        image_id = record if isinstance(record, str) else record.image_id
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("seed0 split requires canonical image identities")
        if image_id in identities:
            raise ValueError("seed0 split identities must be unique")
        identities.add(image_id)
        digest = sha256(f"seed0:{image_id}".encode("utf-8")).hexdigest()
        keyed.append((digest, image_id, record))
    ordered = tuple(item[2] for item in sorted(keyed))
    return ordered[:518], ordered[518:]


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
    "compute_positive_weights",
    "split_seed0_records",
]
