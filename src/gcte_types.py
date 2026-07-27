"""Validated tensor contracts shared by the GCTE network stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


def _require_finite_floating(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class QueryEvidence:
    """A dense batch of decoder-query evidence in normalized ``xywh`` form."""

    queries: torch.Tensor
    logits: torch.Tensor
    boxes: torch.Tensor
    quality: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.queries, torch.Tensor) or self.queries.ndim != 3:
            raise ValueError("queries must be [B,Q,C]")
        batch, count, _ = self.queries.shape
        expected = (batch, count)
        if not isinstance(self.logits, torch.Tensor) or self.logits.shape[:2] != expected or self.logits.ndim != 3:
            raise ValueError("logits must share [B,Q]")
        if not isinstance(self.boxes, torch.Tensor) or self.boxes.shape != (*expected, 4):
            raise ValueError("boxes must be normalized xywh [B,Q,4]")
        if not isinstance(self.quality, torch.Tensor) or self.quality.shape != (*expected, 1):
            raise ValueError("quality must be [B,Q,1]")
        tensors = (self.queries, self.logits, self.boxes, self.quality)
        for name, tensor in zip(
            ("queries", "logits", "boxes", "quality"),
            tensors,
            strict=True,
        ):
            _require_finite_floating(name, tensor)
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1:
            raise ValueError("query evidence tensors must share one device")

    @property
    def batch_size(self) -> int:
        return int(self.queries.shape[0])

    @property
    def query_count(self) -> int:
        return int(self.queries.shape[1])

    @property
    def query_dim(self) -> int:
        return int(self.queries.shape[2])

    @property
    def num_classes(self) -> int:
        return int(self.logits.shape[2])

    def detached(self) -> "QueryEvidence":
        return QueryEvidence(
            queries=self.queries.detach(),
            logits=self.logits.detach(),
            boxes=self.boxes.detach(),
            quality=self.quality.detach(),
        )


@dataclass(frozen=True)
class ViewGeometry:
    """Per-query local-to-global homographies and normalized crop metadata."""

    homography: torch.Tensor
    crop_metadata: torch.Tensor
    view_index: torch.Tensor
    valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        _require_finite_floating("homography", self.homography)
        _require_finite_floating("crop_metadata", self.crop_metadata)
        if self.homography.ndim != 4 or self.homography.shape[-2:] != (3, 3):
            raise ValueError("homography must be [B,Q,3,3]")
        batch, count = self.homography.shape[:2]
        if self.crop_metadata.shape != (batch, count, 6):
            raise ValueError("crop_metadata must be [B,Q,6]")
        if not isinstance(self.view_index, torch.Tensor) or self.view_index.shape != (batch, count):
            raise ValueError("view_index must be [B,Q]")
        if self.view_index.dtype != torch.long:
            raise ValueError("view_index must use torch.long")
        if not isinstance(self.valid_mask, torch.Tensor) or self.valid_mask.shape != (batch, count):
            raise ValueError("valid_mask must be [B,Q]")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must use torch.bool")
        devices = {
            self.homography.device,
            self.crop_metadata.device,
            self.view_index.device,
            self.valid_mask.device,
        }
        if len(devices) != 1:
            raise ValueError("view geometry tensors must share one device")
        determinant = torch.linalg.det(self.homography.detach())
        if bool((determinant[self.valid_mask].abs() <= 1e-12).any()):
            raise ValueError("valid homography entries must be invertible")
        crop_width_height = self.crop_metadata[..., 2:4]
        resize_factors = self.crop_metadata[..., 4:6]
        if bool((crop_width_height[self.valid_mask] <= 0).any()):
            raise ValueError("valid crop width and height must be positive")
        if bool((resize_factors[self.valid_mask] <= 0).any()):
            raise ValueError("valid crop resize factors must be positive")
        if bool((self.view_index[self.valid_mask] < 0).any()):
            raise ValueError("valid view_index entries must be nonnegative")

    @property
    def batch_size(self) -> int:
        return int(self.homography.shape[0])

    @property
    def query_count(self) -> int:
        return int(self.homography.shape[1])


@dataclass(frozen=True)
class GCTEStageOutput:
    """Output of one internal GCTE stage."""

    evidence: QueryEvidence
    diagnostics: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class GCTENetworkOutput:
    """One explicit output object for the top-level GCTE network module."""

    unified_predictions: QueryEvidence
    local_predictions: QueryEvidence
    canonical_queries: QueryEvidence
    gate_outputs: GCTEStageOutput
    losses: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, torch.Tensor]
