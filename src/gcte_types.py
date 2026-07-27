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
class CropGeometry:
    """Per-query crop metadata in the original image pixel frame."""

    crop_xyxy: torch.Tensor
    source_size: torch.Tensor
    view_index: torch.Tensor
    valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        _require_finite_floating("crop_xyxy", self.crop_xyxy)
        _require_finite_floating("source_size", self.source_size)
        if self.crop_xyxy.ndim != 3 or self.crop_xyxy.shape[-1] != 4:
            raise ValueError("crop_xyxy must be [B,Q,4]")
        batch, count, _ = self.crop_xyxy.shape
        if self.source_size.shape != (batch, 2):
            raise ValueError("source_size must be [B,2] as width,height")
        if not isinstance(self.view_index, torch.Tensor) or self.view_index.shape != (batch, count):
            raise ValueError("view_index must be [B,Q]")
        if self.view_index.dtype != torch.long:
            raise ValueError("view_index must use torch.long")
        if not isinstance(self.valid_mask, torch.Tensor) or self.valid_mask.shape != (batch, count):
            raise ValueError("valid_mask must be [B,Q]")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must use torch.bool")
        devices = {
            self.crop_xyxy.device,
            self.source_size.device,
            self.view_index.device,
            self.valid_mask.device,
        }
        if len(devices) != 1:
            raise ValueError("crop geometry tensors must share one device")
        if bool((self.source_size <= 0).any()):
            raise ValueError("source_size must be positive")

        left, top, right, bottom = self.crop_xyxy.unbind(dim=-1)
        source_width = self.source_size[:, 0].unsqueeze(1)
        source_height = self.source_size[:, 1].unsqueeze(1)
        valid_rectangles = (
            (left >= 0)
            & (top >= 0)
            & (right > left)
            & (bottom > top)
            & (right <= source_width)
            & (bottom <= source_height)
        )
        if not bool(valid_rectangles.all()):
            raise ValueError("crop_xyxy must be non-empty and within source bounds")
        if bool((self.view_index[self.valid_mask] < 0).any()):
            raise ValueError("valid view_index entries must be nonnegative")

    @property
    def batch_size(self) -> int:
        return int(self.crop_xyxy.shape[0])

    @property
    def query_count(self) -> int:
        return int(self.crop_xyxy.shape[1])


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
