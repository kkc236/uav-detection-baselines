"""Frozen C0/C1/Q quality-probe math for RT-DETR score reranking."""

from __future__ import annotations

import math
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.rtdetr_quality_oracle import flattened_topk


PROBE_ALPHA = 2.0
PROBE_GATE_MAP_GAIN = Decimal("0.0050")


def _finite(value: torch.Tensor, *, label: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")


def c1_features(
    boxes: torch.Tensor, logits: torch.Tensor, *, num_classes: int
) -> torch.Tensor:
    """Build detached class-conditional probability, entropy, and geometry features."""
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [B,Q,4]")
    if logits.shape != (*boxes.shape[:2], num_classes):
        raise ValueError("logits must have shape [B,Q,C]")
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be positive")
    boxes = boxes.detach().float()
    logits = logits.detach().float()
    _finite(boxes, label="boxes")
    _finite(logits, label="logits")

    probabilities = logits.sigmoid()
    eps = torch.finfo(probabilities.dtype).eps
    bounded = probabilities.clamp(eps, 1.0 - eps)
    entropy = -(
        bounded * bounded.log() + (1.0 - bounded) * (1.0 - bounded).log()
    ).mean(dim=-1, keepdim=True)

    width = boxes[..., 2:3].clamp_min(eps)
    height = boxes[..., 3:4].clamp_min(eps)
    geometry = torch.cat(
        (
            boxes,
            width.log(),
            height.log(),
            width * height,
            width / height,
        ),
        dim=-1,
    )
    batch, queries, _ = boxes.shape
    shared = torch.cat((entropy, geometry), dim=-1).unsqueeze(2).expand(
        batch, queries, num_classes, -1
    )
    one_hot = torch.eye(
        num_classes, dtype=boxes.dtype, device=boxes.device
    ).view(1, 1, num_classes, num_classes).expand(batch, queries, -1, -1)
    result = torch.cat((probabilities.unsqueeze(-1), shared, one_hot), dim=-1)
    _finite(result, label="C1 features")
    return result.contiguous().detach()


class C1QualityProbe(nn.Module):
    """Small geometry/probability control head."""

    def __init__(self, *, feature_dim: int, width: int = 32) -> None:
        super().__init__()
        if feature_dim < 1 or width < 1:
            raise ValueError("probe dimensions must be positive")
        self.feature_dim = feature_dim
        self.network = nn.Sequential(
            nn.Linear(feature_dim, width), nn.SiLU(), nn.Linear(width, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[-1] != self.feature_dim:
            raise ValueError("C1 features have invalid shape")
        _finite(features, label="C1 features")
        return self.network(features.detach()).squeeze(-1)


class QQualityProbe(nn.Module):
    """Decoder-hidden quality head evaluated against C0 and C1 controls."""

    def __init__(
        self, *, feature_dim: int, hidden_dim: int, width: int = 32
    ) -> None:
        super().__init__()
        if min(feature_dim, hidden_dim, width) < 1:
            raise ValueError("probe dimensions must be positive")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.hidden_projection = nn.Sequential(
            nn.Linear(hidden_dim, width), nn.SiLU()
        )
        self.network = nn.Sequential(
            nn.Linear(feature_dim + width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, features: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[-1] != self.feature_dim:
            raise ValueError("Q features have invalid shape")
        if hidden.shape != (*features.shape[:2], self.hidden_dim):
            raise ValueError("decoder hidden has invalid shape")
        _finite(features, label="Q features")
        _finite(hidden, label="decoder hidden")
        features = features.detach()
        projected = self.hidden_projection(hidden.detach()).unsqueeze(2).expand(
            *features.shape[:3], -1
        )
        return self.network(torch.cat((features, projected), dim=-1)).squeeze(-1)


def top_pair_mask(probabilities: torch.Tensor, *, topk: int = 600) -> torch.Tensor:
    """Select flattened Query-by-class pairs using stock probabilities only."""
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape [B,Q,C]")
    if type(topk) is not int or topk < 1 or topk > probabilities.shape[1] * probabilities.shape[2]:
        raise ValueError("topk is outside the flattened pair count")
    _finite(probabilities, label="probabilities")
    indices = probabilities.flatten(1).topk(topk).indices
    flat = torch.zeros_like(probabilities.flatten(1), dtype=torch.bool)
    flat.scatter_(1, indices, True)
    return flat.view_as(probabilities)


def quality_probe_loss(
    predicted_logits: torch.Tensor,
    target_quality: torch.Tensor,
    stock_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Weighted soft-BCE focused on relevant stock pairs and high-quality targets."""
    if predicted_logits.shape != target_quality.shape or predicted_logits.shape != stock_probabilities.shape:
        raise ValueError("probe loss tensors must have identical shapes")
    for label, value in (
        ("predicted logits", predicted_logits),
        ("target quality", target_quality),
        ("stock probabilities", stock_probabilities),
    ):
        _finite(value, label=label)
    if bool((target_quality < 0).any() or (target_quality > 1).any()):
        raise ValueError("target quality must be in [0,1]")
    if bool((stock_probabilities < 0).any() or (stock_probabilities > 1).any()):
        raise ValueError("stock probabilities must be in [0,1]")
    weight = 0.05 + stock_probabilities.detach() + 4.0 * target_quality.detach()
    element = F.binary_cross_entropy_with_logits(
        predicted_logits, target_quality.detach(), reduction="none"
    )
    return (element * weight).sum() / weight.sum().clamp_min(
        torch.finfo(element.dtype).eps
    )


def rerank_with_predicted_quality(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    quality_logits: torch.Tensor,
    *,
    num_classes: int,
    max_det: int = 300,
) -> torch.Tensor:
    if quality_logits.shape != logits.shape:
        raise ValueError("quality logits and detector logits disagree")
    quality = quality_logits.sigmoid()
    scores = logits.sigmoid() * quality.pow(PROBE_ALPHA)
    return flattened_topk(
        boxes, scores, num_classes=num_classes, max_det=max_det
    )


def evaluate_internal_probe_gate(
    *, controls: Mapping[str, Mapping[str, float]], q: Mapping[str, float]
) -> dict[str, Any]:
    if set(controls) != {"c0", "c1"}:
        raise ValueError("controls must contain exactly C0 and C1")
    values = [q, controls["c0"], controls["c1"]]
    if any(
        name not in metrics
        or not isinstance(metrics[name], (int, float))
        or isinstance(metrics[name], bool)
        or not math.isfinite(float(metrics[name]))
        for metrics in values
        for name in ("map", "ap75")
    ):
        raise ValueError("probe metrics are invalid")
    best_map = max(Decimal(str(controls[name]["map"])) for name in controls)
    best_ap75 = max(Decimal(str(controls[name]["ap75"])) for name in controls)
    map_gain = Decimal(str(q["map"])) - best_map
    ap75_gain = Decimal(str(q["ap75"])) - best_ap75
    passed = map_gain >= PROBE_GATE_MAP_GAIN and ap75_gain > Decimal("0")
    return {
        "status": "passed" if passed else "scientific_failed",
        "deltas": {"map": str(map_gain), "ap75": str(ap75_gain)},
        "thresholds": {"map": str(PROBE_GATE_MAP_GAIN), "ap75": "0"},
    }


__all__ = [
    "C1QualityProbe",
    "PROBE_ALPHA",
    "PROBE_GATE_MAP_GAIN",
    "QQualityProbe",
    "c1_features",
    "evaluate_internal_probe_gate",
    "quality_probe_loss",
    "rerank_with_predicted_quality",
    "top_pair_mask",
]
