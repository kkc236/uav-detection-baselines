"""Pure D0 oracle and candidate-pool math for the frozen OAR protocol."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

import torch
from torch import nn

from src.oar_protocol import OAR_GAIN_RECOVERY, OAR_K_GRID
from src.rtdetr_quality_oracle import same_class_iou_quality


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


class OARRanker(nn.Module):
    """Zero-initialized all-pair objective-aligned residual ranker."""

    def __init__(self, feature_dim: int = 276, width: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.network(features.detach()).squeeze(-1)
        return 2.0 * torch.tanh(raw / 2.0)


def apply_oar_r2(
    model: OARRanker,
    features: torch.Tensor,
    stock_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjust every detached stock Query-by-class logit with an OAR residual."""
    residual = model(features)
    adjusted_logits = stock_logits.detach() + residual
    return adjusted_logits.sigmoid(), residual


def _require_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    return value


def _require_floating_tensor(value: Any, name: str) -> torch.Tensor:
    tensor = _require_tensor(value, name)
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be a floating-point tensor")
    return tensor


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def topk_per_class_mask(probabilities: torch.Tensor, k: int) -> torch.Tensor:
    """Select exactly the stock top-K query candidates independently per class."""
    probabilities = _require_floating_tensor(probabilities, "probabilities")
    if probabilities.ndim != 2 or 0 in probabilities.shape:
        raise ValueError("probabilities must have shape [Q, C] with nonzero dimensions")
    if type(k) is not int or k not in OAR_K_GRID:
        raise ValueError(f"k must be in OAR_K_GRID={OAR_K_GRID}")
    if probabilities.shape[0] < k:
        raise ValueError("probabilities must contain at least k queries")
    _require_finite(probabilities, "probabilities")

    indices = torch.argsort(
        probabilities, dim=0, descending=True, stable=True
    )[:k]
    mask = torch.zeros_like(probabilities, dtype=torch.bool)
    mask.scatter_(0, indices, True)
    return mask


def oracle_score_families(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    *,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    """Compute the frozen stock, presence, query-IoU, and same-class scores."""
    boxes = _require_floating_tensor(boxes, "boxes")
    logits = _require_floating_tensor(logits, "logits")
    target_boxes = _require_floating_tensor(target_boxes, "target_boxes")
    target_classes = _require_tensor(target_classes, "target_classes")

    if type(num_classes) is not int or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [Q, 4]")
    if logits.ndim != 2 or logits.shape != (boxes.shape[0], num_classes):
        raise ValueError("logits must have shape [Q, num_classes]")
    if target_boxes.ndim != 2 or target_boxes.shape[1] != 4:
        raise ValueError("target_boxes must have shape [N, 4]")
    if target_classes.ndim != 1:
        raise ValueError("target_classes must have shape [N]")
    if target_boxes.shape[0] != target_classes.shape[0]:
        raise ValueError(
            "target_boxes and target_classes must contain the same number of targets"
        )
    if target_classes.dtype not in _INTEGER_DTYPES:
        raise TypeError("target_classes must contain integer class indices")
    tensors = (boxes, logits, target_boxes, target_classes)
    if any(value.device != boxes.device for value in tensors[1:]):
        raise ValueError(
            "boxes, logits, target_boxes, and target_classes must share a device"
        )
    _require_finite(boxes, "boxes")
    _require_finite(logits, "logits")
    _require_finite(target_boxes, "target_boxes")

    detached_boxes = boxes.detach().float()
    detached_targets = target_boxes.detach().float()
    detached_classes = target_classes.detach().long()
    probabilities = logits.detach().float().sigmoid()
    same_class = same_class_iou_quality(
        detached_boxes,
        detached_targets,
        detached_classes,
        num_classes,
    )
    query_iou = same_class.amax(dim=1, keepdim=True)
    presence = torch.zeros(
        num_classes,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    if detached_classes.numel():
        presence[detached_classes.unique()] = 1

    return {
        "stock": probabilities,
        "presence": probabilities * presence,
        "query_iou": probabilities * query_iou.square(),
        "same_class": probabilities * same_class.square(),
    }


def restrict_oracle(
    stock: torch.Tensor,
    oracle: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply oracle scores only inside a pool, retaining exact stock scores outside."""
    stock = _require_floating_tensor(stock, "stock")
    oracle = _require_floating_tensor(oracle, "oracle")
    mask = _require_tensor(mask, "mask")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be a boolean tensor")
    if stock.ndim != 2:
        raise ValueError("stock, oracle, and mask must have shape [Q, C]")
    if oracle.shape != stock.shape or mask.shape != stock.shape:
        raise ValueError("stock, oracle, and mask must have the same shape")
    if oracle.dtype != stock.dtype:
        raise TypeError("stock and oracle must have the same dtype")
    if oracle.device != stock.device or mask.device != stock.device:
        raise ValueError("stock, oracle, and mask must share a device")
    _require_finite(stock, "stock")
    _require_finite(oracle, "oracle")
    return torch.where(mask, oracle, stock)


def _decimal_metric(value: Any, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not decimal_value.is_finite():
        raise ValueError(f"{name} must be finite")
    return decimal_value


def select_candidate_k(
    *,
    stock_map: Any,
    full_map: Any,
    restricted_map: Mapping[int, Any],
) -> dict[str, Any]:
    """Choose the smallest frozen K recovering at least 90% of positive full gain."""
    if not isinstance(restricted_map, Mapping):
        raise TypeError("restricted_map must be a mapping")
    keys = tuple(restricted_map)
    if (
        len(keys) != len(OAR_K_GRID)
        or any(type(k) is not int for k in keys)
        or set(keys) != set(OAR_K_GRID)
    ):
        raise ValueError(f"restricted_map must contain exactly OAR_K_GRID={OAR_K_GRID}")

    stock = _decimal_metric(stock_map, "stock_map")
    full = _decimal_metric(full_map, "full_map")
    restricted = {
        k: _decimal_metric(restricted_map[k], f"restricted_map[{k}]")
        for k in OAR_K_GRID
    }
    total_gain = full - stock
    if total_gain <= 0:
        return {"status": "scientific_failed", "selected_k": None}

    for k in OAR_K_GRID:
        recovered = (restricted[k] - stock) / total_gain
        if recovered >= OAR_GAIN_RECOVERY:
            return {
                "status": "passed",
                "selected_k": k,
                "recovered": str(recovered),
            }
    return {"status": "scientific_failed", "selected_k": None}
