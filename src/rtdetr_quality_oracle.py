"""Mathematical core for the frozen RT-DETR quality-reranking oracle."""

from __future__ import annotations

from decimal import Decimal

import torch


ALPHA_GRID = (0.25, 0.5, 1.0, 2.0)
DEV_COUNT = 129
MAP_GAIN_THRESHOLD = Decimal("0.0050")

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _require_positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_floating_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating-point tensor")


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def same_class_iou_quality(
    boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Return the maximum same-class IoU for every query and class."""
    if not isinstance(boxes, torch.Tensor):
        raise TypeError("boxes must be a tensor")
    if not isinstance(target_boxes, torch.Tensor):
        raise TypeError("target_boxes must be a tensor")
    if not isinstance(target_classes, torch.Tensor):
        raise TypeError("target_classes must be a tensor")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [Q, 4]")
    if target_boxes.ndim != 2 or target_boxes.shape[1] != 4:
        raise ValueError("target_boxes must have shape [N, 4]")
    if target_classes.ndim != 1:
        raise ValueError("target_classes must have shape [N]")
    if target_boxes.shape[0] != target_classes.shape[0]:
        raise ValueError(
            "target_boxes and target_classes must contain the same number of targets"
        )
    _require_positive_int(num_classes, "num_classes")
    _require_floating_tensor(boxes, "boxes")
    _require_floating_tensor(target_boxes, "target_boxes")
    if target_classes.dtype not in _INTEGER_DTYPES:
        raise TypeError("target_classes must contain integer class indices")
    if boxes.device != target_boxes.device or boxes.device != target_classes.device:
        raise ValueError("boxes, target_boxes, and target_classes must share a device")
    _require_finite(boxes, "boxes")
    _require_finite(target_boxes, "target_boxes")
    if not bool(((boxes >= 0) & (boxes <= 1)).all()):
        raise ValueError("boxes must be normalized to [0, 1]")
    if not bool(((target_boxes >= 0) & (target_boxes <= 1)).all()):
        raise ValueError("target_boxes must be normalized to [0, 1]")
    if target_classes.numel() and not bool(
        ((target_classes >= 0) & (target_classes < num_classes)).all()
    ):
        raise ValueError("target_classes must be in [0, num_classes)")

    compute_dtype = torch.promote_types(boxes.dtype, target_boxes.dtype)
    if compute_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    quality = torch.zeros(
        (boxes.shape[0], num_classes), dtype=compute_dtype, device=boxes.device
    )
    if target_boxes.shape[0] == 0:
        return quality

    query_boxes = boxes.detach().to(dtype=compute_dtype)
    ground_truth = target_boxes.detach().to(dtype=compute_dtype)
    query_center, query_size = query_boxes.split(2, dim=-1)
    target_center, target_size = ground_truth.split(2, dim=-1)
    query_lower = query_center - query_size / 2
    query_upper = query_center + query_size / 2
    target_lower = target_center - target_size / 2
    target_upper = target_center + target_size / 2

    intersection = (
        torch.minimum(query_upper[:, None], target_upper[None])
        - torch.maximum(query_lower[:, None], target_lower[None])
    ).clamp_min(0).prod(dim=-1)
    query_area = query_size.prod(dim=-1)
    target_area = target_size.prod(dim=-1)
    union = query_area[:, None] + target_area[None] - intersection
    iou = torch.where(union > 0, intersection / union, torch.zeros_like(union))
    iou = torch.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0, 1)

    class_indices = target_classes.detach().to(dtype=torch.long)
    for class_index in range(num_classes):
        selected = class_indices == class_index
        if bool(selected.any()):
            quality[:, class_index] = iou[:, selected].amax(dim=1)
    return quality


def flattened_topk(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    num_classes: int,
    max_det: int = 300,
) -> torch.Tensor:
    """Apply Ultralytics 8.4.90's flattened query-by-class Top-K."""
    if not isinstance(boxes, torch.Tensor):
        raise TypeError("boxes must be a tensor")
    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be a tensor")
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [B, Q, 4]")
    if scores.ndim != 3:
        raise ValueError("scores must have shape [B, Q, C]")
    _require_positive_int(num_classes, "num_classes")
    _require_positive_int(max_det, "max_det")
    if scores.shape != (*boxes.shape[:2], num_classes):
        raise ValueError("scores must have shape [B, Q, num_classes]")
    _require_floating_tensor(boxes, "boxes")
    _require_floating_tensor(scores, "scores")
    if boxes.device != scores.device:
        raise ValueError("boxes and scores must share a device")
    if boxes.dtype != scores.dtype:
        raise ValueError("boxes and scores must share a dtype")
    _require_finite(boxes, "boxes")
    _require_finite(scores, "scores")
    if max_det > scores.shape[1] * num_classes:
        raise ValueError("max_det cannot exceed the flattened score count")

    selected_scores, index = scores.flatten(1).topk(max_det)
    query_index = torch.div(index, num_classes, rounding_mode="floor")
    selected_boxes = boxes.gather(
        dim=1,
        index=query_index.unsqueeze(-1).expand(-1, -1, 4).long(),
    )
    class_index = (index - query_index * num_classes)[..., None].float()
    return torch.cat(
        [selected_boxes, selected_scores[..., None], class_index], dim=-1
    )


def oracle_topk(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    qualities: torch.Tensor,
    alpha: float,
    num_classes: int,
    max_det: int = 300,
) -> torch.Tensor:
    """Rerank sigmoid class scores by perfect same-class IoU quality."""
    if isinstance(alpha, bool) or alpha not in ALPHA_GRID:
        raise ValueError(f"alpha must be one of ALPHA_GRID={ALPHA_GRID}")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a tensor")
    if not isinstance(qualities, torch.Tensor):
        raise TypeError("qualities must be a tensor")
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, Q, C]")
    if qualities.shape != logits.shape:
        raise ValueError("qualities and logits must have identical shapes")
    if logits.shape[-1] != num_classes:
        raise ValueError("logits class dimension must equal num_classes")
    _require_floating_tensor(logits, "logits")
    _require_floating_tensor(qualities, "qualities")
    if logits.device != qualities.device:
        raise ValueError("logits and qualities must share a device")
    if logits.dtype != qualities.dtype:
        raise ValueError("logits and qualities must share a dtype")
    _require_finite(logits, "logits")
    _require_finite(qualities, "qualities")
    if not bool(((qualities >= 0) & (qualities <= 1)).all()):
        raise ValueError("qualities must be in [0, 1]")

    reranked_scores = logits.sigmoid() * qualities.pow(alpha)
    return flattened_topk(
        boxes,
        reranked_scores,
        num_classes=num_classes,
        max_det=max_det,
    )
