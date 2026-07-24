from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from hashlib import sha256
from math import ceil, floor
from pathlib import Path, PurePosixPath
from typing import Sequence

import torch
import torch.nn.functional as F


ASCV_TILE_RATIO = 0.60
ASCV_TINY_BOUNDARY_PX = 16.0
ASCV_LAMBDA = 0.1
ASCV_WARMUP_EPOCHS = 3
ASCV_CROP_PROTOCOL = "ascv-loc/crop-v2"
ASCV_PROTOCOL_VERSION = ASCV_CROP_PROTOCOL
ASCV_IMAGE_SIZE = 640
ASCV_CROP_SIZE = 384


@dataclass(frozen=True)
class LocalTargets:
    boxes: torch.Tensor
    classes: torch.Tensor
    batch_indices: torch.Tensor
    gt_ids: torch.Tensor
    groups: list[int]


@dataclass(frozen=True)
class JoinedMatches:
    batch_indices: torch.Tensor
    full_query_indices: torch.Tensor
    local_query_indices: torch.Tensor
    gt_ids: torch.Tensor
    local_target_indices: torch.Tensor


@dataclass(frozen=True)
class ASCVLocLossResult:
    loss: torch.Tensor
    pair_count: int
    tiny_pair_count: int
    non_tiny_pair_count: int
    tiny_teacher_advantage_sum: torch.Tensor
    tiny_teacher_win_count: int
    non_tiny_teacher_advantage_sum: torch.Tensor
    non_tiny_teacher_win_count: int


def ascv_warmup(epoch: int) -> float:
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    return min((int(epoch) + 1) / ASCV_WARMUP_EPOCHS, 1.0)


def canonical_image_id(image_path: str | Path, *, dataset_root: str | Path) -> str:
    root = Path(dataset_root).resolve()
    candidate = Path(image_path).resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"image is outside dataset root: {candidate}") from error
    return relative.as_posix()


def _u64(payload: bytes) -> int:
    if len(payload) != 8:
        raise ValueError("ASCV crop hash words must contain exactly 8 bytes")
    return int.from_bytes(payload, "big")


def _image_digest(image_key: str) -> bytes:
    path = PurePosixPath(image_key)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"ASCV image key must be dataset-relative: {image_key}")
    return sha256(b"ascv-loc/crop-v2\0image\0" + image_key.encode("utf-8")).digest()


def _origin_digest(image_key: str, annotation_ordinal: int) -> bytes:
    return sha256(
        b"ascv-loc/crop-v2\0origin\0"
        + image_key.encode("utf-8")
        + b"\0"
        + str(annotation_ordinal).encode("ascii")
    ).digest()


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center = boxes[..., :2]
    half_extent = boxes[..., 2:] / 2
    return torch.cat((center - half_extent, center + half_extent), dim=-1)


def _xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    low = boxes[..., :2]
    high = boxes[..., 2:]
    return torch.cat(((low + high) / 2, high - low), dim=-1)


def _crop_origin_for_box(
    box_xyxy: Sequence[float],
    *,
    crop_width: int,
    crop_height: int,
    image_width: int,
    image_height: int,
    x_hash: int,
    y_hash: int,
) -> tuple[int, int] | None:
    x1, y1, x2, y2 = box_xyxy
    lower_x = max(0, ceil(x2 - crop_width))
    upper_x = min(image_width - crop_width, floor(x1))
    lower_y = max(0, ceil(y2 - crop_height))
    upper_y = min(image_height - crop_height, floor(y1))
    if lower_x > upper_x or lower_y > upper_y:
        return None
    x0 = lower_x + x_hash % (upper_x - lower_x + 1)
    y0 = lower_y + y_hash % (upper_y - lower_y + 1)
    return x0, y0


def select_target_anchored_crops(
    *,
    boxes: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_size: int,
    image_hw: tuple[int, int],
    image_keys: Sequence[str] | None = None,
) -> torch.Tensor:
    """Choose one deterministic 0.60 crop per full view.

    The chosen crop contains a complete target whenever at least one target fits
    inside the frozen crop size. Target values affect training augmentation only
    and are never used by validation or inference.
    """

    height, width = (int(image_hw[0]), int(image_hw[1]))
    if (height, width) != (ASCV_IMAGE_SIZE, ASCV_IMAGE_SIZE):
        raise ValueError("ASCV crop-v2 requires an exact 640x640 input")
    crop_height = crop_width = ASCV_CROP_SIZE
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if image_keys is None:
        image_keys = [str(index) for index in range(batch_size)]
    if len(image_keys) != batch_size:
        raise ValueError("image_keys length must equal batch_size")

    device = boxes.device
    normalized_xyxy = _xywh_to_xyxy(boxes.detach()).cpu()
    scale = normalized_xyxy.new_tensor([width, height, width, height])
    absolute_xyxy = normalized_xyxy * scale
    flat_batch_indices = batch_indices.detach().to(dtype=torch.long).view(-1).cpu()
    crops: list[list[int]] = []

    for batch_index in range(batch_size):
        image_key = str(image_keys[batch_index])
        base = _image_digest(image_key)
        target_indices = torch.where(flat_batch_indices == batch_index)[0].tolist()
        ordinals: list[int] = []
        if target_indices:
            first_ordinal = _u64(base[:8]) % len(target_indices)
            ordinals = [
                (first_ordinal + offset) % len(target_indices)
                for offset in range(len(target_indices))
            ]
        origin = None
        for annotation_ordinal in ordinals:
            target_index = target_indices[annotation_ordinal]
            origin_hash = _origin_digest(image_key, annotation_ordinal)
            origin = _crop_origin_for_box(
                absolute_xyxy[target_index].tolist(),
                crop_width=crop_width,
                crop_height=crop_height,
                image_width=width,
                image_height=height,
                x_hash=_u64(origin_hash[:8]),
                y_hash=_u64(origin_hash[8:16]),
            )
            if origin is not None:
                break
        if origin is None:
            max_x = width - crop_width
            max_y = height - crop_height
            origin = (
                _u64(base[8:16]) % (max_x + 1),
                _u64(base[16:24]) % (max_y + 1),
            )
        x0, y0 = origin
        crops.append([x0, y0, x0 + crop_width, y0 + crop_height])

    return torch.tensor(crops, dtype=torch.long, device=device)


def build_local_targets(
    *,
    full_boxes: torch.Tensor,
    classes: torch.Tensor,
    batch_indices: torch.Tensor,
    crops: torch.Tensor,
    image_hw: tuple[int, int],
) -> LocalTargets:
    """Project fully contained targets into local coordinates.

    Boxes intersecting a crop boundary are deliberately absent from the local
    matcher target set.
    """

    height, width = (int(image_hw[0]), int(image_hw[1]))
    boxes = full_boxes.view(-1, 4)
    classes = classes.view(-1)
    batch_indices = batch_indices.to(device=boxes.device, dtype=torch.long).view(-1)
    if boxes.shape[0] != classes.shape[0] or boxes.shape[0] != batch_indices.shape[0]:
        raise ValueError("full target tensors must have the same flattened length")
    if crops.shape != (len(crops), 4):
        raise ValueError("crops must have shape [batch, 4]")
    if boxes.numel() == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=boxes.device)
        return LocalTargets(
            boxes=boxes.new_empty((0, 4)),
            classes=classes[:0],
            batch_indices=empty_long,
            gt_ids=empty_long,
            groups=[0 for _ in range(len(crops))],
        )

    full_xyxy = _xywh_to_xyxy(boxes) * boxes.new_tensor([width, height, width, height])
    target_crops = crops.to(device=boxes.device, dtype=boxes.dtype)[batch_indices]
    epsilon = boxes.new_tensor(1e-6)
    complete = (
        (full_xyxy[:, 0] + epsilon >= target_crops[:, 0])
        & (full_xyxy[:, 1] + epsilon >= target_crops[:, 1])
        & (full_xyxy[:, 2] - epsilon <= target_crops[:, 2])
        & (full_xyxy[:, 3] - epsilon <= target_crops[:, 3])
    )
    gt_ids = torch.arange(boxes.shape[0], device=boxes.device, dtype=torch.long)[complete]
    selected_xyxy = full_xyxy[complete]
    selected_crops = target_crops[complete]
    crop_extent = selected_crops[:, 2:] - selected_crops[:, :2]
    local_xyxy = torch.cat(
        (
            (selected_xyxy[:, :2] - selected_crops[:, :2]) / crop_extent,
            (selected_xyxy[:, 2:] - selected_crops[:, :2]) / crop_extent,
        ),
        dim=-1,
    )
    local_batch_indices = batch_indices[complete]
    groups = [int((local_batch_indices == index).sum().item()) for index in range(len(crops))]
    return LocalTargets(
        boxes=_xyxy_to_xywh(local_xyxy),
        classes=classes[complete],
        batch_indices=local_batch_indices,
        gt_ids=gt_ids,
        groups=groups,
    )


def crop_and_resize(images: torch.Tensor, crops: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    if crops.shape != (images.shape[0], 4):
        raise ValueError("one crop is required per image")
    output_height, output_width = images.shape[-2:]
    local_views = []
    for image, crop in zip(images, crops.detach().cpu().tolist()):
        x1, y1, x2, y2 = (int(value) for value in crop)
        if not (0 <= x1 < x2 <= output_width and 0 <= y1 < y2 <= output_height):
            raise ValueError(f"crop is outside image bounds: {crop}")
        local_views.append(
            F.interpolate(
                image[:, y1:y2, x1:x2].unsqueeze(0),
                size=(output_height, output_width),
                mode="bilinear",
                align_corners=False,
            )[0]
        )
    return torch.stack(local_views) if local_views else images.new_empty(images.shape)


@contextmanager
def preserve_batchnorm_buffers(module: torch.nn.Module):
    """Run the local branch with frozen BN statistics and live affine grads."""

    tracked = []
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm):
            tracked.append((submodule, submodule.training))
            submodule.training = False
    try:
        yield
    finally:
        for submodule, training in tracked:
            submodule.training = training


def local_to_full_xywh(
    local_boxes: torch.Tensor,
    crops: torch.Tensor,
    *,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    height, width = (int(image_hw[0]), int(image_hw[1]))
    local_xyxy = _xywh_to_xyxy(local_boxes)
    crop_values = crops.to(device=local_boxes.device, dtype=local_boxes.dtype)
    crop_low = crop_values[:, :2]
    crop_extent = crop_values[:, 2:] - crop_low
    full_xyxy = torch.cat(
        (
            crop_low + local_xyxy[:, :2] * crop_extent,
            crop_low + local_xyxy[:, 2:] * crop_extent,
        ),
        dim=-1,
    )
    return _xyxy_to_xywh(full_xyxy / local_boxes.new_tensor([width, height, width, height]))


def full_to_local_xywh(
    full_boxes: torch.Tensor,
    crops: torch.Tensor,
    *,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    height, width = (int(image_hw[0]), int(image_hw[1]))
    full_xyxy = _xywh_to_xyxy(full_boxes) * full_boxes.new_tensor([width, height, width, height])
    crop_values = crops.to(device=full_boxes.device, dtype=full_boxes.dtype)
    crop_low = crop_values[:, :2]
    crop_extent = crop_values[:, 2:] - crop_low
    local_xyxy = torch.cat(
        (
            (full_xyxy[:, :2] - crop_low) / crop_extent,
            (full_xyxy[:, 2:] - crop_low) / crop_extent,
        ),
        dim=-1,
    )
    return _xyxy_to_xywh(local_xyxy)


def join_matches_by_target_id(
    *,
    full_matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    local_matches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    local_gt_ids: torch.Tensor,
) -> JoinedMatches:
    if len(full_matches) != len(local_matches):
        raise ValueError("full and local matcher batches must have equal length")
    device = local_gt_ids.device
    batches: list[int] = []
    full_queries: list[int] = []
    local_queries: list[int] = []
    gt_ids: list[int] = []
    local_target_indices: list[int] = []

    for batch_index, ((full_q, full_t), (local_q, local_t)) in enumerate(zip(full_matches, local_matches)):
        full_by_gt = {int(target): int(query) for query, target in zip(full_q.tolist(), full_t.tolist())}
        local_by_gt: dict[int, tuple[int, int]] = {}
        for query, local_target in zip(local_q.tolist(), local_t.tolist()):
            original_gt = int(local_gt_ids[int(local_target)].item())
            if original_gt in local_by_gt:
                raise RuntimeError("local matcher produced duplicate original target identity")
            local_by_gt[original_gt] = (int(query), int(local_target))
        for original_gt, full_query in full_by_gt.items():
            local_match = local_by_gt.get(original_gt)
            if local_match is None:
                continue
            local_query, local_target = local_match
            batches.append(batch_index)
            full_queries.append(full_query)
            local_queries.append(local_query)
            gt_ids.append(original_gt)
            local_target_indices.append(local_target)

    def tensor(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long, device=device)

    return JoinedMatches(
        batch_indices=tensor(batches),
        full_query_indices=tensor(full_queries),
        local_query_indices=tensor(local_queries),
        gt_ids=tensor(gt_ids),
        local_target_indices=tensor(local_target_indices),
    )


def _aligned_giou_xywh(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_xyxy = _xywh_to_xyxy(first)
    second_xyxy = _xywh_to_xyxy(second)
    intersection_low = torch.maximum(first_xyxy[:, :2], second_xyxy[:, :2])
    intersection_high = torch.minimum(first_xyxy[:, 2:], second_xyxy[:, 2:])
    intersection = (intersection_high - intersection_low).clamp(min=0).prod(dim=-1)
    first_area = (first_xyxy[:, 2:] - first_xyxy[:, :2]).clamp(min=0).prod(dim=-1)
    second_area = (second_xyxy[:, 2:] - second_xyxy[:, :2]).clamp(min=0).prod(dim=-1)
    union = first_area + second_area - intersection
    iou = intersection / union.clamp(min=torch.finfo(first.dtype).eps)
    enclosure_low = torch.minimum(first_xyxy[:, :2], second_xyxy[:, :2])
    enclosure_high = torch.maximum(first_xyxy[:, 2:], second_xyxy[:, 2:])
    enclosure = (enclosure_high - enclosure_low).clamp(min=0).prod(dim=-1)
    return iou - (enclosure - union) / enclosure.clamp(min=torch.finfo(first.dtype).eps)


def _box_loss_per_pair(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    l1 = (student - teacher).abs().sum(dim=-1)
    return l1 + 1.0 - _aligned_giou_xywh(student, teacher)


def compute_ascv_loc_loss(
    *,
    full_pred_boxes: torch.Tensor,
    local_pred_boxes: torch.Tensor,
    full_gt_boxes: torch.Tensor,
    pair_crops: torch.Tensor,
    image_hw: tuple[int, int],
) -> ASCVLocLossResult:
    pair_count = int(full_pred_boxes.shape[0])
    if (
        local_pred_boxes.shape != full_pred_boxes.shape
        or full_gt_boxes.shape != full_pred_boxes.shape
        or pair_crops.shape != full_pred_boxes.shape
    ):
        raise ValueError("paired prediction, target, and crop tensors must all have shape [pairs, 4]")
    if pair_count == 0:
        zero = (full_pred_boxes.float().sum() + local_pred_boxes.float().sum()) * 0.0
        return ASCVLocLossResult(zero, 0, 0, 0, zero.detach(), 0, zero.detach(), 0)

    for name, tensor in (
        ("full predictions", full_pred_boxes),
        ("local predictions", local_pred_boxes),
        ("full targets", full_gt_boxes),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"ASCV-Loc received non-finite {name}")
    if bool((full_gt_boxes[:, 2:] <= 0).any()):
        raise RuntimeError("ASCV-Loc received degenerate full targets")

    # Keep geometry and GIoU in FP32 even when the surrounding train step uses
    # autocast. Tensor.float() preserves the prediction gradient paths.
    full_predictions_fp32 = full_pred_boxes.float()
    local_predictions_fp32 = local_pred_boxes.float()
    full_targets_fp32 = full_gt_boxes.float()
    height, width = (int(image_hw[0]), int(image_hw[1]))
    effective_size = torch.sqrt(
        (full_targets_fp32[:, 2] * width).clamp(min=0)
        * (full_targets_fp32[:, 3] * height).clamp(min=0)
    )
    tiny = effective_size <= ASCV_TINY_BOUNDARY_PX
    terms: list[torch.Tensor] = []
    mapped_local_full = local_to_full_xywh(
        local_predictions_fp32.detach(),
        pair_crops,
        image_hw=image_hw,
    )
    full_error = _box_loss_per_pair(full_predictions_fp32.detach(), full_targets_fp32)
    local_error = _box_loss_per_pair(mapped_local_full, full_targets_fp32)
    tiny_advantage = full_error[tiny] - local_error[tiny]
    non_tiny = ~tiny
    non_tiny_advantage = local_error[non_tiny] - full_error[non_tiny]

    if bool(tiny.any()):
        mapped_local_teacher = local_to_full_xywh(
            local_predictions_fp32[tiny].detach(),
            pair_crops[tiny],
            image_hw=image_hw,
        )
        terms.append(_box_loss_per_pair(full_predictions_fp32[tiny], mapped_local_teacher))
    if bool(non_tiny.any()):
        mapped_full_teacher = full_to_local_xywh(
            full_predictions_fp32[non_tiny].detach(),
            pair_crops[non_tiny],
            image_hw=image_hw,
        )
        terms.append(_box_loss_per_pair(local_predictions_fp32[non_tiny], mapped_full_teacher))

    loss = torch.cat(terms).mean()
    if not bool(torch.isfinite(loss).all()):
        raise RuntimeError("ASCV-Loc produced a non-finite auxiliary loss")
    tiny_count = int(tiny.sum().item())
    return ASCVLocLossResult(
        loss=loss,
        pair_count=pair_count,
        tiny_pair_count=tiny_count,
        non_tiny_pair_count=pair_count - tiny_count,
        tiny_teacher_advantage_sum=tiny_advantage.sum().detach(),
        tiny_teacher_win_count=int((tiny_advantage > 0).sum().item()),
        non_tiny_teacher_advantage_sum=non_tiny_advantage.sum().detach(),
        non_tiny_teacher_win_count=int((non_tiny_advantage > 0).sum().item()),
    )
