from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class VSFRMRLossResult:
    local: torch.Tensor
    global_: torch.Tensor
    target_count: torch.Tensor
    center_correlation: torch.Tensor


def scale_targets_from_boxes(
    bboxes: torch.Tensor,
    *,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Convert augmented normalized xywh boxes to the frozen VSF scale target."""
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image_size must contain positive height and width")
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise ValueError("bboxes must have shape [N, 4]")

    boxes = bboxes.float()
    pixel_width = boxes[:, 2].clamp_min(0.0) * float(width)
    pixel_height = boxes[:, 3].clamp_min(0.0) * float(height)
    radius = torch.sqrt((pixel_width * pixel_height).clamp_min(1e-12))
    return torch.log2(radius / 8.0).clamp(min=0.05, max=1.95)


def sample_scale_field(
    scale_field: torch.Tensor,
    centers: torch.Tensor,
    batch_indices: torch.Tensor,
) -> torch.Tensor:
    """Bilinearly sample one scale-field value at each normalized GT center."""
    if scale_field.ndim != 4 or scale_field.shape[1] != 1:
        raise ValueError("scale_field must have shape [B, 1, H, W]")
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must have shape [N, 2]")
    indices = batch_indices.reshape(-1).to(device=scale_field.device, dtype=torch.long)
    if len(indices) != len(centers):
        raise ValueError("centers and batch_indices must have matching lengths")
    if len(indices) == 0:
        return scale_field.float().reshape(-1)[:0]
    if int(indices.min()) < 0 or int(indices.max()) >= scale_field.shape[0]:
        raise ValueError("batch_indices contain an out-of-range image index")

    field = scale_field.float()
    selected = field.index_select(0, indices)
    grid = (centers.to(device=field.device, dtype=torch.float32) * 2.0 - 1.0).reshape(-1, 1, 1, 2)
    return F.grid_sample(
        selected,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).reshape(-1)


def ensure_finite_vsf_losses(**losses: torch.Tensor) -> None:
    invalid = [name for name, value in losses.items() if not bool(torch.isfinite(value).all())]
    if invalid:
        raise FloatingPointError(f"NONFINITE_LOSS: VSF-RMR {', '.join(invalid)}")


def _correlation(first: torch.Tensor, second: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    if len(first) < 2:
        return zero
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = torch.sqrt(first_centered.square().sum() * second_centered.square().sum())
    return torch.where(
        denominator > 1e-12,
        (first_centered * second_centered).sum() / denominator.clamp_min(1e-12),
        zero,
    )


def compute_vsf_rmr_loss(
    *,
    scale_field: torch.Tensor,
    global_scale: torch.Tensor,
    bboxes: torch.Tensor,
    batch_indices: torch.Tensor,
    image_size: tuple[int, int],
) -> VSFRMRLossResult:
    if global_scale.ndim != 4 or global_scale.shape[1:] != (1, 1, 1):
        raise ValueError("global_scale must have shape [B, 1, 1, 1]")
    if global_scale.shape[0] != scale_field.shape[0]:
        raise ValueError("scale_field and global_scale must have matching batch sizes")

    device = scale_field.device
    with torch.autocast(device_type=device.type, enabled=False):
        field = scale_field.float()
        global_prediction = global_scale.float().reshape(-1)
        boxes = bboxes.to(device=device, dtype=torch.float32)
        indices = batch_indices.reshape(-1).to(device=device, dtype=torch.long)
        if len(boxes) != len(indices):
            raise ValueError("bboxes and batch_indices must have matching lengths")

        zero = (field.sum() + global_prediction.sum()) * 0.0
        if len(boxes) == 0:
            return VSFRMRLossResult(
                local=zero,
                global_=zero,
                target_count=torch.zeros((), device=device, dtype=torch.long),
                center_correlation=zero,
            )

        targets = scale_targets_from_boxes(boxes, image_size=image_size)
        sampled = sample_scale_field(field, boxes[:, :2], indices)
        local_terms: list[torch.Tensor] = []
        global_terms: list[torch.Tensor] = []
        for image_index in torch.unique(indices, sorted=True):
            image = int(image_index)
            if image < 0 or image >= field.shape[0]:
                raise ValueError(f"batch index {image} is outside batch size {field.shape[0]}")
            mask = indices == image_index
            local_terms.append(F.smooth_l1_loss(sampled[mask], targets[mask], reduction="mean", beta=1.0))
            global_terms.append(
                F.smooth_l1_loss(
                    global_prediction[image],
                    targets[mask].mean(),
                    reduction="mean",
                    beta=1.0,
                )
            )

        local_loss = torch.stack(local_terms).mean()
        global_loss = torch.stack(global_terms).mean()
        correlation = _correlation(sampled, targets, zero)
        ensure_finite_vsf_losses(local=local_loss, global_=global_loss, correlation=correlation)
        return VSFRMRLossResult(
            local=local_loss,
            global_=global_loss,
            target_count=torch.tensor(len(boxes), device=device, dtype=torch.long),
            center_correlation=correlation,
        )

