"""Official D-FINE fine-grained distribution regression math primitives.

This module is a focused, dependency-light port of the FDR math only. It
intentionally excludes DDF, LQE, teacher distillation, matching, and decoder
integration.
"""

from __future__ import annotations

from numbers import Real

import torch
import torch.nn.functional as F
from torch import Tensor, nn


OFFICIAL_DFINE_COMMIT = "7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6"
OFFICIAL_DFINE_SOURCE_URL = (
    "https://github.com/Peterande/D-FINE/tree/"
    f"{OFFICIAL_DFINE_COMMIT}/src/zoo/dfine"
)
OFFICIAL_DFINE_UTILS_URL = (
    "https://github.com/Peterande/D-FINE/blob/"
    f"{OFFICIAL_DFINE_COMMIT}/src/zoo/dfine/dfine_utils.py"
)
OFFICIAL_DFINE_DECODER_URL = (
    "https://github.com/Peterande/D-FINE/blob/"
    f"{OFFICIAL_DFINE_COMMIT}/src/zoo/dfine/dfine_decoder.py"
)
OFFICIAL_DFINE_CRITERION_URL = (
    "https://github.com/Peterande/D-FINE/blob/"
    f"{OFFICIAL_DFINE_COMMIT}/src/zoo/dfine/dfine_criterion.py"
)

REG_MAX = 32
REG_SCALE = 4.0
UP = 0.5


def _validate_reg_max(reg_max: int) -> None:
    if isinstance(reg_max, bool) or not isinstance(reg_max, int) or reg_max < 4 or reg_max % 2:
        raise ValueError("reg_max must be an even integer greater than or equal to 4")


def _common_scalar_tensors(
    up: Tensor | Real,
    reg_scale: Tensor | Real,
) -> tuple[Tensor, Tensor]:
    reference = up if isinstance(up, Tensor) else reg_scale if isinstance(reg_scale, Tensor) else None
    if isinstance(reference, Tensor):
        if not reference.is_floating_point():
            reference = reference.to(dtype=torch.float32)
        dtype, device = reference.dtype, reference.device
    else:
        dtype, device = torch.float32, torch.device("cpu")
    up_tensor = torch.as_tensor(up, dtype=dtype, device=device).reshape(-1)
    scale_tensor = torch.as_tensor(reg_scale, dtype=dtype, device=device).reshape(-1)
    if up_tensor.numel() != 1 or scale_tensor.numel() != 1:
        raise ValueError("up and reg_scale must each contain exactly one value")
    if not torch.isfinite(up_tensor).all() or not torch.isfinite(scale_tensor).all():
        raise ValueError("up and reg_scale must be finite")
    return up_tensor, scale_tensor


def weighting_function(
    reg_max: int = REG_MAX,
    up: Tensor | Real = UP,
    reg_scale: Tensor | Real = REG_SCALE,
    deploy: bool = False,
) -> Tensor:
    """Return D-FINE's non-uniform ``W(n)`` projection.

    ``deploy`` is accepted for compatibility with the official helper. Both
    paths are numerically identical here because this module has no export-time
    graph rewriting.
    """

    del deploy
    _validate_reg_max(reg_max)
    up_tensor, scale_tensor = _common_scalar_tensors(up, reg_scale)
    upper_bound_1 = up_tensor[0].abs() * scale_tensor[0].abs()
    upper_bound_2 = upper_bound_1 * 2
    step = (upper_bound_1 + 1).pow(2 / (reg_max - 2))
    left_values = torch.stack([-(step**i) + 1 for i in range(reg_max // 2 - 1, 0, -1)])
    right_values = torch.stack([(step**i) - 1 for i in range(1, reg_max // 2)])
    return torch.cat(
        (
            -upper_bound_2.reshape(1),
            left_values,
            up_tensor.new_zeros(1),
            right_values,
            upper_bound_2.reshape(1),
        )
    )


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert normalized ``[cx, cy, w, h]`` boxes to ``[x1, y1, x2, y2]``."""

    center, size = boxes[..., :2], boxes[..., 2:]
    half_size = size / 2
    return torch.cat((center - half_size, center + half_size), dim=-1)


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """Convert normalized ``[x1, y1, x2, y2]`` boxes to ``[cx, cy, w, h]``."""

    top_left, bottom_right = boxes[..., :2], boxes[..., 2:]
    return torch.cat(((top_left + bottom_right) / 2, bottom_right - top_left), dim=-1)


def adjacent_bin_soft_labels(
    values: Tensor,
    reg_max: int = REG_MAX,
    reg_scale: Tensor | Real = REG_SCALE,
    up: Tensor | Real = UP,
) -> tuple[Tensor, Tensor, Tensor]:
    """Map continuous FDR distances to adjacent-bin indices and soft weights."""

    _validate_reg_max(reg_max)
    if not values.is_floating_point():
        raise TypeError("values must be a floating-point tensor")
    flat_values = values.reshape(-1)
    project = weighting_function(reg_max, up, reg_scale).to(
        device=flat_values.device, dtype=flat_values.dtype
    )
    differences = project.unsqueeze(0) - flat_values.unsqueeze(1)
    indices = (differences <= 0).sum(dim=1).sub(1).to(flat_values.dtype)
    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)

    valid = (indices >= 0) & (indices < reg_max)
    valid_indices = indices[valid].long()
    left_values = project[valid_indices]
    right_values = project[valid_indices + 1]
    left_differences = (flat_values[valid] - left_values).abs()
    right_differences = (right_values - flat_values[valid]).abs()
    weight_right[valid] = left_differences / (left_differences + right_differences)
    weight_left[valid] = 1 - weight_right[valid]

    below = indices < 0
    weight_right[below] = 0
    weight_left[below] = 1
    indices[below] = 0

    above = indices >= reg_max
    weight_right[above] = 1
    weight_left[above] = 0
    indices[above] = reg_max - 0.1
    return indices, weight_right, weight_left


def translate_gt(
    gt: Tensor,
    reg_max: int = REG_MAX,
    reg_scale: Tensor | Real = REG_SCALE,
    up: Tensor | Real = UP,
) -> tuple[Tensor, Tensor, Tensor]:
    """Official-name compatibility wrapper for adjacent-bin soft labels."""

    return adjacent_bin_soft_labels(gt, reg_max=reg_max, reg_scale=reg_scale, up=up)


def distance2bbox(
    points: Tensor,
    distance: Tensor,
    reg_scale: Tensor | Real = REG_SCALE,
) -> Tensor:
    """Decode normalized FDR ``[l, t, r, b]`` distances around reference boxes."""

    scale = torch.as_tensor(reg_scale, dtype=points.dtype, device=points.device).abs()
    if scale.numel() != 1 or not torch.isfinite(scale).all() or scale.item() == 0:
        raise ValueError("reg_scale must be a finite non-zero scalar")
    x1 = points[..., 0] - (0.5 * scale + distance[..., 0]) * (points[..., 2] / scale)
    y1 = points[..., 1] - (0.5 * scale + distance[..., 1]) * (points[..., 3] / scale)
    x2 = points[..., 0] + (0.5 * scale + distance[..., 2]) * (points[..., 2] / scale)
    y2 = points[..., 1] + (0.5 * scale + distance[..., 3]) * (points[..., 3] / scale)
    return xyxy_to_cxcywh(torch.stack((x1, y1, x2, y2), dim=-1))


def bbox2distance(
    points: Tensor,
    bbox: Tensor,
    reg_max: int = REG_MAX,
    reg_scale: Tensor | Real = REG_SCALE,
    up: Tensor | Real = UP,
    eps: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    """Encode normalized XYXY targets as official D-FINE adjacent-bin labels."""

    if points.shape != bbox.shape or points.shape[-1] != 4:
        raise ValueError("points and bbox must have identical [..., 4] shapes")
    scale = torch.as_tensor(reg_scale, dtype=points.dtype, device=points.device).abs()
    if scale.numel() != 1 or not torch.isfinite(scale).all() or scale.item() == 0:
        raise ValueError("reg_scale must be a finite non-zero scalar")
    flat_points = points.reshape(-1, 4)
    flat_bbox = bbox.reshape(-1, 4)
    width_unit = flat_points[:, 2] / scale + 1e-16
    height_unit = flat_points[:, 3] / scale + 1e-16
    left = (flat_points[:, 0] - flat_bbox[:, 0]) / width_unit - 0.5 * scale
    top = (flat_points[:, 1] - flat_bbox[:, 1]) / height_unit - 0.5 * scale
    right = (flat_bbox[:, 2] - flat_points[:, 0]) / width_unit - 0.5 * scale
    bottom = (flat_bbox[:, 3] - flat_points[:, 1]) / height_unit - 0.5 * scale
    distances = torch.stack((left, top, right, bottom), dim=-1)
    labels, weight_right, weight_left = translate_gt(
        distances, reg_max=reg_max, reg_scale=scale, up=up
    )
    labels = labels.clamp(min=0, max=reg_max - eps)
    return labels.reshape(-1).detach(), weight_right.detach(), weight_left.detach()


class Integral(nn.Module):
    """Integrate four FDR distributions with D-FINE's non-uniform projection."""

    def __init__(
        self,
        reg_max: int = REG_MAX,
        up: Tensor | Real = UP,
        reg_scale: Tensor | Real = REG_SCALE,
    ) -> None:
        super().__init__()
        _validate_reg_max(reg_max)
        self.reg_max = reg_max
        self.register_buffer("project", weighting_function(reg_max, up, reg_scale))

    def forward(self, logits: Tensor, project: Tensor | None = None) -> Tensor:
        expected = 4 * (self.reg_max + 1)
        if logits.ndim == 0 or logits.shape[-1] != expected:
            raise ValueError(
                f"last dimension must equal 4 * (reg_max + 1) = {expected}"
            )
        active_project = self.project if project is None else project
        if active_project.numel() != self.reg_max + 1:
            raise ValueError(f"project must contain {self.reg_max + 1} values")
        probabilities = F.softmax(logits.reshape(-1, self.reg_max + 1), dim=1)
        active_project = active_project.reshape(-1).to(
            device=probabilities.device, dtype=probabilities.dtype
        )
        distances = (probabilities * active_project).sum(dim=1).reshape(-1, 4)
        return distances.reshape(*logits.shape[:-1], 4)


def fine_grained_localization_loss(
    pred: Tensor,
    label: Tensor,
    weight_right: Tensor,
    weight_left: Tensor,
    weight: Tensor | None = None,
    reduction: str = "sum",
    avg_factor: float | Tensor | None = None,
) -> Tensor:
    """Official D-FINE adjacent-bin cross-entropy localization primitive."""

    if pred.ndim != 2:
        raise ValueError("pred must have shape [num_edges, num_bins]")
    label = label.reshape(-1)
    weight_right = weight_right.reshape(-1)
    weight_left = weight_left.reshape(-1)
    if not (pred.shape[0] == label.numel() == weight_right.numel() == weight_left.numel()):
        raise ValueError("pred and adjacent-bin targets must have the same number of edges")
    dis_left = label.long()
    dis_right = dis_left + 1
    if dis_left.numel() and (dis_left.min() < 0 or dis_right.max() >= pred.shape[1]):
        raise ValueError("adjacent-bin labels are outside the prediction range")
    left_ce = F.cross_entropy(pred, dis_left, reduction="none")
    right_ce = F.cross_entropy(pred, dis_right, reduction="none")
    loss = left_ce * weight_left.to(left_ce) + right_ce * weight_right.to(right_ce)
    if weight is not None:
        loss = loss * weight.reshape(-1).float().to(device=loss.device)

    if avg_factor is not None:
        return loss.sum() / torch.as_tensor(avg_factor, dtype=loss.dtype, device=loss.device)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be one of: none, mean, sum")


def unimodal_distribution_focal_loss(
    pred: Tensor,
    label: Tensor,
    weight_right: Tensor,
    weight_left: Tensor,
    weight: Tensor | None = None,
    reduction: str = "sum",
    avg_factor: float | Tensor | None = None,
) -> Tensor:
    """Official criterion-name compatibility wrapper for FGL."""

    return fine_grained_localization_loss(
        pred,
        label,
        weight_right,
        weight_left,
        weight=weight,
        reduction=reduction,
        avg_factor=avg_factor,
    )


__all__ = [
    "OFFICIAL_DFINE_COMMIT",
    "OFFICIAL_DFINE_SOURCE_URL",
    "OFFICIAL_DFINE_UTILS_URL",
    "OFFICIAL_DFINE_DECODER_URL",
    "OFFICIAL_DFINE_CRITERION_URL",
    "REG_MAX",
    "REG_SCALE",
    "UP",
    "Integral",
    "adjacent_bin_soft_labels",
    "bbox2distance",
    "cxcywh_to_xyxy",
    "distance2bbox",
    "fine_grained_localization_loss",
    "translate_gt",
    "unimodal_distribution_focal_loss",
    "weighting_function",
    "xyxy_to_cxcywh",
]
