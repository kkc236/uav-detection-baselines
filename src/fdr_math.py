"""Pinned official D-FINE fine-grained distribution regression primitives.

Only pure FDR math belongs here. Decoder heads, loss integration, matching,
DDF, LQE, and teacher distillation intentionally live outside this module.
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


def _scalar_tensors(
    up: Tensor | Real,
    reg_scale: Tensor | Real,
) -> tuple[Tensor, Tensor]:
    reference = up if isinstance(up, Tensor) else reg_scale if isinstance(reg_scale, Tensor) else None
    if isinstance(reference, Tensor):
        if not reference.is_floating_point():
            reference = reference.float()
        dtype, device = reference.dtype, reference.device
    else:
        dtype, device = torch.float32, torch.device("cpu")
    up_tensor = torch.as_tensor(up, dtype=dtype, device=device).reshape(-1)
    scale_tensor = torch.as_tensor(reg_scale, dtype=dtype, device=device).reshape(-1)
    if up_tensor.numel() != 1 or scale_tensor.numel() != 1:
        raise ValueError("up and reg_scale must each contain exactly one value")
    return up_tensor, scale_tensor


def weighting_function(
    reg_max: int = REG_MAX,
    up: Tensor | Real = UP,
    reg_scale: Tensor | Real = REG_SCALE,
    deploy: bool = False,
) -> Tensor:
    """Generate the official non-uniform D-FINE weighting function ``W(n)``."""

    _validate_reg_max(reg_max)
    up, reg_scale = _scalar_tensors(up, reg_scale)
    if deploy:
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.tensor(values, dtype=up.dtype, device=up.device)

    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
    values = (
        [-upper_bound2]
        + left_values
        + [torch.zeros_like(up[0][None])]
        + right_values
        + [upper_bound2]
    )
    return torch.cat(values, 0)


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert normalized ``[cx, cy, w, h]`` boxes to XYXY."""

    x_center, y_center, width, height = boxes.unbind(-1)
    converted = [
        x_center - 0.5 * width.clamp(min=0.0),
        y_center - 0.5 * height.clamp(min=0.0),
        x_center + 0.5 * width.clamp(min=0.0),
        y_center + 0.5 * height.clamp(min=0.0),
    ]
    return torch.stack(converted, dim=-1)


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """Convert normalized ``[x1, y1, x2, y2]`` boxes to CXCYWH."""

    x0, y0, x1, y1 = boxes.unbind(-1)
    converted = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    return torch.stack(converted, dim=-1)


def translate_gt(
    gt: Tensor,
    reg_max: int = REG_MAX,
    reg_scale: Tensor | Real = REG_SCALE,
    up: Tensor | Real = UP,
) -> tuple[Tensor, Tensor, Tensor]:
    """Translate continuous distances into adjacent-bin indices and weights."""

    gt = gt.reshape(-1)
    function_values = weighting_function(reg_max, up, reg_scale)
    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
    mask = diffs <= 0
    closest_left_indices = torch.sum(mask, dim=1) - 1
    indices = closest_left_indices.float()
    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)

    valid_idx_mask = (indices >= 0) & (indices < reg_max)
    valid_indices = indices[valid_idx_mask].long()
    left_values = function_values[valid_indices]
    right_values = function_values[valid_indices + 1]
    left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
    right_diffs = torch.abs(right_values - gt[valid_idx_mask])
    weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]

    invalid_idx_mask_neg = indices < 0
    weight_right[invalid_idx_mask_neg] = 0.0
    weight_left[invalid_idx_mask_neg] = 1.0
    indices[invalid_idx_mask_neg] = 0.0

    invalid_idx_mask_pos = indices >= reg_max
    weight_right[invalid_idx_mask_pos] = 1.0
    weight_left[invalid_idx_mask_pos] = 0.0
    indices[invalid_idx_mask_pos] = reg_max - 0.1
    return indices, weight_right, weight_left


def adjacent_bin_soft_labels(
    values: Tensor,
    reg_max: int = REG_MAX,
    reg_scale: Tensor | Real = REG_SCALE,
    up: Tensor | Real = UP,
) -> tuple[Tensor, Tensor, Tensor]:
    """Descriptive alias for the official ``translate_gt`` primitive."""

    return translate_gt(values, reg_max, reg_scale, up)


def distance2bbox(
    points: Tensor,
    distance: Tensor,
    reg_scale: Tensor | Real = REG_SCALE,
) -> Tensor:
    """Decode FDR ``[left, top, right, bottom]`` distances around references."""

    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (
        points[..., 2] / reg_scale
    )
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (
        points[..., 3] / reg_scale
    )
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (
        points[..., 2] / reg_scale
    )
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (
        points[..., 3] / reg_scale
    )
    bboxes = torch.stack([x1, y1, x2, y2], -1)
    return xyxy_to_cxcywh(bboxes)


def bbox2distance(
    points: Tensor,
    bbox: Tensor,
    reg_max: int = REG_MAX,
    reg_scale: Tensor | Real = REG_SCALE,
    up: Tensor | Real = UP,
    eps: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    """Encode normalized XYXY targets as official D-FINE FGL targets."""

    reg_scale = abs(reg_scale)
    left = (points[:, 0] - bbox[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    top = (points[:, 1] - bbox[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    right = (bbox[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    bottom = (bbox[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    four_lens = torch.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = translate_gt(
        four_lens, reg_max, reg_scale, up
    )
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max - eps)
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()


class Integral(nn.Module):
    """Mechanically faithful non-uniform integral from the D-FINE decoder."""

    def __init__(
        self,
        reg_max: int = REG_MAX,
        up: Tensor | Real = UP,
        reg_scale: Tensor | Real = REG_SCALE,
    ) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("project", weighting_function(reg_max, up, reg_scale))

    def forward(self, x: Tensor, project: Tensor | None = None) -> Tensor:
        expected = 4 * (self.reg_max + 1)
        if x.ndim == 0 or x.shape[-1] != expected:
            raise ValueError(
                f"last dimension must equal 4 * (reg_max + 1) = {expected}"
            )
        project = self.project if project is None else project
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


def fine_grained_localization_loss(
    pred: Tensor,
    label: Tensor,
    weight_right: Tensor,
    weight_left: Tensor,
    weight: Tensor | None = None,
    reduction: str = "sum",
    avg_factor: float | Tensor | None = None,
) -> Tensor:
    """Mechanically faithful official adjacent-bin FGL loss primitive."""

    dis_left = label.long()
    dis_right = dis_left + 1
    loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(
        -1
    ) + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)
    if weight is not None:
        weight = weight.float()
        loss = loss * weight
    if avg_factor is not None:
        loss = loss.sum() / avg_factor
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def unimodal_distribution_focal_loss(
    pred: Tensor,
    label: Tensor,
    weight_right: Tensor,
    weight_left: Tensor,
    weight: Tensor | None = None,
    reduction: str = "sum",
    avg_factor: float | Tensor | None = None,
) -> Tensor:
    """Official criterion-name alias for the FGL primitive."""

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
