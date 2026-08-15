"""TEST-ONLY GOLDEN REFERENCE for selected official D-FINE FDR formulas.

Source: https://github.com/Peterande/D-FINE
Commit: 7fe2f8889f0b7b817f20c315b40fc15a4fb64ae6
License: Apache-2.0; see the upstream LICENSE at the pinned commit.

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR);
Copyright (c) 2023 lyuwenyu. All Rights Reserved.

This vendored subset is an immutable test oracle, not production code. It keeps
only the official weighting/Integral, box-transform, target-interpolation, and
Fine-Grained Localization loss formulas needed for parity tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = (
    "Integral",
    "weighting_function",
    "translate_gt",
    "distance2bbox",
    "bbox2distance",
    "unimodal_distribution_focal_loss",
)


def _box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Minimal private dependency for the pinned ``distance2bbox`` formula."""
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0), dim=-1)


def weighting_function(reg_max, up, reg_scale, deploy=False):
    """Generate the official non-uniform weighting sequence W(n)."""
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
    else:
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


def translate_gt(gt, reg_max, reg_scale, up):
    """Map continuous edge offsets to adjacent official non-uniform bins."""
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


def distance2bbox(points, distance, reg_scale):
    """Decode official edge distances relative to ``cxcywh`` reference boxes."""
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
    return _box_xyxy_to_cxcywh(bboxes)


def bbox2distance(points, bbox, reg_max, reg_scale, up, eps=0.1):
    """Encode matched ``xyxy`` boxes as official adjacent-bin edge targets."""
    reg_scale = abs(reg_scale)
    left = (points[:, 0] - bbox[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    top = (points[:, 1] - bbox[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    right = (bbox[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    bottom = (bbox[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    four_lens = torch.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = translate_gt(four_lens, reg_max, reg_scale, up)
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max - eps)
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()


class Integral(nn.Module):
    """Apply the official softmax Integral over non-uniform bin weights."""

    def __init__(self, reg_max=32):
        super(Integral, self).__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


def unimodal_distribution_focal_loss(
    pred,
    label,
    weight_right,
    weight_left,
    weight=None,
    reduction="sum",
    avg_factor=None,
):
    """Compute the official adjacent-bin Fine-Grained Localization loss."""
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
