"""Auxiliary tiny-demand and protected-gate supervision for GCMV-EI."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class GCMVAuxiliaryLoss:
    total: torch.Tensor
    tiny: torch.Tensor
    gate: torch.Tensor
    protect: torch.Tensor


@torch.no_grad()
def build_gcmv_scale_targets(
    *,
    bboxes: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_size: int,
    feature_shape: tuple[int, int],
    image_shape: tuple[int, int],
    tiny_max_size: float = 16.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a tiny Gaussian heatmap and a non-tiny box-region mask."""

    if (
        not isinstance(bboxes, torch.Tensor)
        or bboxes.ndim != 2
        or bboxes.shape[1] != 4
        or not bboxes.is_floating_point()
    ):
        raise ValueError("bboxes must be floating-point normalized xywh [N,4]")
    indices = batch_idx.to(device=bboxes.device, dtype=torch.long).view(-1)
    if indices.numel() != bboxes.shape[0]:
        raise ValueError("batch_idx length must match bboxes")
    feature_height, feature_width = feature_shape
    image_height, image_width = image_shape
    if (
        batch_size <= 0
        or min(
            feature_height,
            feature_width,
            image_height,
            image_width,
        )
        <= 0
        or tiny_max_size <= 0
    ):
        raise ValueError("target dimensions and tiny threshold must be positive")
    if indices.numel() and (
        int(indices.min()) < 0 or int(indices.max()) >= batch_size
    ):
        raise ValueError("batch_idx exceeds batch size")

    tiny = bboxes.new_zeros(
        (batch_size, 1, feature_height, feature_width)
    )
    non_tiny = torch.zeros_like(tiny)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(
            feature_height,
            device=bboxes.device,
            dtype=bboxes.dtype,
        ),
        torch.arange(
            feature_width,
            device=bboxes.device,
            dtype=bboxes.dtype,
        ),
        indexing="ij",
    )
    stride_x = image_width / feature_width
    stride_y = image_height / feature_height

    for image_index in range(batch_size):
        boxes = bboxes[indices == image_index]
        if boxes.numel() == 0:
            continue

        width_pixels = boxes[:, 2] * image_width
        height_pixels = boxes[:, 3] * image_height
        effective_size = torch.sqrt(
            (width_pixels * height_pixels).clamp_min(0.0)
        )
        tiny_mask = effective_size <= tiny_max_size
        if tiny_mask.any():
            tiny_boxes = boxes[tiny_mask]
            tiny_widths = width_pixels[tiny_mask]
            tiny_heights = height_pixels[tiny_mask]
            center_x = (
                tiny_boxes[:, 0, None, None] * feature_width
            )
            center_y = (
                tiny_boxes[:, 1, None, None] * feature_height
            )
            sigma_x = (
                tiny_widths / (6.0 * stride_x)
            ).clamp_min(0.5)[:, None, None]
            sigma_y = (
                tiny_heights / (6.0 * stride_y)
            ).clamp_min(0.5)[:, None, None]
            gaussian = torch.exp(
                -(
                    (grid_x[None] - center_x).square()
                    / (2.0 * sigma_x.square())
                    + (grid_y[None] - center_y).square()
                    / (2.0 * sigma_y.square())
                )
            )
            tiny[image_index, 0] = gaussian.amax(dim=0)

        non_tiny_boxes = boxes[~tiny_mask]
        if non_tiny_boxes.numel() == 0:
            continue
        left = (
            non_tiny_boxes[:, 0] - non_tiny_boxes[:, 2] / 2.0
        )[:, None, None] * feature_width
        right = (
            non_tiny_boxes[:, 0] + non_tiny_boxes[:, 2] / 2.0
        )[:, None, None] * feature_width
        top = (
            non_tiny_boxes[:, 1] - non_tiny_boxes[:, 3] / 2.0
        )[:, None, None] * feature_height
        bottom = (
            non_tiny_boxes[:, 1] + non_tiny_boxes[:, 3] / 2.0
        )[:, None, None] * feature_height
        region = (
            (grid_x[None] >= left)
            & (grid_x[None] <= right)
            & (grid_y[None] >= top)
            & (grid_y[None] <= bottom)
        )
        non_tiny[image_index, 0] = region.any(dim=0).to(
            dtype=non_tiny.dtype
        )
    return tiny, non_tiny


def _focal_bce(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("focal prediction and target shapes must match")
    eps = torch.finfo(prediction.dtype).eps
    probability = prediction.clamp(eps, 1.0 - eps)
    positive = (
        -alpha
        * target
        * (1.0 - probability).pow(gamma)
        * torch.log(probability)
    )
    negative = (
        -(1.0 - alpha)
        * (1.0 - target)
        * probability.pow(gamma)
        * torch.log(1.0 - probability)
    )
    return (positive + negative).mean()


def gcmv_auxiliary_loss(
    *,
    tiny_map: torch.Tensor,
    gate_hat: torch.Tensor,
    gate: torch.Tensor,
    coverage: torch.Tensor,
    tiny_target: torch.Tensor,
    non_tiny_mask: torch.Tensor,
    lambda_tiny: float = 0.25,
    lambda_gate: float = 0.02,
    lambda_protect: float = 0.01,
) -> GCMVAuxiliaryLoss:
    """Return the frozen GCMV auxiliary loss components."""

    reference_shape = tiny_map.shape
    if any(
        value.shape != reference_shape
        for value in (
            gate_hat,
            gate,
            coverage,
            tiny_target,
            non_tiny_mask,
        )
    ):
        raise ValueError("all GCMV auxiliary maps must share shape")
    if any(
        not math.isfinite(weight) or weight < 0
        for weight in (lambda_tiny, lambda_gate, lambda_protect)
    ):
        raise ValueError("GCMV auxiliary weights must be finite and nonnegative")

    tiny_loss = _focal_bce(tiny_map, tiny_target)
    gate_target = tiny_target * coverage
    gate_loss = _focal_bce(gate_hat, gate_target)
    protect_loss = (non_tiny_mask * gate).sum() / non_tiny_mask.sum().clamp_min(
        1.0
    )
    total = (
        lambda_tiny * tiny_loss
        + lambda_gate * gate_loss
        + lambda_protect * protect_loss
    )
    return GCMVAuxiliaryLoss(
        total=total,
        tiny=tiny_loss,
        gate=gate_loss,
        protect=protect_loss,
    )
