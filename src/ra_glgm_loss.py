"""Detached residual-difficulty targets and support supervision for RA-GLGM."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from src.btd_se_loss import binary_focal_loss
from src.fdr_loss import MatchIndices


@dataclass(frozen=True)
class ResidualDifficultyTargets:
    heatmap: Tensor
    valid_mask: Tensor
    difficulty: Tensor
    scale_boxes: Tensor | None = None
    scale_batch_idx: Tensor | None = None
    scale_values: Tensor | None = None


# Frozen 5%-quantile knots from all 343,204 valid VisDrone training instances
# after the protocol's 640-pixel letterbox transform.  Interpolating in log
# area yields an empirical-CDF target whose training prior is approximately
# uniform without imposing artificial tiny/small/regular boundaries.
SCALE_LOG_AREA_KNOTS = (
    -1.0939053609460274,
    2.407853606211777,
    2.911793703975937,
    3.2581411529031463,
    3.5404331464502383,
    3.7883982013862365,
    4.025328833331064,
    4.233540793865287,
    4.433455103796916,
    4.620838915786072,
    4.812900798763111,
    5.0039862478828026,
    5.196857416822457,
    5.400227674165377,
    5.624407461981047,
    5.872858322100196,
    6.155378738116352,
    6.482825175640593,
    6.880816055611381,
    7.4580074436281985,
    11.153424678643981,
)
SCALE_PRIOR_AUDIT_SHA256 = (
    "598487AD96F59D1E4B01DE8AA026D4C9D90251785BFE9D98016CE8A5785A2454"
)


def log_area_to_empirical_cdf(log_area: Tensor) -> Tensor:
    """Map log letterbox area to the frozen training-instance empirical CDF."""

    if not torch.is_floating_point(log_area):
        raise TypeError("log_area must be floating point")
    if not bool(torch.isfinite(log_area).all()):
        raise FloatingPointError("NONFINITE_RA_GLGM_LOG_AREA")
    knots = log_area.new_tensor(SCALE_LOG_AREA_KNOTS)
    upper = torch.bucketize(log_area.contiguous(), knots)
    lower = (upper - 1).clamp(0, len(SCALE_LOG_AREA_KNOTS) - 2)
    upper_clamped = upper.clamp(1, len(SCALE_LOG_AREA_KNOTS) - 1)
    lower_knot = knots[lower]
    upper_knot = knots[upper_clamped]
    fraction = (log_area - lower_knot) / (upper_knot - lower_knot)
    quantile_step = 1.0 / (len(SCALE_LOG_AREA_KNOTS) - 1)
    interpolated = (lower.float() + fraction) * quantile_step
    return torch.where(
        upper == 0,
        torch.zeros_like(interpolated),
        torch.where(
            upper == len(SCALE_LOG_AREA_KNOTS),
            torch.ones_like(interpolated),
            interpolated,
        ),
    ).clamp(0.0, 1.0)


def _scale_target_fields(
    targets: ResidualDifficultyTargets,
) -> tuple[Tensor, Tensor, Tensor]:
    boxes = targets.scale_boxes
    batch_idx = targets.scale_batch_idx
    values = targets.scale_values
    if boxes is None or batch_idx is None or values is None:
        raise ValueError("continuous scale targets are unavailable")
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("scale boxes must have shape [instances,4]")
    if batch_idx.ndim != 1 or values.ndim != 1:
        raise ValueError("scale batch indices and values must be one-dimensional")
    if not (len(boxes) == len(batch_idx) == len(values)):
        raise ValueError("continuous scale target fields have different lengths")
    if any(not bool(torch.isfinite(field).all()) for field in (boxes, values)):
        raise FloatingPointError("NONFINITE_RA_GLGM_SCALE_TARGET")
    if bool((values < 0).any()) or bool((values > 1).any()):
        raise ValueError("continuous scale target values must lie in [0,1]")
    if len(boxes) and (bool((boxes[:, 2:] <= 0).any())):
        raise ValueError("continuous scale target boxes must have positive area")
    return boxes, batch_idx, values


def _gaussian_weights(
    boxes: Tensor,
    *,
    height: int,
    width: int,
) -> Tensor:
    """Return per-instance P3 Gaussians normalized to exactly equal mass."""

    if not len(boxes):
        return boxes.new_empty((0, height, width))
    grid_y = torch.arange(height, device=boxes.device, dtype=torch.float32).view(1, height, 1)
    grid_x = torch.arange(width, device=boxes.device, dtype=torch.float32).view(1, 1, width)
    center_x = (boxes[:, 0] * width).view(-1, 1, 1)
    center_y = (boxes[:, 1] * height).view(-1, 1, 1)
    sigma_x = (boxes[:, 2] * width / 8.0).clamp_min(1.0).view(-1, 1, 1)
    sigma_y = (boxes[:, 3] * height / 8.0).clamp_min(1.0).view(-1, 1, 1)
    dx = (grid_x - center_x) / sigma_x
    dy = (grid_y - center_y) / sigma_y
    gaussian = torch.exp(-0.5 * (dx.square() + dy.square()))
    gaussian = gaussian * ((dx.abs() <= 3) & (dy.abs() <= 3))
    return gaussian / gaussian.sum(dim=(1, 2), keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )


def _paired_iou_xywh(first: Tensor, second: Tensor, eps: float = 1e-7) -> Tensor:
    first_half = first[:, 2:] / 2
    second_half = second[:, 2:] / 2
    first_lo, first_hi = first[:, :2] - first_half, first[:, :2] + first_half
    second_lo, second_hi = second[:, :2] - second_half, second[:, :2] + second_half
    intersection = (
        torch.minimum(first_hi, second_hi) - torch.maximum(first_lo, second_lo)
    ).clamp_min(0).prod(dim=1)
    first_area = (first_hi - first_lo).clamp_min(0).prod(dim=1)
    second_area = (second_hi - second_lo).clamp_min(0).prod(dim=1)
    return intersection / (first_area + second_area - intersection).clamp_min(eps)


def _box_slice(box: Tensor, *, height: int, width: int) -> tuple[slice, slice]:
    cx, cy, box_width, box_height = [float(value) for value in box]
    x0 = max(0, min(width - 1, math.floor((cx - box_width / 2) * width)))
    y0 = max(0, min(height - 1, math.floor((cy - box_height / 2) * height)))
    x1 = max(x0 + 1, min(width, math.ceil((cx + box_width / 2) * width)))
    y1 = max(y0 + 1, min(height, math.ceil((cy + box_height / 2) * height)))
    return slice(y0, y1), slice(x0, x1)


@torch.no_grad()
def build_residual_difficulty_targets(
    *,
    pred_bboxes: Tensor,
    pred_scores: Tensor,
    detection_bboxes: Tensor,
    detection_classes: Tensor,
    detection_batch_idx: Tensor,
    match_indices: MatchIndices,
    all_bboxes: Tensor,
    all_classes: Tensor,
    all_batch_idx: Tensor,
    height: int,
    width: int,
    chunk_size: int = 64,
) -> ResidualDifficultyTargets:
    """Rasterize FDR's detached final-layer residual difficulty on P3.

    The supplied assignment must be the already-recorded final normal decoder
    assignment.  No matcher is accepted or called here.  Ground truths absent
    from that assignment retain difficulty one.
    """

    if pred_bboxes.ndim != 3 or pred_bboxes.shape[-1] != 4:
        raise ValueError("pred_bboxes must have shape [batch,queries,4]")
    if pred_scores.ndim != 3 or pred_scores.shape[:2] != pred_bboxes.shape[:2]:
        raise ValueError("pred_scores must match prediction batch/query axes")
    if height <= 0 or width <= 0 or chunk_size <= 0:
        raise ValueError("target dimensions and chunk_size must be positive")
    batch_size = int(pred_bboxes.shape[0])
    if len(match_indices) != batch_size:
        raise ValueError("assignment batch count does not match predictions")

    device = pred_bboxes.device
    with torch.autocast(device_type=device.type, enabled=False):
        finite_inputs = (
            pred_bboxes,
            pred_scores,
            detection_bboxes,
            detection_classes,
            detection_batch_idx,
            all_bboxes,
            all_classes,
            all_batch_idx,
        )
        if any(not bool(torch.isfinite(value).all()) for value in finite_inputs):
            raise FloatingPointError("NONFINITE_RA_GLGM_TARGET_INPUT")
        boxes = detection_bboxes.detach().to(device=device, dtype=torch.float32)
        classes = detection_classes.detach().to(device=device, dtype=torch.long).view(-1)
        batch_idx = detection_batch_idx.detach().to(device=device, dtype=torch.long).view(-1)
        if not (len(boxes) == len(classes) == len(batch_idx)):
            raise ValueError("detection target fields must have matching lengths")
        if bool((classes < 0).any()):
            raise ValueError("ignored classes must not enter residual targets")

        predictions = pred_bboxes.detach().to(dtype=torch.float32)
        scores = pred_scores.detach().to(dtype=torch.float32)
        difficulty = torch.ones(len(boxes), device=device, dtype=torch.float32)
        assigned = torch.zeros(len(boxes), device=device, dtype=torch.bool)

        for image, (source, destination) in enumerate(match_indices):
            source = source.to(device=device, dtype=torch.long)
            destination = destination.to(device=device, dtype=torch.long)
            if source.ndim != 1 or destination.ndim != 1 or len(source) != len(destination):
                raise ValueError("each assignment must contain equal one-dimensional indices")
            if not len(source):
                continue
            if int(source.min()) < 0 or int(source.max()) >= predictions.shape[1]:
                raise IndexError("assigned query index is out of range")
            if int(destination.min()) < 0 or int(destination.max()) >= len(boxes):
                raise IndexError("assigned target index is out of range")
            if bool(assigned[destination].any()):
                raise ValueError("a target cannot appear twice in the final assignment")
            if bool((batch_idx[destination] != image).any()):
                raise ValueError("assignment target belongs to a different image")
            target_classes = classes[destination]
            if int(target_classes.max()) >= scores.shape[-1]:
                raise IndexError("assigned target class is out of range")
            probability = scores[image, source, target_classes].sigmoid()
            iou = _paired_iou_xywh(predictions[image, source], boxes[destination])
            difficulty[destination] = (
                0.7 * (1.0 - probability) + 0.3 * (1.0 - iou)
            ).clamp(0.25, 1.0)
            assigned[destination] = True

        heatmap = torch.zeros(
            (batch_size, 1, height, width),
            device=device,
            dtype=torch.float32,
        )
        area_640 = boxes[:, 2].clamp_min(0) * boxes[:, 3].clamp_min(0) * (640.0**2)
        scale_values = log_area_to_empirical_cdf(
            area_640.clamp_min(torch.finfo(torch.float32).tiny).log()
        )
        grid_y = torch.arange(height, device=device, dtype=torch.float32).view(1, height, 1)
        grid_x = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, width)
        for image in range(batch_size):
            image_indices = torch.nonzero(batch_idx == image, as_tuple=False).flatten()
            for start in range(0, len(image_indices), chunk_size):
                indices = image_indices[start : start + chunk_size]
                if not len(indices):
                    continue
                chunk = boxes[indices]
                center_x = (chunk[:, 0] * width).view(-1, 1, 1)
                center_y = (chunk[:, 1] * height).view(-1, 1, 1)
                sigma_x = (chunk[:, 2] * width / 8.0).clamp_min(1.0).view(-1, 1, 1)
                sigma_y = (chunk[:, 3] * height / 8.0).clamp_min(1.0).view(-1, 1, 1)
                dx = (grid_x - center_x) / sigma_x
                dy = (grid_y - center_y) / sigma_y
                gaussian = torch.exp(-0.5 * (dx.square() + dy.square()))
                gaussian = gaussian * ((dx.abs() <= 3) & (dy.abs() <= 3))
                weighted = gaussian * difficulty[indices].view(-1, 1, 1)
                heatmap[image, 0] = torch.maximum(
                    heatmap[image, 0],
                    weighted.amax(dim=0),
                )

        valid = torch.ones_like(heatmap, dtype=torch.bool)
        raw_boxes = all_bboxes.detach().to(device=device, dtype=torch.float32)
        raw_classes = all_classes.detach().to(device=device).view(-1)
        raw_batch = all_batch_idx.detach().to(device=device, dtype=torch.long).view(-1)
        if not (len(raw_boxes) == len(raw_classes) == len(raw_batch)):
            raise ValueError("unfiltered target fields must have matching lengths")
        for box, class_id, image in zip(raw_boxes, raw_classes, raw_batch):
            if float(class_id) >= 0:
                continue
            image_index = int(image)
            if image_index < 0 or image_index >= batch_size:
                raise IndexError("ignored target batch index is out of range")
            y_slice, x_slice = _box_slice(box, height=height, width=width)
            valid[image_index, 0, y_slice, x_slice] = False
        # Ignore regions suppress only negative supervision; a real positive
        # target remains valid if annotations overlap an ignored rectangle.
        valid |= heatmap > 0
        return ResidualDifficultyTargets(
            heatmap,
            valid,
            difficulty,
            boxes,
            batch_idx,
            scale_values,
        )


def residual_support_focal_loss(
    support: Tensor,
    targets: ResidualDifficultyTargets,
) -> Tensor:
    """Compute the FP32 soft focal objective on valid P3 pixels."""

    return binary_focal_loss(
        support,
        targets.heatmap,
        alpha=0.25,
        exponent=2.0,
        valid_mask=targets.valid_mask,
    )


def scale_conditioning_loss(
    predictions: Tensor,
    targets: ResidualDifficultyTargets,
    *,
    chunk_size: int = 64,
) -> Tensor:
    """FP32 SmoothL1 with one unit of supervision mass per target instance."""

    boxes, batch_idx, values = _scale_target_fields(targets)
    if predictions.ndim != 4 or predictions.shape[1] != 1:
        raise ValueError("continuous scale predictions must have shape [batch,1,H,W]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if len(batch_idx) and (
        int(batch_idx.min()) < 0 or int(batch_idx.max()) >= predictions.shape[0]
    ):
        raise IndexError("scale target batch index is out of range")
    if not bool(torch.isfinite(predictions).all()):
        raise FloatingPointError("NONFINITE_RA_GLGM_SCALE_PREDICTION")
    if not len(boxes):
        return predictions.sum() * 0.0
    height, width = map(int, predictions.shape[-2:])
    instance_losses: list[Tensor] = []
    with torch.autocast(device_type=predictions.device.type, enabled=False):
        predicted = predictions.float()
        for image in range(int(predictions.shape[0])):
            indices = torch.nonzero(batch_idx == image, as_tuple=False).flatten()
            for start in range(0, len(indices), chunk_size):
                selected = indices[start : start + chunk_size]
                if not len(selected):
                    continue
                weights = _gaussian_weights(
                    boxes[selected].float(), height=height, width=width
                )
                errors = F.smooth_l1_loss(
                    predicted[image, 0].expand(len(selected), -1, -1),
                    values[selected].float().view(-1, 1, 1).expand(-1, height, width),
                    reduction="none",
                )
                instance_losses.append((errors * weights).sum(dim=(1, 2)))
    return torch.cat(instance_losses).mean()


@torch.no_grad()
def instance_scale_predictions(
    predictions: Tensor,
    targets: ResidualDifficultyTargets,
    route_weights: Tensor | None = None,
    *,
    chunk_size: int = 64,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Pool continuous predictions and optional global-route loads per instance."""

    boxes, batch_idx, values = _scale_target_fields(targets)
    if predictions.ndim != 4 or predictions.shape[1] != 1:
        raise ValueError("continuous scale predictions must have shape [batch,1,H,W]")
    if route_weights is not None:
        expected = (predictions.shape[0], 2, predictions.shape[2], predictions.shape[3])
        if route_weights.ndim != 5 or (
            route_weights.shape[0],
            route_weights.shape[1],
            route_weights.shape[3],
            route_weights.shape[4],
        ) != expected:
            raise ValueError("route weights do not match continuous scale predictions")
    if not len(boxes):
        empty = predictions.new_empty(0, dtype=torch.float32)
        route_empty = (
            predictions.new_empty((0, int(route_weights.shape[2])), dtype=torch.float32)
            if route_weights is not None
            else None
        )
        return empty, values.float(), route_empty
    height, width = map(int, predictions.shape[-2:])
    pooled_predictions: list[Tensor] = []
    pooled_routes: list[Tensor] = []
    for image in range(int(predictions.shape[0])):
        indices = torch.nonzero(batch_idx == image, as_tuple=False).flatten()
        for start in range(0, len(indices), chunk_size):
            selected = indices[start : start + chunk_size]
            if not len(selected):
                continue
            weights = _gaussian_weights(boxes[selected].float(), height=height, width=width)
            pooled_predictions.append(
                (predictions[image, 0].float().unsqueeze(0) * weights).sum(dim=(1, 2))
            )
            if route_weights is not None:
                global_routes = route_weights[image, 1].float()
                pooled_routes.append(
                    (global_routes.unsqueeze(0) * weights.unsqueeze(1)).sum(dim=(2, 3))
                )
    return (
        torch.cat(pooled_predictions),
        values.float(),
        torch.cat(pooled_routes) if route_weights is not None else None,
    )


@torch.no_grad()
def scale_prediction_diagnostics(
    predictions: Tensor,
    targets: ResidualDifficultyTargets,
    route_weights: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return instance-balanced continuous-scale and route diagnostics."""

    predicted, target, routes = instance_scale_predictions(
        predictions, targets, route_weights
    )
    device = predictions.device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if not len(predicted):
        return {
            "scale_instances": zero,
            "scale_mae": zero,
            "scale_rmse": zero,
            "scale_prediction_mean": zero,
            "scale_prediction_std": zero,
            "scale_target_mean": zero,
            "scale_target_std": zero,
            "scale_pearson": zero,
        }
    error = predicted - target
    predicted_centered = predicted - predicted.mean()
    target_centered = target - target.mean()
    denominator = (
        predicted_centered.square().sum() * target_centered.square().sum()
    ).sqrt()
    values: dict[str, Tensor] = {
        "scale_instances": predicted.new_tensor(float(len(predicted))),
        "scale_mae": error.abs().mean(),
        "scale_rmse": error.square().mean().sqrt(),
        "scale_prediction_mean": predicted.mean(),
        "scale_prediction_std": predicted.std(unbiased=False),
        "scale_target_mean": target.mean(),
        "scale_target_std": target.std(unbiased=False),
        "scale_pearson": torch.where(
            denominator > 0,
            (predicted_centered * target_centered).sum() / denominator.clamp_min(1e-12),
            zero,
        ),
    }
    if routes is not None:
        route_probabilities = torch.stack((1.0 - routes, routes), dim=1)
        values["route_entropy"] = -(
            route_probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
            * route_probabilities
        ).sum(dim=1).mean()
        values["route_global_mean"] = routes.mean()
        values["route_global_std"] = routes.std(unbiased=False)
    return values


__all__ = [
    "ResidualDifficultyTargets",
    "SCALE_LOG_AREA_KNOTS",
    "SCALE_PRIOR_AUDIT_SHA256",
    "build_residual_difficulty_targets",
    "instance_scale_predictions",
    "log_area_to_empirical_cdf",
    "residual_support_focal_loss",
    "scale_conditioning_loss",
    "scale_prediction_diagnostics",
]
