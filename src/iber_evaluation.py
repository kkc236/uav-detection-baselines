"""Exact same-checkpoint metrics and frozen Gate-2 for IBER-BE v1.0."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class, box_iou

from src.itber_geometry import cxcywh_to_xyxy
from src.itber_metrics import aligned_iou, area_bucket, correction_rms


EVALUATION_CONSTANTS = {
    "seed": 0,
    "imgsz": 640,
    "batch": 8,
    "workers": 8,
    "device": "0",
    "max_det": 300,
    "nms": False,
    "cache": False,
    "conf": 0.001,
    "half": False,
    "repeats": 3,
}
IOU_THRESHOLDS = torch.linspace(0.50, 0.95, 10)
METRIC_NAMES = (
    "map",
    "ap50",
    "ap75",
    "ap_tiny",
    "ap_small",
    "precision",
    "recall",
)
_HEX64 = re.compile(r"[0-9A-Fa-f]{64}")


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_image_record(
    record: Mapping[str, torch.Tensor], *, prediction: bool
) -> None:
    required = {"boxes", "classes"} | ({"scores"} if prediction else set())
    if set(record) != required:
        raise ValueError(f"IBER-BE evaluation keys must be exactly {sorted(required)}")
    boxes = record["boxes"]
    classes = record["classes"]
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("IBER-BE evaluation boxes must have shape [N,4]")
    if classes.ndim != 1 or classes.shape[0] != boxes.shape[0]:
        raise ValueError("IBER-BE evaluation classes must have shape [N]")
    if prediction:
        scores = record["scores"]
        if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
            raise ValueError("IBER-BE evaluation scores must have shape [N]")


def _match_predictions(
    prediction_boxes: torch.Tensor,
    prediction_classes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
) -> np.ndarray:
    correct = np.zeros(
        (prediction_boxes.shape[0], len(IOU_THRESHOLDS)), dtype=bool
    )
    if prediction_boxes.shape[0] == 0 or target_boxes.shape[0] == 0:
        return correct
    iou = box_iou(cxcywh_to_xyxy(target_boxes), cxcywh_to_xyxy(prediction_boxes))
    iou = (
        iou * (target_classes[:, None] == prediction_classes[None, :])
    ).cpu().numpy()
    for column, threshold in enumerate(IOU_THRESHOLDS.tolist()):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if not matches.shape[0]:
            continue
        if matches.shape[0] > 1:
            matches = matches[
                iou[matches[:, 0], matches[:, 1]].argsort()[::-1]
            ]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), column] = True
    return correct


def _area_filtered_records(
    predictions: Sequence[Mapping[str, torch.Tensor]],
    targets: Sequence[Mapping[str, torch.Tensor]],
    *,
    image_size: int,
    bucket: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true_positives: list[np.ndarray] = []
    confidence: list[np.ndarray] = []
    predicted_classes: list[np.ndarray] = []
    target_classes: list[np.ndarray] = []
    for prediction, target in zip(predictions, targets):
        _validate_image_record(prediction, prediction=True)
        _validate_image_record(target, prediction=False)
        pred_boxes = prediction["boxes"].detach().float().cpu()
        pred_scores = prediction["scores"].detach().float().cpu()
        pred_classes = prediction["classes"].detach().long().cpu()
        gt_boxes = target["boxes"].detach().float().cpu()
        gt_classes = target["classes"].detach().long().cpu()
        if bucket is not None:
            pred_mask = area_bucket(pred_boxes, image_size=image_size) == bucket
            gt_mask = area_bucket(gt_boxes, image_size=image_size) == bucket
            pred_boxes, pred_scores, pred_classes = (
                pred_boxes[pred_mask],
                pred_scores[pred_mask],
                pred_classes[pred_mask],
            )
            gt_boxes, gt_classes = gt_boxes[gt_mask], gt_classes[gt_mask]
        true_positives.append(
            _match_predictions(pred_boxes, pred_classes, gt_boxes, gt_classes)
        )
        confidence.append(pred_scores.numpy())
        predicted_classes.append(pred_classes.numpy())
        target_classes.append(gt_classes.numpy())
    tp = (
        np.concatenate(true_positives, axis=0)
        if true_positives
        else np.zeros((0, 10), dtype=bool)
    )
    conf = (
        np.concatenate(confidence)
        if confidence
        else np.zeros(0, dtype=np.float32)
    )
    pred_cls = (
        np.concatenate(predicted_classes)
        if predicted_classes
        else np.zeros(0, dtype=np.int64)
    )
    target_cls = (
        np.concatenate(target_classes)
        if target_classes
        else np.zeros(0, dtype=np.int64)
    )
    return tp, conf, pred_cls, target_cls


def _summarize_ap(
    predictions: Sequence[Mapping[str, torch.Tensor]],
    targets: Sequence[Mapping[str, torch.Tensor]],
    *,
    image_size: int,
    bucket: int | None,
) -> dict[str, float]:
    tp, conf, pred_cls, target_cls = _area_filtered_records(
        predictions,
        targets,
        image_size=image_size,
        bucket=bucket,
    )
    empty = {
        "map": 0.0,
        "ap50": 0.0,
        "ap75": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    if target_cls.size == 0 or pred_cls.size == 0:
        return empty
    result = ap_per_class(tp, conf, pred_cls, target_cls, plot=False)
    precision, recall, ap = result[2], result[3], result[5]
    if ap.size == 0:
        return empty
    summary = {
        "map": float(ap.mean()),
        "ap50": float(ap[:, 0].mean()),
        "ap75": float(ap[:, 5].mean()),
        "precision": float(precision.mean()) if precision.size else 0.0,
        "recall": float(recall.mean()) if recall.size else 0.0,
    }
    if not all(math.isfinite(value) for value in summary.values()):
        raise FloatingPointError("non-finite IBER-BE AP summary")
    return summary


def compute_detection_metrics(
    predictions: Sequence[Mapping[str, torch.Tensor]],
    targets: Sequence[Mapping[str, torch.Tensor]],
    *,
    image_size: int,
) -> dict[str, float]:
    """Compute fixed no-NMS full, tiny, and small AP from normalized boxes."""
    if len(predictions) != len(targets) or not predictions:
        raise ValueError(
            "IBER-BE predictions and targets must have equal nonzero image counts"
        )
    full = _summarize_ap(
        predictions, targets, image_size=image_size, bucket=None
    )
    tiny = _summarize_ap(predictions, targets, image_size=image_size, bucket=0)
    small = _summarize_ap(predictions, targets, image_size=image_size, bucket=1)
    return {
        "map": full["map"],
        "ap50": full["ap50"],
        "ap75": full["ap75"],
        "ap_tiny": tiny["map"],
        "ap_small": small["map"],
        "precision": full["precision"],
        "recall": full["recall"],
    }


def compute_refinement_diagnostics(
    stock_boxes: torch.Tensor,
    refined_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    match_indices: list[tuple[torch.Tensor, torch.Tensor]],
    effective_correction: torch.Tensor,
    gates: torch.Tensor,
    residuals: torch.Tensor,
    f3_boundary_features: torch.Tensor,
    rgb_boundary_features: torch.Tensor,
    *,
    target_group_sizes: Sequence[int],
) -> dict[str, Any]:
    """Measure edge error, matched IoU direction, leakage, and route activity."""
    if (
        stock_boxes.shape != refined_boxes.shape
        or stock_boxes.ndim != 3
        or stock_boxes.shape[-1] != 4
    ):
        raise ValueError("IBER-BE stock/refined boxes must have equal [B,Q,4] shapes")
    for name, value in (
        ("effective_correction", effective_correction),
        ("gates", gates),
        ("residuals", residuals),
    ):
        if value.shape != stock_boxes.shape:
            raise ValueError(f"IBER-BE {name} must have shape [B,Q,4]")
    prefix = (*stock_boxes.shape[:2], 4)
    if f3_boundary_features.shape[:3] != prefix:
        raise ValueError("IBER-BE F3 boundary features have invalid shape")
    if rgb_boundary_features.shape[:3] != prefix:
        raise ValueError("IBER-BE RGB boundary features have invalid shape")
    if len(match_indices) != stock_boxes.shape[0]:
        raise ValueError("IBER-BE match index batch count mismatch")
    if (
        isinstance(target_group_sizes, (str, bytes))
        or not isinstance(target_group_sizes, Sequence)
        or len(target_group_sizes) != stock_boxes.shape[0]
        or any(type(value) is not int or value < 0 for value in target_group_sizes)
        or sum(target_group_sizes) != target_boxes.shape[0]
    ):
        raise ValueError("IBER-BE target group sizes are invalid")

    device = stock_boxes.device
    matched_mask = torch.zeros(
        stock_boxes.shape[:2], dtype=torch.bool, device=device
    )
    stock_iou_parts: list[torch.Tensor] = []
    refined_iou_parts: list[torch.Tensor] = []
    stock_edge_error_parts: list[torch.Tensor] = []
    refined_edge_error_parts: list[torch.Tensor] = []
    target_edges = cxcywh_to_xyxy(
        target_boxes.to(device=device, dtype=stock_boxes.dtype)
    )
    target_offset = 0
    for image_index, (source, destination) in enumerate(match_indices):
        source = source.to(device=device, dtype=torch.long)
        destination = destination.to(device=device, dtype=torch.long)
        if source.numel() != destination.numel():
            raise ValueError("IBER-BE matcher source/target lengths differ")
        if source.numel() and (
            int(source.min()) < 0 or int(source.max()) >= stock_boxes.shape[1]
        ):
            raise ValueError("IBER-BE matcher query index is out of range")
        target_end = target_offset + target_group_sizes[image_index]
        if destination.numel() and (
            int(destination.min()) < target_offset
            or int(destination.max()) >= target_end
        ):
            raise ValueError(
                f"IBER-BE matcher target crosses image boundary at image {image_index}"
            )
        if not len(source):
            target_offset = target_end
            continue
        matched_mask[image_index, source] = True
        selected_target = target_edges[destination]
        selected_stock = cxcywh_to_xyxy(stock_boxes[image_index, source])
        selected_refined = cxcywh_to_xyxy(refined_boxes[image_index, source])
        stock_iou_parts.append(aligned_iou(selected_stock, selected_target))
        refined_iou_parts.append(aligned_iou(selected_refined, selected_target))
        stock_edge_error_parts.append((selected_stock - selected_target).abs())
        refined_edge_error_parts.append((selected_refined - selected_target).abs())
        target_offset = target_end
    if not stock_iou_parts:
        raise ValueError("IBER-BE evaluation has no matched validation targets")
    stock_iou = torch.cat(stock_iou_parts).float()
    refined_iou = torch.cat(refined_iou_parts).float()
    iou_delta = refined_iou - stock_iou
    stock_edge_error = torch.cat(stock_edge_error_parts).float()
    refined_edge_error = torch.cat(refined_edge_error_parts).float()
    correction = effective_correction.float()
    matched_rms = float(correction_rms(correction, matched_mask).detach().cpu())
    unmatched_rms = float(
        correction_rms(correction, ~matched_mask).detach().cpu()
    )
    ratio = unmatched_rms / matched_rms if matched_rms > 0 else math.inf
    gate_values = gates.detach().float().reshape(-1)
    residual_values = residuals.detach().float().reshape(-1)
    report = {
        "matched_count": int(iou_delta.numel()),
        "matched_improved": int((iou_delta > 0).sum().cpu()),
        "matched_degraded": int((iou_delta < 0).sum().cpu()),
        "matched_equal": int((iou_delta == 0).sum().cpu()),
        "stock_iou_mean": float(stock_iou.mean().cpu()),
        "refined_iou_mean": float(refined_iou.mean().cpu()),
        "matched_iou_delta_mean": float(iou_delta.mean().cpu()),
        "stock_edge_mae": float(stock_edge_error.mean().cpu()),
        "refined_edge_mae": float(refined_edge_error.mean().cpu()),
        "edge_mae_delta": float(
            (refined_edge_error.mean() - stock_edge_error.mean()).cpu()
        ),
        "matched_correction_rms": matched_rms,
        "unmatched_correction_rms": unmatched_rms,
        "unmatched_to_matched_rms_ratio": ratio,
        "f3_embedding_rms": float(
            f3_boundary_features.detach().float().square().mean().sqrt().cpu()
        ),
        "rgb_embedding_rms": float(
            rgb_boundary_features.detach().float().square().mean().sqrt().cpu()
        ),
        "gate_mean": float(gate_values.mean().cpu()),
        "gate_std": float(gate_values.std(unbiased=False).cpu()),
        "gate_p05": float(torch.quantile(gate_values, 0.05).cpu()),
        "gate_p95": float(torch.quantile(gate_values, 0.95).cpu()),
        "residual_rms": float(residual_values.square().mean().sqrt().cpu()),
        "residual_abs_p95": float(
            torch.quantile(residual_values.abs(), 0.95).cpu()
        ),
    }
    report["finite"] = all(
        _finite_number(value)
        for value in report.values()
        if not isinstance(value, int)
    )
    return report


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def assert_repeated_evaluations(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(reports) != EVALUATION_CONSTANTS["repeats"]:
        raise ValueError("IBER-BE evaluation requires exactly 3 repeats")
    if not reports:
        raise ValueError("IBER-BE evaluation repeats cannot be empty")
    reference = _canonical_bytes(reports[0])
    for index, report in enumerate(reports, start=1):
        if _canonical_bytes(report) != reference:
            raise ValueError(f"IBER-BE evaluation repeat {index} differs from repeat 1")
    return dict(reports[0])


def _metrics_valid(metrics: Mapping[str, Any]) -> bool:
    return isinstance(metrics, Mapping) and all(
        name in metrics
        and _finite_number(metrics[name])
        and 0.0 <= float(metrics[name]) <= 1.0
        for name in METRIC_NAMES
    )


def _diagnostics_schema_valid(
    diagnostics: Mapping[str, Any], *, require_detector_hashes: bool
) -> bool:
    if not isinstance(diagnostics, Mapping) or diagnostics.get("finite") is not True:
        return False
    count_names = (
        "matched_count",
        "matched_improved",
        "matched_degraded",
        "matched_equal",
    )
    if not all(
        type(diagnostics.get(name)) is int and diagnostics[name] >= 0
        for name in count_names
    ):
        return False
    if (
        diagnostics["matched_count"] <= 0
        or diagnostics["matched_count"]
        != diagnostics["matched_improved"]
        + diagnostics["matched_degraded"]
        + diagnostics["matched_equal"]
    ):
        return False
    finite_names = (
        "stock_iou_mean",
        "refined_iou_mean",
        "matched_iou_delta_mean",
        "stock_edge_mae",
        "refined_edge_mae",
        "edge_mae_delta",
        "matched_correction_rms",
        "unmatched_correction_rms",
        "unmatched_to_matched_rms_ratio",
        "f3_embedding_rms",
        "rgb_embedding_rms",
        "gate_mean",
        "gate_std",
        "gate_p05",
        "gate_p95",
        "residual_rms",
        "residual_abs_p95",
    )
    if not all(_finite_number(diagnostics.get(name)) for name in finite_names):
        return False
    if not (
        0.0 <= diagnostics["stock_iou_mean"] <= 1.0
        and 0.0 <= diagnostics["refined_iou_mean"] <= 1.0
        and -1.0 <= diagnostics["matched_iou_delta_mean"] <= 1.0
        and diagnostics["stock_edge_mae"] >= 0.0
        and diagnostics["refined_edge_mae"] >= 0.0
        and diagnostics["matched_correction_rms"] >= 0.0
        and diagnostics["unmatched_correction_rms"] >= 0.0
        and diagnostics["unmatched_to_matched_rms_ratio"] >= 0.0
        and diagnostics["f3_embedding_rms"] >= 0.0
        and diagnostics["rgb_embedding_rms"] >= 0.0
        and 0.0 <= diagnostics["gate_mean"] <= 1.0
        and diagnostics["gate_std"] >= 0.0
        and 0.0 <= diagnostics["gate_p05"] <= diagnostics["gate_p95"] <= 1.0
        and diagnostics["residual_rms"] >= 0.0
        and 0.0 <= diagnostics["residual_abs_p95"] <= 1.0
    ):
        return False
    if require_detector_hashes:
        before = diagnostics.get("detector_sha_before")
        after = diagnostics.get("detector_sha_after")
        if (
            not isinstance(before, str)
            or _HEX64.fullmatch(before) is None
            or not isinstance(after, str)
            or _HEX64.fullmatch(after) is None
        ):
            return False
    return True


def _repeat_schema_valid(repeats: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        isinstance(report, Mapping)
        and _metrics_valid(report.get("stock"))
        and _metrics_valid(report.get("refined"))
        and _diagnostics_schema_valid(
            report.get("diagnostics"), require_detector_hashes=False
        )
        for report in repeats
    )


def _decimal_delta(
    refined: Mapping[str, Any], stock: Mapping[str, Any], name: str
) -> Decimal:
    return Decimal(str(refined[name])) - Decimal(str(stock[name]))


def finite_noncollapsed_activity(diagnostics: Mapping[str, Any]) -> bool:
    required = (
        "matched_correction_rms",
        "unmatched_correction_rms",
        "f3_embedding_rms",
        "rgb_embedding_rms",
        "gate_mean",
        "gate_std",
        "gate_p05",
        "gate_p95",
        "residual_rms",
        "residual_abs_p95",
    )
    if diagnostics.get("finite") is not True or not all(
        _finite_number(diagnostics.get(name)) for name in required
    ):
        return False
    detector_unchanged = (
        isinstance(diagnostics.get("detector_sha_before"), str)
        and diagnostics.get("detector_sha_before")
        == diagnostics.get("detector_sha_after")
    )
    return bool(
        detector_unchanged
        and diagnostics["matched_correction_rms"] > 1e-6
        and diagnostics["f3_embedding_rms"] > 0.0
        and diagnostics["rgb_embedding_rms"] > 0.0
        and diagnostics["gate_mean"] > 1e-4
        and diagnostics["gate_std"] > 1e-6
        and 0.0 <= diagnostics["gate_p05"] < diagnostics["gate_p95"]
        and 1e-3 < diagnostics["gate_p95"] < 0.999
        and diagnostics["residual_rms"] > 1e-4
        and 1e-4 < diagnostics["residual_abs_p95"] < 0.999
    )


def _exact_repeatability(repeats: Sequence[Mapping[str, Any]]) -> bool:
    try:
        assert_repeated_evaluations(repeats)
    except (TypeError, ValueError):
        return False
    return True


def _valid_last5(values: Sequence[float]) -> bool:
    return (
        isinstance(values, Sequence)
        and not isinstance(values, (str, bytes))
        and len(values) == 5
        and all(_finite_number(value) for value in values)
    )


def evaluate_gate2(
    stock: Mapping[str, Any],
    refined: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    *,
    repeats: Sequence[Mapping[str, Any]],
    last5_stock_map: Sequence[float],
    last5_refined_map: Sequence[float],
    checkpoint_epoch: int,
) -> dict[str, Any]:
    """Apply every frozen epoch-30 Gate-2 condition without rounding."""
    metrics_valid = _metrics_valid(stock) and _metrics_valid(refined)
    repeatability = _exact_repeatability(repeats) and _repeat_schema_valid(repeats)
    valid_last5 = _valid_last5(last5_stock_map) and _valid_last5(
        last5_refined_map
    )
    diagnostics_schema = _diagnostics_schema_valid(
        diagnostics, require_detector_hashes=True
    )
    detector_unchanged = diagnostics_schema and (
        diagnostics.get("detector_sha_before")
        == diagnostics.get("detector_sha_after")
    )
    epoch_valid = type(checkpoint_epoch) is int and checkpoint_epoch == 30
    if metrics_valid:
        exact_delta = {
            name: _decimal_delta(refined, stock, name)
            for name in ("map", "ap50", "ap75", "ap_tiny", "ap_small")
        }
    else:
        exact_delta = {
            name: Decimal("NaN")
            for name in ("map", "ap50", "ap75", "ap_tiny", "ap_small")
        }
    matched_counts = (
        type(diagnostics.get("matched_improved")) is int
        and type(diagnostics.get("matched_degraded")) is int
        and diagnostics["matched_improved"] > diagnostics["matched_degraded"]
    )
    unmatched_safe = (
        _finite_number(diagnostics.get("matched_correction_rms"))
        and _finite_number(diagnostics.get("unmatched_correction_rms"))
        and Decimal(str(diagnostics["unmatched_correction_rms"]))
        <= Decimal("0.25")
        * Decimal(str(diagnostics["matched_correction_rms"]))
    )
    last5_condition = valid_last5 and math.fsum(last5_refined_map) > math.fsum(
        last5_stock_map
    )
    conditions = {
        "map": metrics_valid and exact_delta["map"] >= Decimal("0.0020"),
        "ap75": metrics_valid and exact_delta["ap75"] >= Decimal("0.0030"),
        "ap50": metrics_valid and exact_delta["ap50"] >= Decimal("-0.0005"),
        "tiny_or_small": metrics_valid
        and (
            exact_delta["ap_tiny"] > Decimal("0")
            or exact_delta["ap_small"] > Decimal("0")
        ),
        "matched_counts": matched_counts,
        "unmatched_rms": unmatched_safe,
        "activity": finite_noncollapsed_activity(diagnostics),
        "repeatability": repeatability,
        "last5": last5_condition,
    }
    engineering = {
        "checkpoint_epoch30": epoch_valid,
        "finite_metric_schema": metrics_valid,
        "diagnostics_finite": diagnostics_schema,
        "detector_unchanged": detector_unchanged,
        "repeatability": repeatability,
        "last5_history": valid_last5,
    }
    if not all(engineering.values()):
        status = "engineering_invalid"
    else:
        status = "passed" if all(conditions.values()) else "scientific_failed"
    return {
        "status": status,
        "checkpoint_epoch": checkpoint_epoch,
        "conditions": conditions,
        "engineering": engineering,
        "exact_delta": {
            name: float(value) if value.is_finite() else None
            for name, value in exact_delta.items()
        },
        "last5_stock_map": list(last5_stock_map),
        "last5_refined_map": list(last5_refined_map),
        "last5_stock_mean": (
            math.fsum(last5_stock_map) / 5 if valid_last5 else None
        ),
        "last5_refined_mean": (
            math.fsum(last5_refined_map) / 5 if valid_last5 else None
        ),
    }


__all__ = [
    "EVALUATION_CONSTANTS",
    "assert_repeated_evaluations",
    "compute_detection_metrics",
    "compute_refinement_diagnostics",
    "evaluate_gate2",
    "finite_noncollapsed_activity",
]
