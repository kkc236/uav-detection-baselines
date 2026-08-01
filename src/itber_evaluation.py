"""Exact same-checkpoint evaluation and pre-registered I-TBER decisions."""

from __future__ import annotations

import copy
import json
import math
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

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


def _validate_image_record(record: Mapping[str, torch.Tensor], *, prediction: bool) -> None:
    required = {"boxes", "classes"} | ({"scores"} if prediction else set())
    if set(record) != required:
        raise ValueError(f"evaluation record keys must be exactly {sorted(required)}")
    boxes = record["boxes"]
    classes = record["classes"]
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("evaluation boxes must have shape [N,4]")
    if classes.ndim != 1 or classes.shape[0] != boxes.shape[0]:
        raise ValueError("evaluation classes must have shape [N]")
    if prediction:
        scores = record["scores"]
        if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
            raise ValueError("evaluation scores must have shape [N]")


def _match_predictions(
    prediction_boxes: torch.Tensor,
    prediction_classes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
) -> np.ndarray:
    """Match detections exactly like the Ultralytics non-SciPy validator path."""
    correct = np.zeros((prediction_boxes.shape[0], len(IOU_THRESHOLDS)), dtype=bool)
    if prediction_boxes.shape[0] == 0 or target_boxes.shape[0] == 0:
        return correct
    iou = box_iou(cxcywh_to_xyxy(target_boxes), cxcywh_to_xyxy(prediction_boxes))
    iou = (iou * (target_classes[:, None] == prediction_classes[None, :])).cpu().numpy()
    for column, threshold in enumerate(IOU_THRESHOLDS.tolist()):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if not matches.shape[0]:
            continue
        if matches.shape[0] > 1:
            matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
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
    tp = np.concatenate(true_positives, axis=0) if true_positives else np.zeros((0, 10), dtype=bool)
    conf = np.concatenate(confidence) if confidence else np.zeros(0, dtype=np.float32)
    pred_cls = np.concatenate(predicted_classes) if predicted_classes else np.zeros(0, dtype=np.int64)
    target_cls = np.concatenate(target_classes) if target_classes else np.zeros(0, dtype=np.int64)
    return tp, conf, pred_cls, target_cls


def _summarize_ap(
    predictions: Sequence[Mapping[str, torch.Tensor]],
    targets: Sequence[Mapping[str, torch.Tensor]],
    *,
    image_size: int,
    bucket: int | None,
) -> dict[str, float]:
    tp, conf, pred_cls, target_cls = _area_filtered_records(
        predictions, targets, image_size=image_size, bucket=bucket
    )
    if target_cls.size == 0:
        return {"map": 0.0, "ap50": 0.0, "ap75": 0.0, "precision": 0.0, "recall": 0.0}
    if pred_cls.size == 0:
        return {"map": 0.0, "ap50": 0.0, "ap75": 0.0, "precision": 0.0, "recall": 0.0}
    result = ap_per_class(tp, conf, pred_cls, target_cls, plot=False)
    precision, recall, ap = result[2], result[3], result[5]
    if ap.size == 0:
        return {"map": 0.0, "ap50": 0.0, "ap75": 0.0, "precision": 0.0, "recall": 0.0}
    summary = {
        "map": float(ap.mean()),
        "ap50": float(ap[:, 0].mean()),
        "ap75": float(ap[:, 5].mean()),
        "precision": float(precision.mean()) if precision.size else 0.0,
        "recall": float(recall.mean()) if recall.size else 0.0,
    }
    if not all(math.isfinite(value) for value in summary.values()):
        raise FloatingPointError("non-finite I-TBER AP summary")
    return summary


def compute_detection_metrics(
    predictions: Sequence[Mapping[str, torch.Tensor]],
    targets: Sequence[Mapping[str, torch.Tensor]],
    *,
    image_size: int,
) -> dict[str, float]:
    """Compute fixed no-NMS full and size-stratified AP from normalized boxes.

    Tiny and small AP use both target and prediction area buckets in the fixed
    640x640 letterbox coordinate system: tiny <16^2 px and small <32^2 px.
    """
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("predictions and targets must contain the same non-zero image count")
    full = _summarize_ap(predictions, targets, image_size=image_size, bucket=None)
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
) -> dict[str, Any]:
    """Measure stock-assignment IoU changes, correction leakage, and activity."""
    if stock_boxes.shape != refined_boxes.shape or stock_boxes.ndim != 3 or stock_boxes.shape[-1] != 4:
        raise ValueError("stock/refined boxes must have equal [B,Q,4] shapes")
    if effective_correction.shape != stock_boxes.shape:
        raise ValueError("effective correction must have shape [B,Q,4]")
    if gates.shape != stock_boxes.shape or residuals.shape != stock_boxes.shape:
        raise ValueError("gates and residuals must have shape [B,Q,4]")
    if len(match_indices) != stock_boxes.shape[0]:
        raise ValueError("match index batch count mismatch")
    device = stock_boxes.device
    matched_mask = torch.zeros(stock_boxes.shape[:2], dtype=torch.bool, device=device)
    stock_iou_parts: list[torch.Tensor] = []
    refined_iou_parts: list[torch.Tensor] = []
    target_edges = cxcywh_to_xyxy(target_boxes.to(device=device, dtype=stock_boxes.dtype))
    for image_index, (source, destination) in enumerate(match_indices):
        source = source.to(device=device, dtype=torch.long)
        destination = destination.to(device=device, dtype=torch.long)
        if not len(source):
            continue
        matched_mask[image_index, source] = True
        selected_target = target_edges[destination]
        stock_iou_parts.append(
            aligned_iou(cxcywh_to_xyxy(stock_boxes[image_index, source]), selected_target)
        )
        refined_iou_parts.append(
            aligned_iou(cxcywh_to_xyxy(refined_boxes[image_index, source]), selected_target)
        )
    if stock_iou_parts:
        stock_iou = torch.cat(stock_iou_parts).float()
        refined_iou = torch.cat(refined_iou_parts).float()
        delta = refined_iou - stock_iou
    else:
        stock_iou = stock_boxes.new_zeros(0, dtype=torch.float32)
        refined_iou = stock_boxes.new_zeros(0, dtype=torch.float32)
        delta = stock_iou
    correction = effective_correction.float()
    matched_rms = correction_rms(correction, matched_mask)
    unmatched_rms = correction_rms(correction, ~matched_mask)
    denominator = float(matched_rms.detach().cpu())
    ratio = float(unmatched_rms.detach().cpu()) / denominator if denominator > 0 else math.inf
    gate_values = gates.detach().float().reshape(-1)
    residual_values = residuals.detach().float().reshape(-1)
    finite = all(
        bool(torch.isfinite(value).all())
        for value in (stock_boxes, refined_boxes, target_boxes, effective_correction, gates, residuals)
    ) and math.isfinite(ratio)
    return {
        "finite": finite,
        "matched_count": int(delta.numel()),
        "improvement_count": int((delta > 0).sum().item()),
        "degradation_count": int((delta < 0).sum().item()),
        "unchanged_count": int((delta == 0).sum().item()),
        "stock_matched_iou_mean": float(stock_iou.mean().cpu()) if stock_iou.numel() else 0.0,
        "refined_matched_iou_mean": float(refined_iou.mean().cpu()) if refined_iou.numel() else 0.0,
        "matched_iou_delta_mean": float(delta.mean().cpu()) if delta.numel() else 0.0,
        "matched_correction_rms": denominator,
        "unmatched_correction_rms": float(unmatched_rms.detach().cpu()),
        "unmatched_to_matched_rms_ratio": ratio,
        "gate_mean": float(gate_values.mean().cpu()),
        "gate_std": float(gate_values.std(unbiased=False).cpu()),
        "gate_p05": float(torch.quantile(gate_values, 0.05).cpu()),
        "gate_p95": float(torch.quantile(gate_values, 0.95).cpu()),
        "residual_rms": float(residual_values.square().mean().sqrt().cpu()),
        "residual_abs_p95": float(torch.quantile(residual_values.abs(), 0.95).cpu()),
    }


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def assert_repeated_evaluations(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require exactly three evaluation reports with identical unrounded values."""
    if len(reports) != EVALUATION_CONSTANTS["repeats"]:
        raise ValueError("I-TBER requires exactly three repeated evaluations")
    authority = _canonical_bytes(reports[0])
    for index, report in enumerate(reports[1:], start=2):
        if _canonical_bytes(report) != authority:
            raise ValueError(f"I-TBER evaluation repeat {index} differs from repeat 1")
    return copy.deepcopy(dict(reports[0]))


def _metric(metrics: Mapping[str, Any], name: str) -> float:
    value = float(metrics[name])
    if not math.isfinite(value):
        raise ValueError(f"non-finite I-TBER metric: {name}")
    return value


def _decimal_delta(refined: Mapping[str, Any], stock: Mapping[str, Any], name: str) -> Decimal:
    """Subtract the exact serialized float values without binary cancellation."""
    return Decimal(str(_metric(refined, name))) - Decimal(str(_metric(stock, name)))


def _activity_valid(diagnostics: Mapping[str, Any]) -> bool:
    values = {
        name: float(diagnostics[name])
        for name in (
            "gate_mean",
            "gate_std",
            "gate_p05",
            "gate_p95",
            "residual_rms",
            "residual_abs_p95",
        )
    }
    return bool(diagnostics.get("finite")) and all(math.isfinite(value) for value in values.values()) and (
        1e-4 < values["gate_mean"] < 1.0 - 1e-4
        and values["gate_std"] > 1e-4
        and values["gate_p95"] > 1e-3
        and values["gate_p05"] < 1.0 - 1e-3
        and values["residual_rms"] > 1e-4
        and values["residual_abs_p95"] < 0.9999
    )


def _scientific_status(conditions: Mapping[str, bool], *, engineering_valid: bool) -> str:
    if not engineering_valid:
        return "engineering_invalid"
    return "passed" if all(conditions.values()) else "scientific_failed"


def evaluate_gate2(
    stock: Mapping[str, Any],
    refined: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen 12-epoch screen thresholds without rounded inputs."""
    exact_delta = {
        name: _decimal_delta(refined, stock, name)
        for name in ("map", "ap50", "ap75", "ap_tiny", "ap_small")
    }
    delta = {name: float(value) for name, value in exact_delta.items()}
    engineering_valid = bool(diagnostics.get("finite")) and all(
        math.isfinite(float(diagnostics[name]))
        for name in ("matched_correction_rms", "unmatched_correction_rms", "unmatched_to_matched_rms_ratio")
    )
    conditions = {
        "map_gain": exact_delta["map"] >= Decimal("0.002"),
        "ap75_gain": exact_delta["ap75"] >= Decimal("0.003"),
        "ap50_floor": exact_delta["ap50"] >= Decimal("-0.0005"),
        "tiny_or_small_gain": exact_delta["ap_tiny"] > 0 or exact_delta["ap_small"] > 0,
        "matched_iou_majority": int(diagnostics["improvement_count"]) > int(diagnostics["degradation_count"]),
        "unmatched_correction_safe": float(diagnostics["unmatched_to_matched_rms_ratio"]) <= 0.25,
        "refinement_active_unsaturated": _activity_valid(diagnostics),
    }
    status = _scientific_status(conditions, engineering_valid=engineering_valid)
    return {
        "status": status,
        "passed": status == "passed",
        "engineering_valid": engineering_valid,
        "conditions": conditions,
        "delta": delta,
    }


def evaluate_formal_gate(
    stock: Mapping[str, Any],
    refined: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    *,
    tail5_map_delta: float,
) -> dict[str, Any]:
    """Apply the frozen full-data 30-epoch continuation conditions."""
    exact_delta = {
        name: _decimal_delta(refined, stock, name)
        for name in ("map", "ap50", "ap75", "ap_tiny", "ap_small")
    }
    delta = {name: float(value) for name, value in exact_delta.items()}
    detector_unchanged = diagnostics.get("detector_sha_before") == diagnostics.get("detector_sha_after")
    engineering_valid = bool(diagnostics.get("finite")) and detector_unchanged and math.isfinite(float(tail5_map_delta))
    conditions = {
        "map_gain": exact_delta["map"] >= Decimal("0.003"),
        "ap75_gain": exact_delta["ap75"] >= Decimal("0.005"),
        "tiny_or_small_gain": exact_delta["ap_tiny"] > 0 or exact_delta["ap_small"] > 0,
        "tail5_map_gain": float(tail5_map_delta) > 0.0,
        "matched_iou_majority": int(diagnostics["improvement_count"]) > int(diagnostics["degradation_count"]),
        "detector_unchanged": detector_unchanged,
        "refinement_active_unsaturated": _activity_valid(diagnostics),
    }
    status = _scientific_status(conditions, engineering_valid=engineering_valid)
    return {
        "status": status,
        "passed": status == "passed",
        "engineering_valid": engineering_valid,
        "conditions": conditions,
        "delta": delta,
        "tail5_map_delta": float(tail5_map_delta),
    }


def write_immutable_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Atomically create a report and refuse changed content at the same path."""
    destination = Path(path)
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to replace changed I-TBER report: {destination}")
        return destination
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, destination)
    return destination
