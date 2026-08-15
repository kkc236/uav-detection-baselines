"""Strict checkpoint and metric primitives for BPDD Formal100 evaluation.

Both Formal100 arms deploy the ordinary FDR inference graph.  BPDD is a
training-only loss and must never be instantiated by this module.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils.metrics import ap_per_class, box_iou

from src.fdr_protocol import write_create_only_manifest
from src.rtdetr_fdr import FDRRTDETRDetectionModel


SCALE_NAMES = ("tiny", "small", "medium", "large")
SCALE_DEFINITION = {
    "coordinate_space": "network-input-pixels",
    "tiny": "sqrt(area) < 16",
    "small": "16 <= sqrt(area) < 32",
    "medium": "32 <= sqrt(area) < 96",
    "large": "sqrt(area) >= 96",
}
IOU_THRESHOLDS = torch.linspace(0.50, 0.95, 10)
FORMAL_CLASS_COUNT = 10


@dataclass(frozen=True)
class LoadedFinalCheckpoint:
    model: nn.Module
    metadata: dict[str, Any]


def file_sha256(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash a tensor state by key, dtype, shape, and exact bytes."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state entry {name!r} is not a tensor")
        if value.is_quantized or value.layout is not torch.strided:
            raise TypeError(f"state entry {name!r} cannot be byte-fingerprinted")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest().upper()


def _checkpoint_state(source: Any) -> dict[str, torch.Tensor]:
    if isinstance(source, Mapping):
        state = dict(source)
    elif callable(getattr(source, "state_dict", None)):
        state = dict(source.state_dict())
    else:
        raise TypeError("checkpoint EMA/model does not expose a tensor state_dict")
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("checkpoint EMA/model state must contain only tensors")
    return state


def load_exact_final_checkpoint(
    checkpoint: str | Path,
    *,
    expected_sha256: str,
    model_factory: Callable[..., nn.Module] = FDRRTDETRDetectionModel,
) -> LoadedFinalCheckpoint:
    """Load only exact epoch99 into the ordinary FDR inference graph."""

    path = Path(checkpoint).resolve()
    if path.name != "epoch99.pt":
        raise ValueError("Formal100 evaluation only accepts exact epoch99.pt")
    if not path.is_file():
        raise FileNotFoundError(f"Formal100 checkpoint not found: {path}")
    actual_sha = file_sha256(path)
    if actual_sha != str(expected_sha256).upper():
        raise ValueError(
            f"checkpoint SHA256 mismatch: expected={expected_sha256}, actual={actual_sha}"
        )
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, Mapping) or artifact.get("epoch") != 99:
        raise ValueError("Formal100 checkpoint must contain raw epoch99")
    if artifact.get("optimizer") is None:
        raise ValueError("exact epoch99 training checkpoint optimizer state was stripped")
    source_field = "ema" if artifact.get("ema") is not None else "model"
    source = artifact.get(source_field)
    if source is None:
        raise ValueError("Formal100 checkpoint contains neither EMA nor model state")
    state = _checkpoint_state(source)
    model = model_factory(nc=FORMAL_CLASS_COUNT)
    if model_factory is FDRRTDETRDetectionModel and type(model) is not FDRRTDETRDetectionModel:
        raise TypeError("Formal100 must instantiate the ordinary FDR inference graph")
    model.load_state_dict(state, strict=True)
    model.eval()
    kind = "exact-final-ema" if source_field == "ema" else "exact-final-model"
    return LoadedFinalCheckpoint(
        model=model,
        metadata={
            "kind": kind,
            "completed_epoch": 100,
            "raw_epoch": 99,
            "sha256": actual_sha,
            "sha256_verified": True,
            "source_field": source_field,
            "ema_state_sha256": state_sha256(state),
            "strict_fdr_inference_graph": True,
        },
    )


def fixed_f1(precision: float, recall: float) -> float:
    denominator = float(precision) + float(recall)
    return 0.0 if denominator == 0.0 else 2.0 * float(precision) * float(recall) / denominator


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}")
    return result


def summarize_native_box_metrics(box: Any, class_names: Sequence[str]) -> dict[str, Any]:
    """Read overall and per-class metrics from Ultralytics' native DetMetrics."""

    if len(class_names) != 10:
        raise ValueError("Formal100 requires exactly 10 classes")
    all_ap = np.asarray(box.all_ap, dtype=np.float64)
    indices = np.asarray(box.ap_class_index, dtype=np.int64).reshape(-1)
    if all_ap.ndim != 2 or all_ap.shape[1] != 10 or len(indices) != len(all_ap):
        raise ValueError("Ultralytics per-class AP tensor has an invalid shape")
    if sorted(indices.tolist()) != list(range(10)):
        raise ValueError("official validation must contain all 10 classes")
    details: dict[str, dict[str, float | int]] = {}
    maps: dict[str, float] = {}
    for row, class_index in zip(all_ap, indices, strict=True):
        name = str(class_names[int(class_index)])
        detail = {
            "id": int(class_index),
            "map50": _finite(row[0], f"{name} AP50"),
            "map75": _finite(row[5], f"{name} AP75"),
            "map": _finite(row.mean(), f"{name} mAP"),
        }
        details[name] = detail
        maps[name] = float(detail["map"])
    precision = _finite(box.mp, "precision")
    recall = _finite(box.mr, "recall")
    return {
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": fixed_f1(precision, recall),
            "map50": _finite(box.map50, "AP50"),
            "map75": _finite(box.map75, "AP75"),
            "map": _finite(box.map, "mAP"),
        },
        "classes": maps,
        "class_details": details,
    }


def scale_bucket_from_area(area: float) -> str:
    side = math.sqrt(max(0.0, float(area)))
    if side < 16.0:
        return "tiny"
    if side < 32.0:
        return "small"
    if side < 96.0:
        return "medium"
    return "large"


def _box_scale_indices(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("scale boxes must have shape [N,4]")
    sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0)
    side = sizes.prod(dim=1).sqrt()
    return torch.where(
        side < 16.0,
        torch.zeros_like(side, dtype=torch.long),
        torch.where(
            side < 32.0,
            torch.ones_like(side, dtype=torch.long),
            torch.where(
                side < 96.0,
                torch.full_like(side, 2, dtype=torch.long),
                torch.full_like(side, 3, dtype=torch.long),
            ),
        ),
    )


def _match_predictions(
    pred_boxes: torch.Tensor,
    pred_classes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
) -> np.ndarray:
    correct = np.zeros((len(pred_boxes), 10), dtype=bool)
    if len(pred_boxes) == 0 or len(target_boxes) == 0:
        return correct
    iou = box_iou(target_boxes, pred_boxes)
    same_class = target_classes[:, None] == pred_classes[None, :]
    iou = (iou * same_class).cpu().numpy()
    for column, threshold in enumerate(IOU_THRESHOLDS.tolist()):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if not len(matches):
            continue
        if len(matches) > 1:
            matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), column] = True
    return correct


def _scale_ap(
    predictions: Sequence[Mapping[str, torch.Tensor]],
    targets: Sequence[Mapping[str, torch.Tensor]],
    *,
    scale_index: int,
    class_count: int,
) -> dict[str, float | int]:
    true_positives: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    predicted_classes: list[np.ndarray] = []
    target_classes: list[np.ndarray] = []
    gt_count = 0
    for prediction, target in zip(predictions, targets, strict=True):
        pred_boxes = prediction["bboxes"].detach().float().cpu()
        pred_conf = prediction["conf"].detach().float().cpu()
        pred_cls = prediction["cls"].detach().long().cpu()
        gt_boxes = target["bboxes"].detach().float().cpu()
        gt_cls = target["cls"].detach().long().cpu()
        pred_mask = _box_scale_indices(pred_boxes) == scale_index
        gt_mask = _box_scale_indices(gt_boxes) == scale_index
        pred_boxes, pred_conf, pred_cls = (
            pred_boxes[pred_mask],
            pred_conf[pred_mask],
            pred_cls[pred_mask],
        )
        gt_boxes, gt_cls = gt_boxes[gt_mask], gt_cls[gt_mask]
        if len(pred_cls) and (pred_cls.min() < 0 or pred_cls.max() >= class_count):
            raise ValueError("prediction contains an invalid class id")
        if len(gt_cls) and (gt_cls.min() < 0 or gt_cls.max() >= class_count):
            raise ValueError("target contains an invalid class id")
        gt_count += len(gt_cls)
        true_positives.append(_match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls))
        confidences.append(pred_conf.numpy())
        predicted_classes.append(pred_cls.numpy())
        target_classes.append(gt_cls.numpy())
    if gt_count == 0:
        return {"gt": 0, "map50": 0.0, "map75": 0.0, "map": 0.0}
    tp = np.concatenate(true_positives, axis=0)
    conf = np.concatenate(confidences)
    pred_cls = np.concatenate(predicted_classes)
    target_cls = np.concatenate(target_classes)
    if len(pred_cls) == 0:
        return {"gt": gt_count, "map50": 0.0, "map75": 0.0, "map": 0.0}
    ap = ap_per_class(tp, conf, pred_cls, target_cls, plot=False)[5]
    if ap.size == 0:
        return {"gt": gt_count, "map50": 0.0, "map75": 0.0, "map": 0.0}
    return {
        "gt": int(gt_count),
        "map50": _finite(ap[:, 0].mean(), "scale AP50"),
        "map75": _finite(ap[:, 5].mean(), "scale AP75"),
        "map": _finite(ap.mean(), "scale mAP"),
    }


def summarize_scale_metrics(
    predictions: Sequence[Mapping[str, torch.Tensor]],
    targets: Sequence[Mapping[str, torch.Tensor]],
    *,
    class_count: int,
) -> dict[str, Any]:
    if not predictions or len(predictions) != len(targets):
        raise ValueError("cached predictions and targets must have equal nonzero length")
    details = {
        name: _scale_ap(
            predictions,
            targets,
            scale_index=index,
            class_count=class_count,
        )
        for index, name in enumerate(SCALE_NAMES)
    }
    return {
        "scales": {name: float(detail["map"]) for name, detail in details.items()},
        "scale_details": details,
        "scale_definition": dict(SCALE_DEFINITION),
    }


class CachedScaleRTDETRValidator(RTDETRValidator):
    """Native validator that also caches its exact postprocessed val pass."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.scale_predictions: list[dict[str, torch.Tensor]] = []
        self.scale_targets: list[dict[str, torch.Tensor]] = []

    def update_metrics(
        self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]
    ) -> None:
        for image_index, prediction in enumerate(preds):
            prepared = self._prepare_batch(image_index, batch)
            pred_copy = {
                key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for key, value in prediction.items()
            }
            pred_native = self._prepare_pred(pred_copy)
            self.scale_predictions.append(
                {
                    "bboxes": pred_native["bboxes"].detach().float().cpu(),
                    "conf": pred_native["conf"].detach().float().cpu(),
                    "cls": pred_native["cls"].detach().long().cpu(),
                }
            )
            self.scale_targets.append(
                {
                    "bboxes": prepared["bboxes"].detach().float().cpu(),
                    "cls": prepared["cls"].detach().long().cpu(),
                }
            )
        super().update_metrics(preds, batch)


def write_create_only_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return write_create_only_manifest(Path(path).resolve(), payload).resolve()


__all__ = [
    "CachedScaleRTDETRValidator",
    "LoadedFinalCheckpoint",
    "SCALE_DEFINITION",
    "SCALE_NAMES",
    "file_sha256",
    "fixed_f1",
    "load_exact_final_checkpoint",
    "scale_bucket_from_area",
    "state_sha256",
    "summarize_native_box_metrics",
    "summarize_scale_metrics",
    "write_create_only_json",
]
