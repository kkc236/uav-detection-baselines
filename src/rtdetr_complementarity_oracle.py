"""Pure core for the frozen FrequencyCM complementarity oracle."""

from __future__ import annotations

import math
import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment

from src.rtdetr_quality_oracle import (
    QualityOracleCacheViolation as _QualityOracleCacheViolation,
    _fsync_directory,
    _open_regular_file_nofollow as _quality_open_regular_file_nofollow,
    _publish_directory_no_replace,
)


NUM_CLASSES = 10
IOU_THRESHOLDS = (0.5, 0.75)
VISDRONE_SIZE_BUCKETS = ("tiny", "small", "medium", "large")
MAP_GREEN_THRESHOLD = Decimal("0.010")
MAP_YELLOW_THRESHOLD = Decimal("0.003")
RECALL_GREEN_THRESHOLD = Decimal("0.020")
RECALL_YELLOW_THRESHOLD = Decimal("0.010")

_CACHE_FORMAT_VERSION = 1
_AUTHORITY_FIELDS = (
    "fdr_sha256",
    "frequencycm_sha256",
    "dataset_sha256",
    "evaluator_sha256",
    "source_commit",
)
_RECORD_FIELDS = (
    "image_id",
    "original_shape",
    "resized_shape",
    "fdr_boxes",
    "fdr_logits",
    "frequencycm_boxes",
    "frequencycm_logits",
    "target_boxes",
    "target_classes",
)


@dataclass(frozen=True)
class Assignment:
    """A deterministic set of positive-IoU prediction/target matches."""

    prediction_indices: torch.Tensor
    target_indices: torch.Tensor
    ious: torch.Tensor


class ComplementarityOracleCacheViolation(ValueError):
    """Raised when paired complementarity evidence is unsafe or has drifted."""


def _require_box_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError(f"{name} must have shape [N, 4]")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating-point tensor")
    return value


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _require_normalized(value: torch.Tensor, name: str) -> None:
    if not bool(((value >= 0) & (value <= 1)).all()):
        raise ValueError(f"{name} must be normalized to [0, 1]")


def _require_classes(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [N]")
    if value.dtype != torch.long:
        raise TypeError(f"{name} must use torch.long")
    return value


def _require_class_range(
    value: torch.Tensor, name: str, upper_bound: int = NUM_CLASSES
) -> None:
    if value.numel() and not bool(
        ((value >= 0) & (value < upper_bound)).all()
    ):
        raise ValueError(f"{name} class range must be [0, {upper_bound})")


def candidate_iou_matrix(boxes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return pairwise IoU for normalized ``cxcywh`` boxes."""

    boxes = _require_box_tensor(boxes, "boxes")
    targets = _require_box_tensor(targets, "targets")
    if boxes.device != targets.device:
        raise ValueError("boxes and targets must share a device")
    _require_finite(boxes, "boxes")
    _require_finite(targets, "targets")
    _require_normalized(targets, "targets")

    compute_dtype = torch.promote_types(boxes.dtype, targets.dtype)
    if compute_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    boxes = boxes.detach().to(dtype=compute_dtype)
    targets = targets.detach().to(dtype=compute_dtype)

    box_centers, box_sizes = boxes.split(2, dim=-1)
    target_centers, target_sizes = targets.split(2, dim=-1)
    box_lower = box_centers - box_sizes / 2
    box_upper = box_centers + box_sizes / 2
    target_lower = target_centers - target_sizes / 2
    target_upper = target_centers + target_sizes / 2
    valid_boxes = (box_sizes > 0).all(dim=-1)
    valid_targets = (target_sizes > 0).all(dim=-1)
    intersection = (
        torch.minimum(box_upper[:, None], target_upper[None])
        - torch.maximum(box_lower[:, None], target_lower[None])
    ).clamp_min(0).prod(dim=-1)
    union = (
        box_sizes.clamp_min(0).prod(dim=-1)[:, None]
        + target_sizes.clamp_min(0).prod(dim=-1)[None]
        - intersection
    )
    valid = valid_boxes[:, None] & valid_targets[None, :] & (union > 0)
    return torch.where(
        valid, intersection / union.clamp_min(torch.finfo(union.dtype).tiny), torch.zeros_like(union)
    ).clamp(0, 1)


def one_to_one_same_class_assignment(
    iou: torch.Tensor,
    prediction_classes: torch.Tensor,
    target_classes: torch.Tensor,
) -> Assignment:
    """Maximize total IoU independently per class with deterministic ties."""

    if not isinstance(iou, torch.Tensor):
        raise TypeError("IoU must be a tensor")
    if iou.ndim != 2:
        raise ValueError("IoU must have shape [predictions, targets]")
    if not torch.is_floating_point(iou):
        raise TypeError("IoU must be a floating-point tensor")
    prediction_classes = _require_classes(prediction_classes, "prediction_classes")
    target_classes = _require_classes(target_classes, "target_classes")
    if iou.shape != (prediction_classes.numel(), target_classes.numel()):
        raise ValueError("IoU shape must equal prediction-by-target counts")
    if iou.device != prediction_classes.device or iou.device != target_classes.device:
        raise ValueError("assignment tensors must share a device")
    _require_class_range(prediction_classes, "prediction_classes")
    _require_class_range(target_classes, "target_classes")
    _require_finite(iou, "IoU")
    if not bool(((iou >= 0) & (iou <= 1)).all()):
        raise ValueError("IoU must be in [0, 1]")

    selected: list[tuple[int, int, float]] = []
    prediction_classes_cpu = prediction_classes.detach().cpu()
    target_classes_cpu = target_classes.detach().cpu()
    common_classes = sorted(
        set(prediction_classes_cpu.tolist()) & set(target_classes_cpu.tolist())
    )
    for class_id in common_classes:
        prediction_index = torch.where(prediction_classes_cpu == class_id)[0]
        target_index = torch.where(target_classes_cpu == class_id)[0]
        block = (
            iou.detach()
            .cpu()[prediction_index][:, target_index]
            .to(dtype=torch.float64)
            .numpy()
        )
        rows, columns = linear_sum_assignment(-block)
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            value = float(block[row, column])
            if value > 0:
                selected.append(
                    (
                        int(prediction_index[row]),
                        int(target_index[column]),
                        value,
                    )
                )

    selected.sort(key=lambda item: (item[1], item[0]))
    device = iou.device
    return Assignment(
        prediction_indices=torch.tensor(
            [item[0] for item in selected], dtype=torch.long, device=device
        ),
        target_indices=torch.tensor(
            [item[1] for item in selected], dtype=torch.long, device=device
        ),
        ious=torch.tensor(
            [item[2] for item in selected], dtype=iou.dtype, device=device
        ),
    )


def visdrone_size_bucket(width: float, height: float) -> str:
    """Return the frozen VisDrone bucket for an original-image pixel size."""

    dimensions: list[float] = []
    for value, name in ((width, "width"), (height, "height")):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        dimensions.append(numeric)
    area = dimensions[0] * dimensions[1]
    if area < 256:
        return "tiny"
    if area < 1024:
        return "small"
    if area < 9216:
        return "medium"
    return "large"


def _require_iou_vector(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [N]")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating-point tensor")
    return value


def _require_iou_values(value: torch.Tensor, name: str) -> None:
    _require_finite(value, name)
    if not bool(((value >= 0) & (value <= 1)).all()):
        raise ValueError(f"{name} must be in [0, 1]")


def _coverage_counts(
    fdr_iou: torch.Tensor,
    frequencycm_iou: torch.Tensor,
    union_iou: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, dict[str, int | float]]:
    result: dict[str, dict[str, int | float]] = {}
    total = int(mask.sum().item())
    for threshold in IOU_THRESHOLDS:
        fdr_covered = (fdr_iou >= threshold) & mask
        frequencycm_covered = (frequencycm_iou >= threshold) & mask
        union_covered = (union_iou >= threshold) & mask
        fdr_count = int(fdr_covered.sum().item())
        frequencycm_count = int(frequencycm_covered.sum().item())
        union_count = int(union_covered.sum().item())
        both = int((fdr_covered & frequencycm_covered).sum().item())
        fdr_only = int((fdr_covered & ~frequencycm_covered).sum().item())
        frequencycm_only = int((~fdr_covered & frequencycm_covered).sum().item())
        neither = total - both - fdr_only - frequencycm_only
        denominator = float(total) if total else 1.0
        result[f"iou{int(threshold * 100):02d}"] = {
            "threshold": threshold,
            "total": total,
            "fdr": fdr_count,
            "frequencycm": frequencycm_count,
            "union": union_count,
            "union_gain": union_count - max(fdr_count, frequencycm_count),
            "both": both,
            "fdr_only": fdr_only,
            "frequencycm_only": frequencycm_only,
            "neither": neither,
            "fdr_rate": fdr_count / denominator if total else 0.0,
            "frequencycm_rate": frequencycm_count / denominator if total else 0.0,
            "union_rate": union_count / denominator if total else 0.0,
            "union_gain_rate": (
                (union_count - max(fdr_count, frequencycm_count)) / denominator
                if total
                else 0.0
            ),
        }
    return result


def _coverage_ownership(
    fdr_iou: torch.Tensor, frequencycm_iou: torch.Tensor
) -> dict[str, dict[str, int | float]]:
    mask = torch.ones(fdr_iou.shape, dtype=torch.bool, device=fdr_iou.device)
    result = _coverage_counts(
        fdr_iou,
        frequencycm_iou,
        torch.maximum(fdr_iou, frequencycm_iou),
        mask,
    )
    for threshold in IOU_THRESHOLDS:
        key = f"iou{int(threshold * 100):02d}"
        union_covered = torch.maximum(fdr_iou, frequencycm_iou) >= threshold
        result[key]["fdr_only"] = int(
            (union_covered & (fdr_iou > frequencycm_iou)).sum().item()
        )
        result[key]["frequencycm_only"] = int(
            (union_covered & (frequencycm_iou > fdr_iou)).sum().item()
        )
        result[key]["both"] = int(
            (union_covered & (fdr_iou == frequencycm_iou)).sum().item()
        )
        result[key]["neither"] = int((~union_covered).sum().item())
    return result


def _coverage_view(
    fdr_iou: torch.Tensor,
    frequencycm_iou: torch.Tensor,
    union_iou: torch.Tensor,
    target_scales: tuple[str, ...] | None,
    target_classes: torch.Tensor | None,
) -> dict[str, Any]:
    all_targets = torch.ones(
        fdr_iou.shape, dtype=torch.bool, device=fdr_iou.device
    )
    view: dict[str, Any] = {
        "overall": _coverage_counts(
            fdr_iou, frequencycm_iou, union_iou, all_targets
        ),
        "by_scale": {},
        "by_class": {},
    }
    if target_scales is not None:
        for scale in VISDRONE_SIZE_BUCKETS:
            if scale not in target_scales:
                continue
            mask = torch.tensor(
                [value == scale for value in target_scales],
                dtype=torch.bool,
                device=fdr_iou.device,
            )
            view["by_scale"][scale] = _coverage_counts(
                fdr_iou, frequencycm_iou, union_iou, mask
            )
    if target_classes is not None:
        for class_id in sorted(set(target_classes.detach().cpu().tolist())):
            view["by_class"][class_id] = _coverage_counts(
                fdr_iou,
                frequencycm_iou,
                union_iou,
                target_classes == class_id,
            )
    return view


def coverage_summary(
    fdr_best_iou: torch.Tensor,
    frequencycm_best_iou: torch.Tensor,
    thresholds: Sequence[float] = IOU_THRESHOLDS,
    *,
    fdr_matched_iou: torch.Tensor | None = None,
    frequencycm_matched_iou: torch.Tensor | None = None,
    union_matched_iou: torch.Tensor | None = None,
    target_scales: Sequence[str] | None = None,
    target_classes: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Summarize raw and optional one-to-one coverage by target group.

    The top-level ``iou50`` and ``iou75`` entries preserve the frozen plan's
    stronger-best-IoU ownership example. The structured ``raw`` view uses the
    ordinary covered-by-both categories needed by the missed-target audit.
    Supplying all three matched-IoU vectors adds the exact one-to-one view.
    """

    fdr_best_iou = _require_iou_vector(fdr_best_iou, "fdr_best_iou")
    frequencycm_best_iou = _require_iou_vector(
        frequencycm_best_iou, "frequencycm_best_iou"
    )
    if fdr_best_iou.shape != frequencycm_best_iou.shape:
        raise ValueError("coverage IoU vectors must have the same shape")
    if fdr_best_iou.device != frequencycm_best_iou.device:
        raise ValueError("coverage IoU vectors must share a device")
    _require_iou_values(fdr_best_iou, "fdr_best_iou")
    _require_iou_values(frequencycm_best_iou, "frequencycm_best_iou")
    if isinstance(thresholds, (str, bytes)) or not isinstance(thresholds, Sequence):
        raise TypeError("thresholds must be a sequence")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in thresholds):
        raise TypeError("thresholds must contain real numbers")
    if tuple(float(value) for value in thresholds) != IOU_THRESHOLDS:
        raise ValueError(f"thresholds must equal frozen values {IOU_THRESHOLDS}")

    matched_values = (fdr_matched_iou, frequencycm_matched_iou, union_matched_iou)
    if any(value is not None for value in matched_values) and not all(
        value is not None for value in matched_values
    ):
        raise ValueError("matched IoU vectors must be supplied together")
    validated_matched: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    if all(value is not None for value in matched_values):
        validated = tuple(
            _require_iou_vector(value, name)
            for value, name in zip(
                matched_values,
                (
                    "fdr_matched_iou",
                    "frequencycm_matched_iou",
                    "union_matched_iou",
                ),
                strict=True,
            )
        )
        if any(value.shape != fdr_best_iou.shape for value in validated):
            raise ValueError("matched IoU vectors must have the same shape as raw coverage")
        if any(value.device != fdr_best_iou.device for value in validated):
            raise ValueError("matched IoU vectors must share the raw coverage device")
        for value, name in zip(
            validated,
            (
                "fdr_matched_iou",
                "frequencycm_matched_iou",
                "union_matched_iou",
            ),
            strict=True,
        ):
            _require_iou_values(value, name)
        validated_matched = validated  # type: ignore[assignment]

    normalized_scales: tuple[str, ...] | None = None
    if target_scales is not None:
        if isinstance(target_scales, (str, bytes)) or not isinstance(
            target_scales, Sequence
        ):
            raise TypeError("target_scales must be a sequence")
        normalized_scales = tuple(target_scales)
        if len(normalized_scales) != fdr_best_iou.numel():
            raise ValueError("target scale count must match coverage vectors")
        if any(scale not in VISDRONE_SIZE_BUCKETS for scale in normalized_scales):
            raise ValueError("target scale is not a frozen VisDrone bucket")

    validated_classes: torch.Tensor | None = None
    if target_classes is not None:
        validated_classes = _require_classes(target_classes, "target_classes")
        if validated_classes.numel() != fdr_best_iou.numel():
            raise ValueError("target class count must match coverage vectors")
        if validated_classes.device != fdr_best_iou.device:
            raise ValueError("target classes and coverage vectors must share a device")
        _require_class_range(validated_classes, "target_classes")

    raw = _coverage_view(
        fdr_best_iou,
        frequencycm_best_iou,
        torch.maximum(fdr_best_iou, frequencycm_best_iou),
        normalized_scales,
        validated_classes,
    )
    one_to_one = (
        _coverage_view(
            *validated_matched,
            normalized_scales,
            validated_classes,
        )
        if validated_matched is not None
        else None
    )
    return {
        **_coverage_ownership(fdr_best_iou, frequencycm_best_iou),
        "raw": raw,
        "one_to_one": one_to_one,
    }


def build_matched_quality_arm(
    boxes: torch.Tensor,
    probabilities: torch.Tensor,
    source_ranks: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    max_det: int = 300,
) -> torch.Tensor:
    """Rank flattened query/class candidates by one-to-one matched IoU."""

    if not isinstance(boxes, torch.Tensor):
        raise TypeError("boxes must be a tensor")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [Q, 4]")
    if not torch.is_floating_point(boxes):
        raise TypeError("boxes must be a floating-point tensor")
    if not isinstance(probabilities, torch.Tensor):
        raise TypeError("probabilities must be a tensor")
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [Q, C]")
    if not torch.is_floating_point(probabilities):
        raise TypeError("probabilities must be a floating-point tensor")
    if probabilities.shape[0] != boxes.shape[0] or not (
        1 <= probabilities.shape[1] <= NUM_CLASSES
    ):
        raise ValueError("probabilities must have shape [Q, C] with 1 <= C <= 10")
    if probabilities.dtype != boxes.dtype:
        raise ValueError("boxes and probabilities must share a dtype")
    if probabilities.device != boxes.device:
        raise ValueError("boxes and probabilities must share a device")
    _require_finite(boxes, "boxes")
    _require_finite(probabilities, "probabilities")
    if not bool(((probabilities >= 0) & (probabilities <= 1)).all()):
        raise ValueError("probabilities must be in [0, 1]")

    if not isinstance(source_ranks, torch.Tensor):
        raise TypeError("source_ranks must be a tensor")
    if source_ranks.ndim != 1 or source_ranks.numel() != boxes.shape[0]:
        raise ValueError("source_ranks must have shape [Q]")
    if source_ranks.dtype != torch.long:
        raise TypeError("source_ranks must use torch.long")
    if source_ranks.device != boxes.device:
        raise ValueError("source_ranks and boxes must share a device")
    if source_ranks.numel() and not bool((source_ranks >= 0).all()):
        raise ValueError("source_ranks must be non-negative")

    target_boxes = _require_box_tensor(target_boxes, "target_boxes")
    target_classes = _require_classes(target_classes, "target_classes")
    if target_boxes.shape[0] != target_classes.numel():
        raise ValueError("target_boxes and target_classes counts must match")
    if target_boxes.device != boxes.device or target_classes.device != boxes.device:
        raise ValueError("candidate and target tensors must share a device")
    if target_boxes.dtype != boxes.dtype:
        raise ValueError("candidate and target boxes must share a dtype")
    _require_finite(target_boxes, "target_boxes")
    _require_normalized(target_boxes, "target_boxes")
    num_classes = probabilities.shape[1]
    _require_class_range(target_classes, "target_classes", num_classes)
    if type(max_det) is not int or max_det <= 0:
        raise ValueError("max_det must be a positive integer")

    # A duplicated-detector control must not create extra matching capacity.
    # Collapse bit-identical query geometry before class expansion, preserving
    # the first source/query rank.  Confidence is intentionally excluded from
    # identity because this non-deployable arm ranks only by matched IoU.
    seen_boxes: set[bytes] = set()
    unique_indices: list[int] = []
    for index, row in enumerate(boxes.detach().cpu().contiguous()):
        identity = row.contiguous().numpy().tobytes(order="C")
        if identity not in seen_boxes:
            seen_boxes.add(identity)
            unique_indices.append(index)
    if len(unique_indices) != boxes.shape[0]:
        selected_unique = torch.tensor(
            unique_indices, dtype=torch.long, device=boxes.device
        )
        boxes = boxes[selected_unique]
        probabilities = probabilities[selected_unique]
        source_ranks = source_ranks[selected_unique]

    query_count = boxes.shape[0]
    candidate_count = query_count * num_classes
    if candidate_count == 0:
        return torch.empty((0, 6), dtype=boxes.dtype, device=boxes.device)
    candidate_boxes = boxes.detach().repeat_interleave(num_classes, dim=0)
    candidate_classes = torch.arange(
        num_classes, dtype=torch.long, device=boxes.device
    ).repeat(query_count)
    query_indices = torch.arange(
        query_count, dtype=torch.long, device=boxes.device
    ).repeat_interleave(num_classes)
    candidate_source_ranks = source_ranks.detach().repeat_interleave(
        num_classes
    )
    iou = candidate_iou_matrix(candidate_boxes, target_boxes)
    assignment = one_to_one_same_class_assignment(
        iou, candidate_classes, target_classes
    )
    utility = torch.zeros(candidate_count, dtype=boxes.dtype, device=boxes.device)
    utility[assignment.prediction_indices] = assignment.ious.to(dtype=boxes.dtype)

    utility_cpu = utility.detach().cpu().tolist()
    source_cpu = candidate_source_ranks.detach().cpu().tolist()
    query_cpu = query_indices.detach().cpu().tolist()
    class_cpu = candidate_classes.detach().cpu().tolist()
    order = sorted(
        range(candidate_count),
        key=lambda index: (
            -utility_cpu[index],
            source_cpu[index],
            query_cpu[index],
            class_cpu[index],
        ),
    )[: min(max_det, candidate_count)]
    selected = torch.tensor(order, dtype=torch.long, device=boxes.device)
    return torch.cat(
        (
            candidate_boxes[selected],
            utility[selected, None],
            candidate_classes[selected, None].to(dtype=boxes.dtype),
        ),
        dim=1,
    )


def _is_symlink_or_reparse(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _read_regular_file_nofollow(path: Path, *, label: str) -> bytes:
    if _is_symlink_or_reparse(path):
        raise ComplementarityOracleCacheViolation(
            f"{label} is a symlink or reparse point"
        )
    try:
        with _quality_open_regular_file_nofollow(path, label=label) as stream:
            return stream.read()
    except _QualityOracleCacheViolation as error:
        raise ComplementarityOracleCacheViolation(str(error)) from error


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _normalize_authority(authority: object) -> dict[str, str]:
    from collections.abc import Mapping

    if not isinstance(authority, Mapping) or set(authority) != set(_AUTHORITY_FIELDS):
        raise ComplementarityOracleCacheViolation("authority schema mismatch")
    normalized: dict[str, str] = {}
    for name in _AUTHORITY_FIELDS:
        value = authority[name]
        expected_length = 40 if name == "source_commit" else 64
        if (
            not isinstance(value, str)
            or len(value) != expected_length
            or any(character not in "0123456789abcdefABCDEF" for character in value)
        ):
            raise ComplementarityOracleCacheViolation(f"invalid authority {name}")
        normalized[name] = value.upper()
    return normalized


def _validate_image_shape(value: object, *, label: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ComplementarityOracleCacheViolation(
            f"record {label} must contain height and width"
        )
    values = tuple(value)
    if (
        len(values) != 2
        or any(type(dimension) is not int for dimension in values)
        or any(dimension <= 0 for dimension in values)
    ):
        raise ComplementarityOracleCacheViolation(
            f"record {label} must contain two positive integers"
        )
    return values  # type: ignore[return-value]


def _validate_paired_record(record: object) -> dict[str, Any]:
    from collections.abc import Mapping

    if not isinstance(record, Mapping) or set(record) != set(_RECORD_FIELDS):
        raise ComplementarityOracleCacheViolation("record schema mismatch")
    image_id = record["image_id"]
    if (
        not isinstance(image_id, str)
        or not image_id
        or any(character in image_id for character in "\0\r\n")
    ):
        raise ComplementarityOracleCacheViolation("record image_id is invalid")
    original_shape = _validate_image_shape(
        record["original_shape"], label="original_shape"
    )
    resized_shape = _validate_image_shape(
        record["resized_shape"], label="resized_shape"
    )

    fdr_boxes = record["fdr_boxes"]
    fdr_logits = record["fdr_logits"]
    frequencycm_boxes = record["frequencycm_boxes"]
    frequencycm_logits = record["frequencycm_logits"]
    target_boxes = record["target_boxes"]
    target_classes = record["target_classes"]
    tensors = (
        fdr_boxes,
        fdr_logits,
        frequencycm_boxes,
        frequencycm_logits,
        target_boxes,
        target_classes,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise ComplementarityOracleCacheViolation("record evidence must be tensors")
    if (
        fdr_boxes.shape != (300, 4)
        or fdr_logits.shape != (300, NUM_CLASSES)
        or frequencycm_boxes.shape != (300, 4)
        or frequencycm_logits.shape != (300, NUM_CLASSES)
    ):
        raise ComplementarityOracleCacheViolation(
            "record production tensor shape mismatch"
        )
    if (
        target_boxes.ndim != 2
        or target_boxes.shape[1] != 4
        or target_classes.ndim != 1
        or target_boxes.shape[0] != target_classes.shape[0]
    ):
        raise ComplementarityOracleCacheViolation("record target tensor shape mismatch")
    floating_tensors = (
        fdr_boxes,
        fdr_logits,
        frequencycm_boxes,
        frequencycm_logits,
        target_boxes,
    )
    if any(value.dtype != torch.float32 for value in floating_tensors):
        raise ComplementarityOracleCacheViolation(
            "boxes, logits, and target_boxes dtype must be torch.float32"
        )
    if target_classes.dtype != torch.int64:
        raise ComplementarityOracleCacheViolation(
            "target_classes must use torch.int64"
        )
    if any(value.device.type != "cpu" for value in tensors):
        raise ComplementarityOracleCacheViolation("record tensors must be on CPU")
    if any(value.requires_grad for value in tensors):
        raise ComplementarityOracleCacheViolation("record tensors must be detached")
    if any(not value.is_contiguous() for value in tensors):
        raise ComplementarityOracleCacheViolation("record tensors must be contiguous")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ComplementarityOracleCacheViolation("record tensors must be finite")
    if not bool(((target_boxes >= 0) & (target_boxes <= 1)).all()):
        raise ComplementarityOracleCacheViolation(
            "target_boxes must be normalized to [0, 1]"
        )
    if target_classes.numel() and not bool(
        ((target_classes >= 0) & (target_classes < NUM_CLASSES)).all()
    ):
        raise ComplementarityOracleCacheViolation(
            f"target_classes class range must be [0, {NUM_CLASSES})"
        )
    return {
        "image_id": image_id,
        "original_shape": original_shape,
        "resized_shape": resized_shape,
        "fdr_boxes": fdr_boxes,
        "fdr_logits": fdr_logits,
        "frequencycm_boxes": frequencycm_boxes,
        "frequencycm_logits": frequencycm_logits,
        "target_boxes": target_boxes,
        "target_classes": target_classes,
    }


def _validate_paired_records(records: object) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ComplementarityOracleCacheViolation("records must be a sequence")
    validated = [_validate_paired_record(record) for record in records]
    image_ids = [record["image_id"] for record in validated]
    if len(set(image_ids)) != len(image_ids):
        raise ComplementarityOracleCacheViolation("record image IDs must be unique")
    return validated


def _save_fsynced(path: Path, payload: object) -> None:
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())


def write_paired_cache(
    root: Path,
    records: Sequence[dict[str, Any]],
    authority: dict[str, str],
) -> dict[str, Any]:
    """Atomically create a complete immutable paired-detector cache."""

    root = Path(root)
    if _is_symlink_or_reparse(root):
        raise ComplementarityOracleCacheViolation(
            "cache root is a symlink or reparse point"
        )
    if os.path.lexists(root):
        raise FileExistsError(f"refusing to overwrite cache root: {root}")
    normalized_authority = _normalize_authority(authority)
    validated_records = _validate_paired_records(records)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent)
    )
    published = False
    try:
        artifact_path = staging / "records.pt"
        _save_fsynced(
            artifact_path,
            {
                "format_version": _CACHE_FORMAT_VERSION,
                "authority": normalized_authority,
                "records": validated_records,
            },
        )
        manifest = {
            "format_version": _CACHE_FORMAT_VERSION,
            "complete": True,
            "authority": normalized_authority,
            "record_count": len(validated_records),
            "artifact": {
                "path": artifact_path.name,
                "bytes": artifact_path.stat().st_size,
                "sha256": _file_sha256(artifact_path),
            },
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(_canonical_json(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(staging)
        _fsync_directory(root.parent)
        _publish_directory_no_replace(staging, root)
        published = True
        _fsync_directory(root.parent)
        return manifest
    except Exception:
        if not published and os.path.lexists(staging):
            shutil.rmtree(staging)
            try:
                _fsync_directory(root.parent)
            except OSError:
                pass
        raise


def _load_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    try:
        raw = _read_regular_file_nofollow(manifest_path, label="manifest")
        manifest = json.loads(raw.decode("utf-8"))
    except ComplementarityOracleCacheViolation:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComplementarityOracleCacheViolation(
            f"manifest load failed: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise ComplementarityOracleCacheViolation("manifest schema mismatch")
    try:
        canonical = _canonical_json(manifest)
    except (TypeError, ValueError) as error:
        raise ComplementarityOracleCacheViolation(
            "manifest contains unsafe values"
        ) from error
    if raw != canonical:
        raise ComplementarityOracleCacheViolation("manifest is not canonical")
    return manifest


def load_paired_cache(
    root: Path,
    authority: dict[str, str],
) -> tuple[dict[str, Any], ...]:
    """Verify hashes and exact schemas before loading a paired cache."""

    from collections.abc import Mapping

    root = Path(root)
    if _is_symlink_or_reparse(root):
        raise ComplementarityOracleCacheViolation(
            "cache root is a symlink or reparse point"
        )
    if not root.is_dir():
        raise ComplementarityOracleCacheViolation(
            "cache root is missing or not a directory"
        )
    expected_authority = _normalize_authority(authority)
    try:
        entries = {path.name for path in root.iterdir()}
    except OSError as error:
        raise ComplementarityOracleCacheViolation(
            f"cache root is unreadable: {error}"
        ) from error
    if entries != {"manifest.json", "records.pt"}:
        raise ComplementarityOracleCacheViolation("cache root contents mismatch")

    manifest = _load_manifest(root)
    if set(manifest) != {
        "format_version",
        "complete",
        "authority",
        "record_count",
        "artifact",
    }:
        raise ComplementarityOracleCacheViolation("manifest schema mismatch")
    if (
        type(manifest["format_version"]) is not int
        or manifest["format_version"] != _CACHE_FORMAT_VERSION
    ):
        raise ComplementarityOracleCacheViolation("manifest format version mismatch")
    if manifest["complete"] is not True:
        raise ComplementarityOracleCacheViolation("manifest is not complete")
    actual_authority = manifest["authority"]
    try:
        normalized_actual = _normalize_authority(actual_authority)
    except ComplementarityOracleCacheViolation as error:
        raise ComplementarityOracleCacheViolation(
            "manifest authority schema mismatch"
        ) from error
    if actual_authority != normalized_actual:
        raise ComplementarityOracleCacheViolation(
            "manifest authority is not canonical"
        )
    if normalized_actual != expected_authority:
        differing = [
            name
            for name in _AUTHORITY_FIELDS
            if normalized_actual[name] != expected_authority[name]
        ]
        raise ComplementarityOracleCacheViolation(
            "authority mismatch: " + ",".join(differing)
        )
    record_count = manifest["record_count"]
    if type(record_count) is not int or record_count < 0:
        raise ComplementarityOracleCacheViolation("record count mismatch")

    artifact = manifest["artifact"]
    if not isinstance(artifact, Mapping) or set(artifact) != {
        "path",
        "bytes",
        "sha256",
    }:
        raise ComplementarityOracleCacheViolation("artifact schema mismatch")
    if artifact["path"] != "records.pt":
        raise ComplementarityOracleCacheViolation("artifact path mismatch")
    if type(artifact["bytes"]) is not int or artifact["bytes"] <= 0:
        raise ComplementarityOracleCacheViolation("artifact byte count mismatch")
    expected_sha256 = artifact["sha256"]
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or expected_sha256 != expected_sha256.upper()
        or any(character not in "0123456789ABCDEF" for character in expected_sha256)
    ):
        raise ComplementarityOracleCacheViolation("artifact SHA-256 schema mismatch")

    artifact_path = root / "records.pt"
    try:
        artifact_bytes = _read_regular_file_nofollow(
            artifact_path, label="artifact records.pt"
        )
    except ComplementarityOracleCacheViolation:
        raise
    except OSError as error:
        raise ComplementarityOracleCacheViolation(
            f"artifact preflight failed: {error}"
        ) from error
    if (
        len(artifact_bytes) != artifact["bytes"]
        or _sha256_bytes(artifact_bytes) != expected_sha256
    ):
        raise ComplementarityOracleCacheViolation(
            "artifact bytes or SHA-256 mismatch"
        )

    try:
        payload = torch.load(
            io.BytesIO(artifact_bytes), map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise ComplementarityOracleCacheViolation(
            f"artifact load failed: {error}"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"format_version", "authority", "records"}
        or type(payload["format_version"]) is not int
        or payload["format_version"] != _CACHE_FORMAT_VERSION
        or not isinstance(payload["records"], list)
    ):
        raise ComplementarityOracleCacheViolation("artifact schema mismatch")
    try:
        payload_authority = _normalize_authority(payload["authority"])
    except ComplementarityOracleCacheViolation as error:
        raise ComplementarityOracleCacheViolation(
            "artifact authority schema mismatch"
        ) from error
    if payload["authority"] != payload_authority or payload_authority != expected_authority:
        raise ComplementarityOracleCacheViolation("artifact authority mismatch")
    if len(payload["records"]) != record_count:
        raise ComplementarityOracleCacheViolation("record count mismatch")
    validated = _validate_paired_records(payload["records"])
    return tuple(validated)


def _decimal_delta(value: object, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise TypeError(f"{name} must be a real number")
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not decimal_value.is_finite():
        raise ValueError(f"{name} must be finite")
    return decimal_value


def decide_complementarity(
    map_delta: float,
    tiny_small_recall_delta: float,
) -> dict[str, Any]:
    """Apply the exact frozen Red/Yellow/Green scientific boundary."""

    observed_map = _decimal_delta(map_delta, "map_delta")
    observed_recall = _decimal_delta(
        tiny_small_recall_delta, "tiny_small_recall_delta"
    )
    if (
        observed_map >= MAP_GREEN_THRESHOLD
        or observed_recall >= RECALL_GREEN_THRESHOLD
    ):
        decision = "green"
    elif (
        observed_map >= MAP_YELLOW_THRESHOLD
        or observed_recall >= RECALL_YELLOW_THRESHOLD
    ):
        decision = "yellow"
    else:
        decision = "red"
    return {
        "decision": decision,
        "finite": True,
        "observed": {
            "map_delta": format(observed_map, "f"),
            "tiny_small_recall_delta": format(observed_recall, "f"),
        },
        "thresholds": {
            "green": {
                "map_delta": format(MAP_GREEN_THRESHOLD, "f"),
                "tiny_small_recall_delta": format(RECALL_GREEN_THRESHOLD, "f"),
            },
            "yellow": {
                "map_delta": format(MAP_YELLOW_THRESHOLD, "f"),
                "tiny_small_recall_delta": format(RECALL_YELLOW_THRESHOLD, "f"),
            },
        },
    }
