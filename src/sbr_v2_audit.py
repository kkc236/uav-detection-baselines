"""Deterministic reconstruction primitives for the frozen SBR-V2 audit."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
from numbers import Integral, Real
import struct
from typing import Any

import numpy as np

from src.sbr_fusion import Detection, greedy_ios_clusters
from src.sbr_metrics import evaluate_dataset


Box = tuple[float, float, float, float]
TileBounds = tuple[int, int, int, int]
IdentityKey = tuple[str, int, int, int]
_AUTO_TILE_BOUNDS = object()


def _strict_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_dimension(name: str, value: object) -> int:
    result = _strict_nonnegative_int(name, value)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validated_box(name: str, value: object) -> Box:
    try:
        box = tuple(float(item) for item in value)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an xyxy box") from None
    if (
        len(box) != 4
        or not all(math.isfinite(item) for item in box)
        or box[2] <= box[0]
        or box[3] <= box[1]
    ):
        raise ValueError(f"{name} must be a finite nondegenerate xyxy box")
    return box  # type: ignore[return-value]


def _validated_score(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and within [0,1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and within [0,1]")
    return result


def _validated_tile_bounds(
    value: object, *, width: int, height: int
) -> TileBounds | None:
    if value is None:
        return None
    try:
        bounds = tuple(
            _strict_nonnegative_int("tile bound", item)
            for item in value  # type: ignore[union-attr]
        )
    except TypeError:
        raise ValueError("tile_bounds must contain four integers") from None
    if (
        len(bounds) != 4
        or bounds[2] <= bounds[0]
        or bounds[3] <= bounds[1]
        or bounds[2] > width
        or bounds[3] > height
    ):
        raise ValueError("tile_bounds must be a legal tile inside the image")
    return bounds  # type: ignore[return-value]


@dataclass(frozen=True)
class AuditRawDetection:
    """One immutable A/C raw-cache detection with its original row index."""

    image_id: str
    arm: str
    width: int
    height: int
    source_order: int
    query_index: int
    class_id: int
    score: float
    network_xyxy: Box
    view_xyxy: Box
    global_xyxy: Box
    tile_bounds: TileBounds | None
    original_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.image_id, str) or not self.image_id:
            raise ValueError("image_id must be a nonempty exact string")
        if self.arm not in {"A", "C"}:
            raise ValueError("arm must be exactly 'A' or 'C'")
        width = _positive_dimension("width", self.width)
        height = _positive_dimension("height", self.height)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        for name in ("source_order", "query_index", "class_id", "original_index"):
            object.__setattr__(
                self, name, _strict_nonnegative_int(name, getattr(self, name))
            )
        if isinstance(self.score, bool) or not isinstance(self.score, Real):
            raise ValueError("score must be finite and within [0, 1]")
        score = float(self.score)
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            raise ValueError("score must be finite and within [0, 1]")
        object.__setattr__(self, "score", score)
        for name in ("network_xyxy", "view_xyxy", "global_xyxy"):
            object.__setattr__(
                self, name, _validated_box(name, getattr(self, name))
            )
        object.__setattr__(
            self,
            "tile_bounds",
            _validated_tile_bounds(self.tile_bounds, width=width, height=height),
        )

    @property
    def identity_key(self) -> IdentityKey:
        return (
            self.image_id,
            self.class_id,
            self.source_order,
            self.query_index,
        )

    def to_detection(self) -> Detection:
        """Convert to the existing fusion type without dropping provenance."""

        return Detection(
            box=self.global_xyxy,
            score=self.score,
            class_id=self.class_id,
            source_order=self.source_order,
            query_index=self.query_index,
            view_xyxy=self.view_xyxy,
            global_xyxy=self.global_xyxy,
            network_xyxy=self.network_xyxy,
            tile_local_box=(
                self.view_xyxy if self.tile_bounds is not None else None
            ),
            global_box=self.global_xyxy,
            tile_bounds=self.tile_bounds,
            tile_index=(
                self.source_order - 1 if self.tile_bounds is not None else None
            ),
        )

    @classmethod
    def synthetic(
        detection_type,
        image_id: str,
        arm: str,
        *,
        source: int = 0,
        query: int = 0,
        score: float = 0.8,
        cls: int = 0,
        box: Sequence[float] = (0.0, 0.0, 2.0, 2.0),
        width: int = 100,
        height: int = 100,
        tile_bounds: TileBounds | None | object = _AUTO_TILE_BOUNDS,
        original_index: int | None = None,
    ) -> "AuditRawDetection":
        """Build a legal compact record for deterministic synthetic fixtures."""

        resolved_tile = tile_bounds
        if resolved_tile is _AUTO_TILE_BOUNDS:
            resolved_tile = None if source == 0 else (0, 0, width, height)
        coordinates = tuple(float(item) for item in box)
        return detection_type(
            image_id=image_id,
            arm=arm,
            width=width,
            height=height,
            source_order=source,
            query_index=query,
            class_id=cls,
            score=score,
            network_xyxy=coordinates,  # type: ignore[arg-type]
            view_xyxy=coordinates,  # type: ignore[arg-type]
            global_xyxy=coordinates,  # type: ignore[arg-type]
            tile_bounds=resolved_tile,  # type: ignore[arg-type]
            original_index=source if original_index is None else original_index,
        )


@dataclass(frozen=True)
class RelevantRawRows:
    """Relevant raw-cache rows for exactly one manifest image."""

    image_id: str
    rows: tuple[Mapping[str, Any], ...]


def group_relevant_raw_rows(
    rows: Iterable[Mapping[str, Any]], manifest_image_ids: Iterable[str]
) -> Iterable[RelevantRawRows]:
    """Yield A/C rows one image at a time in exact manifest order.

    Image identifiers are compared verbatim. Missing raw rows produce an empty
    group, while unknown, repeated, or backward-moving groups fail closed.
    """

    manifest = tuple(manifest_image_ids)
    if any(not isinstance(image_id, str) for image_id in manifest):
        raise ValueError("manifest image IDs must be exact strings")
    if len(set(manifest)) != len(manifest):
        raise ValueError("manifest image IDs must be unique")
    positions = {image_id: index for index, image_id in enumerate(manifest)}

    active_index: int | None = None
    active_rows: list[Mapping[str, Any]] = []
    next_to_emit = 0

    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("raw cache rows must be mappings")
        image_id = row.get("image_id")
        if not isinstance(image_id, str) or image_id not in positions:
            raise ValueError(f"unknown image group: {image_id!r}")
        row_index = positions[image_id]

        if active_index is None:
            while next_to_emit < row_index:
                yield RelevantRawRows(manifest[next_to_emit], ())
                next_to_emit += 1
            active_index = row_index
            next_to_emit = row_index + 1
        elif row_index != active_index:
            if row_index < active_index:
                raise ValueError(
                    f"repeated or out-of-order image group: {image_id!r}"
                )
            yield RelevantRawRows(manifest[active_index], tuple(active_rows))
            active_rows = []
            while next_to_emit < row_index:
                yield RelevantRawRows(manifest[next_to_emit], ())
                next_to_emit += 1
            active_index = row_index
            next_to_emit = row_index + 1

        if row.get("arm") in {"A", "C"}:
            active_rows.append(row)

    if active_index is not None:
        yield RelevantRawRows(manifest[active_index], tuple(active_rows))
    while next_to_emit < len(manifest):
        yield RelevantRawRows(manifest[next_to_emit], ())
        next_to_emit += 1


def _canonical_measurement_bytes(detection: AuditRawDetection) -> bytes:
    payload = {
        "global_xyxy": detection.global_xyxy,
        "network_xyxy": detection.network_xyxy,
        "score": detection.score,
        "view_xyxy": detection.view_xyxy,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _unique_full_records(
    detections: Iterable[AuditRawDetection], *, arm: str
) -> dict[IdentityKey, AuditRawDetection]:
    records: dict[IdentityKey, AuditRawDetection] = {}
    for detection in detections:
        if not isinstance(detection, AuditRawDetection) or detection.arm != arm:
            raise ValueError(f"expected only Arm-{arm} audit detections")
        if detection.source_order != 0:
            continue
        if detection.identity_key in records:
            raise ValueError(
                f"full-view identity collision: {detection.identity_key!r}"
            )
        records[detection.identity_key] = detection
    return records


def map_full_a_to_c(
    a_detections: Iterable[AuditRawDetection],
    c_detections: Iterable[AuditRawDetection],
) -> dict[IdentityKey, int]:
    """Map every Arm-A full row to its byte-identical Arm-C raw row index."""

    arm_a = _unique_full_records(a_detections, arm="A")
    arm_c = _unique_full_records(c_detections, arm="C")
    mapping: dict[IdentityKey, int] = {}
    for identity_key, a_detection in arm_a.items():
        c_detection = arm_c.get(identity_key)
        if c_detection is None:
            raise ValueError(f"missing Arm-C full-view identity: {identity_key!r}")
        if _canonical_measurement_bytes(a_detection) != _canonical_measurement_bytes(
            c_detection
        ):
            raise ValueError(
                f"Arm-A/Arm-C measurement disagreement: {identity_key!r}"
            )
        mapping[identity_key] = c_detection.original_index
    return mapping


@dataclass(frozen=True)
class CClusterReconstruction:
    """Standard Arm-C predictions before and after the frozen final cap."""

    pre_cap_predictions: tuple[Detection, ...]
    standard_predictions: tuple[Detection, ...]
    cluster_members: tuple[tuple[int, ...], ...]

    @property
    def top300_predictions(self) -> tuple[Detection, ...]:
        return self.standard_predictions


def effective_size(box: Sequence[float], *, width: int, height: int) -> float:
    """Return the frozen 640-pixel effective target size."""
    b = _validated_box("box", box)
    w = _positive_dimension("width", width)
    h = _positive_dimension("height", height)
    gain = min(640.0 / float(w), 640.0 / float(h), 1.0)
    return float(np.sqrt(np.float64(b[2] - b[0]) * np.float64(b[3] - b[1])) * np.float64(gain))


class AttributionCategory(str, Enum):
    MIXED_CLUSTER_LOCALIZATION = "mixed_cluster_localization"
    FINAL_300_TRUNCATION = "final_300_truncation"
    MATCHING_COMPETITION = "matching_competition"
    CLASS_OR_CANDIDATE_LOSS = "class_or_candidate_loss"
    OTHER = "other"


@dataclass(frozen=True)
class LargeMatchResult:
    pred_to_gt: dict[int, int]
    gt_to_prediction: dict[int, int]
    neutral_prediction_indices: tuple[int, ...]
    selected_gt_indices: tuple[int, ...]
    ordered_predictions: tuple[Any, ...]

    @property
    def prediction_to_gt(self) -> dict[int, int]:
        return self.pred_to_gt

    @property
    def neutral_indices(self) -> tuple[int, ...]:
        return self.neutral_prediction_indices

    @property
    def tp_gt_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self.gt_to_prediction))

    @property
    def fn_gt_indices(self) -> tuple[int, ...]:
        return tuple(i for i in self.selected_gt_indices if i not in self.gt_to_prediction)


@dataclass(frozen=True)
class AttributionEvent:
    image_id: str
    gt_index: int
    iou_threshold: float
    category: AttributionCategory
    counterfactual_recovers: bool = False


@dataclass(frozen=True)
class ImageAuditResult:
    events: tuple[AttributionEvent, ...]
    arm_a_match: LargeMatchResult
    arm_c_match: LargeMatchResult


@dataclass(frozen=True)
class AuditImage:
    """Convenient input record for :func:`audit_image_at_threshold`."""

    image_id: str
    width: int
    height: int
    gt_boxes: tuple[Box, ...]
    gt_classes: tuple[int, ...]
    a_detections: tuple[AuditRawDetection, ...] = ()
    c_detections: tuple[AuditRawDetection, ...] = ()
    ignore_boxes: tuple[Box, ...] = ()


@dataclass(frozen=True)
class PreparedImageAudit:
    """One-image trusted audit state reused across IoU thresholds."""

    image: AuditImage
    standard: CClusterReconstruction
    guarded: CClusterReconstruction
    invariants: Mapping[str, bool | float]
    c_match_predictions: tuple[Mapping[str, Any], ...]
    c_pre_cap_match_predictions: tuple[Mapping[str, Any], ...]
    full_a_to_c: tuple[tuple[IdentityKey, int], ...]
    c_by_original_index: tuple[tuple[int, AuditRawDetection], ...]


def _iou64(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (np.float64(x) for x in a)
    bx1, by1, bx2, by2 = (np.float64(x) for x in b)
    inter = max(np.float64(0), min(ax2, bx2) - max(ax1, bx1)) * max(np.float64(0), min(ay2, by2) - max(ay1, by1))
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _prediction_fields(prediction: Any, index: int) -> tuple[Box, float, int, int, int, int]:
    if isinstance(prediction, AuditRawDetection):
        return (
            _validated_box("prediction", prediction.global_xyxy),
            _validated_score("prediction score", prediction.score),
            _strict_nonnegative_int("class_id", prediction.class_id),
            _strict_nonnegative_int("source_order", prediction.source_order),
            _strict_nonnegative_int("query_index", prediction.query_index),
            _strict_nonnegative_int("original_index", prediction.original_index),
        )
    if not isinstance(prediction, Detection):
        if isinstance(prediction, Mapping):
            box = prediction.get("box", prediction.get("global_xyxy"))
            score = prediction.get("score")
            cls = prediction.get("class_id", prediction.get("cls"))
            source = prediction.get("source_order", prediction.get("source", 0))
            query = prediction.get("query_index", prediction.get("query", 0))
            original = prediction.get("original_index", index)
            if box is None or score is None or cls is None:
                raise ValueError("prediction mapping is missing fields")
            return (
                _validated_box("prediction", box),
                _validated_score("prediction score", score),
                _strict_nonnegative_int("class_id", cls),
                _strict_nonnegative_int("source_order", source),
                _strict_nonnegative_int("query_index", query),
                _strict_nonnegative_int("original_index", original),
            )
        # Accept metrics-style prediction objects without coupling this module to them.
        if all(hasattr(prediction, n) for n in ("box", "score")):
            cls = getattr(prediction, "class_id", getattr(prediction, "cls", None))
            source = getattr(prediction, "source_order", getattr(prediction, "source", 0))
            query = getattr(prediction, "query_index", getattr(prediction, "query", 0))
            if cls is not None:
                return (
                    _validated_box("prediction", prediction.box),
                    _validated_score("prediction score", prediction.score),
                    _strict_nonnegative_int("class_id", cls),
                    _strict_nonnegative_int("source_order", source),
                    _strict_nonnegative_int("query_index", query),
                    _strict_nonnegative_int(
                        "original_index",
                        getattr(prediction, "original_index", index),
                    ),
                )
        raise ValueError("predictions must be Detection or prediction-like records")
    if not prediction._metadata_valid:
        raise ValueError("prediction metadata must use strict integer fields")
    return (
        _validated_box("prediction", prediction.box),
        _validated_score("prediction score", prediction.score),
        _strict_nonnegative_int("class_id", prediction.class_id),
        _strict_nonnegative_int("source_order", prediction.source_order),
        _strict_nonnegative_int("query_index", prediction.query_index),
        _strict_nonnegative_int(
            "original_index", getattr(prediction, "original_index", index)
        ),
    )


def _ioa64(pred_box: Sequence[float], ignore_box: Sequence[float]) -> float:
    p = _validated_box("prediction", pred_box); q = _validated_box("ignore", ignore_box)
    inter = max(0.0, min(p[2], q[2]) - max(p[0], q[0])) * max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
    return float(np.float64(inter) / (np.float64(p[2] - p[0]) * np.float64(p[3] - p[1])))


def _match_large_targets(
    predictions: Iterable[Any], gt_boxes: Sequence[Sequence[float]], gt_classes: Sequence[int],
    *, ignore_boxes: Sequence[Sequence[float]] | None = None, width: int, height: int,
    iou_threshold: float, prediction_cap: int | None,
) -> LargeMatchResult:
    validated_width = _positive_dimension("width", width)
    validated_height = _positive_dimension("height", height)
    boxes = tuple(_validated_box("gt box", b) for b in gt_boxes)
    ignore = tuple(_validated_box("ignore box", b) for b in (ignore_boxes or ()))
    classes = tuple(_strict_nonnegative_int("gt class", c) for c in gt_classes)
    if len(boxes) != len(classes):
        raise ValueError("gt_boxes and gt_classes lengths must agree")
    selected = tuple(
        i
        for i, box in enumerate(boxes)
        if effective_size(
            box, width=validated_width, height=validated_height
        )
        > 96.0
    )
    raw = tuple(predictions)
    eligible = []
    for i, p in enumerate(raw):
        box, score, cls, source, query, original = _prediction_fields(p, i)
        if score < 0.001:
            continue
        eligible.append((i, p, box, score, cls, source, query, original))
    eligible.sort(key=lambda x: (-x[3], x[5], x[6], x[7]))
    if prediction_cap is not None:
        eligible = eligible[:prediction_cap]
    neutral: list[int] = []
    for item in eligible:
        if any(_ioa64(item[2], b) >= 0.5 for b in ignore):
            neutral.append(item[0])
    matched_gt: set[int] = set(); p2g: dict[int, int] = {}; g2p: dict[int, int] = {}
    for item in eligible:
        pidx, _, pbox, _, pcls, *_ = item
        if pidx in neutral:
            continue
        candidates = [(i, _iou64(pbox, boxes[i])) for i in selected if i not in matched_gt and classes[i] == pcls]
        candidates = [(i, v) for i, v in candidates if v >= float(iou_threshold)]
        if candidates:
            best = max(candidates, key=lambda x: (x[1], -x[0]))[0]
            matched_gt.add(best); p2g[pidx] = best; g2p[best] = pidx; continue
        # A prediction matching an out-of-bin GT is neutral, as in COCO area ranges.
        if any(classes[i] == pcls and i not in matched_gt and _iou64(pbox, boxes[i]) >= float(iou_threshold) for i in range(len(boxes)) if i not in selected):
            neutral.append(pidx)
    return LargeMatchResult(p2g, g2p, tuple(sorted(set(neutral))), selected, tuple(item[1] for item in eligible))


def match_large_targets(
    predictions: Iterable[Any], gt_boxes: Sequence[Sequence[float]], gt_classes: Sequence[int],
    *, ignore_boxes: Sequence[Sequence[float]] | None = None, width: int, height: int,
    iou_threshold: float, max_det: int = 300, conf_threshold: float = 0.001,
) -> LargeMatchResult:
    """Match predictions to large targets using the frozen evaluator order."""
    if (
        isinstance(iou_threshold, bool)
        or not isinstance(iou_threshold, Real)
        or not math.isfinite(float(iou_threshold))
        or not 0.0 <= float(iou_threshold) <= 1.0
    ):
        raise ValueError("iou_threshold must be finite in [0,1]")
    if isinstance(max_det, bool) or not isinstance(max_det, Integral) or int(max_det) != 300:
        raise ValueError("max_det is frozen at 300")
    if isinstance(conf_threshold, bool) or not isinstance(conf_threshold, (float, np.floating)) or float(conf_threshold) != 0.001:
        raise ValueError("conf_threshold is frozen at 0.001")
    return _match_large_targets(
        predictions,
        gt_boxes,
        gt_classes,
        ignore_boxes=ignore_boxes,
        width=width,
        height=height,
        iou_threshold=float(iou_threshold),
        prediction_cap=300,
    )


def _coerce_image(image: Any) -> AuditImage:
    if isinstance(image, AuditImage):
        return image
    if isinstance(image, Mapping):
        payload = dict(image)
        payload.setdefault("a_detections", payload.pop("a_raw_detections", payload.pop("a_raw", payload.pop("a_predictions", ()))) )
        payload.setdefault("c_detections", payload.pop("c_raw_detections", payload.pop("c_raw", payload.pop("c_predictions", ()))) )
        payload = {k: v for k, v in payload.items() if k in {"image_id", "width", "height", "gt_boxes", "gt_classes", "a_detections", "c_detections", "ignore_boxes"}}
        return AuditImage(**payload)
    attrs = {name: getattr(image, name) for name in ("image_id", "width", "height", "gt_boxes", "gt_classes")}
    attrs.update({"a_detections": getattr(image, "a_detections", getattr(image, "a_raw_detections", getattr(image, "a_raw", getattr(image, "a_predictions", ()))) )})
    attrs.update({"c_detections": getattr(image, "c_detections", getattr(image, "c_raw_detections", getattr(image, "c_raw", getattr(image, "c_predictions", ()))) )})
    attrs["ignore_boxes"] = getattr(image, "ignore_boxes", ())
    return AuditImage(**attrs)


def audit_image_at_threshold(image: Any, iou_threshold: float = 0.75) -> ImageAuditResult:
    """Attribute unique large-target Arm-A TP to Arm-C FN events."""
    fixture = _coerce_image(image)
    a_raw = tuple(fixture.a_detections); c_raw = tuple(fixture.c_detections)
    a_preds = tuple(a_raw)
    if c_raw and all(isinstance(d, AuditRawDetection) for d in c_raw):
        reconstruction = reconstruct_c_clusters(c_raw)
    else:
        reconstruction = CClusterReconstruction(tuple(c_raw), tuple(c_raw), tuple((i,) for i in range(len(c_raw))))
    c_preds = reconstruction.standard_predictions
    c_match_preds = tuple(
        {"box": p.box, "score": p.score, "class_id": p.class_id,
         "source_order": p.source_order, "query_index": p.query_index,
         "original_index": members[0]}
        for p, members in zip(c_preds, reconstruction.cluster_members)
    )
    c_pre_cap_match_preds = tuple(
        {
            "box": prediction.box,
            "score": prediction.score,
            "class_id": prediction.class_id,
            "source_order": prediction.source_order,
            "query_index": prediction.query_index,
            "original_index": members[0],
        }
        for prediction, members in zip(
            reconstruction.pre_cap_predictions, reconstruction.cluster_members
        )
    )
    am = match_large_targets(a_preds, fixture.gt_boxes, fixture.gt_classes, ignore_boxes=fixture.ignore_boxes, width=fixture.width, height=fixture.height, iou_threshold=iou_threshold)
    cm = match_large_targets(c_match_preds, fixture.gt_boxes, fixture.gt_classes, ignore_boxes=fixture.ignore_boxes, width=fixture.width, height=fixture.height, iou_threshold=iou_threshold)
    pre_cap_match = _match_large_targets(
        c_pre_cap_match_preds,
        fixture.gt_boxes,
        fixture.gt_classes,
        ignore_boxes=fixture.ignore_boxes,
        width=fixture.width,
        height=fixture.height,
        iou_threshold=float(iou_threshold),
        prediction_cap=None,
    )
    events: list[AttributionEvent] = []
    c_by_orig = {d.original_index: d for d in c_raw if isinstance(d, AuditRawDetection)}
    for gt_idx in am.tp_gt_indices:
        if gt_idx in cm.gt_to_prediction:
            continue
        category = AttributionCategory.OTHER; recovers = False
        a_pidx = am.gt_to_prediction[gt_idx]
        anchor = a_raw[a_pidx] if a_pidx < len(a_raw) and isinstance(a_raw[a_pidx], AuditRawDetection) else None
        anchor_c_index = None
        if anchor is not None:
            anchor_c_index = map_full_a_to_c(a_raw, c_raw).get(anchor.identity_key)
        cluster_index = next((i for i, members in enumerate(reconstruction.cluster_members) if anchor_c_index in members), None) if anchor_c_index is not None else None
        if cluster_index is not None:
            members = tuple(c_by_orig[idx] for idx in reconstruction.cluster_members[cluster_index] if idx in c_by_orig)
            mixed = any(m.source_order == 0 for m in members) and any(m.source_order > 0 for m in members)
            if mixed:
                if cluster_index < len(reconstruction.standard_predictions):
                    fulls = [m for m in members if m.source_order == 0]
                    best = min(fulls, key=lambda m: (-m.score, m.source_order, m.query_index, m.original_index))
                    cf = list(c_preds)
                    old = cf[cluster_index]
                    standard_fails_geometry = (
                        old.class_id != fixture.gt_classes[gt_idx]
                        or _iou64(old.box, fixture.gt_boxes[gt_idx])
                        < float(iou_threshold)
                    )
                    if standard_fails_geometry:
                        cf[cluster_index] = replace(
                            old, box=best.global_xyxy
                        )
                        cf_records = tuple(
                            {
                                "box": p.box,
                                "score": p.score,
                                "class_id": p.class_id,
                                "source_order": p.source_order,
                                "query_index": p.query_index,
                                "original_index": members_[0],
                            }
                            for p, members_ in zip(
                                cf, reconstruction.cluster_members
                            )
                        )
                        recovers = gt_idx in match_large_targets(
                            cf_records,
                            fixture.gt_boxes,
                            fixture.gt_classes,
                            ignore_boxes=fixture.ignore_boxes,
                            width=fixture.width,
                            height=fixture.height,
                            iou_threshold=iou_threshold,
                        ).gt_to_prediction
                        if recovers:
                            category = (
                                AttributionCategory.MIXED_CLUSTER_LOCALIZATION
                            )
        if category is AttributionCategory.OTHER:
            pre_cap_prediction = pre_cap_match.gt_to_prediction.get(gt_idx)
            if (
                pre_cap_prediction is not None
                and pre_cap_prediction >= len(reconstruction.standard_predictions)
            ):
                category = AttributionCategory.FINAL_300_TRUNCATION
        if category is AttributionCategory.OTHER:
            # Same-class candidates over threshold that lost one-to-one assignment.
            for pidx, p in enumerate(c_match_preds):
                box, _, cls, *_ = _prediction_fields(p, pidx)
                if cls == fixture.gt_classes[gt_idx] and _iou64(box, fixture.gt_boxes[gt_idx]) >= float(iou_threshold) and pidx not in cm.neutral_prediction_indices:
                    category = AttributionCategory.MATCHING_COMPETITION; break
            else:
                if not any(_prediction_fields(p, i)[2] == fixture.gt_classes[gt_idx] and _iou64(_prediction_fields(p, i)[0], fixture.gt_boxes[gt_idx]) >= float(iou_threshold) for i, p in enumerate(c_match_preds)):
                    category = AttributionCategory.CLASS_OR_CANDIDATE_LOSS
        events.append(AttributionEvent(fixture.image_id, gt_idx, float(iou_threshold), category, recovers))
    return ImageAuditResult(tuple(events), am, cm)


def prepare_image_audit(image: Any) -> PreparedImageAudit:
    """Build trusted C reconstruction/guard state exactly once for one image."""

    fixture = _coerce_image(image)
    a_raw = tuple(fixture.a_detections)
    c_raw = tuple(fixture.c_detections)
    if any(
        not isinstance(item, AuditRawDetection) or item.arm != "A"
        for item in a_raw
    ) or any(
        not isinstance(item, AuditRawDetection) or item.arm != "C"
        for item in c_raw
    ):
        raise ValueError("prepared audit requires exact A/C raw detections")
    for item in (*a_raw, *c_raw):
        if (
            item.image_id != fixture.image_id
            or item.width != fixture.width
            or item.height != fixture.height
        ):
            raise ValueError("prepared raw detection image provenance disagrees")
    full_mapping = map_full_a_to_c(a_raw, c_raw)
    standard = reconstruct_c_clusters(c_raw)
    guarded = _apply_guard_prevalidated(standard, c_raw)
    invariants = _verify_guard_prevalidated(
        standard, guarded, c_raw, guarded_raw_detections=c_raw
    )
    c_match_predictions = tuple(
        {
            "box": prediction.box,
            "score": prediction.score,
            "class_id": prediction.class_id,
            "source_order": prediction.source_order,
            "query_index": prediction.query_index,
            "original_index": members[0],
        }
        for prediction, members in zip(
            standard.standard_predictions, standard.cluster_members
        )
    )
    c_pre_cap_match_predictions = tuple(
        {
            "box": prediction.box,
            "score": prediction.score,
            "class_id": prediction.class_id,
            "source_order": prediction.source_order,
            "query_index": prediction.query_index,
            "original_index": members[0],
        }
        for prediction, members in zip(
            standard.pre_cap_predictions, standard.cluster_members
        )
    )
    return PreparedImageAudit(
        image=fixture,
        standard=standard,
        guarded=guarded,
        invariants=invariants,
        c_match_predictions=c_match_predictions,
        c_pre_cap_match_predictions=c_pre_cap_match_predictions,
        full_a_to_c=tuple(full_mapping.items()),
        c_by_original_index=tuple(
            (item.original_index, item) for item in c_raw
        ),
    )


def audit_prepared_image_at_threshold(
    prepared: PreparedImageAudit, iou_threshold: float = 0.75
) -> ImageAuditResult:
    """Attribute one threshold without rebuilding prepared C clusters."""

    if not isinstance(prepared, PreparedImageAudit):
        raise ValueError("prepared audit state is required")
    fixture = prepared.image
    reconstruction = prepared.standard
    a_raw = tuple(fixture.a_detections)
    c_preds = reconstruction.standard_predictions
    c_match_preds = prepared.c_match_predictions
    am = match_large_targets(
        a_raw,
        fixture.gt_boxes,
        fixture.gt_classes,
        ignore_boxes=fixture.ignore_boxes,
        width=fixture.width,
        height=fixture.height,
        iou_threshold=iou_threshold,
    )
    cm = match_large_targets(
        c_match_preds,
        fixture.gt_boxes,
        fixture.gt_classes,
        ignore_boxes=fixture.ignore_boxes,
        width=fixture.width,
        height=fixture.height,
        iou_threshold=iou_threshold,
    )
    pre_cap_match = _match_large_targets(
        prepared.c_pre_cap_match_predictions,
        fixture.gt_boxes,
        fixture.gt_classes,
        ignore_boxes=fixture.ignore_boxes,
        width=fixture.width,
        height=fixture.height,
        iou_threshold=float(iou_threshold),
        prediction_cap=None,
    )
    full_mapping = dict(prepared.full_a_to_c)
    c_by_orig = dict(prepared.c_by_original_index)
    events: list[AttributionEvent] = []
    for gt_idx in am.tp_gt_indices:
        if gt_idx in cm.gt_to_prediction:
            continue
        category = AttributionCategory.OTHER
        recovers = False
        a_pidx = am.gt_to_prediction[gt_idx]
        anchor = (
            a_raw[a_pidx]
            if a_pidx < len(a_raw)
            and isinstance(a_raw[a_pidx], AuditRawDetection)
            else None
        )
        anchor_c_index = (
            full_mapping.get(anchor.identity_key)
            if anchor is not None
            else None
        )
        cluster_index = (
            next(
                (
                    index
                    for index, members in enumerate(
                        reconstruction.cluster_members
                    )
                    if anchor_c_index in members
                ),
                None,
            )
            if anchor_c_index is not None
            else None
        )
        if cluster_index is not None:
            members = tuple(
                c_by_orig[index]
                for index in reconstruction.cluster_members[cluster_index]
                if index in c_by_orig
            )
            mixed = any(
                member.source_order == 0 for member in members
            ) and any(member.source_order > 0 for member in members)
            if mixed and cluster_index < len(c_preds):
                fulls = [
                    member for member in members if member.source_order == 0
                ]
                best = min(
                    fulls,
                    key=lambda member: (
                        -member.score,
                        member.source_order,
                        member.query_index,
                        member.original_index,
                    ),
                )
                counterfactual = list(c_preds)
                old = counterfactual[cluster_index]
                standard_fails_geometry = (
                    old.class_id != fixture.gt_classes[gt_idx]
                    or _iou64(old.box, fixture.gt_boxes[gt_idx])
                    < float(iou_threshold)
                )
                if standard_fails_geometry:
                    counterfactual[cluster_index] = replace(
                        old, box=best.global_xyxy
                    )
                    counterfactual_records = tuple(
                        {
                            "box": prediction.box,
                            "score": prediction.score,
                            "class_id": prediction.class_id,
                            "source_order": prediction.source_order,
                            "query_index": prediction.query_index,
                            "original_index": cluster_members[0],
                        }
                        for prediction, cluster_members in zip(
                            counterfactual,
                            reconstruction.cluster_members,
                        )
                    )
                    recovers = gt_idx in match_large_targets(
                        counterfactual_records,
                        fixture.gt_boxes,
                        fixture.gt_classes,
                        ignore_boxes=fixture.ignore_boxes,
                        width=fixture.width,
                        height=fixture.height,
                        iou_threshold=iou_threshold,
                    ).gt_to_prediction
                    if recovers:
                        category = (
                            AttributionCategory.MIXED_CLUSTER_LOCALIZATION
                        )
        if category is AttributionCategory.OTHER:
            pre_cap_prediction = pre_cap_match.gt_to_prediction.get(gt_idx)
            if (
                pre_cap_prediction is not None
                and pre_cap_prediction
                >= len(reconstruction.standard_predictions)
            ):
                category = AttributionCategory.FINAL_300_TRUNCATION
        if category is AttributionCategory.OTHER:
            for prediction_index, prediction in enumerate(c_match_preds):
                box, _, class_id, *_ = _prediction_fields(
                    prediction, prediction_index
                )
                if (
                    class_id == fixture.gt_classes[gt_idx]
                    and _iou64(box, fixture.gt_boxes[gt_idx])
                    >= float(iou_threshold)
                    and prediction_index
                    not in cm.neutral_prediction_indices
                ):
                    category = AttributionCategory.MATCHING_COMPETITION
                    break
            else:
                if not any(
                    _prediction_fields(prediction, index)[2]
                    == fixture.gt_classes[gt_idx]
                    and _iou64(
                        _prediction_fields(prediction, index)[0],
                        fixture.gt_boxes[gt_idx],
                    )
                    >= float(iou_threshold)
                    for index, prediction in enumerate(c_match_preds)
                ):
                    category = AttributionCategory.CLASS_OR_CANDIDATE_LOSS
        events.append(
            AttributionEvent(
                fixture.image_id,
                gt_idx,
                float(iou_threshold),
                category,
                recovers,
            )
        )
    return ImageAuditResult(tuple(events), am, cm)


def _fuse_float64(cluster: tuple[Detection, ...]) -> Detection:
    seed = cluster[0]
    if len(cluster) == 1:
        return seed
    # Match the frozen ``sbr_fusion._fuse_cluster`` accumulation order
    # bit-for-bit. NumPy reductions may use pairwise summation and differ by
    # an ulp on larger legal clusters.
    total = sum(float(member.score) for member in cluster)
    if not math.isfinite(total) or total <= 0.0:
        box = seed.box
    else:
        box = tuple(
            sum(
                float(member.score) * float(member.box[index])
                for member in cluster
            )
            / total
            for index in range(4)
        )
    return Detection(
        box=box,
        score=max(member.score for member in cluster),
        class_id=seed.class_id,
        source_order=seed.source_order,
        query_index=seed.query_index,
        view_xyxy=seed.view_xyxy,
        global_xyxy=seed.global_xyxy,
        network_xyxy=seed.network_xyxy,
        tile_local_box=seed.tile_local_box,
        global_box=seed.global_box,
        tile_bounds=seed.tile_bounds,
        transform=seed.transform,
        tile_index=seed.tile_index,
        members=cluster,
    )


def reconstruct_c_clusters(
    raw_detections: Iterable[AuditRawDetection],
) -> CClusterReconstruction:
    """Rebuild strict-IoS Arm-C clusters and standard top-300 predictions."""

    raw = tuple(raw_detections)
    if any(
        not isinstance(detection, AuditRawDetection) or detection.arm != "C"
        for detection in raw
    ):
        raise ValueError("cluster reconstruction accepts only Arm-C detections")
    _validate_c_raw_provenance(raw)
    if raw:
        image_signature = {(item.image_id, item.width, item.height) for item in raw}
        if len(image_signature) != 1:
            raise ValueError("cluster reconstruction is limited to one image")
    original_indices = [item.original_index for item in raw]
    if len(set(original_indices)) != len(original_indices):
        raise ValueError("Arm-C original raw indices must be unique")

    # Sort raw records by the frozen key before invoking the fusion primitive.
    # ``greedy_ios_clusters`` only sees enumerate-order as its final tie-break;
    # sorting here preserves immutable original raw indices without changing the
    # public fusion implementation.
    ordered_raw = tuple(sorted(raw, key=lambda item: (-item.score, item.source_order, item.query_index, item.original_index)))
    detections = tuple(item.to_detection() for item in ordered_raw)
    raw_index_by_object = {
        id(detection): item.original_index
        for item, detection in zip(ordered_raw, detections)
    }
    clusters = greedy_ios_clusters(detections, ios_threshold=0.5)
    if sum(len(cluster) for cluster in clusters) != len(detections):
        raise ValueError("an Arm-C raw detection was rejected during clustering")

    sortable: list[
        tuple[int, Detection, tuple[int, ...]]
    ] = []
    for original_cluster_index, cluster in enumerate(clusters):
        prediction = _fuse_float64(cluster)
        members = tuple(raw_index_by_object[id(member)] for member in cluster)
        sortable.append((original_cluster_index, prediction, members))

    ordered = sorted(
        sortable,
        key=lambda item: (
            -float(item[1].score),
            int(item[1].source_order),
            int(item[1].query_index),
            item[0],
        ),
    )
    pre_cap = tuple(item[1] for item in ordered)
    return CClusterReconstruction(
        pre_cap_predictions=pre_cap,
        standard_predictions=pre_cap[:300],
        cluster_members=tuple(item[2] for item in ordered),
    )


def apply_large_view_guard(
    reconstruction: CClusterReconstruction,
    raw_detections: Iterable[AuditRawDetection],
) -> CClusterReconstruction:
    """Replace eligible mixed-cluster coordinates with their full-view anchor."""

    raw = tuple(raw_detections)
    _raw_detection_hash(raw)
    rebuilt = reconstruct_c_clusters(raw)
    if not _same_reconstruction(reconstruction, rebuilt):
        raise ValueError("standard reconstruction does not match supplied raw data")
    return _apply_guard_prevalidated(reconstruction, raw)


def _apply_guard_prevalidated(
    reconstruction: CClusterReconstruction,
    raw_detections: Iterable[AuditRawDetection],
) -> CClusterReconstruction:
    """Apply the guard to a reconstruction created from ``raw_detections``."""

    raw = tuple(raw_detections)
    _raw_detection_hash(raw)
    member_clusters = _validated_cluster_members(reconstruction)
    pre_cap_predictions, selected_predictions = (
        _validated_reconstruction_predictions(reconstruction)
    )
    if (
        len(pre_cap_predictions) != len(member_clusters)
        or len(selected_predictions) != min(300, len(pre_cap_predictions))
        or selected_predictions != pre_cap_predictions[: len(selected_predictions)]
    ):
        raise ValueError("standard reconstruction shape or top-300 is invalid")
    raw_indices = {detection.original_index for detection in raw}
    member_indices = {
        member for cluster in member_clusters for member in cluster
    }
    if raw_indices != member_indices:
        raise ValueError("cluster membership must cover exact raw provenance")
    raw_by_index = {
        detection.original_index: detection for detection in raw
    }
    guarded_pre_cap = list(pre_cap_predictions)
    for cluster_index, cluster_member_indices in enumerate(member_clusters):
        members = tuple(raw_by_index[index] for index in cluster_member_indices)
        full_members = tuple(member for member in members if member.source_order == 0)
        if not full_members or not any(member.source_order > 0 for member in members):
            continue
        anchor = min(
            full_members,
            key=lambda member: (
                -member.score,
                member.source_order,
                member.query_index,
                member.original_index,
            ),
        )
        if effective_size(
            anchor.global_xyxy, width=anchor.width, height=anchor.height
        ) <= 96.0:
            continue
        guarded_pre_cap[cluster_index] = replace(
            guarded_pre_cap[cluster_index], box=anchor.global_xyxy
        )
    pre_cap = tuple(guarded_pre_cap)
    return CClusterReconstruction(
        pre_cap_predictions=pre_cap,
        standard_predictions=pre_cap[: len(reconstruction.standard_predictions)],
        cluster_members=reconstruction.cluster_members,
    )


def _raw_detection_hash(detections: Iterable[AuditRawDetection]) -> str:
    raw = tuple(detections)
    if any(
        not isinstance(detection, AuditRawDetection) or detection.arm != "C"
        for detection in raw
    ):
        raise ValueError("guard provenance accepts only Arm-C detections")
    _validate_c_raw_provenance(raw)
    if len({detection.original_index for detection in raw}) != len(raw):
        raise ValueError("guard provenance raw indices must be unique")
    payload = [
        {
            "image_id": detection.image_id,
            "arm": detection.arm,
            "width": detection.width,
            "height": detection.height,
            "source_order": detection.source_order,
            "query_index": detection.query_index,
            "class_id": detection.class_id,
            "score": detection.score,
            "network_xyxy": detection.network_xyxy,
            "view_xyxy": detection.view_xyxy,
            "global_xyxy": detection.global_xyxy,
            "tile_bounds": detection.tile_bounds,
            "original_index": detection.original_index,
        }
        for detection in raw
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_c_raw_provenance(
    detections: Iterable[AuditRawDetection],
) -> None:
    for detection in detections:
        if not 0 <= detection.source_order <= 4:
            raise ValueError("Arm-C source_order must be within [0,4]")
        if detection.source_order == 0 and detection.tile_bounds is not None:
            raise ValueError("Arm-C source 0 must be a full-view detection")
        if detection.source_order > 0 and detection.tile_bounds is None:
            raise ValueError("Arm-C sources 1..4 must be local detections")


def _validated_cluster_members(
    reconstruction: CClusterReconstruction,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(reconstruction, CClusterReconstruction):
        raise ValueError("guard invariants require cluster reconstructions")
    members = tuple(
        tuple(_strict_nonnegative_int("cluster member", member) for member in cluster)
        for cluster in reconstruction.cluster_members
    )
    if any(not cluster for cluster in members):
        raise ValueError("clusters must not be empty")
    flattened = tuple(member for cluster in members for member in cluster)
    if len(set(flattened)) != len(flattened):
        raise ValueError("cluster raw members must be unique")
    return members


def _validated_reconstruction_predictions(
    reconstruction: CClusterReconstruction,
) -> tuple[tuple[Detection, ...], tuple[Detection, ...]]:
    pre_cap = tuple(reconstruction.pre_cap_predictions)
    selected = tuple(reconstruction.standard_predictions)
    for index, prediction in enumerate(pre_cap):
        _prediction_fields(prediction, index)
    for index, prediction in enumerate(selected):
        _prediction_fields(prediction, index)
    return pre_cap, selected


def _canonical_float(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return struct.pack(">d", result).hex()


def _canonical_metadata(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return ("int", int(value))
    if isinstance(value, Real):
        return ("float64", _canonical_float(value, "metadata"))
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (field.name, _canonical_metadata(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    if isinstance(value, Mapping):
        items = [
            (_canonical_metadata(key), _canonical_metadata(item))
            for key, item in value.items()
        ]
        return ("mapping", tuple(sorted(items, key=repr)))
    if isinstance(value, Sequence):
        return ("sequence", tuple(_canonical_metadata(item) for item in value))
    raise ValueError(f"unsupported provenance metadata: {type(value).__name__}")


def _canonical_optional_box(value: Any, name: str) -> Any:
    if value is None:
        return None
    box = _validated_box(name, value)
    return tuple(_canonical_float(item, name) for item in box)


def _canonical_detection_identity(
    prediction: Detection, *, _seen: set[int] | None = None
) -> tuple[Any, ...]:
    if not isinstance(prediction, Detection):
        raise ValueError("guard output must contain Detection records")
    _prediction_fields(prediction, 0)
    seen = set() if _seen is None else _seen
    identity = id(prediction)
    if identity in seen:
        raise ValueError("cyclic Detection members are invalid")
    seen.add(identity)
    try:
        members = tuple(
            _canonical_detection_identity(member, _seen=seen)
            for member in prediction.members
        )
    finally:
        seen.remove(identity)
    tile_index = (
        None
        if prediction.tile_index is None
        else _strict_nonnegative_int("tile_index", prediction.tile_index)
    )
    return (
        _canonical_optional_box(prediction.box, "box"),
        _canonical_float(prediction.score, "score"),
        _strict_nonnegative_int("class_id", prediction.class_id),
        _strict_nonnegative_int("source_order", prediction.source_order),
        _strict_nonnegative_int("query_index", prediction.query_index),
        _canonical_optional_box(prediction.view_xyxy, "view_xyxy"),
        _canonical_optional_box(prediction.global_xyxy, "global_xyxy"),
        _canonical_optional_box(prediction.network_xyxy, "network_xyxy"),
        _canonical_optional_box(prediction.tile_local_box, "tile_local_box"),
        _canonical_optional_box(prediction.global_box, "global_box"),
        _canonical_metadata(prediction.tile_bounds),
        _canonical_metadata(prediction.transform),
        tile_index,
        members,
    )


def _same_prediction(left: Detection, right: Detection) -> bool:
    return _canonical_detection_identity(left) == _canonical_detection_identity(
        right
    )


def _same_reconstruction(
    left: CClusterReconstruction, right: CClusterReconstruction
) -> bool:
    try:
        left_members = _validated_cluster_members(left)
        right_members = _validated_cluster_members(right)
        left_pre, left_selected = _validated_reconstruction_predictions(left)
        right_pre, right_selected = _validated_reconstruction_predictions(right)
        return (
            left_members == right_members
            and len(left_pre) == len(right_pre)
            and len(left_selected) == len(right_selected)
            and all(
                _same_prediction(left_prediction, right_prediction)
                for left_prediction, right_prediction in zip(left_pre, right_pre)
            )
            and all(
                _same_prediction(left_prediction, right_prediction)
                for left_prediction, right_prediction in zip(
                    left_selected, right_selected
                )
            )
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _verify_guard_invariants_core(
    standard: CClusterReconstruction,
    guarded: CClusterReconstruction,
    raw_detections: Iterable[AuditRawDetection],
    *,
    guarded_raw_detections: Iterable[AuditRawDetection] | None = None,
    prevalidated: bool,
) -> dict[str, bool | float]:
    """Verify that Large-View Guard changed coordinates and nothing else."""

    result = {
        "raw_hash_equal": False,
        "cluster_hash_equal": False,
        "cluster_count_equal": False,
        "scores_equal": False,
        "classes_equal": False,
        "selected_cluster_ids_equal": False,
        "singleton_preservation": 0.0,
        "passed": False,
    }
    original_raw = tuple(raw_detections)
    guarded_raw = (
        original_raw
        if guarded_raw_detections is None
        else tuple(guarded_raw_detections)
    )
    try:
        result["raw_hash_equal"] = (
            _raw_detection_hash(original_raw) == _raw_detection_hash(guarded_raw)
        )
    except (TypeError, ValueError):
        pass

    try:
        standard_members = _validated_cluster_members(standard)
        guarded_members = _validated_cluster_members(guarded)
        result["cluster_count_equal"] = (
            len(standard.pre_cap_predictions)
            == len(standard_members)
            == len(guarded.pre_cap_predictions)
            == len(guarded_members)
        )
    except (AttributeError, TypeError, ValueError):
        standard_members = ()
        guarded_members = ()

    try:
        standard_pre, standard_selected = _validated_reconstruction_predictions(
            standard
        )
        guarded_pre, guarded_selected = _validated_reconstruction_predictions(
            guarded
        )
        expected = (
            _apply_guard_prevalidated(standard, original_raw)
            if prevalidated
            else apply_large_view_guard(standard, original_raw)
        )
        expected_pre, expected_selected = _validated_reconstruction_predictions(
            expected
        )
        expected_members = _validated_cluster_members(expected)
        result["cluster_hash_equal"] = (
            expected_members == guarded_members
            and len(expected_pre) == len(guarded_pre)
            and len(expected_selected) == len(guarded_selected)
            and all(
                _same_prediction(expected_prediction, guarded_prediction)
                for expected_prediction, guarded_prediction in zip(
                    expected_pre, guarded_pre
                )
            )
            and all(
                _same_prediction(expected_prediction, guarded_prediction)
                for expected_prediction, guarded_prediction in zip(
                    expected_selected, guarded_selected
                )
            )
        )
        result["scores_equal"] = (
            tuple(prediction.score for prediction in standard_pre)
            == tuple(prediction.score for prediction in guarded_pre)
            and tuple(prediction.score for prediction in standard_selected)
            == tuple(prediction.score for prediction in guarded_selected)
        )
        result["classes_equal"] = (
            tuple(prediction.class_id for prediction in standard_pre)
            == tuple(prediction.class_id for prediction in guarded_pre)
            and tuple(prediction.class_id for prediction in standard_selected)
            == tuple(prediction.class_id for prediction in guarded_selected)
        )
        standard_prefix_valid = (
            len(standard_selected) <= len(standard_pre)
            and all(
                _same_prediction(selected, pre_cap)
                for selected, pre_cap in zip(standard_selected, standard_pre)
            )
        )
        guarded_prefix_valid = (
            len(guarded_selected) <= len(guarded_pre)
            and all(
                _same_prediction(selected, pre_cap)
                for selected, pre_cap in zip(guarded_selected, guarded_pre)
            )
        )
        result["selected_cluster_ids_equal"] = (
            standard_prefix_valid
            and guarded_prefix_valid
            and tuple(
                (prediction.source_order, prediction.query_index)
                for prediction in standard_pre
            )
            == tuple(
                (prediction.source_order, prediction.query_index)
                for prediction in guarded_pre
            )
            and standard_members[: len(standard_selected)]
            == guarded_members[: len(guarded_selected)]
        )
        singleton_indices = tuple(
            index
            for index, member_ids in enumerate(standard_members)
            if len(member_ids) == 1
        )
        preserved_singletons = (
            sum(
                _same_prediction(standard_pre[index], guarded_pre[index])
                for index in singleton_indices
            )
            if len(standard_pre) == len(guarded_pre) == len(standard_members)
            else 0
        )
        result["singleton_preservation"] = (
            float(preserved_singletons) / float(len(singleton_indices))
            if singleton_indices
            else 1.0
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        pass

    singleton_preservation = result["singleton_preservation"]
    result["passed"] = (
        isinstance(singleton_preservation, float)
        and math.isfinite(singleton_preservation)
        and singleton_preservation == 1.0
        and all(
            value is True
            for key, value in result.items()
            if key not in {"singleton_preservation", "passed"}
        )
    )
    return result


def verify_guard_invariants(
    standard: CClusterReconstruction,
    guarded: CClusterReconstruction,
    raw_detections: Iterable[AuditRawDetection],
    *,
    guarded_raw_detections: Iterable[AuditRawDetection] | None = None,
) -> dict[str, bool | float]:
    """Verify a standalone guard output against a fresh raw reconstruction."""

    return _verify_guard_invariants_core(
        standard,
        guarded,
        raw_detections,
        guarded_raw_detections=guarded_raw_detections,
        prevalidated=False,
    )


def _verify_guard_prevalidated(
    standard: CClusterReconstruction,
    guarded: CClusterReconstruction,
    raw_detections: Iterable[AuditRawDetection],
    *,
    guarded_raw_detections: Iterable[AuditRawDetection] | None = None,
) -> dict[str, bool | float]:
    return _verify_guard_invariants_core(
        standard,
        guarded,
        raw_detections,
        guarded_raw_detections=guarded_raw_detections,
        prevalidated=True,
    )


_GUARD_DELTA_KEYS = (
    "AP-tiny-SBR",
    "mAP50-95",
    "tiny_recall",
    "AP75",
    "AP-large-SBR",
)
_GUARD_INVARIANT_KEYS = (
    "raw_hash_equal",
    "cluster_hash_equal",
    "cluster_count_equal",
    "scores_equal",
    "classes_equal",
    "selected_cluster_ids_equal",
    "singleton_preservation",
    "passed",
)


def _guard_metric_deltas(
    minuend: Mapping[str, Any], subtrahend: Mapping[str, Any]
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in _GUARD_DELTA_KEYS:
        left = minuend.get(key)
        right = subtrahend.get(key)
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not isinstance(left, Real)
            or not isinstance(right, Real)
            or not math.isfinite(float(left))
            or not math.isfinite(float(right))
        ):
            raise ValueError(f"guard metric {key} must be finite")
        deltas[key] = float(left) - float(right)
    return deltas


_EVIDENCE_ROW_KEYS = frozenset(
    {
        "image_id",
        "width",
        "height",
        "pred_boxes",
        "pred_scores",
        "pred_classes",
        "pred_source",
        "pred_query",
        "gt_boxes",
        "gt_classes",
        "ignore_boxes",
        "effective_gain",
    }
)


def _row_sequence(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be an explicit sequence")
    try:
        return tuple(value)
    except TypeError:
        raise ValueError(f"{name} must be an explicit sequence") from None


def _row_boxes(value: Any, name: str) -> tuple[Box, ...]:
    boxes = tuple(
        _validated_box(name, box) for box in _row_sequence(value, name)
    )
    if any(coordinate < 0.0 for box in boxes for coordinate in box):
        raise ValueError(f"{name} must use nonnegative coordinates")
    return boxes


def _validate_evidence_rows(
    rows: Sequence[Mapping[str, Any]], *, arm: str
) -> tuple[dict[str, Any], ...]:
    try:
        raw_rows = tuple(rows)
    except TypeError:
        raise ValueError(f"{arm} evidence rows must be a sequence") from None
    if not raw_rows:
        raise ValueError(f"{arm} evidence rows must be nonempty")
    normalized: list[dict[str, Any]] = []
    image_ids: set[str] = set()
    for row_index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{arm} row {row_index} must be a mapping")
        missing = _EVIDENCE_ROW_KEYS.difference(row)
        if missing:
            raise ValueError(
                f"{arm} row {row_index} is missing explicit keys: "
                + ",".join(sorted(missing))
            )
        image_id = row["image_id"]
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("image_id must be a nonempty exact string")
        if image_id in image_ids:
            raise ValueError(f"duplicate {arm} image_id: {image_id!r}")
        image_ids.add(image_id)
        width = _positive_dimension("width", row["width"])
        height = _positive_dimension("height", row["height"])
        pred_boxes = _row_boxes(row["pred_boxes"], "pred_boxes")
        pred_scores = tuple(
            _validated_score("pred_score", score)
            for score in _row_sequence(row["pred_scores"], "pred_scores")
        )
        pred_classes = tuple(
            _strict_nonnegative_int("pred_class", class_id)
            for class_id in _row_sequence(
                row["pred_classes"], "pred_classes"
            )
        )
        pred_source = tuple(
            _strict_nonnegative_int("pred_source", source)
            for source in _row_sequence(row["pred_source"], "pred_source")
        )
        pred_query = tuple(
            _strict_nonnegative_int("pred_query", query)
            for query in _row_sequence(row["pred_query"], "pred_query")
        )
        prediction_count = len(pred_boxes)
        if any(
            len(values) != prediction_count
            for values in (
                pred_scores,
                pred_classes,
                pred_source,
                pred_query,
            )
        ):
            raise ValueError("prediction evidence lengths must agree")
        if arm == "A":
            if any(source != 0 for source in pred_source):
                raise ValueError("Arm-A evidence pred_source must be all zero")
        elif arm in {"C", "V2"}:
            if any(source > 4 for source in pred_source):
                raise ValueError(f"Arm-{arm} pred_source must be within [0,4]")
        else:
            raise ValueError(f"unsupported evidence arm: {arm}")
        gt_boxes = _row_boxes(row["gt_boxes"], "gt_boxes")
        gt_classes = tuple(
            _strict_nonnegative_int("gt_class", class_id)
            for class_id in _row_sequence(row["gt_classes"], "gt_classes")
        )
        if len(gt_boxes) != len(gt_classes):
            raise ValueError("ground-truth evidence lengths must agree")
        ignore_boxes = _row_boxes(row["ignore_boxes"], "ignore_boxes")
        effective_gain = row["effective_gain"]
        if isinstance(effective_gain, bool) or not isinstance(
            effective_gain, Real
        ):
            raise ValueError("effective_gain must be a finite positive scalar")
        effective_gain = float(effective_gain)
        expected_gain = min(
            640.0 / float(width), 640.0 / float(height), 1.0
        )
        if (
            not math.isfinite(effective_gain)
            or effective_gain <= 0.0
            or not math.isclose(
                effective_gain, expected_gain, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise ValueError("effective_gain disagrees with frozen 640 gain")
        normalized.append(
            {
                "image_id": image_id,
                "width": width,
                "height": height,
                "pred_boxes": pred_boxes,
                "pred_scores": pred_scores,
                "pred_classes": pred_classes,
                "pred_source": pred_source,
                "pred_query": pred_query,
                "gt_boxes": gt_boxes,
                "gt_classes": gt_classes,
                "ignore_boxes": ignore_boxes,
                "effective_gain": effective_gain,
            }
        )
    return tuple(normalized)


def _ground_truth_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["image_id"],
        row["width"],
        row["height"],
        tuple(
            _canonical_optional_box(box, "gt_box")
            for box in row["gt_boxes"]
        ),
        tuple(row["gt_classes"]),
        tuple(
            _canonical_optional_box(box, "ignore_box")
            for box in row["ignore_boxes"]
        ),
        _canonical_float(row["effective_gain"], "effective_gain"),
    )


def _prediction_metadata_signature(
    row: Mapping[str, Any],
) -> tuple[Any, ...]:
    return (
        len(row["pred_boxes"]),
        tuple(
            _canonical_float(score, "pred_score")
            for score in row["pred_scores"]
        ),
        tuple(row["pred_classes"]),
        tuple(row["pred_source"]),
        tuple(row["pred_query"]),
    )


def _validate_aligned_evidence(
    a_rows: Sequence[Mapping[str, Any]],
    c_rows: Sequence[Mapping[str, Any]],
    v2_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    normalized_a = _validate_evidence_rows(a_rows, arm="A")
    normalized_c = _validate_evidence_rows(c_rows, arm="C")
    normalized_v2 = _validate_evidence_rows(v2_rows, arm="V2")
    if not len(normalized_a) == len(normalized_c) == len(normalized_v2):
        raise ValueError("A/C/V2 evidence rows must have equal lengths")
    for a_row, c_row, v2_row in zip(
        normalized_a, normalized_c, normalized_v2
    ):
        signature = _ground_truth_signature(a_row)
        if (
            _ground_truth_signature(c_row) != signature
            or _ground_truth_signature(v2_row) != signature
        ):
            raise ValueError(
                "A/C/V2 image order or ground-truth evidence disagrees"
            )
        if _prediction_metadata_signature(
            c_row
        ) != _prediction_metadata_signature(v2_row):
            raise ValueError(
                "C/V2 prediction metadata must be canonical-identical"
            )
    return normalized_a, normalized_c, normalized_v2


def _evidence_predictions(row: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "box": box,
            "score": score,
            "class_id": class_id,
            "source_order": source,
            "query_index": query,
            "original_index": index,
        }
        for index, (box, score, class_id, source, query) in enumerate(
            zip(
                row["pred_boxes"],
                row["pred_scores"],
                row["pred_classes"],
                row["pred_source"],
                row["pred_query"],
            )
        )
    )


def _recompute_a_tp_to_c_fn(
    a_rows: Sequence[Mapping[str, Any]],
    c_rows: Sequence[Mapping[str, Any]],
) -> int:
    total = 0
    for a_row, c_row in zip(a_rows, c_rows):
        a_match = match_large_targets(
            _evidence_predictions(a_row),
            a_row["gt_boxes"],
            a_row["gt_classes"],
            ignore_boxes=a_row["ignore_boxes"],
            width=a_row["width"],
            height=a_row["height"],
            iou_threshold=0.75,
        )
        c_match = match_large_targets(
            _evidence_predictions(c_row),
            c_row["gt_boxes"],
            c_row["gt_classes"],
            ignore_boxes=c_row["ignore_boxes"],
            width=c_row["width"],
            height=c_row["height"],
            iou_threshold=0.75,
        )
        total += len(
            set(a_match.tp_gt_indices).difference(c_match.tp_gt_indices)
        )
    return total


def evaluate_guard_upper_bound(
    a_rows: Sequence[Mapping[str, Any]],
    c_rows: Sequence[Mapping[str, Any]],
    v2_rows: Sequence[Mapping[str, Any]],
    *,
    mixed_localization_unique_large_gt: int,
    a_tp_to_c_fn_unique_large_gt: int,
    invariants: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the two independent frozen SBR-V2 audit eligibility gates."""

    numerator = _strict_nonnegative_int(
        "mixed_localization_unique_large_gt",
        mixed_localization_unique_large_gt,
    )
    denominator = _strict_nonnegative_int(
        "a_tp_to_c_fn_unique_large_gt",
        a_tp_to_c_fn_unique_large_gt,
    )
    if numerator > denominator:
        raise ValueError(
            "mixed localization count cannot exceed A-TP-to-C-FN count"
        )
    if not isinstance(invariants, Mapping):
        raise ValueError("invariants must be a mapping")

    normalized_a, normalized_c, normalized_v2 = _validate_aligned_evidence(
        a_rows, c_rows, v2_rows
    )
    recomputed_denominator = _recompute_a_tp_to_c_fn(
        normalized_a, normalized_c
    )
    if denominator != recomputed_denominator:
        raise ValueError(
            "A-TP-to-C-FN denominator disagrees with AP75 evidence"
        )

    a_metrics = evaluate_dataset(normalized_a)
    c_metrics = evaluate_dataset(normalized_c)
    v2_metrics = evaluate_dataset(normalized_v2)
    v2_minus_a = _guard_metric_deltas(v2_metrics, a_metrics)
    v2_minus_c = _guard_metric_deltas(v2_metrics, c_metrics)
    mechanism_share = (
        float(numerator) / float(denominator) if denominator > 0 else 0.0
    )
    mechanism_gate = (
        "PASS" if denominator > 0 and mechanism_share >= 0.60 else "FAIL"
    )
    recoverable_gate = (
        "PASS"
        if v2_metrics["AP-large-SBR"] >= a_metrics["AP-large-SBR"] - 0.005
        else "FAIL"
    )
    normalized_invariants = {
        key: invariants.get(key) is True
        for key in _GUARD_INVARIANT_KEYS
        if key not in {"singleton_preservation", "passed"}
    }
    singleton_value = invariants.get("singleton_preservation")
    if (
        isinstance(singleton_value, bool)
        or not isinstance(singleton_value, Real)
        or not math.isfinite(float(singleton_value))
        or not 0.0 <= float(singleton_value) <= 1.0
    ):
        normalized_invariants["singleton_preservation"] = 0.0
    else:
        normalized_invariants["singleton_preservation"] = float(singleton_value)
    normalized_invariants["passed"] = (
        invariants.get("passed") is True
        and normalized_invariants["singleton_preservation"] == 1.0
        and all(
            normalized_invariants[key] is True
            for key in _GUARD_INVARIANT_KEYS
            if key not in {"singleton_preservation", "passed"}
        )
    )
    return {
        "mechanism_share_ap75": mechanism_share,
        "mechanism_gate": mechanism_gate,
        "a_metrics": a_metrics,
        "c_metrics": c_metrics,
        "v2_metrics": v2_metrics,
        "v2_minus_a": v2_minus_a,
        "v2_minus_c": v2_minus_c,
        "recoverable_upper_bound_gate": recoverable_gate,
        "invariants": normalized_invariants,
    }


__all__ = [
    "AuditRawDetection",
    "AuditImage",
    "PreparedImageAudit",
    "AttributionCategory",
    "AttributionEvent",
    "ImageAuditResult",
    "LargeMatchResult",
    "effective_size",
    "match_large_targets",
    "audit_image_at_threshold",
    "prepare_image_audit",
    "audit_prepared_image_at_threshold",
    "group_relevant_raw_rows",
    "map_full_a_to_c",
    "reconstruct_c_clusters",
    "apply_large_view_guard",
    "verify_guard_invariants",
    "evaluate_guard_upper_bound",
]
