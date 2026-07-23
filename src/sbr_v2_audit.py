"""Deterministic reconstruction primitives for the frozen SBR-V2 audit."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import json
import math
from numbers import Integral
from typing import Any

import numpy as np

from src.sbr_fusion import Detection, greedy_ios_clusters


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
        try:
            score = float(self.score)
        except (TypeError, ValueError):
            raise ValueError("score must be finite and within [0, 1]") from None
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
        return prediction.global_xyxy, prediction.score, prediction.class_id, prediction.source_order, prediction.query_index, prediction.original_index
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
            return _validated_box("prediction", box), float(score), int(cls), int(source), int(query), int(original)
        # Accept metrics-style prediction objects without coupling this module to them.
        if all(hasattr(prediction, n) for n in ("box", "score")):
            cls = getattr(prediction, "class_id", getattr(prediction, "cls", None))
            source = getattr(prediction, "source_order", getattr(prediction, "source", 0))
            query = getattr(prediction, "query_index", getattr(prediction, "query", 0))
            if cls is not None:
                return _validated_box("prediction", prediction.box), float(prediction.score), int(cls), int(source), int(query), int(getattr(prediction, "original_index", index))
        raise ValueError("predictions must be Detection or prediction-like records")
    return prediction.box, float(prediction.score), int(prediction.class_id), int(prediction.source_order), int(prediction.query_index), index


def _ioa64(pred_box: Sequence[float], ignore_box: Sequence[float]) -> float:
    p = _validated_box("prediction", pred_box); q = _validated_box("ignore", ignore_box)
    inter = max(0.0, min(p[2], q[2]) - max(p[0], q[0])) * max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
    return float(np.float64(inter) / (np.float64(p[2] - p[0]) * np.float64(p[3] - p[1])))


def match_large_targets(
    predictions: Iterable[Any], gt_boxes: Sequence[Sequence[float]], gt_classes: Sequence[int],
    *, ignore_boxes: Sequence[Sequence[float]] | None = None, width: int, height: int,
    iou_threshold: float, max_det: int = 300, conf_threshold: float = 0.001,
) -> LargeMatchResult:
    """Match predictions to large targets using the frozen evaluator order."""
    if not math.isfinite(float(iou_threshold)) or not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must be finite in [0,1]")
    if isinstance(max_det, bool) or not isinstance(max_det, Integral) or int(max_det) != 300:
        raise ValueError("max_det is frozen at 300")
    if isinstance(conf_threshold, bool) or not isinstance(conf_threshold, (float, np.floating)) or float(conf_threshold) != 0.001:
        raise ValueError("conf_threshold is frozen at 0.001")
    boxes = tuple(_validated_box("gt box", b) for b in gt_boxes)
    classes = tuple(_strict_nonnegative_int("gt class", c) for c in gt_classes)
    if len(boxes) != len(classes):
        raise ValueError("gt_boxes and gt_classes lengths must agree")
    selected = tuple(i for i, b in enumerate(boxes) if effective_size(b, width=width, height=height) > 96.0)
    raw = tuple(predictions)
    eligible = []
    for i, p in enumerate(raw):
        box, score, cls, source, query, original = _prediction_fields(p, i)
        if not math.isfinite(float(score)) or float(score) < float(conf_threshold):
            continue
        eligible.append((i, p, box, float(score), cls, source, query, original))
    eligible.sort(key=lambda x: (-x[3], x[5], x[6], x[7]))
    eligible = eligible[:300]
    neutral: list[int] = []
    ignore = tuple(ignore_boxes or ())
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
    am = match_large_targets(a_preds, fixture.gt_boxes, fixture.gt_classes, ignore_boxes=fixture.ignore_boxes, width=fixture.width, height=fixture.height, iou_threshold=iou_threshold)
    cm = match_large_targets(c_match_preds, fixture.gt_boxes, fixture.gt_classes, ignore_boxes=fixture.ignore_boxes, width=fixture.width, height=fixture.height, iou_threshold=iou_threshold)
    events: list[AttributionEvent] = []
    full_keys = {d.identity_key: d for d in a_raw if isinstance(d, AuditRawDetection) and d.source_order == 0}
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
                    cf[cluster_index] = replace(old, box=best.global_xyxy)
                    cf_records = tuple(
                        {"box": p.box, "score": p.score, "class_id": p.class_id,
                         "source_order": p.source_order, "query_index": p.query_index,
                         "original_index": members_[0]}
                        for p, members_ in zip(cf, reconstruction.cluster_members)
                    )
                    recovers = gt_idx in match_large_targets(cf_records, fixture.gt_boxes, fixture.gt_classes, ignore_boxes=fixture.ignore_boxes, width=fixture.width, height=fixture.height, iou_threshold=iou_threshold).gt_to_prediction
                    if recovers: category = AttributionCategory.MIXED_CLUSTER_LOCALIZATION
                else:
                    category = AttributionCategory.FINAL_300_TRUNCATION
            elif cluster_index >= len(reconstruction.standard_predictions):
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


def _fuse_float64(cluster: tuple[Detection, ...]) -> Detection:
    seed = cluster[0]
    if len(cluster) == 1:
        return seed
    weights = np.asarray([member.score for member in cluster], dtype=np.float64)
    total = np.sum(weights, dtype=np.float64)
    if not np.isfinite(total) or total <= np.float64(0.0):
        box = seed.box
    else:
        coordinates = np.asarray(
            [member.box for member in cluster], dtype=np.float64
        )
        fused = np.sum(
            coordinates * weights[:, np.newaxis], axis=0, dtype=np.float64
        ) / total
        box = tuple(float(value) for value in fused)
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


__all__ = [
    "AuditRawDetection",
    "AuditImage",
    "AttributionCategory",
    "AttributionEvent",
    "ImageAuditResult",
    "LargeMatchResult",
    "effective_size",
    "match_large_targets",
    "audit_image_at_threshold",
    "group_relevant_raw_rows",
    "map_full_a_to_c",
    "reconstruct_c_clusters",
]
