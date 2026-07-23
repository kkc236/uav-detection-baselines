"""Deterministic reconstruction primitives for the frozen SBR-V2 audit."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
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

    detections = tuple(item.to_detection() for item in raw)
    raw_index_by_object = {
        id(detection): item.original_index
        for item, detection in zip(raw, detections)
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
    "group_relevant_raw_rows",
    "map_full_a_to_c",
    "reconstruct_c_clusters",
]
