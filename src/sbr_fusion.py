"""Deterministic, class-aware Greedy NMM for SBR-RTDETR."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import Any, Iterable


def _box_tuple(value: Any) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        return (math.nan, math.nan, math.nan, math.nan)
    if len(values) != 4:
        return (math.nan, math.nan, math.nan, math.nan)
    return values  # type: ignore[return-value]


@dataclass(frozen=True)
class Detection:
    """An immutable prediction and its raw-view provenance.

    ``box`` is the coordinate frame consumed by fusion (normally global image
    coordinates).  Optional frame-specific coordinates are retained verbatim
    so downstream SP-BRF can inspect the same cluster members.
    """

    box: tuple[float, float, float, float]
    score: float
    class_id: int
    source_order: int
    query_index: int
    view_xyxy: tuple[float, float, float, float] | None = None
    global_xyxy: tuple[float, float, float, float] | None = None
    network_xyxy: tuple[float, float, float, float] | None = None
    tile_local_box: tuple[float, float, float, float] | None = None
    global_box: tuple[float, float, float, float] | None = None
    tile_bounds: Any = None
    transform: Any = None
    tile_index: int | None = None
    members: tuple["Detection", ...] = ()
    _metadata_valid: bool = field(default=True, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        metadata_valid = True
        object.__setattr__(self, "box", _box_tuple(self.box))
        try:
            score = float(self.score)
        except (TypeError, ValueError):
            score = math.nan
        object.__setattr__(self, "score", score)
        for name in ("class_id", "source_order", "query_index"):
            original = getattr(self, name)
            if isinstance(original, bool) or not isinstance(original, Integral):
                metadata_valid = False
        for name in ("source_order", "query_index"):
            try:
                value = int(getattr(self, name))
            except (TypeError, ValueError):
                value = -1
                metadata_valid = False
            object.__setattr__(self, name, value)
        try:
            object.__setattr__(self, "class_id", int(self.class_id))
        except (TypeError, ValueError):
            object.__setattr__(self, "class_id", -1)
            metadata_valid = False
        if self.tile_index is not None and (
            isinstance(self.tile_index, bool) or not isinstance(self.tile_index, Integral)
        ):
            metadata_valid = False
        elif self.tile_index is not None and int(self.tile_index) < 0:
            metadata_valid = False
        for name in ("view_xyxy", "global_xyxy", "network_xyxy", "tile_local_box", "global_box"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _box_tuple(value))
        if self.members:
            object.__setattr__(self, "members", tuple(self.members))
        object.__setattr__(self, "_metadata_valid", metadata_valid)

    @property
    def cluster_members(self) -> tuple["Detection", ...]:
        """Return members represented by this fused detection (or itself)."""

        return self.members if self.members else (self,)


def _valid_box(box: Any) -> bool:
    try:
        if len(box) != 4:
            return False
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return False
    return (
        all(math.isfinite(v) for v in (x1, y1, x2, y2))
        and x2 >= x1
        and y2 >= y1
        and x2 > x1
        and y2 > y1
    )


def _valid_detection(detection: Any) -> bool:
    return (
        isinstance(detection, Detection)
        and detection._metadata_valid
        and detection.class_id >= 0
        and detection.source_order >= 0
        and detection.query_index >= 0
        and _valid_box(detection.box)
        and math.isfinite(float(detection.score))
        and float(detection.score) >= 0.0
        and all(
            value is None or _valid_box(value)
            for value in (
                detection.view_xyxy,
                detection.global_xyxy,
                detection.network_xyxy,
                detection.tile_local_box,
                detection.global_box,
            )
        )
    )


def intersection_over_smaller(box_a: Any, box_b: Any) -> float:
    """Return intersection area divided by the smaller box area.

    Invalid, non-finite, or degenerate rectangles fail closed with ``0.0``.
    """

    if not _valid_box(box_a) or not _valid_box(box_b):
        return 0.0
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    return intersection / min(area_a, area_b)


def _sort_key(item: tuple[int, Detection]) -> tuple[float, int, int, int]:
    original_index, detection = item
    return (-float(detection.score), int(detection.source_order), int(detection.query_index), original_index)


def greedy_ios_clusters(
    detections: Iterable[Detection], *, ios_threshold: float = 0.5
) -> tuple[tuple[Detection, ...], ...]:
    """Build seed-only, class-aware Greedy NMM clusters.

    Each seed absorbs currently-unassigned same-class detections whose IoS is
    strictly greater than ``ios_threshold`` against the seed box.  Matching
    is intentionally non-transitive.
    """

    if isinstance(ios_threshold, bool) or not isinstance(ios_threshold, (int, float)) or float(ios_threshold) != 0.5:
        raise ValueError("SBR protocol freezes ios_threshold at 0.5")
    threshold = 0.5
    valid = [(i, d) for i, d in enumerate(detections) if _valid_detection(d)]
    ordered = [d for _, d in sorted(valid, key=_sort_key)]
    remaining = list(ordered)
    clusters: list[tuple[Detection, ...]] = []
    while remaining:
        seed = remaining.pop(0)
        members = [seed]
        keep: list[Detection] = []
        for candidate in remaining:
            if candidate.class_id == seed.class_id and intersection_over_smaller(seed.box, candidate.box) > threshold:
                members.append(candidate)
            else:
                keep.append(candidate)
        remaining = keep
        clusters.append(tuple(members))
    return tuple(clusters)


def _fuse_cluster(cluster: tuple[Detection, ...]) -> Detection:
    seed = cluster[0]
    if len(cluster) == 1:
        return seed
    total = sum(float(member.score) for member in cluster)
    if not math.isfinite(total) or total <= 0.0:
        merged_box = seed.box
    else:
        merged_box = tuple(
            sum(float(member.score) * float(member.box[index]) for member in cluster) / total
            for index in range(4)
        )
    return Detection(
        box=merged_box,
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


def _tile_bounds(tile: Any) -> tuple[float, float, float, float] | None:
    """Normalize Tile-like metadata to ``(left, top, right, bottom)``."""

    if tile is None:
        return None
    value = getattr(tile, "bounds", tile)
    try:
        bounds = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        raise ValueError("invalid tile metadata") from None
    if len(bounds) != 4 or not all(math.isfinite(item) for item in bounds):
        raise ValueError("invalid tile metadata")
    left, top, right, bottom = bounds
    if right <= left or bottom <= top or left < 0 or top < 0:
        raise ValueError("invalid tile metadata")
    return bounds


def _full_shape(full_shape: Any) -> tuple[float, float]:
    try:
        width, height = (float(item) for item in full_shape)
    except (TypeError, ValueError):
        raise ValueError("full_shape must be (width, height)") from None
    if (
        not math.isfinite(width)
        or not math.isfinite(height)
        or width <= 0
        or height <= 0
    ):
        raise ValueError("full_shape must be positive and finite")
    return width, height


def border_reliability(
    detection: Detection, tile: Any, full_shape: Any
) -> float:
    """Return SP-BRF reliability for one full-view or local-tile detection.

    Full-view detections have no tile metadata and receive reliability one.
    Local detections must carry both tile bounds and a tile-local box.  Only
    artificial tile edges are penalized; real image boundaries are ignored.
    """

    if not isinstance(detection, Detection):
        raise ValueError("detection must be a Detection")
    width, height = _full_shape(full_shape)
    bounds = _tile_bounds(tile)
    if bounds is None:
        if detection.tile_local_box is not None:
            raise ValueError("local detection is missing tile metadata")
        return 1.0
    if detection.tile_local_box is None:
        raise ValueError("local detection is missing tile-local box")
    if not _valid_box(detection.tile_local_box):
        raise ValueError("invalid tile-local box")
    left, top, right, bottom = bounds
    if right > width or bottom > height:
        raise ValueError("tile exceeds full image bounds")
    tile_w = right - left
    tile_h = bottom - top
    overlap_x = 2.0 * tile_w - width
    overlap_y = 2.0 * tile_h - height
    internal_x = left > 0 or right < width
    internal_y = top > 0 or bottom < height
    if internal_x and overlap_x <= 0:
        raise ValueError("horizontal tile overlap must be positive")
    if internal_y and overlap_y <= 0:
        raise ValueError("vertical tile overlap must be positive")
    x1, y1, x2, y2 = (float(value) for value in detection.tile_local_box)
    if x1 < 0.0 or y1 < 0.0 or x2 > tile_w or y2 > tile_h:
        raise ValueError("tile-local box exceeds tile bounds")
    reliabilities: list[float] = []
    if left > 0:
        reliabilities.append(x1 / (overlap_x / 2.0))
    if right < width:
        reliabilities.append((tile_w - x2) / (overlap_x / 2.0))
    if top > 0:
        reliabilities.append(y1 / (overlap_y / 2.0))
    if bottom < height:
        reliabilities.append((tile_h - y2) / (overlap_y / 2.0))
    if not reliabilities:
        return 1.0
    return min(1.0, max(0.0, min(reliabilities)))


def fuse_sp_brf(
    cluster: Iterable[Detection], *, full_shape: Any
) -> Detection:
    """Fuse one precomputed cluster using SP-BRF coordinate weights."""

    members = tuple(cluster)
    if not members:
        raise ValueError("cluster must not be empty")
    _full_shape(full_shape)
    if not all(isinstance(member, Detection) for member in members):
        raise ValueError("cluster members must be Detection instances")
    if not all(_valid_detection(member) for member in members):
        raise ValueError("invalid detection in SP-BRF cluster")
    if len({member.class_id for member in members}) != 1:
        raise ValueError("SP-BRF clusters must be class-consistent")
    seed = members[0]
    if len(members) == 1:
        return seed
    weights: list[float] = []
    for member in members:
        tile = member.tile_bounds
        weights.append(float(member.score) * (1.0 + border_reliability(member, tile, full_shape)))
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        merged_box = seed.box
    else:
        merged_box = tuple(
            sum(weights[pos] * float(member.box[index]) for pos, member in enumerate(members))
            / total
            for index in range(4)
        )
    return Detection(
        box=merged_box,
        score=max(member.score for member in members),
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
        members=members,
    )


def fuse_sp_brf_from_clusters(
    clusters: Iterable[Iterable[Detection]], *, full_shape: Any, max_det: int = 300
) -> tuple[Detection, ...]:
    """Fuse precomputed clusters without recomputing membership."""

    if isinstance(max_det, bool) or not isinstance(max_det, Integral) or int(max_det) != 300:
        raise ValueError("SBR protocol freezes max_det at 300")
    fused = [
        fuse_sp_brf(cluster, full_shape=full_shape)
        for cluster in clusters
    ]
    return tuple(
        detection for _, detection in sorted(enumerate(fused), key=_sort_key)
    )[:300]


def fuse_standard(
    detections: Iterable[Detection], *, max_det: int = 300, ios_threshold: float = 0.5
) -> tuple[Detection, ...]:
    """Fuse detections with standard score-weighted SBR coordinates."""

    if isinstance(max_det, bool) or not isinstance(max_det, Integral) or int(max_det) != 300:
        raise ValueError("SBR protocol freezes max_det at 300")
    limit = 300
    fused = [_fuse_cluster(cluster) for cluster in greedy_ios_clusters(detections, ios_threshold=ios_threshold)]
    return tuple(
        detection
        for _, detection in sorted(enumerate(fused), key=_sort_key)
    )[:limit]


__all__ = [
    "Detection",
    "intersection_over_smaller",
    "greedy_ios_clusters",
    "fuse_standard",
    "border_reliability",
    "fuse_sp_brf",
    "fuse_sp_brf_from_clusters",
]
