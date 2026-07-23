"""Deterministic, class-aware Greedy NMM for SBR-RTDETR."""

from __future__ import annotations

from dataclasses import dataclass
import math
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "box", _box_tuple(self.box))
        try:
            score = float(self.score)
        except (TypeError, ValueError):
            score = math.nan
        object.__setattr__(self, "score", score)
        try:
            object.__setattr__(self, "class_id", int(self.class_id))
        except (TypeError, ValueError):
            object.__setattr__(self, "class_id", -1)
        for name in ("source_order", "query_index"):
            try:
                value = int(getattr(self, name))
            except (TypeError, ValueError):
                value = -1
            object.__setattr__(self, name, value)
        for name in ("view_xyxy", "global_xyxy", "network_xyxy", "tile_local_box", "global_box"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _box_tuple(value))
        if self.members:
            object.__setattr__(self, "members", tuple(self.members))

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
        and _valid_box(detection.box)
        and math.isfinite(float(detection.score))
        and float(detection.score) >= 0.0
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

    try:
        threshold = float(ios_threshold)
    except (TypeError, ValueError):
        return ()
    if not math.isfinite(threshold):
        return ()
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


def fuse_standard(
    detections: Iterable[Detection], *, max_det: int = 300, ios_threshold: float = 0.5
) -> tuple[Detection, ...]:
    """Fuse detections with standard score-weighted SBR coordinates."""

    try:
        limit = int(max_det)
    except (TypeError, ValueError):
        return ()
    if limit <= 0:
        return ()
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
]
