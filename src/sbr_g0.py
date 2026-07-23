"""Frozen zero-training SBR-RTDETR G0 view pipeline.

The module is intentionally detector-agnostic: callers inject a predictor that
accepts an already-square NumPy image and returns prediction records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .sbr_fusion import Detection, fuse_sp_brf_from_clusters, fuse_standard, greedy_ios_clusters
from .sbr_geometry import (
    LetterboxTransform,
    Tile,
    inverse_letterbox_xyxy,
    non_overlapping_tiles,
    overlapping_tiles,
    tile_to_global_xyxy,
)


@dataclass(frozen=True)
class FrozenSBRProtocol:
    imgsz: int = 640
    high_imgsz: int = 1088
    conf: float = 0.001
    max_det: int = 300
    ios_threshold: float = 0.5
    tile_ratio: float = 0.60
    padding_value: int = 114

    def __post_init__(self) -> None:
        expected = {
            "imgsz": 640,
            "high_imgsz": 1088,
            "conf": 0.001,
            "max_det": 300,
            "ios_threshold": 0.5,
            "tile_ratio": 0.60,
            "padding_value": 114,
        }
        for name, value in expected.items():
            if float(getattr(self, name)) != float(value):
                raise ValueError(f"SBR protocol freezes {name} at {value}")


@dataclass(frozen=True)
class ViewSpec:
    arm: str
    view_id: str
    source_order: int
    imgsz: int
    tile: Tile | None
    width: int
    height: int

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "view_id": self.view_id,
            "source_order": self.source_order,
            "imgsz": self.imgsz,
            "tile_bounds": None if self.tile is None else list(self.tile.bounds),
            "width": self.width,
            "height": self.height,
        }


def build_arm_views(
    arm: str, width: int, height: int, protocol: FrozenSBRProtocol | None = None
) -> tuple[ViewSpec, ...]:
    """Construct deterministic full/tiled view specifications for Arm A-F."""

    p = protocol or FrozenSBRProtocol()
    if not isinstance(arm, str) or arm.upper() not in {"A", "B", "C", "D", "E", "F"}:
        raise ValueError("arm must be one of A, B, C, D, E, F")
    if isinstance(width, bool) or isinstance(height, bool) or int(width) != width or int(height) != height:
        raise TypeError("width and height must be integers")
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    name = arm.upper()
    full = ViewSpec(name, "full", 0, p.high_imgsz if name == "E" else p.imgsz, None, width, height)
    if name in {"A", "E"}:
        return (full,)
    tiles = non_overlapping_tiles(width, height) if name == "F" else overlapping_tiles(width, height)
    labels = ("TL", "TR", "BL", "BR")
    local = tuple(
        ViewSpec(name, label, i + 1, p.imgsz, tile, width, height)
        for i, (label, tile) in enumerate(zip(labels, tiles))
    )
    return local if name == "B" or name == "F" else (full,) + local


@dataclass(frozen=True)
class RawViewRecord:
    image_id: str
    width: int
    height: int
    arm: str
    view_id: str
    source_order: int
    query_index: int
    tile_bounds: tuple[int, int, int, int] | None
    transform: LetterboxTransform
    network_xyxy: tuple[float, float, float, float]
    view_xyxy: tuple[float, float, float, float]
    global_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int

    @property
    def image_width(self) -> int:
        return self.width

    @property
    def image_height(self) -> int:
        return self.height

    @property
    def source(self) -> int:
        return self.source_order

    @property
    def query(self) -> int:
        return self.query_index

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(v)) for v in (*self.network_xyxy, *self.view_xyxy, *self.global_xyxy, self.score)):
            raise ValueError("raw-view values must be finite")
        for box in (self.network_xyxy, self.view_xyxy, self.global_xyxy):
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("raw-view boxes must be legal xyxy rectangles")
        if self.score < 0 or self.score > 1 or self.class_id < 0 or self.query_index < 0:
            raise ValueError("invalid score/class/query metadata")

    @classmethod
    def from_prediction(
        cls,
        view: ViewSpec,
        network_xyxy: Sequence[float],
        score: float,
        class_id: int,
        query_index: int,
        width: int,
        height: int,
        *,
        image_id: str = "image",
    ) -> "RawViewRecord":
        transform = LetterboxTransform.from_view(
            width=view.tile.width if view.tile is not None else width,
            height=view.tile.height if view.tile is not None else height,
            imgsz=view.imgsz,
        )
        transform = LetterboxTransform(
            source_shape=transform.source_shape,
            network_shape=(view.imgsz, view.imgsz),
            gain_x=transform.gain_x,
            gain_y=transform.gain_y,
            pad_x=transform.pad_x,
            pad_y=transform.pad_y,
            resized_width=transform.resized_width,
            resized_height=transform.resized_height,
            auto=False,
            scale_fill=False,
            scaleup=False,
            center=True,
            padding_value=114,
        )
        net = tuple(float(x) for x in network_xyxy)
        view_box = tuple(float(x) for x in np.asarray(inverse_letterbox_xyxy(net, transform)).tolist())
        # Predictions on letterbox padding can map entirely outside the
        # source view. Clip in the view frame before applying the tile offset;
        # a fully collapsed box is a harmless padding artifact and is
        # discarded by the caller.
        view_w = float(view.tile.width if view.tile is not None else width)
        view_h = float(view.tile.height if view.tile is not None else height)
        clipped_view_box = tuple(
            float(x)
            for x in np.asarray(view_box, dtype=float)
            .reshape(1, 4)
            .clip([0.0, 0.0, 0.0, 0.0], [view_w, view_h, view_w, view_h])[0]
        )
        if clipped_view_box[2] <= clipped_view_box[0] or clipped_view_box[3] <= clipped_view_box[1]:
            raise ValueError("prediction lies outside source frame")
        if view.tile is None:
            global_box = tuple(
                float(x)
                for x in np.asarray(clipped_view_box, dtype=float)
                .reshape(1, 4)
                .clip([0.0, 0.0, 0.0, 0.0], [float(width), float(height), float(width), float(height)])[0]
            )
        else:
            global_box = tuple(
                float(x) for x in np.asarray(tile_to_global_xyxy(clipped_view_box, view.tile, width, height)).tolist()
            )
        return cls(
            image_id=str(image_id), width=int(width), height=int(height), arm=view.arm,
            view_id=view.view_id, source_order=view.source_order, query_index=int(query_index),
            tile_bounds=None if view.tile is None else tuple(view.tile.bounds),
            transform=transform, network_xyxy=net, view_xyxy=view_box,
            global_xyxy=global_box, score=float(score), class_id=int(class_id),
        )

    def to_dict(self) -> dict[str, Any]:
        t = asdict(self.transform)
        return {
            "image_id": self.image_id, "width": self.width, "height": self.height,
            "arm": self.arm, "view_id": self.view_id, "source_order": self.source_order,
            "query_index": self.query_index, "tile_bounds": self.tile_bounds,
            "transform": t, "network_xyxy": self.network_xyxy, "view_xyxy": self.view_xyxy,
            "global_xyxy": self.global_xyxy, "score": self.score, "class_id": self.class_id,
        }


def _letterbox(image: np.ndarray, imgsz: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    try:
        from ultralytics.data.augment import LetterBox
    except Exception as exc:
        raise RuntimeError("Ultralytics LetterBox is required for production SBR inference") from exc
    out = LetterBox(
        new_shape=(imgsz, imgsz), auto=False, scale_fill=False,
        scaleup=False, center=True, padding_value=114,
    )(image=arr)
    return np.asarray(out)


def _prediction_rows(result: Any) -> list[tuple[Sequence[float], float, int, int]]:
    def to_numpy(value: Any) -> np.ndarray:
        # Ultralytics keeps Results tensors on the inference device.  Convert
        # explicitly through CPU so the artifact pipeline is device-agnostic.
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    if hasattr(result, "boxes"):
        b = result.boxes
        xyxy = to_numpy(getattr(b, "xyxy"))
        conf = to_numpy(getattr(b, "conf")).reshape(-1)
        cls = to_numpy(getattr(b, "cls")).reshape(-1)
        return [(xyxy[i], float(conf[i]), int(cls[i]), i) for i in range(len(xyxy))]
    if isinstance(result, np.ndarray):
        result = result.tolist()
    if isinstance(result, Mapping):
        result = [result]
    rows = []
    for i, item in enumerate(result or []):
        if hasattr(item, "boxes"):
            rows.extend(_prediction_rows(item))
            continue
        if isinstance(item, Mapping):
            box = item.get("xyxy", item.get("box", item.get("bbox")))
            score = item.get("score", item.get("conf", 0.0))
            cls = item.get("class_id", item.get("cls", item.get("class", 0)))
            query = item.get("query_index", item.get("query", i))
        else:
            vals = list(item)
            if len(vals) >= 6:
                box, score, cls, query = vals[:4], vals[4], vals[5], i
            elif len(vals) >= 4:
                box, score, cls, query = vals[:4]
            else:
                continue
        rows.append((box, float(score), int(cls), int(query)))
    return rows


def collect_raw_views(
    image: Any,
    arm: str,
    predict_square: Callable[[np.ndarray, int], Any],
    *,
    image_id: str = "image",
    protocol: FrozenSBRProtocol | None = None,
    return_manifest: bool = False,
) -> tuple[RawViewRecord, ...] | tuple[tuple[RawViewRecord, ...], list[dict[str, Any]]]:
    """Run each unique arm view once and return serializable raw records."""

    arr = np.asarray(image)
    if arr.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    height, width = arr.shape[:2]
    records: list[RawViewRecord] = []
    manifest: list[dict[str, Any]] = []
    for view in build_arm_views(arm, width, height, protocol):
        crop = arr if view.tile is None else arr[view.tile.top : view.tile.bottom, view.tile.left : view.tile.right]
        square = _letterbox(crop, view.imgsz)
        rows = _prediction_rows(predict_square(square, view.imgsz))
        manifest.append({"view_id": view.view_id, "source_order": view.source_order, "executed": True})
        validated = []
        for box, score, cls, query in rows:
            try:
                coords = tuple(float(x) for x in box)
            except (TypeError, ValueError):
                raise ValueError("predictor box must contain four numeric values") from None
            if len(coords) != 4 or not all(math.isfinite(x) for x in coords):
                raise ValueError("predictor box must be finite xyxy")
            if coords[2] <= coords[0] or coords[3] <= coords[1]:
                raise ValueError("predictor box must be a non-empty xyxy rectangle")
            if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
                raise ValueError("predictor score must be finite in [0,1]")
            if isinstance(cls, bool) or int(cls) != cls or int(cls) < 0:
                raise ValueError("predictor class must be a non-negative integer")
            if isinstance(query, bool) or int(query) != query or int(query) < 0:
                raise ValueError("predictor query must be a non-negative integer")
            validated.append((coords, float(score), int(cls), int(query)))
        kept = [r for r in validated if r[1] >= FrozenSBRProtocol().conf]
        kept.sort(key=lambda r: (-r[1], r[3]))
        for box, score, cls, query in kept[: FrozenSBRProtocol().max_det]:
            try:
                record = RawViewRecord.from_prediction(
                    view, box, score, cls, query, width, height, image_id=image_id
                )
            except ValueError as exc:
                if str(exc) == "prediction lies outside source frame":
                    continue
                raise
            records.append(record)
    return (tuple(records), manifest) if return_manifest else tuple(records)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def assemble_arm(
    raw_records: Iterable[RawViewRecord],
    arm: str,
    *,
    width: int,
    height: int,
    protocol: FrozenSBRProtocol | None = None,
    view_manifest: Sequence[Mapping[str, Any]] | None = None,
    _paired_c_raw: bool = False,
) -> dict[str, Any]:
    """Fuse cached raw records into one arm result with deterministic hashes."""

    p = protocol or FrozenSBRProtocol()
    records = tuple(raw_records)
    if len(records) > p.max_det * max(1, len(build_arm_views(arm, width, height, p))):
        raise ValueError("raw records exceed protocol maximum")
    expected_views = {v.view_id: v for v in build_arm_views(arm, width, height, p)}
    if view_manifest is not None:
        manifest_ids = {str(item.get("view_id")) for item in view_manifest if item.get("executed")}
        if manifest_ids != set(expected_views):
            raise ValueError("view execution manifest is incomplete")
    per_view: dict[str, int] = {}
    for r in records:
        if not isinstance(r, RawViewRecord):
            raise ValueError("assemble_arm requires RawViewRecord values")
        if arm.upper() != "D" and r.arm != arm.upper():
            raise ValueError("raw record arm mismatch")
        if arm.upper() == "D" and not _paired_c_raw:
            raise ValueError("Arm D must be assembled through assemble_paired_arms")
        if arm.upper() == "D" and r.arm != "C":
            raise ValueError("Arm D accepts only explicit Arm C raw records")
        if r.width != width or r.height != height:
            raise ValueError("raw record image dimensions mismatch")
        if r.view_id not in expected_views or r.source_order != expected_views[r.view_id].source_order:
            # Arm D explicitly consumes Arm C records.
            if arm.upper() == "D" and r.arm == "C":
                c_views = {v.view_id: v for v in build_arm_views("C", width, height, p)}
                if r.view_id not in c_views or r.source_order != c_views[r.view_id].source_order:
                    raise ValueError("raw record view metadata mismatch")
            else:
                raise ValueError("raw record view metadata mismatch")
        expected_tile = expected_views.get(r.view_id)
        if expected_tile is None and arm.upper() == "D" and r.arm == "C":
            expected_tile = {v.view_id: v for v in build_arm_views("C", width, height, p)}.get(r.view_id)
        expected_bounds = None if expected_tile is None or expected_tile.tile is None else expected_tile.tile.bounds
        if r.tile_bounds != expected_bounds:
            raise ValueError("raw record tile metadata mismatch")
        per_view[r.view_id] = per_view.get(r.view_id, 0) + 1
    if any(count > p.max_det for count in per_view.values()):
        raise ValueError("per-view raw records exceed max_det")
    if view_manifest is None and set(per_view) != set(expected_views):
        raise ValueError("view execution manifest required for zero-detection views")
    for r in records:
        if r.tile_bounds is not None:
            tw = r.tile_bounds[2] - r.tile_bounds[0]
            th = r.tile_bounds[3] - r.tile_bounds[1]
            x1, y1, x2, y2 = r.view_xyxy
            if x1 < 0 or y1 < 0 or x2 > tw or y2 > th:
                raise ValueError("tile-local box exceeds tile bounds")
    records_dict = [r.to_dict() if isinstance(r, RawViewRecord) else dict(r) for r in records]
    raw_bytes = _canonical(records_dict)
    detections = tuple(
        Detection(
            box=r.global_xyxy, score=r.score, class_id=r.class_id,
            source_order=r.source_order, query_index=r.query_index,
            view_xyxy=r.view_xyxy, global_xyxy=r.global_xyxy,
            network_xyxy=r.network_xyxy,
            tile_local_box=r.view_xyxy if r.tile_bounds is not None else None,
            global_box=r.global_xyxy, tile_bounds=r.tile_bounds,
            transform=r.transform, tile_index=(r.source_order - 1 if r.tile_bounds else None),
        )
        for r in records if isinstance(r, RawViewRecord)
    )
    clusters = greedy_ios_clusters(detections, ios_threshold=p.ios_threshold)
    identity_index = {id(detection): index for index, detection in enumerate(detections)}
    cluster_index = [
        [identity_index[id(member)] for member in cluster] for cluster in clusters
    ]
    cluster_bytes = _canonical(cluster_index)
    if arm.upper() == "D":
        fused = fuse_sp_brf_from_clusters(clusters, full_shape=(width, height), max_det=p.max_det)
    elif arm.upper() in {"B", "C", "F"}:
        fused = fuse_standard(detections, max_det=p.max_det, ios_threshold=p.ios_threshold)
    else:
        fused = tuple(sorted(detections, key=lambda d: (-d.score, d.source_order, d.query_index)))[: p.max_det]
    return {
        "arm": arm.upper(), "records": records_dict, "predictions": tuple(fused),
        "raw_bytes": raw_bytes, "raw_hash": hashlib.sha256(raw_bytes).hexdigest(),
        "cluster_hash": hashlib.sha256(cluster_bytes).hexdigest(),
        "cluster_members": cluster_index,
    }


def assemble_paired_arms(
    raw_records: Iterable[RawViewRecord], *, width: int, height: int,
    protocol: FrozenSBRProtocol | None = None,
    view_manifest: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Assemble C and D from one immutable C raw-view collection."""

    records = tuple(raw_records)
    if any(r.arm != "C" for r in records):
        raise ValueError("paired C/D assembly requires Arm C raw records")
    c = assemble_arm(records, "C", width=width, height=height, protocol=protocol, view_manifest=view_manifest)
    d = assemble_arm(records, "D", width=width, height=height, protocol=protocol, view_manifest=view_manifest, _paired_c_raw=True)
    if c["raw_hash"] != d["raw_hash"] or c["cluster_hash"] != d["cluster_hash"]:
        raise ValueError("C/D raw or cluster hashes differ")
    return {"C": c, "D": d}


__all__ = [
    "FrozenSBRProtocol", "ViewSpec", "RawViewRecord", "build_arm_views",
    "collect_raw_views", "assemble_arm",
    "assemble_paired_arms",
]
