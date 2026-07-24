#!/usr/bin/env python3
"""Standalone adjudicator for frozen SBR score-oracle evidence.

This file intentionally imports no project module.  Scientific replay,
matching, metrics, integrity checks, and gates are reimplemented here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import numpy as np


CONF = 0.001
MAX_DET = 300
IOS = 0.5
THRESHOLDS = tuple(
    round(0.50 + 0.05 * index, 2) for index in range(10)
)
METRIC_THRESHOLDS = tuple(
    float(value)
    for value in np.arange(0.50, 0.951, 0.05, dtype=float)
)
SIZE_BINS = {
    "tiny": (0.0, 16.0, True, True),
    "small": (16.0, 32.0, False, True),
    "medium": (32.0, 96.0, False, True),
    "large": (96.0, float("inf"), False, False),
}
GATES = {
    "AP-tiny-SBR": 0.010,
    "mAP50-95": 0.003,
    "tiny_recall": 0.020,
    "AP75": -0.002,
    "AP-large-SBR": -0.005,
}
SCHEMA_VERSION = "sbr-score-oracle-evidence/v1"
SCHEMA = {
    "schema_version": SCHEMA_VERSION,
    "required_artifacts": [
        "oracle_manifest.json",
        "unit_events.jsonl.gz",
        "score_patches.jsonl.gz",
        "coverage.json",
        "oracle_metrics.json",
        "invariants.json",
        "primary_gate.json",
        "runtime.json",
        "checksums.sha256",
    ],
    "unit_id_fields": [
        "image_id",
        "stock_member_indices",
        "full_anchor_index",
        "aggressor_indices",
    ],
    "primary_gate_inputs": [
        "joint_minus_a.AP-tiny-SBR",
        "joint_minus_a.mAP50-95",
        "joint_minus_a.tiny_recall",
        "joint_minus_a.AP75",
        "joint_minus_a.AP-large-SBR",
        "invariants.passed",
    ],
    "authoritative_gate_inputs": [
        "primary_gate.proposed_status",
        "independent_adjudication.primary_gate_agrees",
        "independent_adjudication.joint_metrics_agree",
        "independent_adjudication.unit_labels_agree",
    ],
}
REQUIRED_PRIMARY = tuple(SCHEMA["required_artifacts"])
HEX = frozenset("0123456789abcdefABCDEF")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(
    value: Any,
    name: str,
    *,
    lengths: tuple[int, ...] = (64,),
) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in HEX for character in value)
    ):
        raise ValueError(f"{name} is not a hexadecimal digest")
    return value.lower()


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_json(path: Path) -> Any:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )
    _assert_finite(value, path.name)
    return value


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(
            path, "rt", encoding="utf-8", newline=""
        ) as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(
                        f"{path.name}:{line_number} is blank"
                    )
                value = json.loads(
                    line, parse_constant=_reject_constant
                )
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path.name}:{line_number} is not an object"
                    )
                _assert_finite(
                    value, f"{path.name}[{line_number}]"
                )
                rows.append(value)
    except (EOFError, OSError, UnicodeError) as exc:
        raise ValueError(
            f"invalid gzip JSONL: {path.name}"
        ) from exc
    return rows


def _assert_finite(value: Any, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} is non-finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{name}[{index}]")


def _safe_relative(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("unsafe empty checksum path")
    value = Path(relative)
    if (
        value.is_absolute()
        or value.drive
        or ".." in value.parts
        or value.name == "checksums.sha256"
    ):
        raise ValueError(f"unsafe checksum path: {relative}")
    target = (root / value).resolve()
    if root != target.parent and root not in target.parents:
        raise ValueError(f"unsafe checksum path: {relative}")
    return target


def _portable_path(uri: Any, *, base: Path, name: str) -> Path:
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError(f"{name} URI is missing")
    direct = Path(uri)
    if direct.is_absolute():
        return direct.resolve()
    parsed = urlparse(uri)
    if parsed.scheme:
        if parsed.scheme.lower() != "file":
            raise ValueError(
                f"{name} uses a non-local URI scheme"
            )
        value = url2pathname(
            unquote((f"//{parsed.netloc}" if parsed.netloc else "")
                    + parsed.path)
        )
        if value.startswith("/") and len(value) > 2 and value[2] == ":":
            value = value[1:]
        target = Path(value)
        if not target.is_absolute():
            raise ValueError(f"{name} file URI is not absolute")
        return target.resolve()
    return (base / direct).resolve()


def _entry(
    value: Any, *, base: Path, name: str
) -> tuple[Path, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"uri", "sha256"}
    ):
        raise ValueError(f"{name} entry is invalid")
    path = _portable_path(
        value.get("uri"), base=base, name=name
    )
    digest = _digest(value.get("sha256"), f"{name} SHA-256")
    if not path.is_file() or _sha256_file(path) != digest:
        raise ValueError(f"{name} checksum mismatch")
    return path, digest


def _verify_checksums(primary: Path) -> None:
    if (
        not primary.is_dir()
        or primary.is_symlink()
        or {path.name for path in primary.iterdir()}
        != set(REQUIRED_PRIMARY)
        or any(
            path.is_symlink() or not path.is_file()
            for path in primary.iterdir()
        )
    ):
        raise ValueError("primary artifact set/type mismatch")
    path = primary / "checksums.sha256"
    if not path.is_file():
        raise ValueError("primary checksums.sha256 is missing")
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            continue
        try:
            digest_text, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(
                f"invalid primary checksum line {line_number}"
            ) from exc
        digest = _digest(
            digest_text, f"checksum line {line_number}"
        )
        target = _safe_relative(primary, relative)
        normalized = Path(relative).as_posix()
        if normalized in seen:
            raise ValueError(
                f"duplicate checksum path: {normalized}"
            )
        if not target.is_file() or _sha256_file(target) != digest:
            raise ValueError(
                f"primary checksum mismatch: {normalized}"
            )
        seen.add(normalized)
    expected = set(REQUIRED_PRIMARY) - {"checksums.sha256"}
    if seen != expected:
        raise ValueError("primary checksum artifact set mismatch")


def _primary_snapshot(primary: Path) -> list[dict[str, Any]]:
    if (
        not primary.is_dir()
        or primary.is_symlink()
        or {path.name for path in primary.iterdir()}
        != set(REQUIRED_PRIMARY)
        or any(
            path.is_symlink() or not path.is_file()
            for path in primary.iterdir()
        )
    ):
        raise ValueError("primary artifact set/type mismatch")
    rows: list[dict[str, Any]] = []
    paths = [primary, *sorted(primary.iterdir())]
    for path in paths:
        stat = path.stat()
        mode = int(stat.st_mode & 0o7777)
        if mode & 0o222:
            raise ValueError(
                f"writable primary node: {path.name}"
            )
        relative = (
            "."
            if path == primary
            else path.relative_to(primary).as_posix()
        )
        if path.is_dir():
            kind = "directory"
            digest = None
        elif path.is_file():
            kind = "file"
            digest = _sha256_file(path)
        else:
            raise ValueError(
                f"unsupported primary node type: {relative}"
            )
        rows.append(
            {
                "path": relative,
                "type": kind,
                "mode": mode,
                "size": int(stat.st_size),
                "sha256": digest,
            }
        )
    return rows


@dataclass(frozen=True)
class Raw:
    image_id: str
    arm: str
    width: int
    height: int
    source_order: int
    query_index: int
    class_id: int
    score: float
    network_xyxy: tuple[float, float, float, float]
    view_xyxy: tuple[float, float, float, float]
    global_xyxy: tuple[float, float, float, float]
    tile_bounds: tuple[int, int, int, int] | None
    original_index: int


@dataclass(frozen=True)
class Prediction:
    box: tuple[float, float, float, float]
    global_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    source_order: int
    query_index: int
    members: tuple[int, ...]


@dataclass(frozen=True)
class Image:
    image_id: str
    width: int
    height: int
    gt_boxes: tuple[tuple[float, float, float, float], ...]
    gt_classes: tuple[int, ...]
    ignore_boxes: tuple[
        tuple[float, float, float, float], ...
    ]


def _strict_int(value: Any, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _box(
    value: Any, name: str
) -> tuple[float, float, float, float]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be an xyxy sequence")
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an xyxy sequence") from None
    if (
        len(box) != 4
        or not all(math.isfinite(item) for item in box)
        or box[2] <= box[0]
        or box[3] <= box[1]
    ):
        raise ValueError(f"{name} is not a legal finite box")
    return box  # type: ignore[return-value]


def _parse_raw(row: Mapping[str, Any], index: int) -> Raw:
    image_id = row.get("image_id")
    arm = row.get("arm")
    if not isinstance(image_id, str) or arm not in {"A", "C"}:
        raise ValueError("raw A/C identity is invalid")
    width = _strict_int(row.get("width"), "raw width")
    height = _strict_int(row.get("height"), "raw height")
    if width == 0 or height == 0:
        raise ValueError("raw dimensions must be positive")
    source = _strict_int(
        row.get("source_order"), "raw source_order"
    )
    if source > 4 or (arm == "A" and source != 0):
        raise ValueError("raw source is outside frozen A/C views")
    if row.get("view_id") != ("full", "TL", "TR", "BL", "BR")[
        source
    ]:
        raise ValueError("raw view_id/source mismatch")
    manifest_value = row.get("view_manifest")
    if isinstance(manifest_value, (str, bytes, Mapping)):
        raise ValueError("raw view_manifest is invalid")
    try:
        view_manifest = tuple(manifest_value)
    except TypeError:
        raise ValueError("raw view_manifest is missing") from None
    expected_sources = (0,) if arm == "A" else (0, 1, 2, 3, 4)
    if len(view_manifest) != len(expected_sources):
        raise ValueError("raw view_manifest is incomplete")
    seen_sources: set[int] = set()
    for item in view_manifest:
        if not isinstance(item, Mapping):
            raise ValueError("raw view_manifest item is invalid")
        item_source = _strict_int(
            item.get("source_order"), "view_manifest source"
        )
        if (
            item_source not in expected_sources
            or item_source in seen_sources
            or item.get("view_id")
            != ("full", "TL", "TR", "BL", "BR")[item_source]
            or item.get("executed") is not True
        ):
            raise ValueError("raw view_manifest provenance mismatch")
        seen_sources.add(item_source)
    if seen_sources != set(expected_sources):
        raise ValueError("raw view_manifest source set mismatch")
    score = row.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
    ):
        raise ValueError("raw score is invalid")
    tile_value = row.get("tile_bounds")
    tile = None
    if source == 0:
        if tile_value is not None:
            raise ValueError("full raw record has tile bounds")
    else:
        if isinstance(tile_value, (str, bytes, Mapping)):
            raise ValueError("local tile bounds are invalid")
        try:
            parsed_tile = tuple(
                _strict_int(item, "tile bound")
                for item in tile_value
            )
        except TypeError:
            raise ValueError("local tile bounds are invalid") from None
        if (
            len(parsed_tile) != 4
            or parsed_tile[2] <= parsed_tile[0]
            or parsed_tile[3] <= parsed_tile[1]
            or parsed_tile[2] > width
            or parsed_tile[3] > height
        ):
            raise ValueError("local tile bounds are illegal")
        tile = parsed_tile  # type: ignore[assignment]
    network_box = _box(
        row.get("network_xyxy"), "network_xyxy"
    )
    view_box = _box(row.get("view_xyxy"), "view_xyxy")
    global_box = _box(
        row.get("global_xyxy"), "global_xyxy"
    )
    if source == 0:
        expected_global = (
            max(0.0, view_box[0]),
            max(0.0, view_box[1]),
            min(float(width), view_box[2]),
            min(float(height), view_box[3]),
        )
    else:
        assert tile is not None
        tile_width = tile[2] - tile[0]
        tile_height = tile[3] - tile[1]
        tile_width_frozen = int(math.ceil(0.60 * width))
        tile_height_frozen = int(math.ceil(0.60 * height))
        x_origins = (0, width - tile_width_frozen)
        y_origins = (0, height - tile_height_frozen)
        tile_position = source - 1
        expected_tile = (
            x_origins[tile_position % 2],
            y_origins[tile_position // 2],
            x_origins[tile_position % 2] + tile_width_frozen,
            y_origins[tile_position // 2] + tile_height_frozen,
        )
        if tile != expected_tile:
            raise ValueError("raw tile bounds mismatch")
        if (
            view_box[0] < 0.0
            or view_box[1] < 0.0
            or view_box[2] > tile_width
            or view_box[3] > tile_height
        ):
            raise ValueError("raw local view coordinates are outside tile")
        expected_global = (
            view_box[0] + tile[0],
            view_box[1] + tile[1],
            view_box[2] + tile[0],
            view_box[3] + tile[1],
        )
    if any(
        not math.isclose(
            left, right, rel_tol=0.0, abs_tol=1e-9
        )
        for left, right in zip(expected_global, global_box)
    ):
        raise ValueError("raw coordinate frames disagree")
    return Raw(
        image_id=image_id,
        arm=arm,
        width=width,
        height=height,
        source_order=source,
        query_index=_strict_int(
            row.get("query_index"), "raw query_index"
        ),
        class_id=_strict_int(
            row.get("class_id"), "raw class_id"
        ),
        score=float(score),
        network_xyxy=network_box,
        view_xyxy=view_box,
        global_xyxy=global_box,
        tile_bounds=tile,
        original_index=index,
    )


def _rank(record: Raw) -> tuple[float, int, int, int]:
    return (
        -record.score,
        record.source_order,
        record.query_index,
        record.original_index,
    )


def _ios(
    left: Sequence[float], right: Sequence[float]
) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    intersection *= max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = (lx2 - lx1) * (ly2 - ly1)
    right_area = (rx2 - rx1) * (ry2 - ry1)
    return intersection / min(left_area, right_area)


def _clusters(
    records: Sequence[Raw],
) -> tuple[tuple[Raw, ...], ...]:
    remaining = list(sorted(records, key=_rank))
    clusters: list[tuple[Raw, ...]] = []
    while remaining:
        seed = remaining.pop(0)
        members = [seed]
        keep: list[Raw] = []
        for candidate in remaining:
            if (
                candidate.class_id == seed.class_id
                and _ios(
                    seed.global_xyxy, candidate.global_xyxy
                )
                > IOS
            ):
                members.append(candidate)
            else:
                keep.append(candidate)
        remaining = keep
        clusters.append(tuple(members))
    return tuple(clusters)


def _reconstruct(
    records: Sequence[Raw],
) -> tuple[
    tuple[Prediction, ...], tuple[tuple[int, ...], ...]
]:
    if any(record.arm != "C" for record in records):
        raise ValueError("fusion accepts only Arm C")
    identities = [record.original_index for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("raw identity collision")
    fused: list[tuple[int, Prediction]] = []
    for cluster_index, cluster in enumerate(_clusters(records)):
        seed = cluster[0]
        total = sum(member.score for member in cluster)
        weighted = (
            seed.global_xyxy
            if len(cluster) == 1
            or not math.isfinite(total)
            or total <= 0.0
            else tuple(
                sum(
                    member.score * member.global_xyxy[axis]
                    for member in cluster
                )
                / total
                for axis in range(4)
            )
        )
        fused.append(
            (
                cluster_index,
                Prediction(
                    box=weighted,
                    global_xyxy=seed.global_xyxy,
                    score=max(
                        member.score for member in cluster
                    ),
                    class_id=seed.class_id,
                    source_order=seed.source_order,
                    query_index=seed.query_index,
                    members=tuple(
                        member.original_index
                        for member in cluster
                    ),
                ),
            )
        )
    ordered = sorted(
        fused,
        key=lambda item: (
            -item[1].score,
            item[1].source_order,
            item[1].query_index,
            item[0],
        ),
    )
    return (
        tuple(item[1] for item in ordered[:MAX_DET]),
        tuple(item[1].members for item in ordered),
    )


def _arr_boxes(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.empty((0, 4), dtype=float)
    if array.ndim == 1 and array.shape == (4,):
        array = array.reshape(1, 4)
    if (
        array.ndim != 2
        or array.shape[1] != 4
        or not np.isfinite(array).all()
        or (array[:, 2:] < array[:, :2]).any()
        or (array < 0).any()
    ):
        raise ValueError(f"{name} contains illegal boxes")
    return array


def _iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.zeros((len(left), len(right)), dtype=float)
    lt = np.maximum(left[:, None, :2], right[None, :, :2])
    rb = np.minimum(left[:, None, 2:], right[None, :, 2:])
    inter = np.prod(np.maximum(rb - lt, 0.0), axis=2)
    area_left = np.prod(
        np.maximum(left[:, 2:] - left[:, :2], 0.0), axis=1
    )
    area_right = np.prod(
        np.maximum(right[:, 2:] - right[:, :2], 0.0),
        axis=1,
    )
    union = (
        area_left[:, None] + area_right[None, :] - inter
    )
    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter),
        where=union > 0,
    )


def _neutral_ignore(
    predictions: np.ndarray, ignores: np.ndarray
) -> np.ndarray:
    if len(predictions) == 0 or len(ignores) == 0:
        return np.zeros(len(predictions), dtype=bool)
    lt = np.maximum(
        predictions[:, None, :2], ignores[None, :, :2]
    )
    rb = np.minimum(
        predictions[:, None, 2:], ignores[None, :, 2:]
    )
    inter = np.prod(np.maximum(rb - lt, 0.0), axis=2)
    area = np.prod(
        predictions[:, 2:] - predictions[:, :2], axis=1
    )
    ioa = np.divide(
        inter,
        area[:, None],
        out=np.zeros_like(inter),
        where=area[:, None] > 0,
    )
    return np.any(ioa >= 0.50, axis=1)


def _in_bin(radius: np.ndarray, name: str) -> np.ndarray:
    lo, hi, lo_inclusive, hi_inclusive = SIZE_BINS[name]
    left = radius >= lo if lo_inclusive else radius > lo
    right = radius <= hi if hi_inclusive else radius > lo
    return left & right


def _evaluate_threshold(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    neutral: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    selected: np.ndarray,
    ious: np.ndarray,
    threshold: float,
) -> tuple[dict[str, int], list[tuple[float, int, int, int]]]:
    matched = np.zeros(len(gt_boxes), dtype=bool)
    records: list[tuple[float, int, int, int]] = []
    tp = fp = neutralized = 0
    for index in range(len(boxes)):
        if neutral[index]:
            neutralized += 1
            continue
        same = (
            (gt_classes == classes[index])
            & selected
            & ~matched
            & (ious[index] >= threshold)
        )
        if np.any(same):
            candidates = np.flatnonzero(same)
            chosen = int(
                candidates[
                    np.argmax(ious[index, candidates])
                ]
            )
            matched[chosen] = True
            tp += 1
            records.append(
                (
                    float(scores[index]),
                    int(classes[index]),
                    1,
                    0,
                )
            )
            continue
        outside = (
            (gt_classes == classes[index])
            & ~selected
            & (ious[index] >= threshold)
        )
        if np.any(outside):
            neutralized += 1
            continue
        fp += 1
        records.append(
            (
                float(scores[index]),
                int(classes[index]),
                0,
                1,
            )
        )
    return (
        {
            "tp": tp,
            "fp": fp,
            "fn": int(np.count_nonzero(selected & ~matched)),
            "neutralized": neutralized,
            "predictions": len(boxes),
        },
        records,
    )


def _compute_ap(
    recall: Sequence[float], precision: Sequence[float]
) -> float:
    recall_array = np.asarray(recall, dtype=float).reshape(-1)
    precision_array = np.asarray(
        precision, dtype=float
    ).reshape(-1)
    if len(recall_array) == 0:
        return 0.0
    mrec = np.concatenate(
        ([0.0], np.clip(recall_array, 0, 1), [1.0])
    )
    mpre = np.concatenate(
        ([1.0], np.clip(precision_array, 0, 1), [0.0])
    )
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(
        np.trapezoid(np.interp(x, mrec, mpre), x)
    )


def _ap_from_records(
    records: list[tuple[float, int, int, int]],
    gt_count: Mapping[int, int],
) -> float:
    by_class: dict[int, list[tuple[float, int, int]]] = {}
    for score, class_id, tp, fp in records:
        by_class.setdefault(class_id, []).append(
            (score, tp, fp)
        )
    aps: list[float] = []
    for class_id, count in gt_count.items():
        if count <= 0:
            continue
        rows = sorted(
            by_class.get(class_id, []),
            key=lambda row: -row[0],
        )
        if not rows:
            aps.append(0.0)
            continue
        tp = np.cumsum([row[1] for row in rows], dtype=float)
        fp = np.cumsum([row[2] for row in rows], dtype=float)
        aps.append(
            _compute_ap(
                tp / max(float(count), 1.0),
                tp / np.maximum(tp + fp, 1e-12),
            )
        )
    return float(np.mean(aps)) if aps else 0.0


def _prepared(
    image: Image, predictions: Sequence[Prediction]
) -> tuple[np.ndarray, ...]:
    ordered = sorted(
        enumerate(predictions),
        key=lambda item: (
            -item[1].score,
            item[1].source_order,
            item[1].query_index,
            item[0],
        ),
    )
    kept = [
        prediction
        for _, prediction in ordered
        if prediction.score >= CONF
    ][:MAX_DET]
    boxes = _arr_boxes(
        [prediction.global_xyxy for prediction in kept],
        "predictions",
    )
    scores = np.asarray(
        [prediction.score for prediction in kept], dtype=float
    )
    classes = np.asarray(
        [prediction.class_id for prediction in kept], dtype=int
    )
    gt_boxes = _arr_boxes(image.gt_boxes, "gt_boxes")
    gt_classes = np.asarray(image.gt_classes, dtype=int)
    ignores = _arr_boxes(image.ignore_boxes, "ignore_boxes")
    neutral = _neutral_ignore(boxes, ignores)
    ious = _iou(boxes, gt_boxes)
    gain = min(
        640.0 / float(image.width),
        640.0 / float(image.height),
        1.0,
    )
    wh = np.maximum(
        gt_boxes[:, 2:] - gt_boxes[:, :2], 0.0
    )
    radius = np.sqrt(wh[:, 0] * wh[:, 1]) * gain
    return (
        boxes,
        scores,
        classes,
        gt_boxes,
        gt_classes,
        neutral,
        ious,
        radius,
    )


def _profile(
    image: Image, predictions: Sequence[Prediction]
) -> dict[str, dict[str, dict[str, int]]]:
    (
        boxes,
        scores,
        classes,
        gt_boxes,
        gt_classes,
        neutral,
        ious,
        radius,
    ) = _prepared(image, predictions)
    profile: dict[str, dict[str, dict[str, int]]] = {}
    for threshold in THRESHOLDS:
        masks = {"all": np.ones(len(gt_classes), dtype=bool)}
        masks.update(
            {
                name: _in_bin(radius, name)
                for name in SIZE_BINS
            }
        )
        key = f"{threshold:.2f}"
        profile[key] = {}
        for name, selected in masks.items():
            counts, _ = _evaluate_threshold(
                boxes,
                scores,
                classes,
                neutral,
                gt_boxes,
                gt_classes,
                selected,
                ious,
                threshold,
            )
            counts["gt"] = int(np.count_nonzero(selected))
            profile[key][name] = counts
    return profile


def _evaluate_dataset(
    rows: Sequence[tuple[Image, tuple[Prediction, ...]]],
) -> dict[str, Any]:
    pooled = {
        name: {threshold: [] for threshold in THRESHOLDS}
        for name in SIZE_BINS
    }
    pooled_global = {
        threshold: [] for threshold in THRESHOLDS
    }
    gt_counts: dict[str, dict[int, int]] = {
        name: {} for name in SIZE_BINS
    }
    gt_global: dict[int, int] = {}
    sum_counts = {
        key: 0
        for key in (
            "tp",
            "fp",
            "fn",
            "neutralized",
            "predictions",
        )
    }
    bin_counts = {
        name: {
            threshold: {
                key: 0
                for key in (
                    "tp",
                    "fp",
                    "fn",
                    "neutralized",
                    "predictions",
                    "gt",
                )
            }
            for threshold in THRESHOLDS
        }
        for name in SIZE_BINS
    }
    for image, predictions in rows:
        (
            boxes,
            scores,
            classes,
            gt_boxes,
            gt_classes,
            neutral,
            ious,
            radius,
        ) = _prepared(image, predictions)
        classes_present = sorted(set(gt_classes.tolist()))
        for class_id in classes_present:
            gt_global[class_id] = (
                gt_global.get(class_id, 0)
                + int(np.count_nonzero(gt_classes == class_id))
            )
        for name in SIZE_BINS:
            selected = _in_bin(radius, name)
            for class_id in classes_present:
                gt_counts[name][class_id] = (
                    gt_counts[name].get(class_id, 0)
                    + int(
                        np.count_nonzero(
                            selected & (gt_classes == class_id)
                        )
                    )
                )
        for metric_threshold in METRIC_THRESHOLDS:
            threshold = float(round(metric_threshold, 2))
            all_selected = np.ones(
                len(gt_classes), dtype=bool
            )
            counts, records = _evaluate_threshold(
                boxes,
                scores,
                classes,
                neutral,
                gt_boxes,
                gt_classes,
                all_selected,
                ious,
                metric_threshold,
            )
            pooled_global[threshold].extend(records)
            if threshold == 0.50:
                for key in sum_counts:
                    sum_counts[key] += counts[key]
            for name in SIZE_BINS:
                selected = _in_bin(radius, name)
                counts, records = _evaluate_threshold(
                    boxes,
                    scores,
                    classes,
                    neutral,
                    gt_boxes,
                    gt_classes,
                    selected,
                    ious,
                    metric_threshold,
                )
                pooled[name][threshold].extend(records)
                for key in bin_counts[name][threshold]:
                    bin_counts[name][threshold][key] += counts.get(
                        key, 0
                    )
                bin_counts[name][threshold]["gt"] += int(
                    np.count_nonzero(selected)
                )
    output: dict[str, Any] = {
        "counts": sum_counts,
        "per_threshold": {},
    }
    overall_aps = [
        _ap_from_records(
            pooled_global[threshold], gt_global
        )
        for threshold in THRESHOLDS
    ]
    output["AP50"] = overall_aps[0]
    output["AP75"] = overall_aps[5]
    output["mAP50-95"] = float(np.mean(overall_aps))
    output["AP50-95"] = output["mAP50-95"]
    for name in SIZE_BINS:
        aps = [
            _ap_from_records(
                pooled[name][threshold], gt_counts[name]
            )
            for threshold in THRESHOLDS
        ]
        output[f"AP-{name}-SBR"] = float(np.mean(aps))
        output[f"AP-{name}"] = float(np.mean(aps))
        output[f"AP50-{name}-SBR"] = float(aps[0])
        output[f"AP75-{name}-SBR"] = float(aps[5])
        output["per_threshold"][name] = bin_counts[name]
    tiny = bin_counts["tiny"][0.50]
    output["tiny_recall"] = (
        float(tiny["tp"] / tiny["gt"])
        if tiny["gt"]
        else 0.0
    )
    return _jsonable(output)


def _yaml_scalar_mapping(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if raw_line[:1].isspace():
            continue
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(
                f"unsupported dataset YAML line {line_number}"
            )
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if (
            not key
            or not value
            or value in {"|", ">"}
            or value.startswith(("[", "{"))
        ):
            continue
        if value[0] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError("unterminated dataset YAML scalar")
            value = value[1:-1]
        if value.startswith("[") or value.startswith("-"):
            raise ValueError(
                "multi-path dataset YAML is not supported"
            )
        values[key] = value
    return values


def _parse_label(
    path: Path,
    width: int,
    height: int,
    *,
    ignore: bool,
) -> tuple[
    tuple[tuple[float, float, float, float], ...],
    tuple[int, ...],
]:
    boxes: list[tuple[float, float, float, float]] = []
    classes: list[int] = []
    if not path.exists():
        return (), ()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = line.split()
        if ignore:
            if len(fields) < 4:
                raise ValueError(
                    f"invalid ignore label {path}:{line_number}"
                )
            values = fields[:4]
            class_id = 0
        else:
            if len(fields) < 5:
                raise ValueError(
                    f"invalid label {path}:{line_number}"
                )
            try:
                class_id = int(fields[0])
            except ValueError:
                raise ValueError(
                    f"invalid label class {path}:{line_number}"
                ) from None
            if class_id < 0:
                raise ValueError("negative label class")
            values = fields[1:5]
        try:
            cx, cy, box_width, box_height = (
                float(value) for value in values
            )
        except ValueError:
            raise ValueError(
                f"non-numeric label {path}:{line_number}"
            ) from None
        if (
            not all(
                math.isfinite(value)
                for value in (cx, cy, box_width, box_height)
            )
            or box_width <= 0
            or box_height <= 0
        ):
            raise ValueError("illegal label geometry")
        x1 = (cx - box_width / 2) * width
        y1 = (cy - box_height / 2) * height
        x2 = (cx + box_width / 2) * width
        y2 = (cy + box_height / 2) * height
        box = (
            max(0.0, x1),
            max(0.0, y1),
            min(float(width), x2),
            min(float(height), y2),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("empty label box")
        boxes.append(box)
        classes.append(class_id)
    return tuple(boxes), tuple(classes)


def _dataset_signature(root: Path, split: str) -> str:
    paths: list[Path] = []
    for folder in (
        root / "images" / split,
        root / "labels" / split,
        root / "labels_ignore" / split,
    ):
        if folder.exists():
            paths.extend(
                path
                for path in folder.rglob("*")
                if path.is_file()
            )
    lines = [
        f"{_sha256_file(path)}  "
        f"{path.relative_to(root).as_posix()}"
        for path in sorted(
            paths,
            key=lambda item: item.relative_to(root).as_posix(),
        )
    ]
    payload = (
        "\n".join(lines) + ("\n" if lines else "")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _uri_values(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "uri" and isinstance(item, str):
                yield item
            yield from _uri_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _uri_values(item)


def _reject_forbidden_uris(*values: Any) -> None:
    for value in values:
        for uri in _uri_values(value):
            lowered = uri.lower()
            if (
                "test-dev" in lowered
                or "external-dataset" in lowered
            ):
                raise ValueError(
                    "test-dev or external-dataset URI is forbidden"
                )


def _frozen_rule() -> dict[str, Any]:
    return {
        "conf": CONF,
        "max_det": MAX_DET,
        "ios": IOS,
        "thresholds": list(THRESHOLDS),
        "gates": GATES,
        "group_rule": (
            "mixed-cluster-all-local-strictly-above-best-full"
        ),
        "demotion": (
            "float64-nextafter-anchor-toward-negative-infinity"
        ),
        "selection": (
            "all-threshold-all-tiny-large-nondecrease-"
            "and-large-sum-positive"
        ),
    }


def _verify_original_checksums(
    evidence_root: Path, checksum_path: Path
) -> None:
    seen: set[str] = set()
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line:
            continue
        try:
            digest_text, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(
                f"invalid upstream checksum line {line_number}"
            ) from exc
        digest = _digest(
            digest_text, f"upstream checksum {line_number}"
        )
        target = _safe_relative(evidence_root, relative)
        normalized = Path(relative).as_posix()
        if normalized in seen:
            raise ValueError("duplicate upstream checksum path")
        if not target.is_file() or _sha256_file(target) != digest:
            raise ValueError("upstream checksum mismatch")
        seen.add(normalized)


def _verify_manifest_chain(
    root: Path,
    primary: Path,
    manifest: Any,
    source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ValueError("oracle manifest is not an object")
    if set(manifest) != {
        "schema_version",
        "schema",
        "schema_hash",
        "wrapper",
        "upstream_input",
        "approved_spec",
        "source",
        "frozen_rule",
        "frozen_rule_hash",
        "primary_script_sha256",
        "dataset_signature",
        "image_count",
        "image_order_hash",
        "workers",
    }:
        raise ValueError("oracle manifest field set mismatch")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("schema") != SCHEMA
        or manifest.get("schema_hash") != _sha256_json(SCHEMA)
    ):
        raise ValueError("oracle schema contract mismatch")
    recorded_source = manifest.get("source")
    if not isinstance(recorded_source, Mapping):
        raise ValueError("primary source provenance is missing")
    primary_source = {
        "commit": _digest(
            recorded_source.get("commit"),
            "primary source commit",
            lengths=(40, 64),
        ),
        "tree": _digest(
            recorded_source.get("tree"),
            "primary source tree",
            lengths=(40, 64),
        ),
    }
    if source is not None and any(
        primary_source[key] != source[key]
        for key in ("commit", "tree")
    ):
        raise ValueError("primary/source provenance mismatch")
    if (
        manifest.get("frozen_rule") != _frozen_rule()
        or manifest.get("frozen_rule_hash")
        != _sha256_json(_frozen_rule())
    ):
        raise ValueError("frozen oracle rule mismatch")
    repo = Path(__file__).resolve().parents[1]
    primary_script = repo / "scripts" / "run_sbr_score_oracle.py"
    if (
        not primary_script.is_file()
        or manifest.get("primary_script_sha256")
        != _sha256_file(primary_script)
    ):
        raise ValueError("primary script hash mismatch")
    wrapper_path, wrapper_hash = _entry(
        manifest.get("wrapper"),
        base=root,
        name="protocol wrapper",
    )
    upstream_path, upstream_hash = _entry(
        manifest.get("upstream_input"),
        base=root,
        name="upstream input",
    )
    spec_path, spec_hash = _entry(
        manifest.get("approved_spec"),
        base=root,
        name="approved spec",
    )
    wrapper = _read_json(wrapper_path)
    if (
        not isinstance(wrapper, Mapping)
        or wrapper.get("schema_version")
        != "sbr-score-oracle-input/v1"
        or wrapper.get("frozen_rule") != _frozen_rule()
        or wrapper.get("forbidden_inputs")
        != ["test-dev", "external-dataset"]
        or wrapper.get("expected_source") != primary_source
    ):
        raise ValueError("protocol wrapper contract mismatch")
    wrapped_upstream, wrapped_upstream_hash = _entry(
        wrapper.get("upstream_input"),
        base=wrapper_path.parent,
        name="wrapped upstream input",
    )
    wrapped_spec, wrapped_spec_hash = _entry(
        wrapper.get("approved_spec"),
        base=wrapper_path.parent,
        name="wrapped approved spec",
    )
    if (
        wrapped_upstream != upstream_path
        or wrapped_upstream_hash != upstream_hash
        or wrapped_spec != spec_path
        or wrapped_spec_hash != spec_hash
        or wrapper_hash != _sha256_file(wrapper_path)
    ):
        raise ValueError("wrapper hash/URI chain mismatch")
    upstream = _read_json(upstream_path)
    if (
        not isinstance(upstream, Mapping)
        or upstream.get("schema_version")
        != "sbr-v2-audit-input/v1"
        or upstream.get("dataset", {}).get("split") != "val"
    ):
        raise ValueError("upstream must be frozen val input")
    _reject_forbidden_uris(manifest, wrapper, upstream)
    files = upstream.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("upstream file mapping is missing")
    resolved: dict[str, Path] = {}
    for key, entry_value in files.items():
        path, _ = _entry(
            entry_value,
            base=upstream_path.parent,
            name=f"upstream file {key}",
        )
        resolved[str(key)] = path
    required = {
        "g0_manifest",
        "raw_views",
        "arm_predictions",
        "g0_metrics",
        "g0_gate",
        "independent_adjudication",
        "original_checksums",
        "checkpoint",
        "image_list",
        "dataset_yaml",
    }
    if set(resolved) != required:
        raise ValueError("upstream file set mismatch")
    evidence_entry = upstream.get("original_evidence_root")
    if isinstance(evidence_entry, str):
        evidence_uri = evidence_entry
    elif (
        isinstance(evidence_entry, Mapping)
        and set(evidence_entry) == {"uri"}
    ):
        evidence_uri = evidence_entry.get("uri")
    else:
        raise ValueError("original evidence root is invalid")
    evidence_root = _portable_path(
        evidence_uri,
        base=upstream_path.parent,
        name="original evidence root",
    )
    if not evidence_root.is_dir():
        raise ValueError("original evidence root is missing")
    _verify_original_checksums(
        evidence_root, resolved["original_checksums"]
    )
    dataset_entry = upstream.get("dataset", {}).get("root")
    if (
        not isinstance(dataset_entry, Mapping)
        or set(dataset_entry) != {"uri", "sha256"}
    ):
        raise ValueError("dataset root entry is invalid")
    dataset_root = _portable_path(
        dataset_entry.get("uri"),
        base=upstream_path.parent,
        name="dataset root",
    )
    dataset_signature = _digest(
        dataset_entry.get("sha256"), "dataset signature"
    )
    if (
        not dataset_root.is_dir()
        or _dataset_signature(dataset_root, "val")
        != dataset_signature
        or manifest.get("dataset_signature") != dataset_signature
    ):
        raise ValueError("dataset signature mismatch")
    image_list = _read_json(resolved["image_list"])
    if (
        not isinstance(image_list, list)
        or not image_list
        or any(
            not isinstance(image_id, str) or not image_id
            for image_id in image_list
        )
        or len(image_list) != len(set(image_list))
    ):
        raise ValueError("frozen image list is invalid")
    image_order_hash = hashlib.sha256(
        _canonical(image_list)
    ).hexdigest()
    if (
        manifest.get("image_count") != len(image_list)
        or manifest.get("image_order_hash") != image_order_hash
    ):
        raise ValueError("primary image count/order mismatch")
    workers = manifest.get("workers")
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers < 0
    ):
        raise ValueError("primary worker count is invalid")
    config = _yaml_scalar_mapping(resolved["dataset_yaml"])
    val_value = config.get("val")
    if val_value is None:
        raise ValueError("dataset YAML lacks val path")
    val_root = Path(val_value)
    if not val_root.is_absolute():
        val_root = dataset_root / val_root
    val_root = val_root.resolve()
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }
    actual_images = [
        path.relative_to(val_root).as_posix()
        for path in sorted(
            (
                path
                for path in val_root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in extensions
            ),
            key=lambda item: item.relative_to(val_root).as_posix(),
        )
    ]
    if actual_images != image_list:
        raise ValueError("dataset/image list order mismatch")
    return {
        "wrapper_path": wrapper_path,
        "upstream_path": upstream_path,
        "spec_path": spec_path,
        "upstream": upstream,
        "files": resolved,
        "dataset_root": dataset_root,
        "dataset_signature": dataset_signature,
        "image_list": tuple(image_list),
        "primary_source": primary_source,
        "workers": workers,
    }


def _delta(
    before: Mapping[str, Mapping[str, Mapping[str, int]]],
    after: Mapping[str, Mapping[str, Mapping[str, int]]],
    field: str,
) -> dict[str, dict[str, int]]:
    return {
        f"{threshold:.2f}": {
            name: int(
                after[f"{threshold:.2f}"][name][field]
                - before[f"{threshold:.2f}"][name][field]
            )
            for name in ("all", *SIZE_BINS)
        }
        for threshold in THRESHOLDS
    }


def _selected(
    tp_delta: Mapping[str, Mapping[str, int]],
) -> tuple[bool, str]:
    safe = all(
        tp_delta[f"{threshold:.2f}"][name] >= 0
        for threshold in THRESHOLDS
        for name in ("all", "tiny", "large")
    )
    large_gain = (
        sum(
            tp_delta[f"{threshold:.2f}"]["large"]
            for threshold in THRESHOLDS
        )
        > 0
    )
    if safe and large_gain:
        return True, "SAFE_LARGE_GAIN"
    if not safe:
        return False, "TP_SAFETY_FAIL"
    return False, "NO_LARGE_GAIN"


def _groups(
    image_id: str,
    records: Sequence[Raw],
) -> list[dict[str, Any]]:
    _, cluster_members = _reconstruct(records)
    by_index = {
        record.original_index: record for record in records
    }
    groups: list[dict[str, Any]] = []
    for position, member_indices in enumerate(cluster_members):
        members = tuple(by_index[index] for index in member_indices)
        full = tuple(
            sorted(
                (
                    member
                    for member in members
                    if member.source_order == 0
                ),
                key=_rank,
            )
        )
        if (
            not full
            or not any(
                member.source_order > 0 for member in members
            )
        ):
            continue
        anchor = full[0]
        aggressors = tuple(
            sorted(
                (
                    member
                    for member in members
                    if member.source_order > 0
                    and member.score > anchor.score
                ),
                key=_rank,
            )
        )
        if not aggressors:
            continue
        payload = {
            "image_id": image_id,
            "members": list(member_indices),
            "anchor": anchor.original_index,
            "aggressors": [
                member.original_index for member in aggressors
            ],
        }
        unit_id = (
            f"{image_id}:"
            f"{hashlib.sha256(_canonical(payload)).hexdigest()[:24]}"
        )
        groups.append(
            {
                "image_id": image_id,
                "unit_id": unit_id,
                "stock_cluster_position": position,
                "stock_member_indices": list(member_indices),
                "full_anchor_index": anchor.original_index,
                "aggressor_indices": [
                    member.original_index
                    for member in aggressors
                ],
                "anchor_score": anchor.score,
            }
        )
    return groups


def _overlay(
    records: Sequence[Raw],
    groups: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Raw, ...], list[dict[str, Any]]]:
    replacements: dict[int, tuple[float, Mapping[str, Any]]] = {}
    by_index = {
        record.original_index: record for record in records
    }
    for group in groups:
        anchor = by_index[group["full_anchor_index"]]
        new_score = math.nextafter(
            float(group["anchor_score"]), -math.inf
        )
        if (
            anchor.source_order != 0
            or anchor.score != group["anchor_score"]
            or not new_score < anchor.score
        ):
            raise ValueError("forged oracle anchor")
        for index in group["aggressor_indices"]:
            record = by_index[index]
            if (
                record.source_order == 0
                or record.score <= anchor.score
                or index in replacements
            ):
                raise ValueError("forged oracle group membership")
            replacements[index] = (new_score, group)
    overlaid: list[Raw] = []
    patches: list[dict[str, Any]] = []
    for record in records:
        change = replacements.get(record.original_index)
        if change is None:
            overlaid.append(record)
            continue
        new_score, group = change
        overlaid.append(replace(record, score=new_score))
        patches.append(
            {
                "image_id": record.image_id,
                "unit_id": group["unit_id"],
                "original_index": record.original_index,
                "full_anchor_index": group[
                    "full_anchor_index"
                ],
                "old_score": record.score,
                "new_score": new_score,
            }
        )
    return tuple(overlaid), patches


def _prediction_digest(
    predictions: Sequence[Prediction],
) -> str:
    rows = [
        {
            "box": list(prediction.box),
            "global_xyxy": list(prediction.global_xyxy),
            "score": prediction.score,
            "class_id": prediction.class_id,
            "source_order": prediction.source_order,
            "query_index": prediction.query_index,
            "original_index": index,
        }
        for index, prediction in enumerate(predictions)
    ]
    return hashlib.sha256(_canonical(rows)).hexdigest()


def _a_predictions(
    records: Sequence[Raw],
) -> tuple[Prediction, ...]:
    return tuple(
        Prediction(
            box=record.global_xyxy,
            global_xyxy=record.global_xyxy,
            score=record.score,
            class_id=record.class_id,
            source_order=record.source_order,
            query_index=record.query_index,
            members=(record.original_index,),
        )
        for record in records
    )


def _input_image_hash(
    image: Image, raw_rows: Sequence[Mapping[str, Any]]
) -> str:
    payload = {
        "image_id": image.image_id,
        "width": image.width,
        "height": image.height,
        "gt_boxes": _jsonable(image.gt_boxes),
        "gt_classes": _jsonable(image.gt_classes),
        "ignore_boxes": _jsonable(image.ignore_boxes),
        "raw_rows": [
            {
                key: value
                for key, value in row.items()
                if key != "_audit_original_index"
            }
            for row in raw_rows
        ],
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _load_images_and_raw(
    chain: Mapping[str, Any],
) -> tuple[
    dict[str, Image],
    dict[str, tuple[Mapping[str, Any], ...]],
    dict[str, tuple[Raw, ...]],
]:
    image_list = tuple(chain["image_list"])
    positions = {
        image_id: index
        for index, image_id in enumerate(image_list)
    }
    raw_source = _read_jsonl_gz(chain["files"]["raw_views"])
    source_by_image: dict[str, list[Mapping[str, Any]]] = {
        image_id: [] for image_id in image_list
    }
    parsed_by_image: dict[str, list[Raw]] = {
        image_id: [] for image_id in image_list
    }
    last_position = -1
    for original_index, source_row in enumerate(raw_source):
        image_id = source_row.get("image_id")
        if (
            not isinstance(image_id, str)
            or image_id not in positions
        ):
            raise ValueError("unknown raw image ID")
        position = positions[image_id]
        if position < last_position:
            raise ValueError("raw image order is not monotonic")
        last_position = position
        if source_row.get("arm") not in {"A", "C"}:
            continue
        row = dict(source_row)
        row["_audit_original_index"] = original_index
        source_by_image[image_id].append(row)
        parsed_by_image[image_id].append(
            _parse_raw(row, original_index)
        )
    images: dict[str, Image] = {}
    dataset_root = Path(chain["dataset_root"])
    for image_id in image_list:
        records = parsed_by_image[image_id]
        signatures = {
            (record.width, record.height) for record in records
        }
        if len(signatures) != 1:
            raise ValueError(
                f"raw dimensions missing/disagree for {image_id}"
            )
        width, height = next(iter(signatures))
        label = (
            dataset_root
            / "labels"
            / "val"
            / Path(image_id).with_suffix(".txt")
        )
        ignore = (
            dataset_root
            / "labels_ignore"
            / "val"
            / Path(image_id).with_suffix(".txt")
        )
        gt_boxes, gt_classes = _parse_label(
            label, width, height, ignore=False
        )
        ignore_boxes, _ = _parse_label(
            ignore, width, height, ignore=True
        )
        images[image_id] = Image(
            image_id=image_id,
            width=width,
            height=height,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
            ignore_boxes=ignore_boxes,
        )
    return (
        images,
        {
            key: tuple(value)
            for key, value in source_by_image.items()
        },
        {
            key: tuple(value)
            for key, value in parsed_by_image.items()
        },
    )


def _verify_primary_invariants(
    primary: Path,
    *,
    image_list: Sequence[str],
    expected_rows: Sequence[Mapping[str, Any]],
) -> None:
    invariants = _read_json(primary / "invariants.json")
    if (
        not isinstance(invariants, Mapping)
        or invariants.get("passed") is not True
        or invariants.get("image_count") != len(image_list)
        or invariants.get("expected_image_count")
        != len(image_list)
        or invariants.get("baseline_a_metrics_reproduced")
        is not True
        or invariants.get("baseline_c_metrics_reproduced")
        is not True
        or invariants.get("selection_rounds") != 1
    ):
        raise ValueError("primary invariant aggregate is invalid")
    per_image = invariants.get("per_image")
    if (
        not isinstance(per_image, list)
        or len(per_image) != len(expected_rows)
    ):
        raise ValueError("primary per-image invariants are incomplete")
    required_true = {
        "retained_identity_count_unchanged",
        "active_exclusions_exact",
        "modified_records_exact_selected_aggressors",
        "modified_scores_exact_predecessor",
        "unselected_scores_bit_identical",
        "non_score_fields_bit_identical",
        "complete_single_replays",
        "complete_joint_replay",
        "no_direct_prediction_editing",
        "no_op_digest_exact",
        "patches_exact",
        "finite_and_legal",
        "selection_is_one_frozen_round",
        "absolute_profiles_complete",
        "passed",
    }
    for order, (actual, expected) in enumerate(
        zip(per_image, expected_rows)
    ):
        if not isinstance(actual, Mapping):
            raise ValueError("primary per-image invariant is invalid")
        if any(actual.get(key) is not True for key in required_true):
            raise ValueError("primary scientific invariant failed")
        exact = {
            "image_order": order,
            "image_id": expected["image_id"],
            "input_image_hash": expected["input_image_hash"],
            "a_prediction_digest": expected[
                "a_prediction_digest"
            ],
            "c_prediction_digest": expected[
                "c_prediction_digest"
            ],
            "selection_rounds": 1,
            "eligible_units": expected["eligible_units"],
            "selected_units": expected["selected_units"],
        }
        if any(actual.get(key) != value for key, value in exact.items()):
            raise ValueError(
                "primary per-image invariant replay mismatch"
            )


def _verify_runtime(primary: Path, workers: int) -> None:
    runtime = _read_json(primary / "runtime.json")
    if (
        not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "seconds",
            "workers",
            "peak_rss_bytes",
            "parent_peak_rss_bytes",
            "max_worker_peak_rss_bytes",
            "environment",
        }
    ):
        raise ValueError("runtime artifact field set is invalid")
    seconds = runtime.get("seconds")
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(float(seconds))
        or float(seconds) < 0.0
        or runtime.get("workers") != workers
        or not isinstance(runtime.get("environment"), Mapping)
    ):
        raise ValueError("runtime artifact values are invalid")
    peaks: list[int] = []
    for name in (
        "peak_rss_bytes",
        "parent_peak_rss_bytes",
        "max_worker_peak_rss_bytes",
    ):
        value = runtime.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(f"runtime {name} is invalid")
        peaks.append(value)
    if peaks[0] != max(peaks[1:]):
        raise ValueError("runtime peak RSS aggregate disagrees")


def _gate_metrics(
    a_metrics: Mapping[str, Any],
    joint_metrics: Mapping[str, Any],
    selected_units: int,
) -> tuple[str, dict[str, float], dict[str, bool]]:
    deltas = {
        name: float(joint_metrics[name])
        - float(a_metrics[name])
        for name in GATES
    }
    if not all(math.isfinite(value) for value in deltas.values()):
        raise ValueError("oracle gate delta is non-finite")
    gates = {
        name: deltas[name] >= threshold
        for name, threshold in GATES.items()
    }
    status = (
        "SBR_SCORE_ORACLE_GO"
        if selected_units > 0 and all(gates.values())
        else "SBR_SCORE_ORACLE_STOP"
    )
    return status, deltas, gates


def replay_primary_evidence(
    root: Path,
    *,
    source: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Independently recompute unit labels, patches, metrics, and gate."""

    evidence_root = Path(root).resolve()
    primary = evidence_root / "primary"
    _verify_checksums(primary)
    manifest = _read_json(primary / "oracle_manifest.json")
    chain = _verify_manifest_chain(
        evidence_root, primary, manifest, source
    )
    _verify_runtime(primary, int(chain["workers"]))
    (
        images,
        source_rows_by_image,
        raw_by_image,
    ) = _load_images_and_raw(chain)
    expected_events: list[dict[str, Any]] = []
    expected_patches: list[dict[str, Any]] = []
    a_metric_rows: list[
        tuple[Image, tuple[Prediction, ...]]
    ] = []
    c_metric_rows: list[
        tuple[Image, tuple[Prediction, ...]]
    ] = []
    joint_metric_rows: list[
        tuple[Image, tuple[Prediction, ...]]
    ] = []
    invariant_rows: list[dict[str, Any]] = []
    eligible_units = selected_units = 0
    eligible_members = patched_members = 0
    affected_images: set[str] = set()
    large_positive_affected_images: set[str] = set()
    by_class: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_sequence: Counter[str] = Counter()

    for order, image_id in enumerate(chain["image_list"]):
        image = images[image_id]
        raw_records = raw_by_image[image_id]
        a_raw = tuple(
            record for record in raw_records if record.arm == "A"
        )
        c_raw = tuple(
            record for record in raw_records if record.arm == "C"
        )
        a_predictions = _a_predictions(a_raw)
        stock_active = tuple(
            record for record in c_raw if record.score >= CONF
        )
        stock_predictions, _ = _reconstruct(stock_active)
        stock_profile = _profile(image, stock_predictions)
        groups = _groups(image_id, c_raw)
        selected_groups: list[Mapping[str, Any]] = []
        by_index = {
            record.original_index: record for record in c_raw
        }
        for group in groups:
            overlaid, patches = _overlay(c_raw, [group])
            active = tuple(
                record
                for record in overlaid
                if record.score >= CONF
            )
            predictions, _ = _reconstruct(active)
            after_profile = _profile(image, predictions)
            tp_delta = _delta(
                stock_profile, after_profile, "tp"
            )
            fp_delta = _delta(
                stock_profile, after_profile, "fp"
            )
            take, reason = _selected(tp_delta)
            if take:
                selected_groups.append(group)
            expected_events.append(
                {
                    "unit_id": group["unit_id"],
                    "selected": take,
                    "reason": reason,
                    "before_profile": stock_profile,
                    "after_profile": after_profile,
                    "tp_delta": tp_delta,
                    "fp_delta": fp_delta,
                    "group": dict(group),
                    "patches": patches,
                    "image_order": order,
                }
            )
            anchor = by_index[group["full_anchor_index"]]
            by_class[str(anchor.class_id)] += 1
            for aggressor in group["aggressor_indices"]:
                by_source[str(by_index[aggressor].source_order)] += 1
            by_sequence[
                Path(image_id).name.split("_", 1)[0]
            ] += 1
        joint_overlaid, joint_patches = _overlay(
            c_raw, selected_groups
        )
        joint_active = tuple(
            record
            for record in joint_overlaid
            if record.score >= CONF
        )
        joint_predictions, _ = _reconstruct(joint_active)
        joint_profile = _profile(image, joint_predictions)
        for patch in joint_patches:
            expected_patches.append(
                {**patch, "image_order": order}
            )
        selected_count = len(selected_groups)
        eligible_units += len(groups)
        selected_units += selected_count
        eligible_members += sum(
            len(group["aggressor_indices"]) for group in groups
        )
        patched_members += len(joint_patches)
        if selected_count:
            affected_images.add(image_id)
            if stock_profile["0.50"]["large"]["gt"] > 0:
                large_positive_affected_images.add(image_id)
        a_metric_rows.append((image, a_predictions))
        c_metric_rows.append((image, stock_predictions))
        joint_metric_rows.append((image, joint_predictions))
        invariant_rows.append(
            {
                "image_id": image_id,
                "input_image_hash": _input_image_hash(
                    image, source_rows_by_image[image_id]
                ),
                "a_prediction_digest": _prediction_digest(
                    a_predictions
                ),
                "c_prediction_digest": _prediction_digest(
                    stock_predictions
                ),
                "eligible_units": len(groups),
                "selected_units": selected_count,
                "joint_profile": joint_profile,
            }
        )

    recorded_events = _read_jsonl_gz(
        primary / "unit_events.jsonl.gz"
    )
    event_ids = [row.get("unit_id") for row in recorded_events]
    if (
        any(not isinstance(unit_id, str) for unit_id in event_ids)
        or len(event_ids) != len(set(event_ids))
    ):
        raise ValueError("duplicate or invalid unit label ID")
    if recorded_events != _jsonable(expected_events):
        raise ValueError(
            "unit label rows disagree with independent replay"
        )
    recorded_patches = _read_jsonl_gz(
        primary / "score_patches.jsonl.gz"
    )
    patch_ids = [
        (
            row.get("image_id"),
            row.get("original_index"),
            row.get("unit_id"),
        )
        for row in recorded_patches
    ]
    if len(patch_ids) != len(set(patch_ids)):
        raise ValueError("duplicate joint patch identity")
    if recorded_patches != _jsonable(expected_patches):
        raise ValueError(
            "joint patch rows disagree with independent replay"
        )

    a_metrics = _evaluate_dataset(a_metric_rows)
    c_metrics = _evaluate_dataset(c_metric_rows)
    joint_metrics = _evaluate_dataset(joint_metric_rows)
    g0_metrics = _read_json(chain["files"]["g0_metrics"])
    if (
        not isinstance(g0_metrics, Mapping)
        or a_metrics != g0_metrics.get("A")
        or c_metrics != g0_metrics.get("C")
    ):
        raise ValueError(
            "baseline A/C metrics disagree with sealed G0 metrics"
        )
    status, deltas, gates = _gate_metrics(
        a_metrics, joint_metrics, selected_units
    )
    joint_minus_c = {
        name: float(joint_metrics[name])
        - float(c_metrics[name])
        for name in GATES
    }
    expected_metrics = {
        "A": a_metrics,
        "C": c_metrics,
        "joint": joint_metrics,
        "joint_minus_a": deltas,
        "joint_minus_c": joint_minus_c,
    }
    recorded_metrics = _read_json(
        primary / "oracle_metrics.json"
    )
    if recorded_metrics != _jsonable(expected_metrics):
        raise ValueError(
            "oracle metric artifact disagrees with independent replay"
        )
    _verify_primary_invariants(
        primary,
        image_list=chain["image_list"],
        expected_rows=invariant_rows,
    )
    expected_coverage = {
        "eligible_units": eligible_units,
        "selected_units": selected_units,
        "eligible_members": eligible_members,
        "patched_members": patched_members,
        "affected_images": len(affected_images),
        "large_positive_affected_images": len(
            large_positive_affected_images
        ),
        "by_class": dict(sorted(by_class.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_sequence_token": dict(sorted(by_sequence.items())),
    }
    if _read_json(primary / "coverage.json") != expected_coverage:
        raise ValueError(
            "coverage artifact disagrees with independent replay"
        )
    expected_gate = {
        "proposed_status": status,
        "joint_minus_a": deltas,
        "thresholds": GATES,
        "gates": gates,
        "invariants_passed": True,
        "selected_units": selected_units,
        "independent_adjudication": "PENDING",
    }
    recorded_gate = _read_json(primary / "primary_gate.json")
    if recorded_gate != expected_gate:
        raise ValueError(
            "primary gate disagrees with independent replay"
        )
    return {
        "status": status,
        "unit_labels": expected_events,
        "patches": expected_patches,
        "metrics": expected_metrics,
        "gate": expected_gate,
        "coverage": expected_coverage,
        "unit_labels_agree": True,
        "joint_metrics_agree": True,
        "primary_gate_agrees": True,
        "selected_units": selected_units,
        "eligible_units": eligible_units,
        "image_count": len(chain["image_list"]),
        "dataset_signature": chain["dataset_signature"],
        "primary_source": chain["primary_source"],
    }


def _capture_self_state(script_path: Path) -> dict[str, Any]:
    repo = script_path.resolve().parents[1]

    def run(argv: list[str]) -> str:
        completed = subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
        return completed.stdout.strip()

    prefix = ["git", "-C", str(repo)]
    commit = run([*prefix, "rev-parse", "HEAD"])
    tree = run([*prefix, "rev-parse", "HEAD^{tree}"])
    status = run(
        [
            *prefix,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ]
    )
    return {
        "commit": commit,
        "tree": tree,
        "clean": status == "",
        "script_sha256": _sha256_file(script_path),
        "repo_root": str(repo),
    }


def _validated_source(
    value: Any, name: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"adjudicator source {name} is missing")
    source = {
        "commit": _digest(
            value.get("commit"),
            f"adjudicator {name} commit",
            lengths=(40, 64),
        ),
        "tree": _digest(
            value.get("tree"),
            f"adjudicator {name} tree",
            lengths=(40, 64),
        ),
        "clean": value.get("clean") is True,
        "script_sha256": _digest(
            value.get("script_sha256"),
            f"adjudicator {name} script SHA-256",
        ),
        "repo_root": str(
            Path(str(value.get("repo_root", ""))).resolve()
        ),
    }
    if not source["clean"]:
        raise ValueError(
            f"adjudicator source {name} is not clean"
        )
    return source


def _assert_same_source(
    expected: Mapping[str, Any], actual: Any, name: str
) -> None:
    normalized = _validated_source(actual, name)
    for key in (
        "commit",
        "tree",
        "script_sha256",
        "repo_root",
    ):
        if normalized[key] != expected[key]:
            raise ValueError(
                f"adjudicator source changed at {name}: {key}"
            )


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical(value))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_root_checksums(root: Path) -> None:
    if (
        {path.name for path in root.iterdir()}
        != {
            "primary",
            "independent_adjudication.json",
            "final_status.json",
        }
        or (root / "primary").is_symlink()
        or not (root / "primary").is_dir()
        or any(
            path.is_symlink() or not path.is_file()
            for path in (
                root / "independent_adjudication.json",
                root / "final_status.json",
            )
        )
    ):
        raise ValueError(
            "pre-checksum root artifact set/type mismatch"
        )
    targets = (
        root / "primary" / "checksums.sha256",
        root / "independent_adjudication.json",
        root / "final_status.json",
    )
    if any(not path.is_file() for path in targets):
        raise ValueError("authoritative output file set is incomplete")
    text = "".join(
        f"{_sha256_file(path)}  "
        f"{path.relative_to(root).as_posix()}\n"
        for path in targets
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".checksums.",
        suffix=".tmp",
        dir=str(root),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, root / "checksums.sha256")
    finally:
        if temporary.exists():
            temporary.unlink()
    if (
        {path.name for path in root.iterdir()}
        != {
            "primary",
            "independent_adjudication.json",
            "final_status.json",
            "checksums.sha256",
        }
        or any(
            path.is_symlink()
            for path in root.iterdir()
        )
        or not (root / "primary").is_dir()
        or any(
            not path.is_file()
            for path in (
                root / "independent_adjudication.json",
                root / "final_status.json",
                root / "checksums.sha256",
            )
        )
    ):
        raise ValueError(
            "final root artifact set/type mismatch"
        )


def _environment() -> dict[str, str]:
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def _finish_report(report: dict[str, Any]) -> dict[str, Any]:
    report["output_hash_semantics"] = (
        "sha256(canonical-json(report without output_hash))"
    )
    report["output_hash"] = _sha256_json(report)
    return report


def adjudicate_evidence(
    root: Path,
    primary_checksums_sha256: str,
) -> dict[str, object]:
    """Fail closed and write the authoritative independent decision."""

    evidence_root = Path(root).resolve()
    primary = evidence_root / "primary"
    script_path = Path(__file__).resolve()
    start_state: dict[str, Any] | None = None
    anchor: str | None = None
    can_write = evidence_root.is_dir()
    report: dict[str, Any]
    try:
        if (
            not evidence_root.is_dir()
            or evidence_root.is_symlink()
            or {path.name for path in evidence_root.iterdir()}
            != {"primary"}
            or (evidence_root / "primary").is_symlink()
            or not (evidence_root / "primary").is_dir()
        ):
            raise ValueError(
                "root artifact set/type mismatch at start"
            )
        anchor = _digest(
            primary_checksums_sha256,
            "external primary checksum anchor",
        )
        start_state = _validated_source(
            _capture_self_state(script_path), "at start"
        )
        repo_root = Path(start_state["repo_root"])
        if (
            evidence_root == repo_root
            or repo_root in evidence_root.parents
        ):
            raise ValueError(
                "evidence output must be outside source repo"
            )
        before_snapshot = _primary_snapshot(primary)
        checksum_path = primary / "checksums.sha256"
        if (
            not checksum_path.is_file()
            or _sha256_file(checksum_path) != anchor
        ):
            raise ValueError(
                "external primary checksum anchor mismatch"
            )
        replay = replay_primary_evidence(
            evidence_root,
            source=start_state,
        )
        after_snapshot = _primary_snapshot(primary)
        if after_snapshot != before_snapshot:
            raise ValueError(
                "primary snapshot changed during adjudication"
            )
        _assert_same_source(
            start_state,
            _capture_self_state(script_path),
            "before output write",
        )
        authoritative_status = str(replay["status"])
        if authoritative_status not in {
            "SBR_SCORE_ORACLE_GO",
            "SBR_SCORE_ORACLE_STOP",
        }:
            raise ValueError("independent status is invalid")
        snapshot_hash = _sha256_json(before_snapshot)
        report = {
            "decision": "PASS",
            "authoritative_status": authoritative_status,
            "primary_gate_agrees": True,
            "joint_metrics_agree": True,
            "unit_labels_agree": True,
            "checksums_verified": True,
            "primary_checksum_anchor_verified": True,
            "primary_checksums_sha256": anchor,
            "primary_snapshot_unchanged": True,
            "primary_snapshot_sha256": snapshot_hash,
            "primary_snapshot": before_snapshot,
            "selected_units": replay["selected_units"],
            "eligible_units": replay["eligible_units"],
            "image_count": replay["image_count"],
            "dataset_signature": replay["dataset_signature"],
            "adjudicator_source": start_state,
            "adjudicator_script_sha256": start_state[
                "script_sha256"
            ],
            "source_stability_verified": True,
            "environment": _environment(),
        }
    except Exception as exc:
        report = {
            "decision": "FAIL",
            "authoritative_status": "SBR_SCORE_ORACLE_INVALID",
            "primary_gate_agrees": False,
            "joint_metrics_agree": False,
            "unit_labels_agree": False,
            "checksums_verified": False,
            "primary_checksum_anchor_verified": False,
            "primary_checksums_sha256": anchor,
            "primary_snapshot_unchanged": False,
            "adjudicator_source": start_state,
            "adjudicator_script_sha256": (
                start_state["script_sha256"]
                if start_state is not None
                else _sha256_file(script_path)
            ),
            "source_stability_verified": False,
            "environment": _environment(),
            "error": str(exc),
        }
    _finish_report(report)
    if can_write:
        _atomic_write_json(
            evidence_root / "independent_adjudication.json",
            report,
        )
        final_status = {
            "status": report["authoritative_status"],
            "decision": report["decision"],
            "primary_checksums_sha256": anchor,
            "independent_adjudication_sha256": _sha256_file(
                evidence_root / "independent_adjudication.json"
            ),
            "report_output_hash": report["output_hash"],
        }
        _atomic_write_json(
            evidence_root / "final_status.json",
            final_status,
        )
        if report["decision"] == "PASS":
            try:
                if start_state is None:
                    raise ValueError(
                        "adjudicator source start state is missing"
                    )
                _assert_same_source(
                    start_state,
                    _capture_self_state(script_path),
                    "after output write",
                )
                if _primary_snapshot(primary) != report[
                    "primary_snapshot"
                ]:
                    raise ValueError(
                        "primary snapshot changed after output write"
                    )
                _write_root_checksums(evidence_root)
            except Exception as exc:
                report = {
                    **report,
                    "decision": "FAIL",
                    "authoritative_status": (
                        "SBR_SCORE_ORACLE_INVALID"
                    ),
                    "primary_gate_agrees": False,
                    "joint_metrics_agree": False,
                    "unit_labels_agree": False,
                    "source_stability_verified": False,
                    "error": str(exc),
                }
                report.pop("output_hash", None)
                report.pop("output_hash_semantics", None)
                _finish_report(report)
                _atomic_write_json(
                    evidence_root
                    / "independent_adjudication.json",
                    report,
                )
                _atomic_write_json(
                    evidence_root / "final_status.json",
                    {
                        "status": "SBR_SCORE_ORACLE_INVALID",
                        "decision": "FAIL",
                        "primary_checksums_sha256": anchor,
                        "independent_adjudication_sha256": (
                            _sha256_file(
                                evidence_root
                                / "independent_adjudication.json"
                            )
                        ),
                        "report_output_hash": report[
                            "output_hash"
                        ],
                    },
                )
                checksum_output = (
                    evidence_root / "checksums.sha256"
                )
                if checksum_output.exists():
                    checksum_output.unlink()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independently adjudicate frozen SBR score-oracle evidence"
        )
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--primary-checksums-sha256",
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = adjudicate_evidence(
        args.evidence,
        args.primary_checksums_sha256,
    )
    print(
        json.dumps(
            report,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0 if report.get("decision") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
