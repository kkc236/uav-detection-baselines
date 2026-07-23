#!/usr/bin/env python3
"""Primary, fail-closed causal audit for the frozen SBR-V2 guard."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import gzip
import json
import math
from numbers import Real
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import numpy as np

from src.sbr_artifacts import (
    atomic_write_json,
    atomic_write_jsonl_gz,
    canonical_json_bytes,
    environment_info,
    git_provenance,
    load_dataset,
    protocol_signature,
    sha256_bytes,
    sha256_file,
    write_checksums,
)
from src.sbr_g0 import FrozenSBRProtocol, build_arm_views
from src.sbr_v2_audit import (
    AuditImage,
    AuditRawDetection,
    AttributionCategory,
    audit_prepared_image_at_threshold,
    evaluate_guard_upper_bound,
    group_relevant_raw_rows,
    prepare_image_audit,
)


INPUT_SCHEMA_VERSION = "sbr-v2-audit-input/v1"
OUTPUT_SCHEMA_VERSION = "sbr-v2-audit-evidence/v1"
FROZEN_IOU_THRESHOLDS = tuple(round(0.50 + index * 0.05, 2) for index in range(10))
FROZEN = {
    "conf_threshold": 0.001,
    "max_det": 300,
    "ios_threshold": 0.5,
    "large_effective_size": 96.0,
    "mechanism_share_threshold": 0.60,
    "large_ap_tolerance": -0.005,
    "primary_iou_threshold": 0.75,
    "secondary_iou_thresholds": list(FROZEN_IOU_THRESHOLDS),
}
OUTPUT_SCHEMA = {
    "schema_version": OUTPUT_SCHEMA_VERSION,
    "required_artifacts": [
        "audit_manifest.json",
        "attribution_events.jsonl.gz",
        "attribution_summary.json",
        "upper_bound_metrics.json",
        "invariants.json",
        "primary_gate.json",
        "checksums.sha256",
    ],
    "primary_event_id": ["image_id", "gt_index", "iou_threshold"],
    "primary_gate_inputs": [
        "mechanism_gate",
        "recoverable_upper_bound_gate",
        "invariants.passed",
    ],
}
ORIGINAL_G0_FILE_KEYS = (
    "g0_manifest",
    "raw_views",
    "arm_predictions",
    "g0_metrics",
    "g0_gate",
    "independent_adjudication",
    "original_checksums",
)
ROOT_CHECKSUM_SEALED_KEYS = ORIGINAL_G0_FILE_KEYS[:-1]
ALL_FILE_KEYS = ORIGINAL_G0_FILE_KEYS + (
    "checkpoint",
    "image_list",
    "dataset_yaml",
)
HEX = frozenset("0123456789abcdefABCDEF")
VIEW_BY_SOURCE = ("full", "TL", "TR", "BL", "BR")
INVARIANT_BOOL_KEYS = (
    "raw_hash_equal",
    "cluster_hash_equal",
    "cluster_count_equal",
    "scores_equal",
    "classes_equal",
    "selected_cluster_ids_equal",
)


@dataclass(frozen=True)
class ValidatedAuditInput:
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    evidence_root: Path
    paths: dict[str, Path]
    hashes: dict[str, str]
    dataset_root: Path
    dataset_signature: str
    image_list: tuple[str, ...]
    g0_manifest: dict[str, Any]
    g0_metrics: dict[str, Any]
    g0_gate: dict[str, Any]
    independent_adjudication: dict[str, Any]
    g0_gate_sha256: str
    independent_adjudication_sha256: str
    original_checksums_sha256: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class FrozenArmImage:
    records: tuple[Mapping[str, Any], ...]
    predictions: tuple[Mapping[str, Any], ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen primary SBR-V2 causal audit"
    )
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=0)
    return parser


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def _digest(value: Any, name: str, *, lengths: tuple[int, ...] = (64,)) -> str:
    if not isinstance(value, str) or len(value) not in lengths or any(
        character not in HEX for character in value
    ):
        raise ValueError(f"{name} must be a hexadecimal digest")
    return value.lower()


def _entry_uri(value: Any, name: str) -> str:
    if isinstance(value, str):
        uri = value
    elif isinstance(value, Mapping):
        uri = value.get("uri")
    else:
        uri = None
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError(f"{name} portable URI is missing")
    return uri


def _entry_hash(value: Any, name: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} expected sha256 is missing")
    return _digest(value.get("sha256"), f"{name}.sha256")


def _portable_path(uri: str, *, base: Path) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme.lower() != "file":
        raise ValueError(f"unsupported portable URI scheme: {parsed.scheme}")
    if parsed.scheme.lower() == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError("remote file URI authorities are forbidden")
        value = url2pathname(unquote(parsed.path))
        # url2pathname('/C:/x') retains the leading slash on some platforms.
        if os.name == "nt" and len(value) >= 3 and value[0] in "/\\" and value[2] == ":":
            value = value[1:]
        path = Path(value)
    else:
        path = Path(uri)
        if not path.is_absolute():
            path = base / path
    return path.resolve()


def _inside_or_equal(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _overlaps(left: Path, right: Path) -> bool:
    return _inside_or_equal(left, right) or _inside_or_equal(right, left)


def _manifest_file_entries(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    entries = manifest.get("files")
    if not isinstance(entries, Mapping):
        raise ValueError("input manifest files mapping is missing")
    return entries


def _verify_original_checksums(root: Path, checksum_path: Path) -> set[str]:
    if not checksum_path.is_file():
        raise ValueError("original evidence checksums.sha256 is missing")
    listed: set[str] = set()
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("cannot read original evidence checksums") from exc
    for line in lines:
        if not line:
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError("invalid original checksums line") from exc
        expected = _digest(digest, f"original checksum {relative}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("original checksum path traversal is forbidden")
        target = (root / relative_path).resolve()
        if not _inside_or_equal(target, root) or not target.is_file():
            raise ValueError(f"original checksum target is outside evidence: {relative}")
        if sha256_file(target).lower() != expected:
            raise ValueError(f"original evidence checksum mismatch: {relative}")
        normalized = relative_path.as_posix()
        if normalized in listed:
            raise ValueError(f"duplicate original checksum entry: {normalized}")
        listed.add(normalized)
    return listed


def _validate_output_path(output: Path, evidence_root: Path, inputs: Iterable[Path]) -> None:
    resolved = output.resolve()
    if _inside_or_equal(resolved, evidence_root):
        raise ValueError("output must be outside original G0 evidence")
    for input_path in inputs:
        if _overlaps(resolved, input_path.resolve()):
            raise ValueError("output must not overlap any input")
    if resolved.exists():
        if not resolved.is_dir() or any(resolved.iterdir()):
            raise FileExistsError("output must not exist or must be an empty directory")


def validate_input_manifest(
    manifest_path: Path | str, output: Path | str
) -> ValidatedAuditInput:
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw_manifest = _read_json(path)
    if not isinstance(raw_manifest, Mapping):
        raise ValueError("input manifest must be a JSON object")
    manifest = dict(raw_manifest)
    if manifest.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError("unsupported input manifest schema_version")
    protocol_hash = _digest(manifest.get("protocol_hash"), "protocol_hash")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("source provenance is missing")
    source_commit = _digest(
        source.get("commit"), "source.commit", lengths=(40, 64)
    )
    source_tree = _digest(source.get("tree"), "source.tree")

    evidence_root = _portable_path(
        _entry_uri(manifest.get("original_evidence_root"), "original_evidence_root"),
        base=path.parent,
    )
    if not evidence_root.is_dir():
        raise ValueError("original evidence root does not exist")
    entries = _manifest_file_entries(manifest)
    resolved_paths: dict[str, Path] = {}
    expected_hashes: dict[str, str] = {}
    for key in ALL_FILE_KEYS:
        if key not in entries:
            raise ValueError(f"input manifest file entry is missing: {key}")
        entry = entries[key]
        target = _portable_path(_entry_uri(entry, key), base=path.parent)
        if key in ORIGINAL_G0_FILE_KEYS and not _inside_or_equal(
            target, evidence_root
        ):
            raise ValueError(f"{key} path escapes original evidence root")
        expected = _entry_hash(entry, key)
        if not target.is_file():
            raise ValueError(f"input file does not exist: {key}")
        # Every file is authenticated before its contents are opened.
        if sha256_file(target).lower() != expected:
            raise ValueError(f"input checksum mismatch: {key}")
        resolved_paths[key] = target
        expected_hashes[key] = expected

    dataset_spec = manifest.get("dataset")
    if not isinstance(dataset_spec, Mapping):
        raise ValueError("dataset provenance is missing")
    root_entry = dataset_spec.get("root")
    dataset_root = _portable_path(
        _entry_uri(root_entry, "dataset.root"), base=path.parent
    )
    if not dataset_root.is_dir():
        raise ValueError("dataset root does not exist")
    dataset_signature = _entry_hash(root_entry, "dataset.root")

    _validate_output_path(
        Path(output), evidence_root, [path, *resolved_paths.values(), dataset_root]
    )
    listed = _verify_original_checksums(
        evidence_root, resolved_paths["original_checksums"]
    )
    for key in ROOT_CHECKSUM_SEALED_KEYS:
        relative = resolved_paths[key].relative_to(evidence_root).as_posix()
        if relative not in listed:
            raise ValueError(f"{key} is not sealed by original checksums")

    g0_manifest_raw = _read_json(resolved_paths["g0_manifest"])
    g0_metrics_raw = _read_json(resolved_paths["g0_metrics"])
    gate_path = resolved_paths["g0_gate"]
    independent_path = resolved_paths["independent_adjudication"]
    g0_gate_raw = _read_json(gate_path)
    independent = _read_json(independent_path)
    if not isinstance(g0_manifest_raw, Mapping) or g0_manifest_raw.get("mode") != "g0-a":
        raise ValueError("g0_manifest is not frozen G0-A evidence")
    if not isinstance(g0_metrics_raw, Mapping) or not all(
        isinstance(g0_metrics_raw.get(arm), Mapping) for arm in ("A", "C")
    ):
        raise ValueError("g0_metrics must contain A and C")
    g0_manifest = dict(g0_manifest_raw)
    g0_metrics = dict(g0_metrics_raw)
    if not isinstance(g0_gate_raw, Mapping):
        raise ValueError("original G0 gate must be an object")
    g0_gate = dict(g0_gate_raw)
    if (
        str(g0_manifest.get("source_hash", "")).lower() != source_commit
        or str(g0_manifest.get("protocol_hash", "")).lower() != protocol_hash
    ):
        raise ValueError("source/protocol provenance disagrees with G0 evidence")
    g0_source = g0_manifest.get("source")
    if not isinstance(g0_source, Mapping):
        raise ValueError("G0 source provenance is incomplete")
    if str(g0_source.get("commit", "")).lower() != source_commit:
        raise ValueError("G0 source commit provenance disagrees")
    g0_tree = _digest(
        g0_source.get("source_tree_hash", g0_source.get("tree")),
        "G0 source tree",
    )
    if g0_tree != source_tree:
        raise ValueError("G0 source tree provenance disagrees")
    g0_protocol = g0_manifest.get("protocol")
    frozen_protocol = dict(FrozenSBRProtocol().__dict__)
    if not isinstance(g0_protocol, Mapping) or canonical_json_bytes(
        dict(g0_protocol)
    ) != canonical_json_bytes(frozen_protocol):
        raise ValueError("G0 protocol is not canonical-exact frozen SBR")
    if protocol_signature(dict(g0_protocol)) != protocol_hash:
        raise ValueError("G0 protocol payload/hash disagreement")
    if (
        not isinstance(independent, Mapping)
        or independent.get("checksums_verified") is not True
        or str(independent.get("source_hash", "")).lower() != source_commit
        or str(independent.get("protocol_hash", "")).lower() != protocol_hash
    ):
        raise ValueError("independent adjudication provenance is incomplete")
    expected_dataset_signature = str(
        g0_manifest.get("dataset_signature", "")
    ).lower()
    checkpoint_hash = expected_hashes["checkpoint"]
    if any(
        str(record.get("checkpoint_hash", "")).lower() != checkpoint_hash
        for record in (g0_manifest, g0_gate, independent)
    ):
        raise ValueError(
            "checkpoint bytes and G0 manifest/gate/adjudication disagree"
        )
    if (
        g0_gate.get("status") != "SBR_G0A_FAIL"
        or str(g0_gate.get("source_hash", "")).lower() != source_commit
        or str(g0_gate.get("protocol_hash", "")).lower() != protocol_hash
        or str(g0_gate.get("dataset_signature", "")).lower()
        != expected_dataset_signature
    ):
        raise ValueError("original sealed G0 gate must remain SBR_G0A_FAIL")
    if (
        independent.get("status") != "SBR_G0A_INDEPENDENT_FAIL"
        or independent.get("decision") != "FAIL"
        or independent.get("independent_gate") != "SBR_G0A_FAIL"
        or independent.get("runner_status") != "SBR_G0A_FAIL"
    ):
        raise ValueError(
            "original independent adjudication must agree on immutable G0 FAIL"
        )

    image_list_raw = _read_json(resolved_paths["image_list"])
    if (
        not isinstance(image_list_raw, list)
        or not image_list_raw
        or any(not isinstance(item, str) or not item for item in image_list_raw)
        or len(set(image_list_raw)) != len(image_list_raw)
    ):
        raise ValueError("image_list must be a nonempty unique JSON string list")
    image_list = tuple(image_list_raw)
    if (
        list(g0_manifest.get("image_list", ())) != list(image_list)
        or g0_manifest.get("image_count") != len(image_list)
    ):
        raise ValueError("image_list disagrees with G0 manifest")

    split = dataset_spec.get("split", "val")
    if not isinstance(split, str) or not split:
        raise ValueError("dataset split must be an exact string")
    dataset = load_dataset(
        resolved_paths["dataset_yaml"],
        split=split,
        root_override=dataset_root,
    )
    if dataset["dataset_signature"].lower() != dataset_signature:
        raise ValueError("dataset root checksum/signature mismatch")
    if (
        str(g0_manifest.get("dataset_signature", "")).lower()
        != dataset_signature
    ):
        raise ValueError("dataset signature disagrees with G0 evidence")
    if (
        str(g0_gate.get("dataset_signature", "")).lower()
        != dataset_signature
        or str(independent.get("dataset_signature", "")).lower()
        != dataset_signature
    ):
        raise ValueError(
            "dataset signature disagrees across G0 gate/adjudication"
        )
    if tuple(dataset["image_list"]) != image_list:
        raise ValueError("dataset image order disagrees with frozen image_list")
    by_image = {record["relative_path"]: record for record in dataset["images"]}
    if set(by_image) != set(image_list):
        raise ValueError("dataset image set disagrees with exact image_list")

    return ValidatedAuditInput(
        manifest_path=path,
        manifest=manifest,
        manifest_sha256=sha256_file(path),
        evidence_root=evidence_root,
        paths=resolved_paths,
        hashes=expected_hashes,
        dataset_root=dataset_root,
        dataset_signature=dataset_signature,
        image_list=image_list,
        g0_manifest=g0_manifest,
        g0_metrics=g0_metrics,
        g0_gate=g0_gate,
        independent_adjudication=dict(independent),
        g0_gate_sha256=sha256_file(gate_path),
        independent_adjudication_sha256=sha256_file(independent_path),
        original_checksums_sha256=sha256_file(
            resolved_paths["original_checksums"]
        ),
        checkpoint_sha256=checkpoint_hash,
    )


def _iter_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"{path.name}:{line_number} is blank")
                try:
                    row = json.loads(line, parse_constant=_reject_constant)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path.name}:{line_number} is malformed JSON"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path.name}:{line_number} is not an object")
                row["_audit_original_index"] = line_number - 1
                yield row
    except (EOFError, OSError, UnicodeError) as exc:
        raise ValueError(f"invalid or truncated gzip JSONL: {path}") from exc


def _explicit_mapping_sequence(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be an explicit sequence")
    try:
        rows = tuple(value)
    except TypeError:
        raise ValueError(f"{name} must be an explicit sequence") from None
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{name} rows must be objects")
    return rows  # type: ignore[return-value]


def _load_frozen_arm_predictions(
    path: Path, image_list: Sequence[str]
) -> dict[str, dict[str, FrozenArmImage]]:
    block_arms = ("A", "B", "C", "D", "E", "F")
    image_ids = tuple(image_list)
    expected_rows = len(block_arms) * len(image_ids)
    selected: dict[str, dict[str, FrozenArmImage]] = {"A": {}, "C": {}}
    seen = 0
    for index, row in enumerate(_iter_jsonl_gz(path)):
        if index >= expected_rows:
            raise ValueError("arm_predictions has more than exact 6*N rows")
        block_index, image_index = divmod(index, len(image_ids))
        block_arm = block_arms[block_index]
        image_id = image_ids[image_index]
        if row.get("image_id") != image_id:
            raise ValueError(
                f"arm_predictions {block_arm} block image order disagrees"
            )
        top_arm = row.get("arm")
        if top_arm is not None and top_arm != block_arm:
            raise ValueError("arm_predictions top-level arm/block disagrees")
        records = _explicit_mapping_sequence(
            row.get("records"), "arm_predictions.records"
        )
        predictions = _explicit_mapping_sequence(
            row.get("predictions"), "arm_predictions.predictions"
        )
        if block_arm in selected:
            if any(record.get("arm") != block_arm for record in records):
                raise ValueError(
                    f"arm_predictions {block_arm} records arm disagrees"
                )
            selected[block_arm][image_id] = FrozenArmImage(
                records=records,
                predictions=predictions,
            )
        seen += 1
    if seen != expected_rows:
        raise ValueError(
            f"arm_predictions requires exact 6*N rows, got {seen}"
        )
    if any(len(selected[arm]) != len(image_ids) for arm in ("A", "C")):
        raise ValueError("arm_predictions A/C blocks are incomplete")
    return selected


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _box(value: Any, name: str) -> tuple[float, float, float, float]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be an xyxy sequence")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an xyxy sequence") from None
    if (
        len(result) != 4
        or not all(math.isfinite(item) for item in result)
        or result[2] <= result[0]
        or result[3] <= result[1]
    ):
        raise ValueError(f"{name} must be a finite nondegenerate xyxy box")
    return result  # type: ignore[return-value]


def _same_float_box(
    left: Sequence[float], right: Sequence[float], *, tolerance: float = 1e-9
) -> bool:
    return all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)
        for a, b in zip(left, right)
    )


def _validate_view_manifest(row: Mapping[str, Any], arm: str) -> None:
    value = row.get("view_manifest")
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError("view_manifest must be an explicit sequence")
    try:
        items = tuple(value)
    except TypeError:
        raise ValueError("view_manifest is required for every A/C raw row") from None
    expected_sources = (0,) if arm == "A" else (0, 1, 2, 3, 4)
    if len(items) != len(expected_sources):
        raise ValueError("view_manifest executed source set is incomplete")
    seen: set[int] = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("view_manifest items must be objects")
        source = _strict_int(item.get("source_order"), "view_manifest source")
        if (
            source not in expected_sources
            or source in seen
            or item.get("view_id") != VIEW_BY_SOURCE[source]
            or item.get("executed") is not True
        ):
            raise ValueError("view_manifest source/view/executed provenance is invalid")
        seen.add(source)
    if seen != set(expected_sources):
        raise ValueError("view_manifest executed source set is incomplete")


def _parse_raw_detection(
    row: Mapping[str, Any], *, expected_image_id: str
) -> AuditRawDetection:
    image_id = row.get("image_id")
    arm = row.get("arm")
    if image_id != expected_image_id or arm not in {"A", "C"}:
        raise ValueError("raw A/C row image/arm provenance is invalid")
    width = _strict_int(row.get("width"), "width")
    height = _strict_int(row.get("height"), "height")
    if width == 0 or height == 0:
        raise ValueError("raw dimensions must be positive")
    source = _strict_int(row.get("source_order"), "source_order")
    if source > 4 or (arm == "A" and source != 0):
        raise ValueError("raw view source must follow frozen A/C sources 0..4")
    if row.get("view_id") != VIEW_BY_SOURCE[source]:
        raise ValueError("raw view_id/source_order disagreement")
    _validate_view_manifest(row, arm)
    # The frozen runner retained finite detector boxes slightly outside the
    # network canvas, then clamped only the inverse-mapped view/global boxes.
    network = _box(row.get("network_xyxy"), "network_xyxy")
    view = _box(row.get("view_xyxy"), "view_xyxy")
    global_box = _box(row.get("global_xyxy"), "global_xyxy")
    tile_value = row.get("tile_bounds")
    expected_view = {
        view.source_order: view
        for view in build_arm_views(arm, width, height)
    }[source]
    expected_tile = (
        None
        if expected_view.tile is None
        else tuple(expected_view.tile.bounds)
    )
    tile: tuple[int, int, int, int] | None
    if source == 0:
        if tile_value is not None:
            raise ValueError("full view must not have tile_bounds")
        tile = None
        clipped = (
            max(0.0, view[0]),
            max(0.0, view[1]),
            min(float(width), view[2]),
            min(float(height), view[3]),
        )
        if not _same_float_box(clipped, global_box):
            raise ValueError("full-view coordinate frames disagree")
    else:
        if isinstance(tile_value, (str, bytes, Mapping)):
            raise ValueError("local view must have four tile bounds")
        try:
            values = tuple(
                _strict_int(item, "tile bound") for item in tile_value
            )
        except TypeError:
            raise ValueError("local view must have four tile bounds") from None
        if (
            len(values) != 4
            or values[2] <= values[0]
            or values[3] <= values[1]
            or values[2] > width
            or values[3] > height
        ):
            raise ValueError("tile bounds are outside the image")
        tile = values  # type: ignore[assignment]
        if tile != expected_tile:
            raise ValueError("tile_bounds disagree with the frozen view geometry")
        tile_width = tile[2] - tile[0]
        tile_height = tile[3] - tile[1]
        if (
            view[0] < 0.0
            or view[1] < 0.0
            or view[2] > tile_width
            or view[3] > tile_height
        ):
            raise ValueError("view_xyxy is outside the tile frame")
        expected_global = (
            view[0] + tile[0],
            view[1] + tile[1],
            view[2] + tile[0],
            view[3] + tile[1],
        )
        if not _same_float_box(expected_global, global_box):
            raise ValueError("local/global coordinate frames disagree")
    return AuditRawDetection(
        image_id=image_id,
        arm=arm,
        width=width,
        height=height,
        source_order=source,
        query_index=_strict_int(row.get("query_index"), "query_index"),
        class_id=_strict_int(row.get("class_id"), "class_id"),
        score=row.get("score"),  # type: ignore[arg-type]
        network_xyxy=network,
        view_xyxy=view,
        global_xyxy=global_box,
        tile_bounds=tile,
        original_index=_strict_int(
            row.get("_audit_original_index"), "original row index"
        ),
    )


def _metric_row(
    image: Mapping[str, Any],
    predictions: Sequence[Any],
    *,
    frozen_global_xyxy: bool = False,
) -> dict[str, Any]:
    boxes = [
        (
            getattr(prediction, "global_xyxy", None)
            if frozen_global_xyxy
            else prediction.box
        )
        for prediction in predictions
    ]
    return {
        "image_id": image["relative_path"],
        "width": int(image["width"]),
        "height": int(image["height"]),
        "pred_boxes": [list(box) for box in boxes],
        "pred_scores": [float(prediction.score) for prediction in predictions],
        "pred_classes": [int(prediction.class_id) for prediction in predictions],
        "pred_source": [int(prediction.source_order) for prediction in predictions],
        "pred_query": [int(prediction.query_index) for prediction in predictions],
        "gt_boxes": [list(box) for box in image["gt_boxes"]],
        "gt_classes": [int(class_id) for class_id in image["gt_classes"]],
        "ignore_boxes": [list(box) for box in image["ignore_boxes"]],
        "effective_gain": min(
            640.0 / float(image["width"]),
            640.0 / float(image["height"]),
            1.0,
        ),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, AttributionCategory):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _raw_record_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key != "view_manifest" and not key.startswith("_audit_")
    }


def _assert_frozen_records(
    arm: str,
    image_id: str,
    raw_rows: Sequence[Mapping[str, Any]],
    frozen: FrozenArmImage,
) -> None:
    expected = [_raw_record_payload(row) for row in raw_rows]
    if canonical_json_bytes(expected) != canonical_json_bytes(
        list(frozen.records)
    ):
        raise ValueError(
            f"frozen {arm} records disagree with raw cache for {image_id}"
        )


def _strict_recursive_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(
            _strict_recursive_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(
            right, (list, tuple)
        ):
            return False
        return len(left) == len(right) and all(
            _strict_recursive_equal(a, b) for a, b in zip(left, right)
        )
    if (
        isinstance(left, Real)
        and not isinstance(left, bool)
        and isinstance(right, Real)
        and not isinstance(right, bool)
    ):
        return (
            math.isfinite(float(left))
            and math.isfinite(float(right))
            and float(left) == float(right)
        )
    return type(left) is type(right) and left == right


def _prediction_identity_from_mapping(
    prediction: Mapping[str, Any], name: str
) -> dict[str, Any]:
    required = (
        "box",
        "global_xyxy",
        "score",
        "class_id",
        "source_order",
        "query_index",
    )
    if any(key not in prediction for key in required):
        raise ValueError(f"{name} prediction identity is incomplete")
    box = _box(prediction["box"], f"{name}.box")
    global_box = _box(
        prediction["global_xyxy"], f"{name}.global_xyxy"
    )
    score = prediction["score"]
    if (
        isinstance(score, bool)
        or not isinstance(score, Real)
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
    ):
        raise ValueError(f"{name}.score is invalid")
    return {
        "box": box,
        "global_xyxy": global_box,
        "score": float(score),
        "class_id": _strict_int(prediction["class_id"], f"{name}.class_id"),
        "source_order": _strict_int(
            prediction["source_order"], f"{name}.source_order"
        ),
        "query_index": _strict_int(
            prediction["query_index"], f"{name}.query_index"
        ),
    }


def _prediction_identity_from_detection(
    prediction: Any, name: str
) -> dict[str, Any]:
    global_box = getattr(prediction, "global_xyxy", None)
    return _prediction_identity_from_mapping(
        {
            "box": getattr(prediction, "box", None),
            "global_xyxy": global_box,
            "score": getattr(prediction, "score", None),
            "class_id": getattr(prediction, "class_id", None),
            "source_order": getattr(prediction, "source_order", None),
            "query_index": getattr(prediction, "query_index", None),
        },
        name,
    )


def _assert_frozen_predictions(
    arm: str,
    image_id: str,
    predictions: Sequence[Any],
    frozen: FrozenArmImage,
) -> None:
    actual = [
        _prediction_identity_from_detection(
            prediction, f"recomputed {arm}[{index}]"
        )
        for index, prediction in enumerate(predictions)
    ]
    expected = [
        _prediction_identity_from_mapping(
            prediction, f"frozen {arm}[{index}]"
        )
        for index, prediction in enumerate(frozen.predictions)
    ]
    if not _strict_recursive_equal(actual, expected):
        raise ValueError(
            f"frozen {arm} predictions disagree for {image_id}"
        )


def _aggregate_invariants(
    per_image: Sequence[tuple[str, Mapping[str, Any], int, int]]
) -> dict[str, Any]:
    bools = {
        key: all(result.get(key) is True for _, result, _, _ in per_image)
        for key in INVARIANT_BOOL_KEYS
    }
    singleton_total = sum(total for _, _, total, _ in per_image)
    singleton_preserved = sum(preserved for _, _, _, preserved in per_image)
    singleton_ratio = (
        float(singleton_preserved) / float(singleton_total)
        if singleton_total
        else 1.0
    )
    per_image_passed = all(
        result.get("passed") is True
        and result.get("singleton_preservation") == 1.0
        for _, result, _, _ in per_image
    )
    passed = (
        bool(per_image)
        and all(value is True for value in bools.values())
        and singleton_ratio == 1.0
        and per_image_passed
    )
    return {
        **bools,
        "singleton_preservation": singleton_ratio,
        "passed": passed,
        "singleton_total": singleton_total,
        "singleton_preserved": singleton_preserved,
        "image_count": len(per_image),
        "per_image": [
            {
                "image_id": image_id,
                **dict(result),
                "singleton_total": total,
                "singleton_preserved": preserved,
            }
            for image_id, result, total, preserved in per_image
        ],
    }


def primary_gate_status(
    upper_bound: Mapping[str, Any], invariants: Mapping[str, Any]
) -> str:
    eligible = (
        upper_bound.get("mechanism_gate") == "PASS"
        and upper_bound.get("recoverable_upper_bound_gate") == "PASS"
        and invariants.get("passed") is True
    )
    return "SBR_V2_AUDIT_ELIGIBLE" if eligible else "SBR_V2_AUDIT_STOP"


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except Exception:
        try:
            import psutil

            return int(psutil.Process().memory_info().rss)
        except Exception:
            return None


def _clean_audit_provenance(repo_root: Path) -> dict[str, Any]:
    provenance = git_provenance(repo_root)
    commit = _digest(
        provenance.get("commit"), "audit source commit", lengths=(40, 64)
    )
    tree = _digest(
        provenance.get("source_tree_hash"), "audit source tree"
    )
    if (
        provenance.get("clean_tracked") is not True
        or provenance.get("untracked") is not False
    ):
        raise ValueError("audit source worktree must be fully clean")
    return {**provenance, "commit": commit, "source_tree_hash": tree}


def _run_audit(
    validated: ValidatedAuditInput,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    dataset_spec = validated.manifest["dataset"]
    dataset = load_dataset(
        validated.paths["dataset_yaml"],
        split=dataset_spec.get("split", "val"),
        root_override=validated.dataset_root,
    )
    image_by_id = {
        image["relative_path"]: image for image in dataset["images"]
    }
    a_rows: list[dict[str, Any]] = []
    c_rows: list[dict[str, Any]] = []
    v2_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    invariant_rows: list[tuple[str, Mapping[str, Any], int, int]] = []
    seen_event_ids: set[tuple[str, int, float]] = set()
    frozen_arms = _load_frozen_arm_predictions(
        validated.paths["arm_predictions"], validated.image_list
    )

    grouped = group_relevant_raw_rows(
        _iter_jsonl_gz(validated.paths["raw_views"]),
        validated.image_list,
    )
    for group in grouped:
        image = image_by_id[group.image_id]
        parsed = tuple(
            _parse_raw_detection(row, expected_image_id=group.image_id)
            for row in group.rows
        )
        for arm in ("A", "C"):
            signatures = {
                canonical_json_bytes(row.get("view_manifest"))
                for row in group.rows
                if row.get("arm") == arm
            }
            if len(signatures) > 1:
                raise ValueError(
                    f"{arm} view_manifest is inconsistent within {group.image_id}"
                )
        a_raw = tuple(item for item in parsed if item.arm == "A")
        c_raw = tuple(item for item in parsed if item.arm == "C")
        a_frozen = frozen_arms["A"][group.image_id]
        c_frozen = frozen_arms["C"][group.image_id]
        _assert_frozen_records(
            "A",
            group.image_id,
            tuple(row for row in group.rows if row.get("arm") == "A"),
            a_frozen,
        )
        _assert_frozen_records(
            "C",
            group.image_id,
            tuple(row for row in group.rows if row.get("arm") == "C"),
            c_frozen,
        )
        fixture = AuditImage(
            image_id=group.image_id,
            width=int(image["width"]),
            height=int(image["height"]),
            gt_boxes=tuple(tuple(box) for box in image["gt_boxes"]),
            gt_classes=tuple(int(cls) for cls in image["gt_classes"]),
            a_detections=a_raw,
            c_detections=c_raw,
            ignore_boxes=tuple(tuple(box) for box in image["ignore_boxes"]),
        )
        prepared = prepare_image_audit(fixture)
        standard = prepared.standard
        guarded = prepared.guarded
        image_invariants = prepared.invariants
        singleton_indices = [
            index
            for index, members in enumerate(standard.cluster_members)
            if len(members) == 1
        ]
        singleton_preserved = sum(
            standard.pre_cap_predictions[index]
            == guarded.pre_cap_predictions[index]
            for index in singleton_indices
        )
        invariant_rows.append(
            (
                group.image_id,
                image_invariants,
                len(singleton_indices),
                singleton_preserved,
            )
        )

        a_predictions = tuple(item.to_detection() for item in a_raw)
        _assert_frozen_predictions(
            "A", group.image_id, a_predictions, a_frozen
        )
        _assert_frozen_predictions(
            "C",
            group.image_id,
            standard.standard_predictions,
            c_frozen,
        )
        a_rows.append(
            _metric_row(
                image, a_predictions, frozen_global_xyxy=True
            )
        )
        c_rows.append(
            _metric_row(
                image,
                standard.standard_predictions,
                frozen_global_xyxy=True,
            )
        )
        v2_rows.append(_metric_row(image, guarded.standard_predictions))
        for threshold in FROZEN_IOU_THRESHOLDS:
            result = audit_prepared_image_at_threshold(
                prepared, threshold
            )
            for event in result.events:
                event_id = (
                    event.image_id,
                    int(event.gt_index),
                    float(event.iou_threshold),
                )
                if event_id in seen_event_ids:
                    raise ValueError(f"duplicate attribution event: {event_id!r}")
                seen_event_ids.add(event_id)
                event_rows.append(_jsonable(event))

    if len(a_rows) != len(validated.image_list):
        raise ValueError("raw stream did not produce the exact manifest image order")
    invariants = _aggregate_invariants(invariant_rows)
    primary_events = [
        event
        for event in event_rows
        if float(event["iou_threshold"]) == FROZEN["primary_iou_threshold"]
    ]
    denominator = len(primary_events)
    mixed = sum(
        event["category"] == AttributionCategory.MIXED_CLUSTER_LOCALIZATION.value
        for event in primary_events
    )
    upper_bound = evaluate_guard_upper_bound(
        a_rows,
        c_rows,
        v2_rows,
        mixed_localization_unique_large_gt=mixed,
        a_tp_to_c_fn_unique_large_gt=denominator,
        invariants=invariants,
    )
    if not _strict_recursive_equal(
        _jsonable(upper_bound["a_metrics"]),
        validated.g0_metrics["A"],
    ) or not _strict_recursive_equal(
        _jsonable(upper_bound["c_metrics"]),
        validated.g0_metrics["C"],
    ):
        raise ValueError("recomputed A/C metrics disagree with sealed g0_metrics")
    category_names = [category.value for category in AttributionCategory]
    primary_counts = Counter(event["category"] for event in primary_events)
    secondary: dict[str, Any] = {}
    for threshold in FROZEN_IOU_THRESHOLDS:
        rows = [
            event
            for event in event_rows
            if float(event["iou_threshold"]) == threshold
        ]
        counts = Counter(event["category"] for event in rows)
        secondary[f"{threshold:.2f}"] = {
            "denominator": len(rows),
            "category_counts": {
                name: int(counts.get(name, 0)) for name in category_names
            },
        }
    summary = {
        "primary_ap75": {
            "unique_event_key": ["image_id", "gt_index", "iou_threshold"],
            "denominator": denominator,
            "mixed_cluster_localization": mixed,
            "mechanism_share": upper_bound["mechanism_share_ap75"],
            "category_counts": {
                name: int(primary_counts.get(name, 0))
                for name in category_names
            },
        },
        "secondary_repeated_measures": {
            "note": "The ten thresholds are pooled repeated measures, not independent samples.",
            "thresholds": secondary,
        },
    }
    status = primary_gate_status(upper_bound, invariants)
    primary_gate = {
        "status": status,
        "mechanism_gate": upper_bound["mechanism_gate"],
        "recoverable_upper_bound_gate": upper_bound[
            "recoverable_upper_bound_gate"
        ],
        "invariants_passed": invariants["passed"],
    }
    return (
        event_rows,
        _jsonable(summary),
        _jsonable(upper_bound),
        _jsonable(invariants),
        _jsonable(primary_gate),
    )


def _write_final_evidence(
    validated: ValidatedAuditInput,
    output: Path,
    workers: int,
    started: float,
    audit_provenance: Mapping[str, Any],
    artifacts: tuple[
        list[dict[str, Any]],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.audit-tmp-", dir=str(output.parent)
        )
    )
    try:
        events, summary, upper_bound, invariants, gate = artifacts
        deterministic_paths = {
            "attribution_events.jsonl.gz": temporary
            / "attribution_events.jsonl.gz",
            "attribution_summary.json": temporary / "attribution_summary.json",
            "upper_bound_metrics.json": temporary / "upper_bound_metrics.json",
            "invariants.json": temporary / "invariants.json",
            "primary_gate.json": temporary / "primary_gate.json",
        }
        atomic_write_jsonl_gz(
            deterministic_paths["attribution_events.jsonl.gz"], events
        )
        atomic_write_json(
            deterministic_paths["attribution_summary.json"], summary
        )
        atomic_write_json(
            deterministic_paths["upper_bound_metrics.json"], upper_bound
        )
        atomic_write_json(deterministic_paths["invariants.json"], invariants)
        atomic_write_json(deterministic_paths["primary_gate.json"], gate)
        deterministic_hashes = {
            name: sha256_file(path)
            for name, path in sorted(deterministic_paths.items())
        }
        deterministic_evidence_hash = sha256_bytes(
            canonical_json_bytes(deterministic_hashes)
        )
        script_path = Path(__file__).resolve()
        image_order_hash = sha256_bytes(
            canonical_json_bytes(list(validated.image_list))
        )
        elapsed = time.perf_counter() - started
        audit_manifest = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "schema": OUTPUT_SCHEMA,
            "schema_hash": sha256_bytes(canonical_json_bytes(OUTPUT_SCHEMA)),
            "frozen_constants": FROZEN,
            "input_manifest": {
                "uri": str(validated.manifest_path),
                "sha256": validated.manifest_sha256,
            },
            "inputs": {
                key: {
                    "uri": str(validated.paths[key]),
                    "sha256": validated.hashes[key],
                }
                for key in ALL_FILE_KEYS
            },
            "source_g0": {
                "commit": validated.manifest["source"]["commit"],
                "tree": validated.manifest["source"]["tree"],
                "protocol_hash": validated.manifest["protocol_hash"],
                "checkpoint_sha256": validated.checkpoint_sha256,
                "original_evidence_root_uri": _entry_uri(
                    validated.manifest["original_evidence_root"],
                    "original_evidence_root",
                ),
            },
            "audit_source": {
                **dict(audit_provenance),
                "script_uri": str(script_path),
                "script_sha256": sha256_file(script_path),
            },
            "original_g0_decision": {
                "gate_status": validated.g0_gate["status"],
                "gate_sha256": validated.g0_gate_sha256,
                "adjudication_status": validated.independent_adjudication[
                    "status"
                ],
                "adjudication_decision": validated.independent_adjudication[
                    "decision"
                ],
                "adjudication_sha256": (
                    validated.independent_adjudication_sha256
                ),
                "checksums_sha256": validated.original_checksums_sha256,
                "checkpoint_sha256": validated.checkpoint_sha256,
            },
            "limitations": [
                "A/C arm-image pairs with zero retained raw detections cannot carry a row-level view_manifest; their execution completeness is inherited from the sealed immutable G0-A evidence."
            ],
            "image_count": len(validated.image_list),
            "image_order_hash": image_order_hash,
            "deterministic_artifact_hashes": deterministic_hashes,
            "deterministic_evidence_hash": deterministic_evidence_hash,
            "primary_command": {
                "input_manifest": str(validated.manifest_path),
                "output": str(output),
                "workers": workers,
                "effective_workers": 0,
            },
            "nondeterministic_runtime": {
                "seconds": elapsed,
                "peak_rss_bytes": _peak_rss_bytes(),
                "environment": environment_info(),
            },
        }
        atomic_write_json(temporary / "audit_manifest.json", audit_manifest)
        files = [
            path
            for path in temporary.iterdir()
            if path.is_file() and path.name != "checksums.sha256"
        ]
        write_checksums(
            temporary / "checksums.sha256", files, root=temporary
        )
        expected_names = set(OUTPUT_SCHEMA["required_artifacts"])
        if {path.name for path in temporary.iterdir()} != expected_names:
            raise ValueError("temporary evidence contract is incomplete")
        if output.exists():
            output.rmdir()
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def run(args: argparse.Namespace) -> int:
    workers = _strict_int(args.workers, "workers")
    if workers != 0:
        raise ValueError("workers is frozen at 0 for deterministic streaming")
    output = Path(args.output).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    audit_provenance = _clean_audit_provenance(repo_root)
    validated = validate_input_manifest(args.input_manifest, output)
    started = time.perf_counter()
    artifacts = _run_audit(validated)
    final_provenance = _clean_audit_provenance(repo_root)
    if (
        final_provenance["commit"] != audit_provenance["commit"]
        or final_provenance["source_tree_hash"]
        != audit_provenance["source_tree_hash"]
    ):
        raise ValueError("audit source HEAD/tree changed during execution")
    _write_final_evidence(
        validated,
        output,
        workers,
        started,
        audit_provenance,
        artifacts,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return run(args)
    except Exception as exc:
        print(f"SBR_V2_AUDIT_FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
