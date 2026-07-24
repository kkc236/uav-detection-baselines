#!/usr/bin/env python3
"""Create the prediction-only SP-PPAF route closure without loading GT."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import gzip
import json
import math
from numbers import Real
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sbr_artifacts import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl_gz,
    canonical_json_bytes,
    git_provenance,
    protocol_signature,
    sha256_bytes,
    sha256_file,
    write_checksums,
)
from src.sbr_fusion import Detection  # noqa: E402
from src.sbr_g0 import FrozenSBRProtocol, build_arm_views  # noqa: E402
from src.sbr_ppaf import (  # noqa: E402
    A_FLOOR,
    ARM_NAMES,
    C_CEILING,
    CONF_THRESHOLD,
    FRAGMENT_IOS,
    LARGE_EFFECTIVE_SIZE,
    MAX_DET,
    build_ppaf_arms,
    verify_a_floor,
    verify_tail_score_domain,
)
from src.sbr_v2_audit import (  # noqa: E402
    AuditRawDetection,
    group_relevant_raw_rows,
    map_full_a_to_c,
    reconstruct_c_clusters,
)


INPUT_SCHEMA_VERSION = "sbr-v2-audit-input/v1"
ROUTE_SCHEMA_VERSION = "sbr-sp-ppaf-route/v1"
ROUTE_ARTIFACTS = (
    "route_manifest.json",
    "predictions.jsonl.gz",
    "coverage.json",
    "route_invariants.json",
)
REQUIRED_INPUT_FILES = (
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
)
ORIGINAL_EVIDENCE_KEYS = REQUIRED_INPUT_FILES[:7]
VIEW_BY_SOURCE = ("full", "TL", "TR", "BL", "BR")
HEX = frozenset("0123456789abcdefABCDEF")


@dataclass(frozen=True)
class ValidatedRouteInput:
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    paths: Mapping[str, Path]
    hashes: Mapping[str, str]
    evidence_root: Path
    dataset_root: Path
    dataset_signature: str
    image_list: tuple[str, ...]
    g0_manifest: Mapping[str, Any]
    g0_metrics: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenArmImage:
    records: tuple[Mapping[str, Any], ...]
    predictions: tuple[Mapping[str, Any], ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal GT-free SP-PPAF prediction routes"
    )
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def _digest(value: object, name: str, lengths: tuple[int, ...] = (64,)) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in HEX for character in value)
    ):
        raise ValueError(f"{name} must be a hexadecimal digest")
    return value.lower()


def _entry_uri(value: object, name: str) -> str:
    uri = value if isinstance(value, str) else (
        value.get("uri") if isinstance(value, Mapping) else None
    )
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError(f"{name} portable URI is missing")
    return uri


def _entry_hash(value: object, name: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} expected sha256 is missing")
    return _digest(value.get("sha256"), f"{name}.sha256")


def _portable_path(uri: str, *, base: Path) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        raise ValueError(f"unsupported portable URI scheme: {parsed.scheme}")
    if parsed.scheme == "file":
        raw = url2pathname(unquote(parsed.path))
        if parsed.netloc:
            raw = f"//{parsed.netloc}{raw}"
        path = Path(raw)
    else:
        path = Path(unquote(uri))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _inside_or_equal(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _verify_checksum_file(root: Path, checksum_path: Path) -> set[str]:
    listed: set[str] = set()
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise ValueError(f"invalid checksum line {line_number}")
        expected = _digest(parts[0], "original checksum")
        relative = Path(parts[1])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("original checksum path escapes evidence root")
        target = (root / relative).resolve()
        if not _inside_or_equal(target, root) or not target.is_file():
            raise ValueError("original checksum target is invalid")
        if sha256_file(target) != expected:
            raise ValueError(f"original evidence checksum mismatch: {relative}")
        label = relative.as_posix()
        if label in listed:
            raise ValueError(f"duplicate original checksum entry: {label}")
        listed.add(label)
    return listed


def validate_route_input(manifest_path: Path | str) -> ValidatedRouteInput:
    """Authenticate prediction inputs without opening dataset annotations."""

    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw_manifest = _read_json(path)
    if (
        not isinstance(raw_manifest, Mapping)
        or raw_manifest.get("schema_version") != INPUT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported input manifest")
    manifest = dict(raw_manifest)
    _digest(manifest.get("protocol_hash"), "protocol_hash")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("source provenance is missing")
    _digest(source.get("commit"), "source.commit", (40, 64))
    _digest(source.get("tree"), "source.tree")

    evidence_root = _portable_path(
        _entry_uri(
            manifest.get("original_evidence_root"),
            "original_evidence_root",
        ),
        base=path.parent,
    )
    if not evidence_root.is_dir():
        raise ValueError("original evidence root does not exist")
    entries = manifest.get("files")
    if not isinstance(entries, Mapping):
        raise ValueError("input manifest files are missing")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for key in REQUIRED_INPUT_FILES:
        if key not in entries:
            raise ValueError(f"input manifest file is missing: {key}")
        target = _portable_path(
            _entry_uri(entries[key], key),
            base=path.parent,
        )
        expected = _entry_hash(entries[key], key)
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"input checksum mismatch: {key}")
        if key in ORIGINAL_EVIDENCE_KEYS and not _inside_or_equal(
            target, evidence_root
        ):
            raise ValueError(f"{key} escapes original evidence root")
        paths[key] = target
        hashes[key] = expected

    listed = _verify_checksum_file(
        evidence_root,
        paths["original_checksums"],
    )
    for key in ORIGINAL_EVIDENCE_KEYS[:-1]:
        relative = paths[key].relative_to(evidence_root).as_posix()
        if relative not in listed:
            raise ValueError(f"{key} is not sealed by original checksums")

    image_list_raw = _read_json(paths["image_list"])
    if (
        not isinstance(image_list_raw, list)
        or not image_list_raw
        or any(not isinstance(item, str) or not item for item in image_list_raw)
        or len(set(image_list_raw)) != len(image_list_raw)
    ):
        raise ValueError("image_list must be nonempty unique strings")
    image_list = tuple(image_list_raw)

    g0_manifest = _read_json(paths["g0_manifest"])
    g0_metrics = _read_json(paths["g0_metrics"])
    if (
        not isinstance(g0_manifest, Mapping)
        or g0_manifest.get("mode") != "g0-a"
        or list(g0_manifest.get("image_list", ())) != list(image_list)
        or g0_manifest.get("image_count") != len(image_list)
    ):
        raise ValueError("G0 manifest/image list disagreement")
    if not isinstance(g0_metrics, Mapping) or not all(
        isinstance(g0_metrics.get(arm), Mapping) for arm in ("A", "C")
    ):
        raise ValueError("G0 metrics must contain A and C")
    frozen_protocol = dict(FrozenSBRProtocol().__dict__)
    if (
        not isinstance(g0_manifest.get("protocol"), Mapping)
        or canonical_json_bytes(dict(g0_manifest["protocol"]))
        != canonical_json_bytes(frozen_protocol)
        or protocol_signature(frozen_protocol)
        != str(manifest["protocol_hash"]).lower()
    ):
        raise ValueError("G0 protocol is not canonical frozen SBR")

    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("dataset provenance is missing")
    root_entry = dataset.get("root")
    dataset_root = _portable_path(
        _entry_uri(root_entry, "dataset.root"),
        base=path.parent,
    )
    if not dataset_root.is_dir():
        raise ValueError("dataset root does not exist")
    dataset_signature = _entry_hash(root_entry, "dataset.root")
    if (
        str(g0_manifest.get("dataset_signature", "")).lower()
        != dataset_signature
    ):
        raise ValueError("dataset signature disagrees with G0 manifest")

    return ValidatedRouteInput(
        manifest_path=path,
        manifest=manifest,
        manifest_sha256=sha256_file(path),
        paths=paths,
        hashes=hashes,
        evidence_root=evidence_root,
        dataset_root=dataset_root,
        dataset_signature=dataset_signature,
        image_list=image_list,
        g0_manifest=dict(g0_manifest),
        g0_metrics=dict(g0_metrics),
    )


def _iter_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"{path.name}:{line_number} is blank")
                row = json.loads(line, parse_constant=_reject_constant)
                if not isinstance(row, dict):
                    raise ValueError(f"{path.name}:{line_number} is not an object")
                row["_route_original_index"] = line_number - 1
                yield row
    except (EOFError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid gzip JSONL: {path}") from exc


def _mapping_rows(value: object, name: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be an explicit sequence")
    try:
        rows = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        raise ValueError(f"{name} must be an explicit sequence") from None
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{name} rows must be objects")
    return rows  # type: ignore[return-value]


def _load_frozen_arms(
    path: Path,
    image_list: Sequence[str],
) -> dict[str, dict[str, FrozenArmImage]]:
    block_arms = ("A", "B", "C", "D", "E", "F")
    expected = len(block_arms) * len(image_list)
    selected: dict[str, dict[str, FrozenArmImage]] = {"A": {}, "C": {}}
    seen = 0
    for index, row in enumerate(_iter_jsonl_gz(path)):
        if index >= expected:
            raise ValueError("arm_predictions has too many rows")
        block, image_index = divmod(index, len(image_list))
        arm = block_arms[block]
        image_id = image_list[image_index]
        if row.get("image_id") != image_id:
            raise ValueError("arm_predictions image order disagrees")
        if arm in selected:
            selected[arm][image_id] = FrozenArmImage(
                records=_mapping_rows(row.get("records"), "records"),
                predictions=_mapping_rows(
                    row.get("predictions"),
                    "predictions",
                ),
            )
        seen += 1
    if seen != expected:
        raise ValueError("arm_predictions row count disagrees")
    return selected


def _strict_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _box(value: object, name: str) -> tuple[float, float, float, float]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be an xyxy sequence")
    try:
        result = tuple(float(item) for item in value)  # type: ignore[arg-type]
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


def _same_box(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)
        for a, b in zip(left, right)
    )


def _validate_view_manifest(row: Mapping[str, Any], arm: str) -> None:
    items = _mapping_rows(row.get("view_manifest"), "view_manifest")
    expected = (0,) if arm == "A" else (0, 1, 2, 3, 4)
    if len(items) != len(expected):
        raise ValueError("view_manifest source set is incomplete")
    seen: set[int] = set()
    for item in items:
        source = _strict_int(item.get("source_order"), "view source")
        if (
            source not in expected
            or source in seen
            or item.get("view_id") != VIEW_BY_SOURCE[source]
            or item.get("executed") is not True
        ):
            raise ValueError("view_manifest provenance is invalid")
        seen.add(source)


def _parse_raw(
    row: Mapping[str, Any],
    *,
    image_id: str,
) -> AuditRawDetection:
    arm = row.get("arm")
    if row.get("image_id") != image_id or arm not in {"A", "C"}:
        raise ValueError("raw image/arm provenance is invalid")
    width = _strict_int(row.get("width"), "width")
    height = _strict_int(row.get("height"), "height")
    if not width or not height:
        raise ValueError("raw dimensions must be positive")
    source = _strict_int(row.get("source_order"), "source_order")
    if source > 4 or (arm == "A" and source != 0):
        raise ValueError("raw source is invalid")
    if row.get("view_id") != VIEW_BY_SOURCE[source]:
        raise ValueError("view_id/source disagreement")
    _validate_view_manifest(row, arm)
    network = _box(row.get("network_xyxy"), "network_xyxy")
    view = _box(row.get("view_xyxy"), "view_xyxy")
    global_box = _box(row.get("global_xyxy"), "global_xyxy")
    expected_view = {
        item.source_order: item for item in build_arm_views(arm, width, height)
    }[source]
    expected_tile = (
        None
        if expected_view.tile is None
        else tuple(expected_view.tile.bounds)
    )
    tile_value = row.get("tile_bounds")
    if source == 0:
        if tile_value is not None:
            raise ValueError("full view must not have tile bounds")
        tile = None
        clipped = (
            max(0.0, view[0]),
            max(0.0, view[1]),
            min(float(width), view[2]),
            min(float(height), view[3]),
        )
        if not _same_box(clipped, global_box):
            raise ValueError("full-view frames disagree")
    else:
        try:
            tile = tuple(
                _strict_int(item, "tile bound")
                for item in tile_value  # type: ignore[union-attr]
            )
        except TypeError:
            raise ValueError("local view must have tile bounds") from None
        if len(tile) != 4 or tile != expected_tile:
            raise ValueError("tile bounds disagree with frozen geometry")
        expected_global = (
            view[0] + tile[0],
            view[1] + tile[1],
            view[2] + tile[0],
            view[3] + tile[1],
        )
        if not _same_box(expected_global, global_box):
            raise ValueError("local/global frames disagree")
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
        tile_bounds=tile,  # type: ignore[arg-type]
        original_index=_strict_int(
            row.get("_route_original_index"),
            "original_index",
        ),
    )


def _raw_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key != "view_manifest" and not key.startswith("_route_")
    }


def _prediction_payload(prediction: Detection) -> dict[str, Any]:
    if prediction.global_xyxy is None:
        raise ValueError("prediction is missing global_xyxy")
    return {
        "box": list(prediction.box),
        "global_xyxy": list(prediction.global_xyxy),
        "score": float(prediction.score),
        "class_id": int(prediction.class_id),
        "source_order": int(prediction.source_order),
        "query_index": int(prediction.query_index),
    }


def _frozen_prediction_payload(
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    score = prediction.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, Real)
        or not math.isfinite(float(score))
    ):
        raise ValueError("frozen prediction score is invalid")
    return {
        "box": list(_box(prediction.get("box"), "prediction.box")),
        "global_xyxy": list(
            _box(prediction.get("global_xyxy"), "prediction.global_xyxy")
        ),
        "score": float(score),
        "class_id": _strict_int(
            prediction.get("class_id"),
            "prediction.class_id",
        ),
        "source_order": _strict_int(
            prediction.get("source_order"),
            "prediction.source_order",
        ),
        "query_index": _strict_int(
            prediction.get("query_index"),
            "prediction.query_index",
        ),
    }


def _assert_frozen(
    arm: str,
    image_id: str,
    raw_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Detection],
    frozen: FrozenArmImage,
) -> None:
    if canonical_json_bytes([_raw_payload(row) for row in raw_rows]) != (
        canonical_json_bytes(list(frozen.records))
    ):
        raise ValueError(f"frozen {arm} records disagree for {image_id}")
    actual = [_prediction_payload(item) for item in predictions]
    expected = [
        _frozen_prediction_payload(item) for item in frozen.predictions
    ]
    if canonical_json_bytes(actual) != canonical_json_bytes(expected):
        raise ValueError(f"frozen {arm} predictions disagree for {image_id}")


def _source_state(*, require_clean: bool) -> dict[str, Any]:
    state = git_provenance(REPO_ROOT)
    if require_clean and (
        state.get("clean_tracked") is not True
        or state.get("untracked") is not False
    ):
        raise ValueError("route source worktree must be fully clean")
    return state


def _same_source_state(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = (
        "commit",
        "source_tree_hash",
        "clean_tracked",
        "untracked",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def _score_order_hash(rows: Sequence[tuple[str, int]]) -> str:
    return sha256_bytes(canonical_json_bytes(list(rows)))


def _aggregate_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals: dict[str, dict[str, int]] = {}
    for arm in ARM_NAMES:
        keys = tuple(rows[0]["coverage"][arm]) if rows else ()
        totals[arm] = {
            key: sum(int(row["coverage"][arm][key]) for row in rows)
            for key in keys
        }
    return {
        "image_count": len(rows),
        "per_image": [
            {
                "image_id": row["image_id"],
                "coverage": row["coverage"],
            }
            for row in rows
        ],
        "totals": totals,
    }


def route_replay(
    input_manifest: Path | str,
    output: Path | str,
    *,
    require_clean: bool = True,
) -> Path:
    """Run and atomically seal one prediction-only replay."""

    if require_clean and "src.sbr_metrics" in sys.modules:
        raise ValueError("evaluator was imported before routing")
    before_source = _source_state(require_clean=require_clean)
    validated = validate_route_input(input_manifest)
    output_root = Path(output).resolve()
    if output_root.exists():
        raise FileExistsError("output must not exist")
    for input_path in (
        validated.manifest_path,
        validated.evidence_root,
        validated.dataset_root,
        *validated.paths.values(),
    ):
        if _inside_or_equal(output_root, Path(input_path).resolve()) or (
            _inside_or_equal(Path(input_path).resolve(), output_root)
        ):
            raise ValueError("output overlaps an input")

    frozen = _load_frozen_arms(
        validated.paths["arm_predictions"],
        validated.image_list,
    )
    route_rows: list[dict[str, Any]] = []
    all_a_scores: list[float] = []
    all_tail_scores: list[float] = []
    original_order: list[tuple[str, int]] = []
    mapped_order: list[tuple[str, int]] = []
    grouped = group_relevant_raw_rows(
        _iter_jsonl_gz(validated.paths["raw_views"]),
        validated.image_list,
    )
    for group in grouped:
        parsed = tuple(
            _parse_raw(row, image_id=group.image_id) for row in group.rows
        )
        a_raw = tuple(item for item in parsed if item.arm == "A")
        c_raw = tuple(item for item in parsed if item.arm == "C")
        if not a_raw:
            raise ValueError(f"Arm A is empty for {group.image_id}")
        width = a_raw[0].width
        height = a_raw[0].height
        if any(
            item.width != width or item.height != height for item in parsed
        ):
            raise ValueError("raw image dimensions disagree")
        a_predictions = tuple(item.to_detection() for item in a_raw)
        map_full_a_to_c(a_raw, c_raw)
        c_reconstruction = reconstruct_c_clusters(c_raw)
        _assert_frozen(
            "A",
            group.image_id,
            tuple(row for row in group.rows if row.get("arm") == "A"),
            a_predictions,
            frozen["A"][group.image_id],
        )
        _assert_frozen(
            "C",
            group.image_id,
            tuple(row for row in group.rows if row.get("arm") == "C"),
            c_reconstruction.standard_predictions,
            frozen["C"][group.image_id],
        )
        result = build_ppaf_arms(
            image_id=group.image_id,
            width=width,
            height=height,
            a_final=a_predictions,
            c_reconstruction=c_reconstruction,
            c_raw=c_raw,
        )
        all_a_scores.extend(float(item.score) for item in a_predictions)
        all_tail_scores.extend(
            item.original_score for item in result.eligible_tail
        )
        original_order.extend(
            (group.image_id, item.cluster_rank)
            for item in result.eligible_tail
        )
        mapped_sorted = sorted(
            result.eligible_tail,
            key=lambda item: (
                -float(item.prediction.score),
                int(item.prediction.source_order),
                int(item.prediction.query_index),
                int(item.cluster_rank),
            ),
        )
        mapped_order.extend(
            (group.image_id, item.cluster_rank) for item in mapped_sorted
        )
        arms = {
            "A": [_prediction_payload(item) for item in result.arms["A"]],
            "C": [
                _prediction_payload(item)
                for item in c_reconstruction.standard_predictions
            ],
            **{
                arm: [_prediction_payload(item) for item in result.arms[arm]]
                for arm in ("All-A", "P1", "P2", "P3")
            },
        }
        route_rows.append(
            {
                "image_id": group.image_id,
                "width": width,
                "height": height,
                "arms": arms,
                "eligible_clusters": [
                    {
                        "cluster_rank": item.cluster_rank,
                        "original_score": item.original_score,
                        "mapped_score": item.prediction.score,
                        "member_indices": list(item.member_indices),
                        "member_identities": [
                            list(identity)
                            for identity in sorted(item.member_identities)
                        ],
                        "tile_only": item.tile_only,
                    }
                    for item in result.eligible_tail
                ],
                "selected_cluster_ranks": {
                    arm: list(ranks)
                    for arm, ranks in result.selected_cluster_ranks.items()
                },
                "coverage": result.coverage,
                "invariants": result.invariants,
            }
        )

    if tuple(row["image_id"] for row in route_rows) != validated.image_list:
        raise ValueError("route image order disagrees with manifest")
    floor = verify_a_floor(tuple(all_a_scores))
    score_domain = verify_tail_score_domain(tuple(all_tail_scores))
    original_order_hash = _score_order_hash(original_order)
    mapped_order_hash = _score_order_hash(mapped_order)
    invariants = {
        "image_count": len(route_rows),
        "expected_image_count": len(validated.image_list),
        "image_order_exact": tuple(
            row["image_id"] for row in route_rows
        )
        == validated.image_list,
        "per_image_passed": all(
            row["invariants"].get("passed") is True for row in route_rows
        ),
        "a_floor": floor,
        "tail_score_domain": score_domain,
        "original_score_order_hash": original_order_hash,
        "mapped_score_order_hash": mapped_order_hash,
        "mapped_order_exact": original_order_hash == mapped_order_hash,
        "c_ceiling_below_actual_a_min": (
            C_CEILING < float(floor["actual_a_min"])
        ),
    }
    invariants["passed"] = (
        invariants["image_count"] == invariants["expected_image_count"]
        and invariants["image_order_exact"] is True
        and invariants["per_image_passed"] is True
        and floor["passed"] is True
        and score_domain["passed"] is True
        and invariants["mapped_order_exact"] is True
        and invariants["c_ceiling_below_actual_a_min"] is True
    )
    if invariants["passed"] is not True:
        raise ValueError("route invariants failed")
    coverage = _aggregate_coverage(route_rows)

    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.route-staging-",
            dir=parent,
        )
    )
    try:
        route_dir = staging / "route"
        route_dir.mkdir()
        predictions_path = atomic_write_jsonl_gz(
            route_dir / "predictions.jsonl.gz",
            route_rows,
        )
        coverage_path = atomic_write_json(
            route_dir / "coverage.json",
            coverage,
        )
        invariants_path = atomic_write_json(
            route_dir / "route_invariants.json",
            invariants,
        )
        after_source = _source_state(require_clean=require_clean)
        if not _same_source_state(before_source, after_source):
            raise ValueError("source state changed during routing")
        route_manifest = {
            "schema_version": ROUTE_SCHEMA_VERSION,
            "input_manifest_sha256": validated.manifest_sha256,
            "input_file_sha256": dict(validated.hashes),
            "original_source": dict(validated.manifest["source"]),
            "route_source": after_source,
            "dataset_signature": validated.dataset_signature,
            "image_count": len(route_rows),
            "image_list_sha256": validated.hashes["image_list"],
            "constants": {
                "conf": CONF_THRESHOLD,
                "max_det": MAX_DET,
                "large_effective_size": LARGE_EFFECTIVE_SIZE,
                "fragment_ios": FRAGMENT_IOS,
                "a_floor": A_FLOOR,
                "c_ceiling": C_CEILING,
            },
            "arms": ["A", "C", "All-A", "P1", "P2", "P3"],
            "required_artifacts": list(ROUTE_ARTIFACTS)
            + ["checksums.sha256"],
            "route_invariants_sha256": sha256_file(invariants_path),
            "coverage_sha256": sha256_file(coverage_path),
            "predictions_sha256": sha256_file(predictions_path),
        }
        manifest_path = atomic_write_json(
            route_dir / "route_manifest.json",
            route_manifest,
        )
        checksum_path = write_checksums(
            route_dir / "checksums.sha256",
            [
                manifest_path,
                predictions_path,
                coverage_path,
                invariants_path,
            ],
            root=route_dir,
        )
        anchor = {
            "schema_version": "sbr-sp-ppaf-route-anchor/v1",
            "route_checksums_sha256": sha256_file(checksum_path),
            "route_manifest_sha256": sha256_file(manifest_path),
            "predictions_sha256": sha256_file(predictions_path),
            "input_manifest_sha256": validated.manifest_sha256,
        }
        atomic_write_json(staging / "route_anchor.json", anchor)
        staging.rename(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_root


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        route_replay(args.input_manifest, args.output)
    except Exception as exc:
        print(f"SP_PPAF_ROUTE_INVALID: {exc}", file=sys.stderr)
        return 2
    print("SP_PPAF_ROUTE_SEALED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
