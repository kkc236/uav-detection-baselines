#!/usr/bin/env python3
"""Authenticate two endpoint caches and seal one GT-free SADED route."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gzip
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import statistics
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.saded_stage import (  # noqa: E402
    CACHE_ROW_KEYS,
    prediction_payload,
    route_paired_caches,
)
from src.saded_stage_protocol import stage_source_state  # noqa: E402
from src.sbr_artifacts import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl_gz,
    sha256_file,
    write_checksums,
)
from src.sbr_fusion import Detection  # noqa: E402
from src.sbr_g0 import (  # noqa: E402
    RawViewRecord,
    assemble_paired_arms,
)
from src.sbr_geometry import LetterboxTransform  # noqa: E402
from src.tascv_protocol import reject_forbidden_path  # noqa: E402


CACHE_FILES = {
    "cache_manifest.json",
    "predictions.jsonl.gz",
    "raw_views.jsonl.gz",
    "view_manifests.jsonl.gz",
    "cache_invariants.json",
    "checksums.sha256",
}
ROUTE_FILES = (
    "route_manifest.json",
    "predictions.jsonl.gz",
    "capacity.json",
    "route_invariants.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal one GT-free paired SADED route."
    )
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument(
        "--baseline-anchor-sha256",
        required=True,
    )
    parser.add_argument("--treatment-cache", type=Path, required=True)
    parser.add_argument(
        "--treatment-anchor-sha256",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _iter_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(
                        f"blank JSONL row at line {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"non-object JSONL row at line {line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid gzip JSONL: {path}") from error
    return rows


def _digest(value: object) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError("invalid SHA256 digest")
    return text


def _parse_checksums(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise ValueError("invalid checksum row")
        digest, name = parts
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or name != relative.as_posix()
            or name in parsed
        ):
            raise ValueError("invalid checksum artifact name")
        parsed[name] = _digest(digest)
    if not parsed:
        raise ValueError("empty checksum file")
    return parsed


def _snapshot(paths: Sequence[Path]) -> dict[str, str]:
    return {
        path.resolve().as_posix(): sha256_file(path)
        for path in sorted(
            {Path(path).resolve() for path in paths},
            key=lambda item: item.as_posix(),
        )
    }


RAW_VIEW_KEYS = {
    "image_id",
    "width",
    "height",
    "arm",
    "view_id",
    "source_order",
    "query_index",
    "tile_bounds",
    "transform",
    "network_xyxy",
    "view_xyxy",
    "global_xyxy",
    "score",
    "class_id",
}


def _raw_view(record: Mapping[str, Any]) -> RawViewRecord:
    if set(record) != RAW_VIEW_KEYS:
        raise ValueError("SADED raw-view schema drift")
    return RawViewRecord(
        image_id=str(record["image_id"]),
        width=int(record["width"]),
        height=int(record["height"]),
        arm=str(record["arm"]),
        view_id=str(record["view_id"]),
        source_order=int(record["source_order"]),
        query_index=int(record["query_index"]),
        tile_bounds=(
            None
            if record["tile_bounds"] is None
            else tuple(record["tile_bounds"])
        ),
        transform=LetterboxTransform(**record["transform"]),
        network_xyxy=tuple(record["network_xyxy"]),
        view_xyxy=tuple(record["view_xyxy"]),
        global_xyxy=tuple(record["global_xyxy"]),
        score=float(record["score"]),
        class_id=int(record["class_id"]),
    )


def _raw_detection(record: RawViewRecord) -> Detection:
    return Detection(
        box=record.global_xyxy,
        global_xyxy=record.global_xyxy,
        score=record.score,
        class_id=record.class_id,
        source_order=record.source_order,
        query_index=record.query_index,
        view_xyxy=record.view_xyxy,
        network_xyxy=record.network_xyxy,
        tile_bounds=record.tile_bounds,
        transform=record.transform,
        tile_index=(
            record.source_order - 1
            if record.tile_bounds is not None
            else None
        ),
    )


def _replay_cache_predictions(
    *,
    cache_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    raw_rows = _iter_jsonl_gz(cache_dir / "raw_views.jsonl.gz")
    view_rows = _iter_jsonl_gz(
        cache_dir / "view_manifests.jsonl.gz"
    )
    if (
        len(view_rows) != len(rows)
        or [record.get("image_id") for record in view_rows]
        != [record["image_id"] for record in rows]
    ):
        raise ValueError("SADED view-manifest identity drift")
    grouped: dict[str, list[RawViewRecord]] = {
        str(row["image_id"]): [] for row in rows
    }
    for record in raw_rows:
        parsed = _raw_view(record)
        if (
            parsed.image_id not in grouped
            or parsed.arm != "C"
        ):
            raise ValueError("SADED raw-view identity drift")
        grouped[parsed.image_id].append(parsed)
    for row, view_record in zip(rows, view_rows):
        if (
            set(view_record)
            != {"image_id", "width", "height", "view_manifest"}
            or int(view_record["width"]) != int(row["width"])
            or int(view_record["height"]) != int(row["height"])
        ):
            raise ValueError("SADED view-manifest schema drift")
        raw = tuple(grouped[str(row["image_id"])])
        full = [
            prediction_payload(_raw_detection(record))
            for record in raw
            if record.source_order == 0
        ]
        local = [
            prediction_payload(detection)
            for detection in assemble_paired_arms(
                raw,
                width=int(row["width"]),
                height=int(row["height"]),
                view_manifest=view_record["view_manifest"],
            )["C"]["predictions"]
        ]
        if (
            full != row["full_predictions"]
            or local != row["local_fused_predictions"]
        ):
            raise ValueError("SADED cache prediction replay drift")


def _inside_or_equal(parent: Path, child: Path) -> bool:
    parent = parent.resolve()
    child = child.resolve()
    return child == parent or parent in child.parents


def _verify_cache_root(
    root: Path,
    *,
    expected_anchor_sha256: str,
    evaluation_source: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]]:
    root = reject_forbidden_path(root, context="SADED cache root")
    if (
        not root.is_dir()
        or {path.name for path in root.iterdir()}
        != {"cache", "cache_anchor.json"}
    ):
        raise ValueError("SADED cache root closure drift")
    anchor_path = root / "cache_anchor.json"
    if sha256_file(anchor_path) != _digest(expected_anchor_sha256):
        raise ValueError("SADED cache external anchor drift")
    cache_dir = root / "cache"
    if (
        not cache_dir.is_dir()
        or {path.name for path in cache_dir.iterdir()} != CACHE_FILES
    ):
        raise ValueError("SADED cache artifact closure drift")
    anchor = _read_json(anchor_path)
    if set(anchor) != {
        "schema_version",
        "cache_manifest_sha256",
        "cache_checksums_sha256",
        "training_protocol_sha256",
        "training_source_commit",
        "evaluation_source_commit",
    }:
        raise ValueError("SADED cache anchor schema drift")
    manifest_path = cache_dir / "cache_manifest.json"
    checksums_path = cache_dir / "checksums.sha256"
    if (
        anchor["schema_version"] != "saded-endpoint-cache-anchor/v1"
        or sha256_file(manifest_path)
        != _digest(anchor["cache_manifest_sha256"])
        or sha256_file(checksums_path)
        != _digest(anchor["cache_checksums_sha256"])
    ):
        raise ValueError("SADED cache anchor binding drift")
    checksums = _parse_checksums(checksums_path)
    if set(checksums) != CACHE_FILES - {"checksums.sha256"}:
        raise ValueError("SADED cache checksum closure drift")
    for name, digest in checksums.items():
        if sha256_file(cache_dir / name) != digest:
            raise ValueError(f"SADED cache checksum drift: {name}")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != "saded-endpoint-cache/v1"
        or manifest.get("required_artifacts")
        != [
            "cache_manifest.json",
            "predictions.jsonl.gz",
            "raw_views.jsonl.gz",
            "view_manifests.jsonl.gz",
            "cache_invariants.json",
            "checksums.sha256",
        ]
        or manifest.get("evaluation_source") != dict(evaluation_source)
        or anchor["evaluation_source_commit"]
        != evaluation_source["commit"]
    ):
        raise ValueError("SADED cache manifest identity drift")
    artifact_bindings = manifest.get("artifacts")
    expected_artifact_bindings = {
        "predictions_sha256": sha256_file(
            cache_dir / "predictions.jsonl.gz"
        ),
        "raw_views_sha256": sha256_file(
            cache_dir / "raw_views.jsonl.gz"
        ),
        "view_manifests_sha256": sha256_file(
            cache_dir / "view_manifests.jsonl.gz"
        ),
        "invariants_sha256": sha256_file(
            cache_dir / "cache_invariants.json"
        ),
    }
    if artifact_bindings != expected_artifact_bindings:
        raise ValueError("SADED cache manifest artifact drift")
    invariants = _read_json(cache_dir / "cache_invariants.json")
    expected_image_count = int(manifest["dataset"]["image_count"])
    if (
        invariants.get("passed") is not True
        or expected_image_count <= 0
        or invariants.get("image_count") != expected_image_count
        or invariants.get("expected_image_count")
        != expected_image_count
    ):
        raise ValueError("SADED cache invariants did not pass")
    external_paths: list[Path] = []
    bindings = (
        (
            manifest["training_protocol"]["path"],
            manifest["training_protocol"]["sha256"],
        ),
        (
            manifest["training_summary"]["path"],
            manifest["training_summary"]["sha256"],
        ),
        (
            manifest["checkpoint"]["path"],
            manifest["checkpoint"]["sha256"],
        ),
        (
            manifest["dataset"]["image_list"],
            manifest["dataset"]["image_list_sha256"],
        ),
    )
    for raw_path, expected_sha in bindings:
        path = reject_forbidden_path(
            raw_path,
            context="SADED cache external input",
        )
        if not path.is_file() or sha256_file(path) != _digest(expected_sha):
            raise ValueError("SADED cache external input drift")
        external_paths.append(path)
    if (
        anchor["training_protocol_sha256"]
        != manifest["training_protocol"]["sha256"]
        or anchor["training_source_commit"]
        != manifest["training_protocol"]["source_commit"]
    ):
        raise ValueError("SADED cache training source binding drift")
    rows = _iter_jsonl_gz(cache_dir / "predictions.jsonl.gz")
    if (
        len(rows) != expected_image_count
        or any(set(row) != CACHE_ROW_KEYS for row in rows)
        or [row["image_id"] for row in rows]
        != json.loads(external_paths[-1].read_text(encoding="utf-8"))
    ):
        raise ValueError("SADED cache prediction identity drift")
    _replay_cache_predictions(cache_dir=cache_dir, rows=rows)
    snapshot_paths = [
        anchor_path,
        *(cache_dir / name for name in CACHE_FILES),
        *external_paths,
    ]
    return manifest, rows, snapshot_paths


def _summary(values: Sequence[int]) -> dict[str, int | float]:
    return {
        "min": min(values) if values else 0,
        "median": statistics.median(values) if values else 0,
        "max": max(values) if values else 0,
        "total": sum(values),
    }


def _aggregate_capacity(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    for output_name, coverage_name in (
        ("route_control", "control"),
        ("route_treatment", "treatment"),
    ):
        coverage = [
            dict(row["coverage"][coverage_name]) for row in rows
        ]
        systems[output_name] = {
            key: _summary([int(item[key]) for item in coverage])
            for key in (
                "protected_baseline",
                "remaining_tiny_slots",
                "accepted_local",
                "capacity_rejected",
            )
        }
    return {
        "image_count": len(rows),
        "systems": systems,
        "per_image": [
            {
                "image_id": row["image_id"],
                **dict(row["coverage"]),
            }
            for row in rows
        ],
    }


def route_pair(args: argparse.Namespace) -> Path:
    if "src.sbr_metrics" in sys.modules:
        raise ValueError("GT-aware evaluator imported before SADED routing")
    for value in (
        args.baseline_cache,
        args.treatment_cache,
        args.output,
    ):
        reject_forbidden_path(value, context="SADED paired route")
    output_root = args.output.resolve()
    if output_root.exists():
        raise FileExistsError("SADED route output already exists")
    baseline_root = args.baseline_cache.resolve()
    treatment_root = args.treatment_cache.resolve()
    if baseline_root == treatment_root:
        raise ValueError("SADED paired caches must be distinct")
    if any(
        _inside_or_equal(root, output_root)
        or _inside_or_equal(output_root, root)
        for root in (baseline_root, treatment_root)
    ):
        raise ValueError("SADED route output overlaps a cache input")
    evaluation_source = stage_source_state(REPO_ROOT)
    baseline_manifest, baseline_rows, baseline_paths = _verify_cache_root(
        baseline_root,
        expected_anchor_sha256=args.baseline_anchor_sha256,
        evaluation_source=evaluation_source,
    )
    treatment_manifest, treatment_rows, treatment_paths = (
        _verify_cache_root(
            treatment_root,
            expected_anchor_sha256=args.treatment_anchor_sha256,
            evaluation_source=evaluation_source,
        )
    )
    baseline_identity = baseline_manifest.get("identity")
    treatment_identity = treatment_manifest.get("identity")
    if (
        not isinstance(baseline_identity, dict)
        or not isinstance(treatment_identity, dict)
        or baseline_identity.get("stage")
        not in {"SCREEN_10", "FORMAL_100"}
        or baseline_identity.get("seed") not in {0, 1, 2}
        or baseline_identity.get("arm") != "control"
        or treatment_identity
        != {
            "stage": baseline_identity["stage"],
            "seed": baseline_identity["seed"],
            "arm": "tascv",
        }
    ):
        raise ValueError("SADED paired endpoint cache identity drift")
    paired_fields = (
        "training_protocol",
        "dataset",
        "protocol",
        "evaluation_source",
    )
    if any(
        baseline_manifest.get(field) != treatment_manifest.get(field)
        for field in paired_fields
    ):
        raise ValueError("SADED cache pair binding drift")
    baseline_summary = _read_json(
        Path(baseline_manifest["training_summary"]["path"])
    )
    treatment_summary = _read_json(
        Path(treatment_manifest["training_summary"]["path"])
    )
    paired_summary_fields = (
        "stage",
        "seed",
        "initial_state",
        "initial_state_sha256",
        "initial_state_common_fingerprint",
        "data",
        "data_sha256",
        "subset_binding",
        "batch",
        "workers",
        "loader",
        "optimizer",
        "batch_canaries",
        "successful_batches",
        "optimizer_attempts",
    )
    if (
        any(
            baseline_summary.get(field)
            != treatment_summary.get(field)
            for field in paired_summary_fields
        )
        or baseline_manifest["checkpoint"]["sha256"].lower()
        == treatment_manifest["checkpoint"]["sha256"].lower()
    ):
        raise ValueError(
            "SADED paired training initialization/canary drift"
        )
    input_paths = baseline_paths + treatment_paths
    before_snapshot = _snapshot(input_paths)
    route_rows, invariants = route_paired_caches(
        baseline_rows,
        treatment_rows,
    )
    expected_image_count = int(
        baseline_manifest["dataset"]["image_count"]
    )
    if (
        invariants.get("passed") is not True
        or invariants.get("image_count") != expected_image_count
        or "src.sbr_metrics" in sys.modules
    ):
        raise ValueError("SADED paired route invariants failed")
    capacity = _aggregate_capacity(route_rows)
    after_snapshot = _snapshot(input_paths)
    invariants = {
        **invariants,
        "input_snapshot_unchanged": after_snapshot == before_snapshot,
        "evaluation_source_unchanged": (
            stage_source_state(REPO_ROOT) == evaluation_source
        ),
        "gt_module_absent": "src.sbr_metrics" not in sys.modules,
    }
    invariants["passed"] = all(
        value is True
        for key, value in invariants.items()
        if key != "image_count"
    ) and invariants["image_count"] == expected_image_count
    if not invariants["passed"]:
        raise ValueError("SADED route closure changed during execution")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.route-staging-",
            dir=output_root.parent,
        )
    )
    try:
        route_dir = staging / "route"
        route_dir.mkdir()
        predictions_path = atomic_write_jsonl_gz(
            route_dir / "predictions.jsonl.gz",
            route_rows,
        )
        capacity_path = atomic_write_json(
            route_dir / "capacity.json",
            capacity,
        )
        invariants_path = atomic_write_json(
            route_dir / "route_invariants.json",
            invariants,
        )
        manifest_path = atomic_write_json(
            route_dir / "route_manifest.json",
            {
                "schema_version": "saded-paired-route/v1",
                "evaluation_source": evaluation_source,
                "training_protocol": baseline_manifest[
                    "training_protocol"
                ],
                "identity": {
                    "stage": baseline_identity["stage"],
                    "seed": baseline_identity["seed"],
                },
                "dataset": baseline_manifest["dataset"],
                "router_protocol": baseline_manifest["protocol"],
                "cache_inputs": {
                    "baseline": {
                        "root": baseline_root.as_posix(),
                        "anchor_sha256": _digest(
                            args.baseline_anchor_sha256
                        ),
                        "manifest_sha256": sha256_file(
                            baseline_root
                            / "cache/cache_manifest.json"
                        ),
                        "checkpoint": baseline_manifest["checkpoint"],
                    },
                    "treatment": {
                        "root": treatment_root.as_posix(),
                        "anchor_sha256": _digest(
                            args.treatment_anchor_sha256
                        ),
                        "manifest_sha256": sha256_file(
                            treatment_root
                            / "cache/cache_manifest.json"
                        ),
                        "checkpoint": treatment_manifest["checkpoint"],
                    },
                },
                "input_snapshot": before_snapshot,
                "artifacts": {
                    "predictions_sha256": sha256_file(predictions_path),
                    "capacity_sha256": sha256_file(capacity_path),
                    "invariants_sha256": sha256_file(invariants_path),
                },
                "required_artifacts": list(ROUTE_FILES)
                + ["checksums.sha256"],
            },
        )
        checksums_path = write_checksums(
            route_dir / "checksums.sha256",
            [
                manifest_path,
                predictions_path,
                capacity_path,
                invariants_path,
            ],
            root=route_dir,
        )
        atomic_write_json(
            staging / "route_anchor.json",
            {
                "schema_version": "saded-paired-route-anchor/v1",
                "route_manifest_sha256": sha256_file(manifest_path),
                "route_checksums_sha256": sha256_file(checksums_path),
                "baseline_cache_anchor_sha256": _digest(
                    args.baseline_anchor_sha256
                ),
                "treatment_cache_anchor_sha256": _digest(
                    args.treatment_anchor_sha256
                ),
                "training_protocol_sha256": baseline_manifest[
                    "training_protocol"
                ]["sha256"],
                "training_source_commit": baseline_manifest[
                    "training_protocol"
                ]["source_commit"],
                "evaluation_source_commit": evaluation_source["commit"],
            },
        )
        shutil.move(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_root


def main() -> None:
    print(route_pair(build_parser().parse_args()))


if __name__ == "__main__":
    main()
