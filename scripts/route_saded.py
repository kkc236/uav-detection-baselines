#!/usr/bin/env python3
"""Seal a prediction-only SADED route-control replay without loading GT."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.route_sbr_ppaf import (  # noqa: E402
    _assert_frozen,
    _inside_or_equal,
    _iter_jsonl_gz,
    _load_frozen_arms,
    _parse_raw,
    _prediction_payload,
    _same_source_state,
    _source_state,
    validate_route_input,
)
from src.saded import (  # noqa: E402
    CONF_THRESHOLD,
    FRAGMENT_IOS,
    LARGE_EFFECTIVE_SIZE,
    MATCH_IOU,
    MAX_DET,
    ROUTER_K,
    TINY_EFFECTIVE_SIZE,
    ExpertCandidate,
    route_saded_image,
)
from src.sbr_artifacts import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl_gz,
    sha256_file,
    write_checksums,
)
from src.sbr_v2_audit import (  # noqa: E402
    group_relevant_raw_rows,
    map_full_a_to_c,
    reconstruct_c_clusters,
)


ROUTE_SCHEMA_VERSION = "sbr-saded-route/v1"
ROUTE_ARTIFACTS = (
    "route_manifest.json",
    "predictions.jsonl.gz",
    "capacity.json",
    "route_invariants.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal a GT-free SADED route-control replay"
    )
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _aggregate_capacity(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    remaining = [
        int(row["coverage"]["remaining_tiny_slots"])
        for row in rows
    ]
    protected = [
        int(row["coverage"]["protected_baseline"])
        for row in rows
    ]
    appended = [
        int(row["coverage"]["accepted_local"])
        for row in rows
    ]
    capacity_rejected = [
        int(row["coverage"]["capacity_rejected"])
        for row in rows
    ]

    def summary(values: Sequence[int]) -> dict[str, int | float]:
        return {
            "min": min(values) if values else 0,
            "median": statistics.median(values) if values else 0,
            "max": max(values) if values else 0,
            "total": sum(values),
        }

    return {
        "image_count": len(rows),
        "per_image": [
            {
                "image_id": row["image_id"],
                **dict(row["coverage"]),
            }
            for row in rows
        ],
        "protected_baseline": summary(protected),
        "remaining_tiny_slots": summary(remaining),
        "accepted_local": summary(appended),
        "capacity_rejected": summary(capacity_rejected),
    }


def route_replay(
    input_manifest: Path | str,
    output: Path | str,
    *,
    require_clean: bool = True,
) -> Path:
    """Authenticate, route, and atomically seal one SADED R0 closure."""

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
        resolved = Path(input_path).resolve()
        if _inside_or_equal(output_root, resolved) or _inside_or_equal(
            resolved,
            output_root,
        ):
            raise ValueError("output overlaps an input")

    frozen = _load_frozen_arms(
        validated.paths["arm_predictions"],
        validated.image_list,
    )
    route_rows: list[dict[str, Any]] = []
    grouped = group_relevant_raw_rows(
        _iter_jsonl_gz(validated.paths["raw_views"]),
        validated.image_list,
    )
    for group in grouped:
        parsed = tuple(
            _parse_raw(row, image_id=group.image_id)
            for row in group.rows
        )
        a_raw = tuple(item for item in parsed if item.arm == "A")
        c_raw = tuple(item for item in parsed if item.arm == "C")
        if not a_raw:
            raise ValueError(f"Arm A is empty for {group.image_id}")
        width = a_raw[0].width
        height = a_raw[0].height
        if any(
            item.width != width or item.height != height
            for item in parsed
        ):
            raise ValueError("raw image dimensions disagree")
        a_predictions = tuple(item.to_detection() for item in a_raw)
        map_full_a_to_c(a_raw, c_raw)
        c_reconstruction = reconstruct_c_clusters(c_raw)
        _assert_frozen(
            "A",
            group.image_id,
            tuple(
                row
                for row in group.rows
                if row.get("arm") == "A"
            ),
            a_predictions,
            frozen["A"][group.image_id],
        )
        _assert_frozen(
            "C",
            group.image_id,
            tuple(
                row
                for row in group.rows
                if row.get("arm") == "C"
            ),
            c_reconstruction.standard_predictions,
            frozen["C"][group.image_id],
        )
        baseline = tuple(
            ExpertCandidate(
                detection=detection,
                image_id=group.image_id,
                original_index=index,
            )
            for index, detection in enumerate(a_predictions)
        )
        local = tuple(
            ExpertCandidate(
                detection=detection,
                image_id=group.image_id,
                original_index=index,
            )
            for index, detection in enumerate(
                c_reconstruction.standard_predictions
            )
        )
        result = route_saded_image(
            image_id=group.image_id,
            width=width,
            height=height,
            baseline=baseline,
            local_fused=local,
        )
        route_rows.append(
            {
                "image_id": group.image_id,
                "width": width,
                "height": height,
                "arms": {
                    "A": [
                        _prediction_payload(item)
                        for item in a_predictions
                    ],
                    "route_control": [
                        _prediction_payload(item)
                        for item in result.predictions
                    ],
                },
                "coverage": dict(result.coverage),
                "invariants": dict(result.invariants),
            }
        )

    if tuple(row["image_id"] for row in route_rows) != (
        validated.image_list
    ):
        raise ValueError("route image order disagrees with manifest")
    invariants = {
        "image_count": len(route_rows),
        "expected_image_count": len(validated.image_list),
        "image_order_exact": tuple(
            row["image_id"] for row in route_rows
        )
        == validated.image_list,
        "per_image_passed": all(
            row["invariants"].get("passed") is True
            for row in route_rows
        ),
        "gt_module_absent": (
            "src.sbr_metrics" not in sys.modules or not require_clean
        ),
    }
    invariants["passed"] = (
        invariants["image_count"]
        == invariants["expected_image_count"]
        and invariants["image_order_exact"] is True
        and invariants["per_image_passed"] is True
        and invariants["gt_module_absent"] is True
    )
    if invariants["passed"] is not True:
        raise ValueError("route invariants failed")
    capacity = _aggregate_capacity(route_rows)

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
        capacity_path = atomic_write_json(
            route_dir / "capacity.json",
            capacity,
        )
        invariants_path = atomic_write_json(
            route_dir / "route_invariants.json",
            invariants,
        )
        after_source = _source_state(require_clean=require_clean)
        if not _same_source_state(before_source, after_source):
            raise ValueError("source state changed during routing")
        manifest = {
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
                "tiny_effective_size": TINY_EFFECTIVE_SIZE,
                "large_effective_size": LARGE_EFFECTIVE_SIZE,
                "match_iou_strictly_greater_than": MATCH_IOU,
                "fragment_ios": FRAGMENT_IOS,
                "router_k": ROUTER_K,
            },
            "arms": ["A", "route_control"],
            "required_artifacts": list(ROUTE_ARTIFACTS)
            + ["checksums.sha256"],
            "predictions_sha256": sha256_file(predictions_path),
            "capacity_sha256": sha256_file(capacity_path),
            "route_invariants_sha256": sha256_file(invariants_path),
        }
        manifest_path = atomic_write_json(
            route_dir / "route_manifest.json",
            manifest,
        )
        checksum_path = write_checksums(
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
                "schema_version": "sbr-saded-route-anchor/v1",
                "route_checksums_sha256": sha256_file(checksum_path),
                "route_manifest_sha256": sha256_file(manifest_path),
                "predictions_sha256": sha256_file(predictions_path),
                "input_manifest_sha256": validated.manifest_sha256,
            },
        )
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
        print(f"SADED_ROUTE_INVALID: {exc}", file=sys.stderr)
        return 2
    print("SADED_ROUTE_SEALED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
