#!/usr/bin/env python3
"""Verify and evaluate one sealed SADED R0 route in a GT-aware process."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_sbr_ppaf import (  # noqa: E402
    _jsonable,
    _metric_row,
    _parse_checksum_file,
    _read_jsonl_gz,
    _snapshot,
    _strict_recursive_equal,
)
from scripts.route_saded import (  # noqa: E402
    ROUTE_ARTIFACTS,
    ROUTE_SCHEMA_VERSION,
)
from scripts.route_sbr_ppaf import (  # noqa: E402
    _same_source_state,
    _source_state,
    validate_route_input,
)
from src.sbr_artifacts import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    write_checksums,
)
from src.sbr_ppaf import metric_deltas  # noqa: E402


EVALUATION_SCHEMA_VERSION = "sbr-saded-r0-evaluation/v1"
EVALUATION_ARTIFACTS = (
    "evaluation_manifest.json",
    "metrics.json",
    "deltas.json",
    "capacity.json",
    "evaluation_invariants.json",
    "r0_gate.json",
)
PREDICTION_KEYS = {
    "box",
    "global_xyxy",
    "score",
    "class_id",
    "source_order",
    "query_index",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one sealed SADED R0 route"
    )
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--route", required=True, type=Path)
    parser.add_argument("--route-anchor-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def adjudicate_r0(
    *,
    deltas: Mapping[str, float],
    aggregate_remaining_slots: int,
    invariants_passed: bool,
) -> dict[str, Any]:
    """Apply only the pre-registered SADED router safety thresholds."""

    failures: list[str] = []
    if not invariants_passed:
        return {
            "schema_version": "sbr-saded-r0-gate/v1",
            "decision": "INVALID",
            "failures": ["evaluation_invariants_failed"],
        }
    if float(deltas["AP75"]) < -0.002:
        failures.append("AP75_delta<-0.002")
    if float(deltas["AP-large-SBR"]) < -0.005:
        failures.append("AP-large-SBR_delta<-0.005")
    if int(aggregate_remaining_slots) <= 0:
        failures.append("aggregate_remaining_tiny_slots<=0")
    return {
        "schema_version": "sbr-saded-r0-gate/v1",
        "decision": "R0_GO" if not failures else "R0_STOP",
        "failures": failures,
        "thresholds": {
            "AP75": -0.002,
            "AP-large-SBR": -0.005,
            "aggregate_remaining_tiny_slots": 1,
        },
    }


def _verify_route(
    input_manifest: Path | str,
    route: Path | str,
    route_anchor_sha256: str,
) -> tuple[
    Any,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    route_dir = Path(route).resolve()
    expected_names = set(ROUTE_ARTIFACTS) | {"checksums.sha256"}
    if not route_dir.is_dir() or {
        item.name for item in route_dir.iterdir()
    } != expected_names:
        raise ValueError("route closure artifact set is not exact")
    anchor_path = route_dir.parent / "route_anchor.json"
    if (
        not anchor_path.is_file()
        or sha256_file(anchor_path) != route_anchor_sha256.lower()
    ):
        raise ValueError("route external anchor checksum mismatch")
    paths = [route_dir / name for name in sorted(expected_names)]
    snapshot = _snapshot(
        [*paths, anchor_path],
        root=route_dir.parent,
    )
    checksums = _parse_checksum_file(
        route_dir / "checksums.sha256"
    )
    if set(checksums) != set(ROUTE_ARTIFACTS):
        raise ValueError("route checksum target set is not exact")
    for label, expected in checksums.items():
        if sha256_file(route_dir / label) != expected:
            raise ValueError(f"route checksum mismatch: {label}")
    anchor = _read_json(anchor_path)
    if (
        anchor.get("schema_version") != "sbr-saded-route-anchor/v1"
        or anchor.get("route_checksums_sha256")
        != sha256_file(route_dir / "checksums.sha256")
        or anchor.get("route_manifest_sha256")
        != sha256_file(route_dir / "route_manifest.json")
        or anchor.get("predictions_sha256")
        != sha256_file(route_dir / "predictions.jsonl.gz")
        or anchor.get("input_manifest_sha256")
        != sha256_file(input_manifest)
    ):
        raise ValueError("route anchor binding failed")
    manifest = _read_json(route_dir / "route_manifest.json")
    capacity = _read_json(route_dir / "capacity.json")
    route_invariants = _read_json(
        route_dir / "route_invariants.json"
    )
    if (
        manifest.get("schema_version") != ROUTE_SCHEMA_VERSION
        or manifest.get("input_manifest_sha256")
        != sha256_file(input_manifest)
        or manifest.get("predictions_sha256")
        != sha256_file(route_dir / "predictions.jsonl.gz")
        or manifest.get("capacity_sha256")
        != sha256_file(route_dir / "capacity.json")
        or manifest.get("route_invariants_sha256")
        != sha256_file(route_dir / "route_invariants.json")
    ):
        raise ValueError("route manifest binding failed")
    if route_invariants.get("passed") is not True:
        raise ValueError("route invariants are not closed")
    validated = validate_route_input(input_manifest)
    if (
        manifest.get("input_file_sha256") != dict(validated.hashes)
        or manifest.get("dataset_signature")
        != validated.dataset_signature
        or manifest.get("image_count") != len(validated.image_list)
    ):
        raise ValueError("route/input provenance binding failed")
    rows = _read_jsonl_gz(route_dir / "predictions.jsonl.gz")
    if len(rows) != len(validated.image_list):
        raise ValueError("route prediction count mismatch")
    for index, row in enumerate(rows):
        if (
            row.get("image_id") != validated.image_list[index]
            or set(row.get("arms", {})) != {"A", "route_control"}
        ):
            raise ValueError("route prediction identity mismatch")
        for predictions in row["arms"].values():
            if any(
                not isinstance(prediction, Mapping)
                or set(prediction) != PREDICTION_KEYS
                for prediction in predictions
            ):
                raise ValueError("route prediction schema drift")
    if _snapshot([*paths, anchor_path], root=route_dir.parent) != snapshot:
        raise ValueError("route changed during verification")
    return validated, rows, manifest, capacity, snapshot


def evaluate_replay(
    input_manifest: Path | str,
    route: Path | str,
    route_anchor_sha256: str,
    output: Path | str,
    *,
    require_clean: bool = True,
) -> Path:
    """Verify first, then load GT once and seal one R0 decision."""

    before_source = _source_state(require_clean=require_clean)
    validated, route_rows, route_manifest, capacity, route_snapshot = (
        _verify_route(
            input_manifest,
            route,
            route_anchor_sha256,
        )
    )
    if not _same_source_state(
        before_source,
        route_manifest["route_source"],
    ):
        raise ValueError("evaluation source does not match route source")
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError("evaluation output must not exist")
    anchor_path = output_path.parent / (
        f"{output_path.name}_anchor.json"
    )
    if anchor_path.exists():
        raise FileExistsError("evaluation anchor must not exist")

    # GT-aware imports and annotation reads are delayed until route closure,
    # source, checksum, schema, and external-anchor verification is complete.
    from src.sbr_artifacts import load_dataset
    from src.sbr_metrics import evaluate_dataset

    dataset_spec = validated.manifest["dataset"]
    dataset = load_dataset(
        validated.paths["dataset_yaml"],
        split=dataset_spec.get("split", "val"),
        root_override=validated.dataset_root,
    )
    if (
        dataset["dataset_signature"] != validated.dataset_signature
        or tuple(dataset["image_list"]) != validated.image_list
    ):
        raise ValueError("loaded dataset disagrees with sealed route input")
    image_by_id = {
        image["relative_path"]: image
        for image in dataset["images"]
    }
    evaluator_rows = {"A": [], "route_control": []}
    for route_row in route_rows:
        image = image_by_id[route_row["image_id"]]
        if (
            int(image["width"]) != int(route_row["width"])
            or int(image["height"]) != int(route_row["height"])
        ):
            raise ValueError("route/dataset dimensions disagree")
        for arm in evaluator_rows:
            evaluator_rows[arm].append(
                _metric_row(
                    image,
                    route_row["arms"][arm],
                    frozen_global=True,
                )
            )
    metrics = {
        arm: _jsonable(evaluate_dataset(rows))
        for arm, rows in evaluator_rows.items()
    }
    sealed_g0_metrics = _read_json(validated.paths["g0_metrics"])
    arm_a_reproduced = _strict_recursive_equal(
        metrics["A"],
        sealed_g0_metrics["A"],
    )
    deltas = metric_deltas(metrics["route_control"], metrics["A"])
    route_dir = Path(route).resolve()
    route_paths = [
        route_dir / name
        for name in sorted(set(ROUTE_ARTIFACTS) | {"checksums.sha256"})
    ]
    route_snapshot_after = _snapshot(
        [*route_paths, route_dir.parent / "route_anchor.json"],
        root=route_dir.parent,
    )
    after_source = _source_state(require_clean=require_clean)
    invariants = {
        "route_snapshot_unchanged": (
            route_snapshot_after == route_snapshot
        ),
        "source_state_unchanged": _same_source_state(
            before_source,
            after_source,
        ),
        "dataset_signature_exact": (
            dataset["dataset_signature"]
            == validated.dataset_signature
        ),
        "image_order_exact": (
            tuple(dataset["image_list"])
            == validated.image_list
        ),
        "arm_a_baseline_reproduced": arm_a_reproduced,
        "single_row_set_per_arm": all(
            len(rows) == len(validated.image_list)
            for rows in evaluator_rows.values()
        ),
    }
    invariants["passed"] = all(invariants.values())
    gate = adjudicate_r0(
        deltas=deltas,
        aggregate_remaining_slots=int(
            capacity["remaining_tiny_slots"]["total"]
        ),
        invariants_passed=invariants["passed"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.evaluation-staging-",
            dir=output_path.parent,
        )
    )
    try:
        metrics_path = atomic_write_json(
            staging / "metrics.json",
            metrics,
        )
        deltas_path = atomic_write_json(
            staging / "deltas.json",
            deltas,
        )
        capacity_path = atomic_write_json(
            staging / "capacity.json",
            capacity,
        )
        invariants_path = atomic_write_json(
            staging / "evaluation_invariants.json",
            invariants,
        )
        gate_path = atomic_write_json(
            staging / "r0_gate.json",
            gate,
        )
        manifest_path = atomic_write_json(
            staging / "evaluation_manifest.json",
            {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "input_manifest_sha256": validated.manifest_sha256,
                "route_checksums_sha256": sha256_file(
                    route_dir / "checksums.sha256"
                ),
                "route_anchor_sha256": sha256_file(
                    route_dir.parent / "route_anchor.json"
                ),
                "route_snapshot_verified": True,
                "source": after_source,
                "dataset_signature": validated.dataset_signature,
                "image_count": len(validated.image_list),
                "decision": gate["decision"],
                "required_artifacts": list(EVALUATION_ARTIFACTS)
                + ["checksums.sha256"],
            },
        )
        checksums_path = write_checksums(
            staging / "checksums.sha256",
            [
                manifest_path,
                metrics_path,
                deltas_path,
                capacity_path,
                invariants_path,
                gate_path,
            ],
            root=staging,
        )
        atomic_write_json(
            anchor_path,
            {
                "schema_version": "sbr-saded-r0-anchor/v1",
                "evaluation_checksums_sha256": sha256_file(
                    checksums_path
                ),
                "evaluation_manifest_sha256": sha256_file(
                    manifest_path
                ),
                "route_anchor_sha256": route_anchor_sha256.lower(),
                "decision": gate["decision"],
            },
        )
        staging.rename(output_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if anchor_path.exists():
            anchor_path.unlink()
        raise
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = evaluate_replay(
            args.input_manifest,
            args.route,
            args.route_anchor_sha256,
            args.output,
        )
        gate = _read_json(output / "r0_gate.json")
    except Exception as exc:
        print(f"SADED_R0_INVALID: {exc}", file=sys.stderr)
        return 2
    print(gate["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
