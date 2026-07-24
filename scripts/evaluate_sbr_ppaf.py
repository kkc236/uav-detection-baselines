#!/usr/bin/env python3
"""Evaluate one checksum-sealed SP-PPAF route in a separate GT-aware process."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gzip
import json
import math
from numbers import Real
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sbr_artifacts import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    git_provenance,
    sha256_bytes,
    sha256_file,
    write_checksums,
)
from src.sbr_ppaf import (  # noqa: E402
    A_FLOOR,
    C_CEILING,
    CONF_THRESHOLD,
    FRAGMENT_IOS,
    LARGE_EFFECTIVE_SIZE,
    MAX_DET,
    decide_ppaf,
    metric_deltas,
)
from scripts.route_sbr_ppaf import (  # noqa: E402
    ROUTE_ARTIFACTS,
    ROUTE_SCHEMA_VERSION,
    ValidatedRouteInput,
    validate_route_input,
)


EVALUATION_SCHEMA_VERSION = "sbr-sp-ppaf-evaluation/v1"
EVALUATION_ARTIFACTS = (
    "evaluation_manifest.json",
    "metrics.json",
    "deltas.json",
    "evaluation_invariants.json",
    "primary_gate.json",
)
ROUTE_ROW_KEYS = {
    "image_id",
    "width",
    "height",
    "arms",
    "eligible_clusters",
    "selected_cluster_ranks",
    "coverage",
    "invariants",
}
PREDICTION_KEYS = {
    "box",
    "global_xyxy",
    "score",
    "class_id",
    "source_order",
    "query_index",
}
EVALUATED_ARMS = ("A", "C", "All-A", "P1", "P2", "P3")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a sealed SP-PPAF prediction route"
    )
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--route", required=True, type=Path)
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


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"blank prediction row: {line_number}")
                value = json.loads(line, parse_constant=_reject_constant)
                if not isinstance(value, dict):
                    raise ValueError(f"prediction row is not an object: {line_number}")
                rows.append(value)
    except (EOFError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid prediction JSONL: {path}") from exc
    return rows


def _inside_or_equal(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _snapshot(paths: Sequence[Path], *, root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in sorted(paths, key=lambda item: item.as_posix()):
        resolved = path.resolve()
        metadata = resolved.stat()
        label = (
            resolved.relative_to(root).as_posix()
            if _inside_or_equal(resolved, root)
            else resolved.as_posix()
        )
        result[label] = {
            "size": metadata.st_size,
            "mode": stat.S_IMODE(metadata.st_mode),
            "sha256": sha256_file(resolved),
        }
    return result


def _snapshot_digest(snapshot: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(snapshot)))


def _parse_checksum_file(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        parts = line.split("  ", 1)
        if (
            len(parts) != 2
            or len(parts[0]) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in parts[0])
        ):
            raise ValueError(f"invalid route checksum line: {line_number}")
        label = Path(parts[1])
        if label.is_absolute() or ".." in label.parts:
            raise ValueError("route checksum path escapes closure")
        normalized = label.as_posix()
        if normalized in checksums:
            raise ValueError("duplicate route checksum label")
        checksums[normalized] = parts[0].lower()
    return checksums


def _strict_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _box(value: object, name: str) -> list[float]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be an xyxy sequence")
    try:
        result = [float(item) for item in value]  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an xyxy sequence") from None
    if (
        len(result) != 4
        or not all(math.isfinite(item) for item in result)
        or result[2] <= result[0]
        or result[3] <= result[1]
    ):
        raise ValueError(f"{name} is not a finite nondegenerate box")
    return result


def _validate_prediction(
    value: object,
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != PREDICTION_KEYS:
        raise ValueError(f"{name} prediction schema is not exact")
    score = value.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, Real)
        or not math.isfinite(float(score))
        or not CONF_THRESHOLD <= float(score) <= 1.0
    ):
        raise ValueError(f"{name}.score is invalid")
    return {
        "box": _box(value.get("box"), f"{name}.box"),
        "global_xyxy": _box(
            value.get("global_xyxy"),
            f"{name}.global_xyxy",
        ),
        "score": float(score),
        "class_id": _strict_int(value.get("class_id"), f"{name}.class_id"),
        "source_order": _strict_int(
            value.get("source_order"),
            f"{name}.source_order",
        ),
        "query_index": _strict_int(
            value.get("query_index"),
            f"{name}.query_index",
        ),
    }


def _validate_route_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    validated: ValidatedRouteInput,
    coverage: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if len(rows) != len(validated.image_list):
        raise ValueError("route row count disagrees with image list")
    normalized: list[dict[str, Any]] = []
    coverage_rows = coverage.get("per_image")
    if not isinstance(coverage_rows, list) or len(coverage_rows) != len(rows):
        raise ValueError("coverage per-image rows are incomplete")
    for index, (row, coverage_row) in enumerate(zip(rows, coverage_rows)):
        if set(row) != ROUTE_ROW_KEYS:
            raise ValueError("route row schema is not exact")
        image_id = validated.image_list[index]
        if row.get("image_id") != image_id:
            raise ValueError("route row image order disagrees")
        width = _strict_int(row.get("width"), "route width")
        height = _strict_int(row.get("height"), "route height")
        if not width or not height:
            raise ValueError("route dimensions must be positive")
        arms = row.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != set(EVALUATED_ARMS):
            raise ValueError("route arm schema is not exact")
        normalized_arms: dict[str, list[dict[str, Any]]] = {}
        for arm in EVALUATED_ARMS:
            arm_rows = arms[arm]
            if isinstance(arm_rows, (str, bytes, Mapping)):
                raise ValueError(f"{arm} predictions must be a sequence")
            normalized_arms[arm] = [
                _validate_prediction(item, name=f"{image_id}.{arm}[{item_index}]")
                for item_index, item in enumerate(arm_rows)
            ]
            if len(normalized_arms[arm]) > MAX_DET:
                raise ValueError(f"{arm} exceeds max_det")
        row_coverage = row.get("coverage")
        if (
            not isinstance(row_coverage, Mapping)
            or not isinstance(coverage_row, Mapping)
            or coverage_row.get("image_id") != image_id
            or coverage_row.get("coverage") != row_coverage
        ):
            raise ValueError("coverage.json disagrees with route row")
        for arm in ("A", "All-A", "P1", "P2", "P3"):
            arm_coverage = row_coverage.get(arm)
            if not isinstance(arm_coverage, Mapping):
                raise ValueError("route arm coverage is missing")
            if arm_coverage.get("output") != len(normalized_arms[arm]):
                raise ValueError("route output count disagrees with coverage")
            prefix = _strict_int(arm_coverage.get("prefix"), "prefix")
            appended = _strict_int(arm_coverage.get("appended"), "appended")
            if prefix + appended != len(normalized_arms[arm]):
                raise ValueError("route prefix/appended arithmetic disagrees")
        a = normalized_arms["A"]
        if normalized_arms["All-A"][: len(a)] != a:
            raise ValueError("All-A prefix is not exact Arm A")
        p_prefix = int(row_coverage["P3"]["prefix"])
        if not (
            normalized_arms["P1"][:p_prefix]
            == normalized_arms["P2"][:p_prefix]
            == normalized_arms["P3"][:p_prefix]
        ):
            raise ValueError("P1/P2/P3 prefixes disagree")
        if not isinstance(row.get("invariants"), Mapping) or (
            row["invariants"].get("passed") is not True
        ):
            raise ValueError("per-image route invariants failed")
        normalized.append(
            {
                "image_id": image_id,
                "width": width,
                "height": height,
                "arms": normalized_arms,
            }
        )
    return normalized


def _verify_route_closure(
    input_manifest: Path | str,
    route: Path | str,
) -> tuple[
    ValidatedRouteInput,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    route_dir = Path(route).resolve()
    if not route_dir.is_dir():
        raise FileNotFoundError(route_dir)
    names = {item.name for item in route_dir.iterdir()}
    expected_names = set(ROUTE_ARTIFACTS) | {"checksums.sha256"}
    if names != expected_names:
        raise ValueError("route closure artifact set is not exact")
    anchor_path = route_dir.parent / "route_anchor.json"
    if not anchor_path.is_file():
        raise ValueError("route external anchor is missing")
    route_paths = [route_dir / name for name in sorted(expected_names)]
    snapshot_before = _snapshot(
        [*route_paths, anchor_path],
        root=route_dir.parent,
    )

    checksum_path = route_dir / "checksums.sha256"
    checksums = _parse_checksum_file(checksum_path)
    if set(checksums) != set(ROUTE_ARTIFACTS):
        raise ValueError("route checksum target set is not exact")
    for label, expected in checksums.items():
        if sha256_file(route_dir / label) != expected:
            raise ValueError(f"route checksum mismatch: {label}")
    anchor = _read_json(anchor_path)
    if not isinstance(anchor, Mapping) or set(anchor) != {
        "schema_version",
        "route_checksums_sha256",
        "route_manifest_sha256",
        "predictions_sha256",
        "input_manifest_sha256",
    }:
        raise ValueError("route anchor schema is not exact")
    if (
        anchor.get("schema_version") != "sbr-sp-ppaf-route-anchor/v1"
        or anchor.get("route_checksums_sha256") != sha256_file(checksum_path)
        or anchor.get("route_manifest_sha256")
        != sha256_file(route_dir / "route_manifest.json")
        or anchor.get("predictions_sha256")
        != sha256_file(route_dir / "predictions.jsonl.gz")
        or anchor.get("input_manifest_sha256")
        != sha256_file(input_manifest)
    ):
        raise ValueError("route anchor binding failed")

    route_manifest = _read_json(route_dir / "route_manifest.json")
    route_invariants = _read_json(route_dir / "route_invariants.json")
    coverage = _read_json(route_dir / "coverage.json")
    if (
        not isinstance(route_manifest, Mapping)
        or route_manifest.get("schema_version") != ROUTE_SCHEMA_VERSION
        or route_manifest.get("input_manifest_sha256")
        != sha256_file(input_manifest)
        or route_manifest.get("predictions_sha256")
        != sha256_file(route_dir / "predictions.jsonl.gz")
        or route_manifest.get("coverage_sha256")
        != sha256_file(route_dir / "coverage.json")
        or route_manifest.get("route_invariants_sha256")
        != sha256_file(route_dir / "route_invariants.json")
    ):
        raise ValueError("route manifest binding failed")
    expected_constants = {
        "conf": CONF_THRESHOLD,
        "max_det": MAX_DET,
        "large_effective_size": LARGE_EFFECTIVE_SIZE,
        "fragment_ios": FRAGMENT_IOS,
        "a_floor": A_FLOOR,
        "c_ceiling": C_CEILING,
    }
    if route_manifest.get("constants") != expected_constants:
        raise ValueError("route constants drifted")
    if (
        not isinstance(route_invariants, Mapping)
        or route_invariants.get("passed") is not True
        or not isinstance(coverage, Mapping)
    ):
        raise ValueError("route invariants/coverage are invalid")

    validated = validate_route_input(input_manifest)
    if (
        route_manifest.get("input_file_sha256") != dict(validated.hashes)
        or route_manifest.get("dataset_signature")
        != validated.dataset_signature
        or route_manifest.get("image_count") != len(validated.image_list)
    ):
        raise ValueError("route/input provenance binding failed")
    raw_rows = _read_jsonl_gz(route_dir / "predictions.jsonl.gz")
    normalized = _validate_route_rows(
        raw_rows,
        validated=validated,
        coverage=coverage,
    )
    snapshot_verified = _snapshot(
        [*route_paths, anchor_path],
        root=route_dir.parent,
    )
    if snapshot_verified != snapshot_before:
        raise ValueError("route snapshot changed during verification")
    return (
        validated,
        normalized,
        dict(route_manifest),
        dict(route_invariants),
        snapshot_before,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _strict_recursive_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_strict_recursive_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_strict_recursive_equal(a, b) for a, b in zip(left, right))
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


def _metric_row(
    image: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    frozen_global: bool,
) -> dict[str, Any]:
    return {
        "image_id": image["relative_path"],
        "width": int(image["width"]),
        "height": int(image["height"]),
        "pred_boxes": [
            prediction["global_xyxy"] if frozen_global else prediction["box"]
            for prediction in predictions
        ],
        "pred_scores": [prediction["score"] for prediction in predictions],
        "pred_classes": [prediction["class_id"] for prediction in predictions],
        "pred_source": [prediction["source_order"] for prediction in predictions],
        "pred_query": [prediction["query_index"] for prediction in predictions],
        "gt_boxes": [list(box) for box in image["gt_boxes"]],
        "gt_classes": [int(item) for item in image["gt_classes"]],
        "ignore_boxes": [list(box) for box in image["ignore_boxes"]],
        "effective_gain": min(
            640.0 / float(image["width"]),
            640.0 / float(image["height"]),
            1.0,
        ),
    }


def _source_state(*, require_clean: bool) -> dict[str, Any]:
    state = git_provenance(REPO_ROOT)
    if require_clean and (
        state.get("clean_tracked") is not True
        or state.get("untracked") is not False
    ):
        raise ValueError("evaluation source worktree must be fully clean")
    return state


def _same_source_state(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = ("commit", "source_tree_hash", "clean_tracked", "untracked")
    return all(left.get(key) == right.get(key) for key in keys)


def evaluate_replay(
    input_manifest: Path | str,
    route: Path | str,
    output: Path | str,
    *,
    require_clean: bool = True,
) -> Path:
    """Verify a route closure, then load labels and evaluate it exactly once."""

    before_source = _source_state(require_clean=require_clean)
    (
        validated,
        route_rows,
        route_manifest,
        route_invariants,
        route_snapshot,
    ) = _verify_route_closure(input_manifest, route)
    route_dir = Path(route).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError("evaluation output must not exist")
    if _inside_or_equal(output_path, route_dir) or _inside_or_equal(
        route_dir, output_path
    ):
        raise ValueError("evaluation output overlaps route closure")

    # GT-aware imports and annotation reads are intentionally delayed until
    # every route checksum, anchor, schema, invariant, and input binding passed.
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
        raise ValueError("loaded dataset disagrees with sealed input")
    image_by_id = {
        image["relative_path"]: image for image in dataset["images"]
    }
    evaluator_rows: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in EVALUATED_ARMS
    }
    for route_row in route_rows:
        image = image_by_id.get(route_row["image_id"])
        if image is None:
            raise ValueError("route image is absent from loaded dataset")
        if (
            int(image["width"]) != route_row["width"]
            or int(image["height"]) != route_row["height"]
        ):
            raise ValueError("route/dataset dimensions disagree")
        for arm in EVALUATED_ARMS:
            evaluator_rows[arm].append(
                _metric_row(
                    image,
                    route_row["arms"][arm],
                    frozen_global=arm in {"A", "C"},
                )
            )

    metrics = {
        arm: _jsonable(evaluate_dataset(evaluator_rows[arm]))
        for arm in EVALUATED_ARMS
    }
    a_reproduced = _strict_recursive_equal(
        metrics["A"],
        validated.g0_metrics["A"],
    )
    c_reproduced = _strict_recursive_equal(
        metrics["C"],
        validated.g0_metrics["C"],
    )
    route_snapshot_after = _snapshot(
        [
            *(route_dir / name for name in sorted(set(ROUTE_ARTIFACTS) | {"checksums.sha256"})),
            route_dir.parent / "route_anchor.json",
        ],
        root=route_dir.parent,
    )
    source_after = _source_state(require_clean=require_clean)
    invariants = {
        "route_checksum_and_anchor_verified": True,
        "route_invariants_passed": route_invariants.get("passed") is True,
        "input_manifest_bound": (
            route_manifest.get("input_manifest_sha256")
            == validated.manifest_sha256
        ),
        "dataset_signature_exact": (
            dataset["dataset_signature"] == validated.dataset_signature
        ),
        "image_order_exact": tuple(dataset["image_list"])
        == validated.image_list,
        "arm_a_baseline_reproduced": a_reproduced,
        "arm_c_baseline_reproduced": c_reproduced,
        "route_snapshot_unchanged": route_snapshot_after == route_snapshot,
        "source_state_unchanged": _same_source_state(
            before_source,
            source_after,
        ),
        "single_row_set_per_arm": all(
            len(rows) == len(validated.image_list)
            for rows in evaluator_rows.values()
        ),
    }
    invariants["passed"] = all(invariants.values())
    decision = decide_ppaf(
        metrics["A"],
        metrics["P3"],
        metrics["All-A"],
        invariants_passed=invariants["passed"],
    )
    deltas = {
        arm: metric_deltas(metrics[arm], metrics["A"])
        for arm in ("C", "All-A", "P1", "P2", "P3")
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.evaluation-staging-",
            dir=output_path.parent,
        )
    )
    try:
        metrics_path = atomic_write_json(staging / "metrics.json", metrics)
        deltas_path = atomic_write_json(staging / "deltas.json", deltas)
        invariants_path = atomic_write_json(
            staging / "evaluation_invariants.json",
            invariants,
        )
        gate_path = atomic_write_json(
            staging / "primary_gate.json",
            decision,
        )
        evaluation_manifest = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "input_manifest_sha256": validated.manifest_sha256,
            "route_checksums_sha256": sha256_file(
                route_dir / "checksums.sha256"
            ),
            "route_manifest_sha256": sha256_file(
                route_dir / "route_manifest.json"
            ),
            "route_predictions_sha256": sha256_file(
                route_dir / "predictions.jsonl.gz"
            ),
            "route_anchor_sha256": sha256_file(
                route_dir.parent / "route_anchor.json"
            ),
            "route_snapshot_sha256": _snapshot_digest(route_snapshot),
            "route_snapshot_after_sha256": _snapshot_digest(
                route_snapshot_after
            ),
            "source": source_after,
            "dataset_signature": validated.dataset_signature,
            "image_count": len(validated.image_list),
            "selected_arm": decision["selected_arm"],
            "status": decision["status"],
            "required_artifacts": list(EVALUATION_ARTIFACTS)
            + ["checksums.sha256"],
        }
        manifest_path = atomic_write_json(
            staging / "evaluation_manifest.json",
            evaluation_manifest,
        )
        write_checksums(
            staging / "checksums.sha256",
            [
                manifest_path,
                metrics_path,
                deltas_path,
                invariants_path,
                gate_path,
            ],
            root=staging,
        )
        staging.rename(output_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = evaluate_replay(
            args.input_manifest,
            args.route,
            args.output,
        )
        gate = _read_json(output / "primary_gate.json")
    except Exception as exc:
        print(f"SP_PPAF_INVALID: {exc}", file=sys.stderr)
        return 2
    print(gate["status"])
    return 2 if gate["status"] == "SP_PPAF_INVALID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
