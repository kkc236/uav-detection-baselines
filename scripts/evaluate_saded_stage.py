#!/usr/bin/env python3
"""Verify a paired SADED route, then evaluate it in a GT-aware process."""

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

from scripts.route_saded_pair import (  # noqa: E402
    ROUTE_FILES,
    _iter_jsonl_gz,
    _parse_checksums,
    _read_json,
    _snapshot,
    _verify_cache_root,
)
from src.saded_stage import (  # noqa: E402
    PREDICTION_KEYS,
    ROUTE_ARMS,
    route_paired_caches,
)
from src.saded_stage_protocol import stage_source_state  # noqa: E402
from src.sbr_artifacts import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    write_checksums,
)
from src.sbr_ppaf import metric_deltas  # noqa: E402
from src.tascv_protocol import reject_forbidden_path  # noqa: E402


EVALUATION_FILES = (
    "evaluation_manifest.json",
    "metrics.json",
    "deltas.json",
    "capacity.json",
    "evaluation_invariants.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one checksum-sealed paired SADED route."
    )
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--route-anchor-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _three_way_deltas(
    metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    if set(metrics) != set(ROUTE_ARMS):
        raise ValueError("SADED evaluation metric arm drift")
    return {
        "route_control_vs_A": metric_deltas(
            metrics["route_control"],
            metrics["A"],
        ),
        "route_treatment_vs_A": metric_deltas(
            metrics["route_treatment"],
            metrics["A"],
        ),
        "route_treatment_vs_route_control": metric_deltas(
            metrics["route_treatment"],
            metrics["route_control"],
        ),
    }


def _verify_route(
    route_root: Path,
    *,
    expected_anchor_sha256: str,
    evaluation_source: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, str],
]:
    root = reject_forbidden_path(
        route_root,
        context="SADED evaluation route root",
    )
    if (
        not root.is_dir()
        or {path.name for path in root.iterdir()}
        != {"route", "route_anchor.json"}
    ):
        raise ValueError("SADED route root closure drift")
    anchor_path = root / "route_anchor.json"
    expected_anchor = str(expected_anchor_sha256).lower()
    if (
        len(expected_anchor) != 64
        or sha256_file(anchor_path) != expected_anchor
    ):
        raise ValueError("SADED route external anchor drift")
    route_dir = root / "route"
    expected_files = set(ROUTE_FILES) | {"checksums.sha256"}
    if (
        not route_dir.is_dir()
        or {path.name for path in route_dir.iterdir()} != expected_files
    ):
        raise ValueError("SADED route artifact closure drift")
    route_paths = [
        anchor_path,
        *(route_dir / name for name in expected_files),
    ]
    route_snapshot = _snapshot(route_paths)
    checksums_path = route_dir / "checksums.sha256"
    checksums = _parse_checksums(checksums_path)
    if set(checksums) != set(ROUTE_FILES):
        raise ValueError("SADED route checksum closure drift")
    for name, digest in checksums.items():
        if sha256_file(route_dir / name) != digest:
            raise ValueError(f"SADED route checksum drift: {name}")
    anchor = _read_json(anchor_path)
    if (
        set(anchor)
        != {
            "schema_version",
            "route_manifest_sha256",
            "route_checksums_sha256",
            "baseline_cache_anchor_sha256",
            "treatment_cache_anchor_sha256",
            "training_protocol_sha256",
            "training_source_commit",
            "evaluation_source_commit",
        }
        or anchor["schema_version"]
        != "saded-paired-route-anchor/v1"
        or anchor["route_manifest_sha256"]
        != sha256_file(route_dir / "route_manifest.json")
        or anchor["route_checksums_sha256"]
        != sha256_file(checksums_path)
        or anchor["evaluation_source_commit"]
        != evaluation_source["commit"]
    ):
        raise ValueError("SADED route anchor binding drift")
    manifest = _read_json(route_dir / "route_manifest.json")
    if (
        manifest.get("schema_version") != "saded-paired-route/v1"
        or manifest.get("required_artifacts")
        != list(ROUTE_FILES) + ["checksums.sha256"]
        or manifest.get("evaluation_source")
        != dict(evaluation_source)
        or manifest.get("artifacts")
        != {
            "predictions_sha256": sha256_file(
                route_dir / "predictions.jsonl.gz"
            ),
            "capacity_sha256": sha256_file(
                route_dir / "capacity.json"
            ),
            "invariants_sha256": sha256_file(
                route_dir / "route_invariants.json"
            ),
        }
        or anchor["training_protocol_sha256"]
        != manifest["training_protocol"]["sha256"]
        or anchor["training_source_commit"]
        != manifest["training_protocol"]["source_commit"]
    ):
        raise ValueError("SADED route manifest binding drift")
    cache_inputs = manifest.get("cache_inputs")
    if not isinstance(cache_inputs, dict) or set(cache_inputs) != {
        "baseline",
        "treatment",
    }:
        raise ValueError("SADED route cache input schema drift")
    input_paths: list[Path] = []
    cache_manifests: dict[str, dict[str, Any]] = {}
    cache_rows: dict[str, list[dict[str, Any]]] = {}
    for name in ("baseline", "treatment"):
        binding = cache_inputs[name]
        cache_manifest, verified_cache_rows, paths = _verify_cache_root(
            Path(binding["root"]),
            expected_anchor_sha256=binding["anchor_sha256"],
            evaluation_source=evaluation_source,
        )
        if (
            binding["manifest_sha256"]
            != sha256_file(
                Path(binding["root"])
                / "cache/cache_manifest.json"
            )
            or binding["checkpoint"]
            != cache_manifest["checkpoint"]
            or anchor[f"{name}_cache_anchor_sha256"]
            != binding["anchor_sha256"]
        ):
            raise ValueError("SADED route cache binding drift")
        cache_manifests[name] = cache_manifest
        cache_rows[name] = verified_cache_rows
        input_paths.extend(paths)
    if _snapshot(input_paths) != manifest.get("input_snapshot"):
        raise ValueError("SADED route input snapshot drift")
    route_invariants = _read_json(
        route_dir / "route_invariants.json"
    )
    capacity = _read_json(route_dir / "capacity.json")
    expected_image_count = int(manifest["dataset"]["image_count"])
    if (
        route_invariants.get("passed") is not True
        or route_invariants.get("image_count")
        != expected_image_count
        or capacity.get("image_count") != expected_image_count
    ):
        raise ValueError("SADED route invariants did not pass")
    rows = _iter_jsonl_gz(route_dir / "predictions.jsonl.gz")
    image_list_path = Path(manifest["dataset"]["image_list"]).resolve()
    image_list = json.loads(image_list_path.read_text(encoding="utf-8"))
    if (
        len(rows) != expected_image_count
        or [row.get("image_id") for row in rows] != image_list
    ):
        raise ValueError("SADED route image identity drift")
    for row in rows:
        if (
            set(row)
            != {
                "image_id",
                "width",
                "height",
                "arms",
                "coverage",
                "invariants",
            }
            or set(row["arms"]) != set(ROUTE_ARMS)
            or any(
                len(predictions) > 300
                or any(
                    not isinstance(prediction, Mapping)
                    or set(prediction) != PREDICTION_KEYS
                    for prediction in predictions
                )
                for predictions in row["arms"].values()
            )
        ):
            raise ValueError("SADED route prediction schema drift")
    replayed_rows, replayed_invariants = route_paired_caches(
        cache_rows["baseline"],
        cache_rows["treatment"],
    )
    from scripts.route_saded_pair import _aggregate_capacity

    if (
        replayed_rows != rows
        or _aggregate_capacity(replayed_rows) != capacity
        or any(
            route_invariants.get(key) != value
            for key, value in replayed_invariants.items()
            if key != "passed"
        )
    ):
        raise ValueError("SADED sealed route replay drift")
    if _snapshot(route_paths) != route_snapshot:
        raise ValueError("SADED route changed during verification")
    return manifest, rows, capacity, route_snapshot


def _metric_row(
    image: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "image_id": image["relative_path"],
        "width": int(image["width"]),
        "height": int(image["height"]),
        "pred_boxes": [prediction["box"] for prediction in predictions],
        "pred_scores": [
            prediction["score"] for prediction in predictions
        ],
        "pred_classes": [
            prediction["class_id"] for prediction in predictions
        ],
        "pred_source": [
            prediction["source_order"] for prediction in predictions
        ],
        "pred_query": [
            prediction["query_index"] for prediction in predictions
        ],
        "gt_boxes": [list(box) for box in image["gt_boxes"]],
        "gt_classes": [int(item) for item in image["gt_classes"]],
        "ignore_boxes": [list(box) for box in image["ignore_boxes"]],
        "effective_gain": min(
            640.0 / float(image["width"]),
            640.0 / float(image["height"]),
            1.0,
        ),
    }


def evaluate_stage(args: argparse.Namespace) -> Path:
    for value in (args.route_root, args.output):
        reject_forbidden_path(value, context="SADED stage evaluation")
    output_root = args.output.resolve()
    if output_root.exists():
        raise FileExistsError("SADED evaluation output already exists")
    evaluation_source = stage_source_state(REPO_ROOT)
    route_manifest, route_rows, capacity, route_snapshot = _verify_route(
        args.route_root.resolve(),
        expected_anchor_sha256=args.route_anchor_sha256,
        evaluation_source=evaluation_source,
    )
    if (
        route_manifest["dataset"].get("scope", "development")
        != "development"
        or int(route_manifest["dataset"]["image_count"]) != 548
    ):
        raise ValueError("SADED evaluator accepts only development val")
    route_root = args.route_root.resolve()
    route_paths = [
        route_root / "route_anchor.json",
        *(
            route_root / "route" / name
            for name in set(ROUTE_FILES) | {"checksums.sha256"}
        ),
    ]
    training_protocol_path = Path(
        route_manifest["training_protocol"]["path"]
    ).resolve()
    training_protocol = _read_json(training_protocol_path)
    r0_manifest_path = Path(
        training_protocol["r0_authority"]["evaluation_manifest"]
    ).resolve()
    if (
        sha256_file(r0_manifest_path)
        != str(
            training_protocol["r0_authority"][
                "evaluation_manifest_sha256"
            ]
        ).lower()
    ):
        raise ValueError("SADED R0 dataset authority drift")
    r0_manifest = _read_json(r0_manifest_path)
    full_yaml_path = Path(
        training_protocol["dataset"]["full_yaml"]
    ).resolve()
    if (
        not full_yaml_path.is_file()
        or sha256_file(full_yaml_path)
        != str(
            training_protocol["dataset"]["full_yaml_sha256"]
        ).lower()
    ):
        raise ValueError("SADED full dataset YAML drift")

    # GT-aware imports and annotation reads are deliberately delayed until
    # every route/cache/source/checksum/snapshot verification above passes.
    from src.sbr_artifacts import load_dataset
    from src.sbr_metrics import evaluate_dataset

    dataset = load_dataset(
        full_yaml_path,
        split="val",
        root_override=Path(training_protocol["dataset"]["root"]),
    )
    image_list = json.loads(
        Path(route_manifest["dataset"]["image_list"]).read_text(
            encoding="utf-8"
        )
    )
    if (
        dataset["image_count"] != 548
        or dataset["image_list"] != image_list
        or dataset["dataset_signature"]
        != r0_manifest["dataset_signature"]
    ):
        raise ValueError("SADED loaded dataset identity drift")
    image_by_id = {
        image["relative_path"]: image for image in dataset["images"]
    }
    metric_rows: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in ROUTE_ARMS
    }
    for row in route_rows:
        image = image_by_id[row["image_id"]]
        if (
            int(image["width"]) != int(row["width"])
            or int(image["height"]) != int(row["height"])
        ):
            raise ValueError("SADED route/dataset dimension drift")
        for arm in ROUTE_ARMS:
            metric_rows[arm].append(
                _metric_row(image, row["arms"][arm])
            )
    metrics = {
        arm: _jsonable(evaluate_dataset(rows))
        for arm, rows in metric_rows.items()
    }
    deltas = _three_way_deltas(metrics)
    invariants = {
        "route_snapshot_unchanged": (
            _snapshot(route_paths) == route_snapshot
        ),
        "evaluation_source_unchanged": (
            stage_source_state(REPO_ROOT) == evaluation_source
        ),
        "dataset_signature_exact": (
            dataset["dataset_signature"]
            == r0_manifest["dataset_signature"]
        ),
        "image_order_exact": dataset["image_list"] == image_list,
        "single_row_set_per_arm": all(
            len(rows) == 548 for rows in metric_rows.values()
        ),
        "three_delta_sets_exact": set(deltas)
        == {
            "route_control_vs_A",
            "route_treatment_vs_A",
            "route_treatment_vs_route_control",
        },
    }
    invariants["passed"] = all(invariants.values())
    if not invariants["passed"]:
        raise ValueError("SADED evaluation invariants failed")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.evaluation-staging-",
            dir=output_root.parent,
        )
    )
    try:
        evaluation_dir = staging / "evaluation"
        evaluation_dir.mkdir()
        metrics_path = atomic_write_json(
            evaluation_dir / "metrics.json",
            metrics,
        )
        deltas_path = atomic_write_json(
            evaluation_dir / "deltas.json",
            deltas,
        )
        capacity_path = atomic_write_json(
            evaluation_dir / "capacity.json",
            capacity,
        )
        invariants_path = atomic_write_json(
            evaluation_dir / "evaluation_invariants.json",
            invariants,
        )
        manifest_path = atomic_write_json(
            evaluation_dir / "evaluation_manifest.json",
            {
                "schema_version": "saded-stage-evaluation/v1",
                "evaluation_source": evaluation_source,
                "identity": route_manifest["identity"],
                "training_protocol": route_manifest[
                    "training_protocol"
                ],
                "route": {
                    "root": route_root.as_posix(),
                    "anchor_sha256": str(
                        args.route_anchor_sha256
                    ).lower(),
                    "manifest_sha256": sha256_file(
                        route_root / "route/route_manifest.json"
                    ),
                    "snapshot": route_snapshot,
                },
                "dataset": {
                    "root": dataset["root"].as_posix(),
                    "yaml_path": dataset["yaml_path"].as_posix(),
                    "yaml_hash": dataset["yaml_hash"],
                    "dataset_signature": dataset[
                        "dataset_signature"
                    ],
                    "image_count": dataset["image_count"],
                },
                "artifacts": {
                    "metrics_sha256": sha256_file(metrics_path),
                    "deltas_sha256": sha256_file(deltas_path),
                    "capacity_sha256": sha256_file(capacity_path),
                    "invariants_sha256": sha256_file(invariants_path),
                },
                "required_artifacts": list(EVALUATION_FILES)
                + ["checksums.sha256"],
            },
        )
        checksums_path = write_checksums(
            evaluation_dir / "checksums.sha256",
            [
                manifest_path,
                metrics_path,
                deltas_path,
                capacity_path,
                invariants_path,
            ],
            root=evaluation_dir,
        )
        atomic_write_json(
            staging / "evaluation_anchor.json",
            {
                "schema_version": "saded-stage-evaluation-anchor/v1",
                "evaluation_manifest_sha256": sha256_file(
                    manifest_path
                ),
                "evaluation_checksums_sha256": sha256_file(
                    checksums_path
                ),
                "route_anchor_sha256": str(
                    args.route_anchor_sha256
                ).lower(),
                "training_protocol_sha256": route_manifest[
                    "training_protocol"
                ]["sha256"],
                "training_source_commit": route_manifest[
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
    print(evaluate_stage(build_parser().parse_args()))


if __name__ == "__main__":
    main()
