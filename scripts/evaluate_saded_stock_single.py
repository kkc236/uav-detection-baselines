#!/usr/bin/env python3
"""Evaluate one sealed fresh-stock SADED route after a one-shot claim."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.route_saded_stock_single import (  # noqa: E402
    ROUTE_ARTIFACTS,
    ROUTE_SCHEMA,
    _verify_checksums,
)
from src.saded_stock_evaluation_protocol import (  # noqa: E402
    postprocess_source_state,
    reject_forbidden,
    validate_evaluation_protocol,
)
from src.sbr_artifacts import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    sha256_file,
    write_checksums,
)
from src.sbr_ppaf import metric_deltas  # noqa: E402


EVALUATION_SCHEMA = "saded-fresh-stock-single-evaluation/v1"
EVALUATION_ARTIFACTS = {
    "evaluation_manifest.json",
    "metrics.json",
    "deltas.json",
    "evaluation_invariants.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a sealed fresh-stock single route."
    )
    parser.add_argument(
        "--evaluation-protocol",
        required=True,
        type=Path,
    )
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return [json.loads(line) for line in handle]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _snapshot(paths: Sequence[Path]) -> dict[str, str]:
    return {
        path.resolve().as_posix(): sha256_file(path)
        for path in paths
    }


def create_evaluation_claim(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Create the immutable claim before any GT-aware import or read."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(canonical_json_bytes(dict(payload)))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


def metric_row(
    image: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one metric row without selecting metrics by scale."""

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


def evaluation_invariants_passed(
    invariants: Mapping[str, Any],
) -> bool:
    return bool(invariants) and all(
        value is True for value in invariants.values()
    )


def _verify_route(
    protocol: dict[str, Any],
    protocol_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    route = Path(protocol["outputs"]["route"]).resolve()
    anchor_path = route.parent / "route_anchor.json"
    expected = ROUTE_ARTIFACTS | {"checksums.sha256"}
    if (
        not route.is_dir()
        or {item.name for item in route.iterdir()} != expected
        or not anchor_path.is_file()
    ):
        raise ValueError("fresh route artifact set drift")
    checksums = _verify_checksums(route, ROUTE_ARTIFACTS)
    manifest = _read_json(route / "route_manifest.json")
    invariants = _read_json(route / "route_invariants.json")
    anchor = _read_json(anchor_path)
    capacity = _read_json(route / "capacity.json")
    if (
        manifest.get("schema_version") != ROUTE_SCHEMA
        or manifest.get("evaluation_protocol_sha256")
        != sha256_file(protocol_path)
        or manifest.get("source") != protocol["source"]
        or manifest.get("checkpoint")
        != protocol["training"]["checkpoint"]
        or manifest.get("route_contract") != protocol["route_contract"]
        or manifest.get("arms") != ["A", "route_control"]
        or manifest.get("image_count") != 548
        or manifest.get("predictions_sha256")
        != checksums["predictions.jsonl.gz"]
        or manifest.get("capacity_sha256")
        != checksums["capacity.json"]
        or manifest.get("invariants_sha256")
        != checksums["route_invariants.json"]
        or invariants.get("passed") is not True
        or invariants.get("image_count") != 548
        or capacity.get("image_count") != 548
        or anchor.get("schema_version")
        != "saded-fresh-stock-single-route-anchor/v1"
        or anchor.get("route_checksums_sha256")
        != sha256_file(route / "checksums.sha256")
        or anchor.get("route_manifest_sha256")
        != checksums["route_manifest.json"]
        or anchor.get("predictions_sha256")
        != checksums["predictions.jsonl.gz"]
        or anchor.get("evaluation_protocol_sha256")
        != sha256_file(protocol_path)
    ):
        raise ValueError("fresh route nested closure drift")
    image_list = _read_json(
        Path(
            protocol["protocol_artifacts"]["image_list"]["path"]
        ).resolve()
    )
    rows = _read_jsonl_gz(route / "predictions.jsonl.gz")
    if (
        len(rows) != 548
        or [row.get("image_id") for row in rows] != image_list
        or any(set(row.get("arms", {})) != {"A", "route_control"} for row in rows)
    ):
        raise ValueError("fresh route image or arm identity drift")
    paths = [anchor_path, *(route / name for name in expected)]
    return rows, capacity, _snapshot(paths)


def evaluate(args: argparse.Namespace) -> Path:
    reject_forbidden(vars(args))
    protocol_path = args.evaluation_protocol.resolve()
    protocol = validate_evaluation_protocol(
        protocol_path,
        repo_root=REPO_ROOT,
        verify_images=False,
    )
    output = Path(protocol["outputs"]["evaluation"]).resolve()
    anchor_path = output.parent / "evaluation_anchor.json"
    claim_path = Path(protocol["outputs"]["evaluation_claim"]).resolve()
    if output.exists() or anchor_path.exists() or claim_path.exists():
        raise FileExistsError("fresh evaluation target or claim exists")
    source_before = postprocess_source_state(REPO_ROOT)
    route_rows, capacity, route_snapshot = _verify_route(
        protocol,
        protocol_path,
    )
    claim_payload = {
        "schema_version": "saded-fresh-stock-evaluation-claim/v1",
        "state": "CONSUMED",
        "evaluation_protocol_sha256": sha256_file(protocol_path),
        "route_anchor_sha256": sha256_file(
            Path(protocol["outputs"]["route"]).resolve().parent
            / "route_anchor.json"
        ),
        "route_snapshot": route_snapshot,
        "retry_permitted": False,
    }
    create_evaluation_claim(claim_path, claim_payload)
    claim_sha256 = sha256_file(claim_path)

    # GT-aware imports and annotation reads occur only after the immutable
    # claim and complete route verification above.
    from src.sbr_artifacts import load_dataset
    from src.sbr_metrics import evaluate_dataset

    dataset = load_dataset(
        Path(protocol["dataset"]["yaml"]),
        split="val",
        root_override=Path(protocol["dataset"]["root"]),
    )
    image_list = _read_json(
        Path(
            protocol["protocol_artifacts"]["image_list"]["path"]
        ).resolve()
    )
    if (
        dataset["image_count"] != 548
        or dataset["image_list"] != image_list
        or dataset["dataset_signature"]
        != protocol["dataset"]["signature"]
    ):
        raise ValueError("fresh evaluation dataset authority drift")
    image_by_id = {
        image["relative_path"]: image for image in dataset["images"]
    }
    metric_rows: dict[str, list[dict[str, Any]]] = {
        "A": [],
        "route_control": [],
    }
    for row in route_rows:
        image = image_by_id[row["image_id"]]
        if (
            int(image["width"]) != int(row["width"])
            or int(image["height"]) != int(row["height"])
        ):
            raise ValueError("fresh route/dataset dimension drift")
        for arm in metric_rows:
            metric_rows[arm].append(
                metric_row(image, row["arms"][arm])
            )
    metrics = {
        arm: _jsonable(evaluate_dataset(rows))
        for arm, rows in metric_rows.items()
    }
    deltas = metric_deltas(metrics["route_control"], metrics["A"])
    route_path = Path(protocol["outputs"]["route"]).resolve()
    route_paths = [
        route_path.parent / "route_anchor.json",
        *(
            route_path / name
            for name in ROUTE_ARTIFACTS | {"checksums.sha256"}
        ),
    ]
    invariants = {
        "claim_created_before_gt_import": True,
        "claim_is_immutable_consumed": (
            claim_path.is_file()
            and sha256_file(claim_path) == claim_sha256
        ),
        "route_snapshot_unchanged": (
            _snapshot(route_paths) == route_snapshot
        ),
        "source_unchanged": (
            postprocess_source_state(REPO_ROOT) == source_before
        ),
        "dataset_signature_exact": (
            dataset["dataset_signature"]
            == protocol["dataset"]["signature"]
        ),
        "image_order_exact": dataset["image_list"] == image_list,
        "two_metric_sets_exact": set(metrics) == {"A", "route_control"},
        "single_row_set_per_arm": all(
            len(rows) == 548 for rows in metric_rows.values()
        ),
        "retry_forbidden": True,
    }
    invariants["passed"] = evaluation_invariants_passed(invariants)
    if not invariants["passed"]:
        raise ValueError("fresh evaluation invariants failed")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".evaluation-staging-",
            dir=output.parent,
        )
    )
    try:
        metrics_path = atomic_write_json(staging / "metrics.json", metrics)
        deltas_path = atomic_write_json(staging / "deltas.json", deltas)
        invariants_path = atomic_write_json(
            staging / "evaluation_invariants.json",
            invariants,
        )
        manifest_path = atomic_write_json(
            staging / "evaluation_manifest.json",
            {
                "schema_version": EVALUATION_SCHEMA,
                "evaluation_protocol_sha256": sha256_file(protocol_path),
                "source": source_before,
                "route": {
                    "path": route_path.as_posix(),
                    "anchor_sha256": sha256_file(
                        route_path.parent / "route_anchor.json"
                    ),
                    "snapshot": route_snapshot,
                },
                "claim": {
                    "path": claim_path.as_posix(),
                    "sha256": claim_sha256,
                    "retry_permitted": False,
                },
                "dataset": protocol["dataset"],
                "checkpoint": protocol["training"]["checkpoint"],
                "arms": ["A", "route_control"],
                "image_count": 548,
                "capacity": capacity,
                "artifacts": {
                    "metrics_sha256": sha256_file(metrics_path),
                    "deltas_sha256": sha256_file(deltas_path),
                    "invariants_sha256": sha256_file(invariants_path),
                },
                "required_artifacts": sorted(
                    EVALUATION_ARTIFACTS | {"checksums.sha256"}
                ),
            },
        )
        checksums_path = write_checksums(
            staging / "checksums.sha256",
            [
                manifest_path,
                metrics_path,
                deltas_path,
                invariants_path,
            ],
            root=staging,
        )
        staging.rename(output)
        atomic_write_json(
            anchor_path,
            {
                "schema_version": (
                    "saded-fresh-stock-single-evaluation-anchor/v1"
                ),
                "evaluation_checksums_sha256": sha256_file(
                    output / checksums_path.name
                ),
                "evaluation_manifest_sha256": sha256_file(
                    output / manifest_path.name
                ),
                "metrics_sha256": sha256_file(
                    output / metrics_path.name
                ),
                "route_anchor_sha256": sha256_file(
                    route_path.parent / "route_anchor.json"
                ),
                "claim_sha256": claim_sha256,
                "evaluation_protocol_sha256": sha256_file(protocol_path),
            },
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)
        if anchor_path.exists():
            anchor_path.unlink()
        raise
    return output


def main() -> None:
    print(evaluate(build_parser().parse_args()))


if __name__ == "__main__":
    main()
