"""Fail-closed primary runner for the frozen SBR score oracle."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, is_dataclass
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import sys
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.audit_sbr_v2 import (
    _assert_frozen_predictions,
    _assert_frozen_records,
    _iter_jsonl_gz,
    _load_frozen_arm_predictions,
    _metric_row,
    _parse_raw_detection,
    _strict_recursive_equal,
    validate_input_manifest,
)
from scripts.prepare_sbr_score_oracle_protocol import (
    SCHEMA_VERSION,
    capture_clean_source,
    frozen_rule_payload,
)
from src.sbr_artifacts import (
    atomic_write_json,
    atomic_write_jsonl_gz,
    canonical_json_bytes,
    environment_info,
    load_dataset,
    sha256_bytes,
    sha256_file,
    write_checksums,
)
from src.sbr_metrics import evaluate_dataset
from src.sbr_score_oracle import (
    GATES,
    THRESHOLDS,
    OracleImage,
    evaluate_oracle_image,
    gate_oracle_metrics,
    verify_oracle_image_invariants,
)
from src.sbr_v2_audit import group_relevant_raw_rows


EXPECTED_IMAGE_COUNT = 548
ORACLE_SCHEMA_VERSION = "sbr-score-oracle-evidence/v1"
ORACLE_SCHEMA = {
    "schema_version": ORACLE_SCHEMA_VERSION,
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


def _peak_rss_bytes() -> int:
    """Return the process peak resident memory, failing closed."""

    try:
        import resource

        value = int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
        result = value if sys.platform == "darwin" else value * 1024
    except Exception:
        try:
            import psutil

            memory = psutil.Process().memory_info()
            result = int(
                getattr(memory, "peak_wset", memory.rss)
            )
        except Exception as exc:
            raise RuntimeError(
                "peak RSS measurement is unavailable"
            ) from exc
    if result < 0:
        raise RuntimeError("peak RSS measurement is negative")
    return result


def _entry(
    value: Any,
    *,
    base: Path,
    name: str,
) -> tuple[Path, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"uri", "sha256"}
    ):
        raise ValueError(f"{name} entry is invalid")
    uri = value["uri"]
    digest = str(value["sha256"]).lower()
    if not isinstance(uri, str) or not uri:
        raise ValueError(f"{name} URI is invalid")
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} SHA256 is invalid")
    target = Path(uri)
    if not target.is_absolute():
        target = base / target
    target = target.resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    if sha256_file(target).lower() != digest:
        raise ValueError(f"{name} checksum mismatch")
    return target, digest


def validate_oracle_wrapper(
    manifest_path: Path,
    spec_path: Path,
    *,
    source: Mapping[str, str],
) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    spec = Path(spec_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not spec.is_file():
        raise FileNotFoundError(spec)
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("oracle wrapper is not valid JSON") from exc
    if not isinstance(wrapper, dict):
        raise ValueError("oracle wrapper must be an object")
    if wrapper.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("oracle wrapper schema is invalid")
    upstream, upstream_hash = _entry(
        wrapper.get("upstream_input"),
        base=path.parent,
        name="upstream input",
    )
    sealed_spec, spec_hash = _entry(
        wrapper.get("approved_spec"),
        base=path.parent,
        name="approved spec",
    )
    if sealed_spec != spec or spec_hash != sha256_file(spec):
        raise ValueError("approved spec path/hash mismatch")
    expected_source = wrapper.get("expected_source")
    if (
        not isinstance(expected_source, Mapping)
        or dict(expected_source) != dict(source)
    ):
        raise ValueError("oracle source commit/tree mismatch")
    if wrapper.get("frozen_rule") != frozen_rule_payload():
        raise ValueError("oracle frozen rule mismatch")
    if wrapper.get("forbidden_inputs") != [
        "test-dev",
        "external-dataset",
    ]:
        raise ValueError("oracle forbidden-input declaration mismatch")
    try:
        upstream_payload = json.loads(
            upstream.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ValueError("upstream manifest is not valid JSON") from exc
    dataset = (
        upstream_payload.get("dataset")
        if isinstance(upstream_payload, Mapping)
        else None
    )
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("split") != "val"
    ):
        raise ValueError("oracle upstream split must be exact val")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "payload": wrapper,
        "upstream_path": upstream,
        "upstream_sha256": upstream_hash,
        "spec_path": spec,
        "spec_sha256": spec_hash,
        "source": dict(source),
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
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


SHARD_SCHEMA_VERSION = "sbr-score-oracle-shard/v1"
SHARD_KEYS = {
    "schema_version",
    "run_identity",
    "image_order",
    "image_id",
    "input_image_hash",
    "payload",
    "payload_hash",
}


def build_shard(
    *,
    run_identity: str,
    image_order: int,
    image_id: str,
    input_image_hash: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _jsonable(dict(payload))
    return {
        "schema_version": SHARD_SCHEMA_VERSION,
        "run_identity": str(run_identity),
        "image_order": int(image_order),
        "image_id": str(image_id),
        "input_image_hash": str(input_image_hash),
        "payload": normalized,
        "payload_hash": sha256_bytes(
            canonical_json_bytes(normalized)
        ),
    }


def validate_shard(
    shard: Any,
    *,
    run_identity: str,
    image_order: int,
    image_id: str,
    input_image_hash: str,
) -> dict[str, Any]:
    if not isinstance(shard, Mapping) or set(shard) != SHARD_KEYS:
        raise ValueError("shard schema fields are invalid")
    if shard.get("schema_version") != SHARD_SCHEMA_VERSION:
        raise ValueError("shard schema version is invalid")
    expected = {
        "run_identity": run_identity,
        "image_order": image_order,
        "image_id": image_id,
        "input_image_hash": input_image_hash,
    }
    if any(shard.get(key) != value for key, value in expected.items()):
        raise ValueError("shard manifest identity mismatch")
    payload = shard.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("shard payload must be an object")
    expected_hash = sha256_bytes(canonical_json_bytes(payload))
    if shard.get("payload_hash") != expected_hash:
        raise ValueError("shard payload hash mismatch")
    return dict(payload)


def validate_complete_shards(
    entries: Any,
    *,
    image_ids: tuple[str, ...],
    input_hashes: tuple[str, ...],
) -> tuple[Mapping[str, Any], ...]:
    rows = tuple(entries)
    if len(rows) != len(image_ids):
        raise ValueError("shard set is not complete")
    orders = [
        row.get("image_order")
        if isinstance(row, Mapping)
        else None
        for row in rows
    ]
    if len(set(orders)) != len(orders):
        raise ValueError("duplicate shard image order")
    expected_orders = tuple(range(len(image_ids)))
    if tuple(sorted(orders)) != expected_orders:
        raise ValueError("shard orders are not continuous")
    by_order = {
        int(row["image_order"]): row
        for row in rows
        if isinstance(row, Mapping)
    }
    ordered = tuple(by_order[index] for index in expected_orders)
    for index, row in enumerate(ordered):
        if (
            row.get("image_id") != image_ids[index]
            or row.get("input_image_hash") != input_hashes[index]
        ):
            raise ValueError("shard image identity mismatch")
    return ordered


def _metric_delta(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, float]:
    names = tuple(GATES)
    return {
        name: float(right[name]) - float(left[name])
        for name in names
    }


def _prediction_digest(predictions: Any) -> str:
    rows = []
    for original_index, prediction in enumerate(predictions):
        rows.append(
            {
                "box": list(prediction.box),
                "global_xyxy": list(prediction.global_xyxy),
                "score": float(prediction.score),
                "class_id": int(prediction.class_id),
                "source_order": int(prediction.source_order),
                "query_index": int(prediction.query_index),
                "original_index": original_index,
            }
        )
    return sha256_bytes(canonical_json_bytes(rows))


def _sequence_token(image_id: str) -> str:
    return Path(image_id).name.split("_", 1)[0]


def _seal_primary(primary: Path) -> None:
    for path in sorted(
        primary.rglob("*"),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(primary, 0o555)


def _input_image_hash(
    *,
    group: Any,
    image: Mapping[str, Any],
) -> str:
    payload = {
        "image_id": group.image_id,
        "width": int(image["width"]),
        "height": int(image["height"]),
        "gt_boxes": _jsonable(image["gt_boxes"]),
        "gt_classes": _jsonable(image["gt_classes"]),
        "ignore_boxes": _jsonable(image["ignore_boxes"]),
        "raw_rows": [
            {
                key: value
                for key, value in row.items()
                if key != "_audit_original_index"
            }
            for row in group.rows
        ],
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _evaluate_image_task(task: Mapping[str, Any]) -> dict[str, Any]:
    order = int(task["image_order"])
    group = task["group"]
    image = task["image"]
    a_frozen = task["a_frozen"]
    c_frozen = task["c_frozen"]
    parsed = tuple(
        _parse_raw_detection(
            row, expected_image_id=group.image_id
        )
        for row in group.rows
    )
    a_raw = tuple(item for item in parsed if item.arm == "A")
    c_raw = tuple(item for item in parsed if item.arm == "C")
    a_source_rows = tuple(
        row for row in group.rows if row.get("arm") == "A"
    )
    c_source_rows = tuple(
        row for row in group.rows if row.get("arm") == "C"
    )
    _assert_frozen_records(
        "A", group.image_id, a_source_rows, a_frozen
    )
    _assert_frozen_records(
        "C", group.image_id, c_source_rows, c_frozen
    )
    oracle_image = OracleImage(
        image_id=group.image_id,
        width=int(image["width"]),
        height=int(image["height"]),
        gt_boxes=tuple(
            tuple(float(value) for value in box)
            for box in image["gt_boxes"]
        ),
        gt_classes=tuple(
            int(value) for value in image["gt_classes"]
        ),
        ignore_boxes=tuple(
            tuple(float(value) for value in box)
            for box in image["ignore_boxes"]
        ),
        a_raw=a_raw,
        c_raw=c_raw,
    )
    result = evaluate_oracle_image(oracle_image)
    a_predictions = tuple(item.to_detection() for item in a_raw)
    c_predictions = result.stock.reconstruction.standard_predictions
    o_predictions = result.joint.reconstruction.standard_predictions
    _assert_frozen_predictions(
        "A", group.image_id, a_predictions, a_frozen
    )
    _assert_frozen_predictions(
        "C", group.image_id, c_predictions, c_frozen
    )
    selected = tuple(
        event for event in result.events if event.selected
    )
    by_index = {item.original_index: item for item in c_raw}
    events = []
    by_class: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_sequence: Counter[str] = Counter()
    for event in result.events:
        event_row = _jsonable(event)
        event_row["image_order"] = order
        events.append(event_row)
        anchor = by_index[event.group.full_anchor_index]
        by_class[str(anchor.class_id)] += 1
        for index in event.group.aggressor_indices:
            by_source[str(by_index[index].source_order)] += 1
        by_sequence[_sequence_token(group.image_id)] += 1
    patches = []
    for patch in result.joint.patches:
        patch_row = _jsonable(patch)
        patch_row["image_order"] = order
        patches.append(patch_row)
    verified = verify_oracle_image_invariants(
        oracle_image, result
    )
    invariant = {
        "image_order": order,
        "image_id": group.image_id,
        "input_image_hash": task["input_image_hash"],
        "a_prediction_digest": _prediction_digest(a_predictions),
        "c_prediction_digest": _prediction_digest(c_predictions),
        "selection_rounds": result.selection_rounds,
        "eligible_units": len(result.groups),
        "selected_units": len(selected),
        **verified,
    }
    return {
        "image_order": order,
        "image_id": group.image_id,
        "input_image_hash": task["input_image_hash"],
        "a_metric_row": _metric_row(
            image, a_predictions, frozen_global_xyxy=True
        ),
        "c_metric_row": _metric_row(
            image, c_predictions, frozen_global_xyxy=True
        ),
        "oracle_metric_row": _metric_row(
            image, o_predictions, frozen_global_xyxy=True
        ),
        "events": events,
        "patches": patches,
        "invariant": invariant,
        "peak_rss_bytes": _peak_rss_bytes(),
        "coverage": {
            "eligible_units": len(result.groups),
            "selected_units": len(selected),
            "eligible_members": sum(
                len(item.aggressor_indices)
                for item in result.groups
            ),
            "patched_members": len(result.joint.patches),
            "affected": bool(selected),
            "large_positive_affected": bool(selected)
            and result.stock_profile["0.50"]["large"]["gt"] > 0,
            "by_class": dict(by_class),
            "by_source": dict(by_source),
            "by_sequence_token": dict(by_sequence),
        },
    }


def _read_shard(path: Path) -> dict[str, Any]:
    try:
        with gzip.open(
            path, "rt", encoding="utf-8", newline=""
        ) as handle:
            rows = [
                json.loads(line)
                for line in handle
                if line.strip()
            ]
    except Exception as exc:
        raise ValueError(f"invalid shard: {path.name}") from exc
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(f"shard must contain one object: {path.name}")
    return rows[0]


def _write_shard(path: Path, shard: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    atomic_write_jsonl_gz(temporary, [dict(shard)])
    os.replace(temporary, path)


def _execute_tasks(
    tasks: tuple[Mapping[str, Any], ...],
    *,
    run_identity: str,
    staging: Path,
    workers: int,
) -> tuple[dict[str, Any], ...]:
    identity_payload = {
        "schema_version": "sbr-score-oracle-run-identity/v1",
        "run_identity": run_identity,
        "image_count": len(tasks),
    }
    identity_path = staging / "run_identity.json"
    shards_dir = staging / "shards"
    if staging.exists():
        allowed = {"run_identity.json", "shards"}
        if {path.name for path in staging.iterdir()} != allowed:
            raise ValueError("staging contains unknown entries")
        existing_identity = json.loads(
            identity_path.read_text(encoding="utf-8")
        )
        if existing_identity != identity_payload:
            raise ValueError("staging run identity mismatch")
        if not shards_dir.is_dir():
            raise ValueError("staging shards directory is missing")
    else:
        staging.mkdir(parents=True)
        shards_dir.mkdir()
        atomic_write_json(identity_path, identity_payload)
    known_names = {
        f"{index:06d}.json.gz" for index in range(len(tasks))
    }
    existing_names = {path.name for path in shards_dir.iterdir()}
    if not existing_names <= known_names:
        raise ValueError("staging contains unknown shard")
    by_order: dict[int, dict[str, Any]] = {}
    missing: list[Mapping[str, Any]] = []
    for task in tasks:
        order = int(task["image_order"])
        path = shards_dir / f"{order:06d}.json.gz"
        if path.exists():
            shard = _read_shard(path)
            validate_shard(
                shard,
                run_identity=run_identity,
                image_order=order,
                image_id=str(task["group"].image_id),
                input_image_hash=str(task["input_image_hash"]),
            )
            by_order[order] = shard
        else:
            missing.append(task)
    if workers > 1 and missing:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            payloads = executor.map(_evaluate_image_task, missing)
            computed = zip(missing, payloads)
            for task, payload in computed:
                order = int(task["image_order"])
                shard = build_shard(
                    run_identity=run_identity,
                    image_order=order,
                    image_id=str(task["group"].image_id),
                    input_image_hash=str(task["input_image_hash"]),
                    payload=payload,
                )
                _write_shard(
                    shards_dir / f"{order:06d}.json.gz", shard
                )
                by_order[order] = shard
    else:
        for task in missing:
            order = int(task["image_order"])
            payload = _evaluate_image_task(task)
            shard = build_shard(
                run_identity=run_identity,
                image_order=order,
                image_id=str(task["group"].image_id),
                input_image_hash=str(task["input_image_hash"]),
                payload=payload,
            )
            _write_shard(
                shards_dir / f"{order:06d}.json.gz", shard
            )
            by_order[order] = shard
    ordered_shards = validate_complete_shards(
        tuple(by_order.values()),
        image_ids=tuple(str(task["group"].image_id) for task in tasks),
        input_hashes=tuple(
            str(task["input_image_hash"]) for task in tasks
        ),
    )
    return tuple(
        validate_shard(
            shard,
            run_identity=run_identity,
            image_order=index,
            image_id=str(tasks[index]["group"].image_id),
            input_image_hash=str(tasks[index]["input_image_hash"]),
        )
        for index, shard in enumerate(ordered_shards)
    )


def _build_primary(
    validated: Any,
    wrapper: Mapping[str, Any],
    source: Mapping[str, str],
    *,
    workers: int,
    started: float,
    run_identity: str,
    staging: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    dataset = load_dataset(
        validated.paths["dataset_yaml"],
        split="val",
        root_override=validated.dataset_root,
    )
    image_by_id = {
        image["relative_path"]: image for image in dataset["images"]
    }
    frozen_arms = _load_frozen_arm_predictions(
        validated.paths["arm_predictions"],
        validated.image_list,
    )
    a_rows: list[dict[str, Any]] = []
    c_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    unit_events: list[dict[str, Any]] = []
    joint_patches: list[dict[str, Any]] = []
    per_image_invariants: list[dict[str, Any]] = []
    eligible_units = selected_units = 0
    eligible_members = patched_members = 0
    affected_images: set[str] = set()
    large_positive_affected_images: set[str] = set()
    by_class: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_sequence: Counter[str] = Counter()

    grouped = group_relevant_raw_rows(
        _iter_jsonl_gz(validated.paths["raw_views"]),
        validated.image_list,
    )
    tasks = []
    for order, group in enumerate(grouped):
        image = image_by_id[group.image_id]
        tasks.append(
            {
                "image_order": order,
                "group": group,
                "image": image,
                "a_frozen": frozen_arms["A"][group.image_id],
                "c_frozen": frozen_arms["C"][group.image_id],
                "input_image_hash": _input_image_hash(
                    group=group, image=image
                ),
            }
        )
    payloads = _execute_tasks(
        tuple(tasks),
        run_identity=run_identity,
        staging=staging,
        workers=workers,
    )
    worker_peak_rss_bytes: list[int] = []
    for payload in payloads:
        group_id = str(payload["image_id"])
        a_rows.append(payload["a_metric_row"])
        c_rows.append(payload["c_metric_row"])
        oracle_rows.append(payload["oracle_metric_row"])
        unit_events.extend(payload["events"])
        joint_patches.extend(payload["patches"])
        per_image_invariants.append(payload["invariant"])
        item_coverage = payload["coverage"]
        eligible_units += item_coverage["eligible_units"]
        selected_units += item_coverage["selected_units"]
        eligible_members += item_coverage["eligible_members"]
        patched_members += item_coverage["patched_members"]
        if item_coverage["affected"]:
            affected_images.add(group_id)
        if item_coverage["large_positive_affected"]:
            large_positive_affected_images.add(group_id)
        by_class.update(item_coverage["by_class"])
        by_source.update(item_coverage["by_source"])
        by_sequence.update(item_coverage["by_sequence_token"])
        peak_rss_bytes = payload.get("peak_rss_bytes")
        if (
            isinstance(peak_rss_bytes, bool)
            or not isinstance(peak_rss_bytes, int)
            or peak_rss_bytes < 0
        ):
            raise ValueError("worker peak RSS is invalid")
        worker_peak_rss_bytes.append(peak_rss_bytes)

    if len(a_rows) != len(validated.image_list):
        raise ValueError(
            "raw stream did not produce exact manifest image order"
        )
    a_metrics = evaluate_dataset(a_rows)
    c_metrics = evaluate_dataset(c_rows)
    oracle_metrics = evaluate_dataset(oracle_rows)
    if not _strict_recursive_equal(
        _jsonable(a_metrics), validated.g0_metrics["A"]
    ) or not _strict_recursive_equal(
        _jsonable(c_metrics), validated.g0_metrics["C"]
    ):
        raise ValueError(
            "recomputed A/C metrics disagree with sealed G0"
        )
    gate = gate_oracle_metrics(
        a_metrics,
        oracle_metrics,
        selected_count=selected_units,
    )
    invariant_passed = (
        len(per_image_invariants) == len(validated.image_list)
        and all(
            row["passed"] is True for row in per_image_invariants
        )
    )
    if not invariant_passed:
        raise ValueError("primary oracle invariants failed")
    metrics = {
        "A": _jsonable(a_metrics),
        "C": _jsonable(c_metrics),
        "joint": _jsonable(oracle_metrics),
        "joint_minus_a": gate.deltas,
        "joint_minus_c": _metric_delta(c_metrics, oracle_metrics),
    }
    invariants = {
        "passed": True,
        "image_count": len(per_image_invariants),
        "expected_image_count": EXPECTED_IMAGE_COUNT,
        "baseline_a_metrics_reproduced": True,
        "baseline_c_metrics_reproduced": True,
        "selection_rounds": 1,
        "per_image": per_image_invariants,
    }
    coverage = {
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
    primary_gate = {
        "proposed_status": gate.status,
        "joint_minus_a": gate.deltas,
        "thresholds": GATES,
        "gates": gate.gates,
        "invariants_passed": True,
        "selected_units": selected_units,
        "independent_adjudication": "PENDING",
    }
    manifest = {
        "schema_version": ORACLE_SCHEMA_VERSION,
        "schema": ORACLE_SCHEMA,
        "schema_hash": sha256_bytes(
            canonical_json_bytes(ORACLE_SCHEMA)
        ),
        "wrapper": {
            "uri": str(wrapper["path"]),
            "sha256": wrapper["sha256"],
        },
        "upstream_input": {
            "uri": str(wrapper["upstream_path"]),
            "sha256": wrapper["upstream_sha256"],
        },
        "approved_spec": {
            "uri": str(wrapper["spec_path"]),
            "sha256": wrapper["spec_sha256"],
        },
        "source": dict(source),
        "frozen_rule": frozen_rule_payload(),
        "frozen_rule_hash": sha256_bytes(
            canonical_json_bytes(frozen_rule_payload())
        ),
        "primary_script_sha256": sha256_file(__file__),
        "dataset_signature": validated.dataset_signature,
        "image_count": len(validated.image_list),
        "image_order_hash": sha256_bytes(
            canonical_json_bytes(list(validated.image_list))
        ),
        "workers": workers,
    }
    parent_peak_rss_bytes = _peak_rss_bytes()
    max_worker_peak_rss_bytes = max(
        worker_peak_rss_bytes, default=0
    )
    runtime = {
        "seconds": time.time() - started,
        "workers": workers,
        "peak_rss_bytes": max(
            parent_peak_rss_bytes,
            max_worker_peak_rss_bytes,
        ),
        "parent_peak_rss_bytes": parent_peak_rss_bytes,
        "max_worker_peak_rss_bytes": (
            max_worker_peak_rss_bytes
        ),
        "environment": environment_info(),
    }
    return (
        unit_events,
        joint_patches,
        coverage,
        metrics,
        invariants,
        primary_gate,
        {"manifest": manifest, "runtime": runtime},
    )


def _write_primary(
    output: Path,
    artifacts: tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.oracle-tmp-",
            dir=str(output.parent),
        )
    )
    try:
        primary = temporary / "primary"
        primary.mkdir()
        (
            events,
            patches,
            coverage,
            metrics,
            invariants,
            gate,
            metadata,
        ) = artifacts
        atomic_write_json(
            primary / "oracle_manifest.json",
            metadata["manifest"],
        )
        atomic_write_jsonl_gz(
            primary / "unit_events.jsonl.gz", events
        )
        atomic_write_jsonl_gz(
            primary / "score_patches.jsonl.gz", patches
        )
        atomic_write_json(primary / "coverage.json", coverage)
        atomic_write_json(
            primary / "oracle_metrics.json", metrics
        )
        atomic_write_json(
            primary / "invariants.json", invariants
        )
        atomic_write_json(primary / "primary_gate.json", gate)
        atomic_write_json(
            primary / "runtime.json", metadata["runtime"]
        )
        write_checksums(
            primary / "checksums.sha256",
            sorted(
                path
                for path in primary.iterdir()
                if path.name != "checksums.sha256"
            ),
            root=primary,
        )
        actual = {path.name for path in primary.iterdir()}
        expected = set(ORACLE_SCHEMA["required_artifacts"])
        if actual != expected:
            raise ValueError("primary artifact set is incomplete")
        _seal_primary(primary)
        temporary.replace(output)
    except Exception:
        for path in sorted(
            temporary.rglob("*"),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            try:
                os.chmod(path, 0o755 if path.is_dir() else 0o644)
            except OSError:
                pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen SBR score-only causal oracle"
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> int:
    if (
        isinstance(args.workers, bool)
        or not isinstance(args.workers, int)
        or args.workers < 0
    ):
        raise ValueError("workers must be a non-negative integer")
    repo_root = Path(__file__).resolve().parents[1]
    source = capture_clean_source(repo_root)
    wrapper = validate_oracle_wrapper(
        args.input_manifest,
        args.spec,
        source=source,
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    validated = validate_input_manifest(
        wrapper["upstream_path"], output
    )
    if (
        validated.manifest.get("dataset", {}).get("split")
        != "val"
    ):
        raise ValueError("oracle upstream split must be val")
    if len(validated.image_list) != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            f"oracle requires exactly {EXPECTED_IMAGE_COUNT} images"
        )
    started = time.time()
    run_identity = sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": (
                    "sbr-score-oracle-run-identity/v1"
                ),
                "wrapper_sha256": wrapper["sha256"],
                "source": dict(source),
                "rule_schema_hash": sha256_bytes(
                    canonical_json_bytes(frozen_rule_payload())
                ),
                "primary_script_sha256": sha256_file(__file__),
            }
        )
    )
    staging = output.parent / f".{output.name}.oracle-staging"
    artifacts = _build_primary(
        validated,
        wrapper,
        source,
        workers=args.workers,
        started=started,
        run_identity=run_identity,
        staging=staging,
    )
    _write_primary(output, artifacts)
    shutil.rmtree(staging)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except Exception as exc:
        print(f"SBR_SCORE_ORACLE_INVALID: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
