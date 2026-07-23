#!/usr/bin/env python3
"""Independent, fail-closed adjudicator for SBR-V2 causal-audit evidence.

This module intentionally shares no project implementation code with the
primary audit.  Its only non-standard-library dependency is NumPy, which is
used solely to record the independent process environment.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
from typing import Any, Mapping
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import numpy as np


SCHEMA_VERSION = "sbr-v2-audit-evidence/v1"
SCHEMA = {
    "schema_version": SCHEMA_VERSION,
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
REQUIRED_PRIMARY = tuple(SCHEMA["required_artifacts"])
DETERMINISTIC_ARTIFACTS = (
    "attribution_events.jsonl.gz",
    "attribution_summary.json",
    "upper_bound_metrics.json",
    "invariants.json",
    "primary_gate.json",
)
THRESHOLDS = tuple(round(0.50 + index * 0.05, 2) for index in range(10))
CATEGORIES = (
    "mixed_cluster_localization",
    "final_300_truncation",
    "matching_competition",
    "class_or_candidate_loss",
    "other",
)
INVARIANT_KEYS = (
    "raw_hash_equal",
    "cluster_hash_equal",
    "cluster_count_equal",
    "scores_equal",
    "classes_equal",
    "selected_cluster_ids_equal",
)
HEX = frozenset("0123456789abcdefABCDEF")
OUTPUT_HASH_SEMANTICS = (
    "sha256(canonical-json(report without output_hash))"
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _digest(value: Any, name: str, *, lengths: tuple[int, ...] = (64,)) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} is not a hexadecimal digest")
    if len(value) not in lengths or any(character not in HEX for character in value):
        raise ValueError(f"{name} is not a hexadecimal digest")
    return value.lower()


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value}")


def _read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )


def _assert_finite(value: Any, name: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} is non-finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{name}[{index}]")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not a strict number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is non-finite")
    return result


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} is not a non-negative integer")
    return value


def _close(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=1e-12
        )
    except (TypeError, ValueError):
        return False


def _safe_target(root: Path, relative: str) -> tuple[Path, str]:
    if not isinstance(relative, str) or not relative:
        raise ValueError("unsafe empty checksum path")
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.drive
        or relative_path.name == "checksums.sha256"
    ):
        raise ValueError(f"unsafe checksum path: {relative}")
    target = (root / relative_path).resolve()
    if root != target.parent and root not in target.parents:
        raise ValueError(f"unsafe checksum path: {relative}")
    return target, relative_path.as_posix()


def _portable_path(uri: Any, *, base: Path, name: str) -> Path:
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError(f"{name} local URI is missing")
    direct = Path(uri)
    if direct.is_absolute():
        return direct.resolve()
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme.lower() != "file":
        raise ValueError(f"{name} uses unsupported non-local URI scheme")
    if parsed.scheme.lower() == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError(f"{name} uses a remote file authority")
        value = url2pathname(unquote(parsed.path))
        if (
            os.name == "nt"
            and len(value) >= 3
            and value[0] in "/\\"
            and value[2] == ":"
        ):
            value = value[1:]
        path = Path(value)
        if not path.is_absolute():
            raise ValueError(f"{name} file URI is not absolute")
        return path.resolve()
    return (base / direct).resolve()


def _verify_input_chain(
    audit_manifest: Mapping[str, Any], *, evidence_root: Path
) -> tuple[Path, str, int]:
    input_record = audit_manifest.get("input_manifest")
    if not isinstance(input_record, Mapping):
        raise ValueError("input manifest provenance is missing")
    input_manifest_path = _portable_path(
        input_record.get("uri"),
        base=evidence_root,
        name="audit input manifest",
    )
    if not input_manifest_path.is_file():
        raise ValueError("audit input manifest local file is missing")
    recorded_manifest_hash = _digest(
        input_record.get("sha256"), "input manifest hash"
    )
    actual_manifest_hash = _sha256_file(input_manifest_path)
    if actual_manifest_hash != recorded_manifest_hash:
        raise ValueError("input manifest bytes disagree with recorded SHA-256")
    input_manifest = _read_json(input_manifest_path)
    if not isinstance(input_manifest, Mapping):
        raise ValueError("input manifest is not an object")
    if input_manifest.get("schema_version") != "sbr-v2-audit-input/v1":
        raise ValueError("input manifest schema version mismatch")
    input_files = input_manifest.get("files")
    audit_inputs = audit_manifest.get("inputs")
    if not isinstance(input_files, Mapping) or not isinstance(
        audit_inputs, Mapping
    ):
        raise ValueError("input manifest file mappings are missing")
    if set(input_files) != set(audit_inputs):
        raise ValueError("input manifest and audit manifest input keys disagree")
    for key in sorted(input_files):
        portable_entry = input_files[key]
        audit_entry = audit_inputs[key]
        if not isinstance(portable_entry, Mapping) or not isinstance(
            audit_entry, Mapping
        ):
            raise ValueError(f"input manifest entry {key} is invalid")
        portable_path = _portable_path(
            portable_entry.get("uri"),
            base=input_manifest_path.parent,
            name=f"input manifest file {key}",
        )
        audit_path = _portable_path(
            audit_entry.get("uri"),
            base=evidence_root,
            name=f"audit manifest input {key}",
        )
        if portable_path != audit_path:
            raise ValueError(
                f"input manifest and audit manifest URI disagree: {key}"
            )
        portable_hash = _digest(
            portable_entry.get("sha256"), f"input manifest {key} SHA-256"
        )
        audit_hash = _digest(
            audit_entry.get("sha256"), f"audit manifest {key} SHA-256"
        )
        if portable_hash != audit_hash:
            raise ValueError(
                f"input manifest and audit manifest SHA-256 disagree: {key}"
            )
        if not portable_path.is_file():
            raise ValueError(f"input file is missing: {key}")
        if _sha256_file(portable_path) != portable_hash:
            raise ValueError(f"input file bytes disagree with SHA-256: {key}")
    return input_manifest_path, actual_manifest_hash, len(input_files)


def _verify_checksums(root: Path) -> set[str]:
    checksum_path = root / "checksums.sha256"
    if not checksum_path.is_file():
        raise ValueError("checksums.sha256 is missing")
    seen: set[str] = set()
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            digest_text, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(
                f"invalid checksums.sha256 line {line_number}"
            ) from exc
        digest = _digest(
            digest_text, f"checksum line {line_number}", lengths=(64,)
        )
        target, normalized = _safe_target(root, relative)
        if normalized in seen:
            raise ValueError(f"duplicate checksum path: {normalized}")
        if not target.is_file():
            raise ValueError(f"missing checksum target: {normalized}")
        if _sha256_file(target) != digest:
            raise ValueError(f"artifact checksum mismatch: {normalized}")
        seen.add(normalized)
    required = set(REQUIRED_PRIMARY) - {"checksums.sha256"}
    missing = sorted(required - seen)
    if missing:
        raise ValueError(
            "primary artifacts are not checksum sealed: " + ", ".join(missing)
        )
    return seen


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            value = json.loads(line, parse_constant=_reject_constant)
            if not isinstance(value, dict):
                raise ValueError(f"event {line_number} is not an object")
            _assert_finite(value, f"event[{line_number}]")
            image_id = value.get("image_id")
            gt_index = value.get("gt_index")
            threshold = value.get("iou_threshold")
            category = value.get("category")
            recovers = value.get("counterfactual_recovers")
            if not isinstance(image_id, str) or not image_id:
                raise ValueError(f"event {line_number} has invalid image_id")
            _strict_nonnegative_int(gt_index, f"event {line_number} gt_index")
            numeric_threshold = _finite_number(
                threshold, f"event {line_number} iou_threshold"
            )
            if numeric_threshold not in THRESHOLDS:
                raise ValueError(
                    f"event {line_number} has non-frozen IoU threshold"
                )
            if category not in CATEGORIES:
                raise ValueError(f"event {line_number} has invalid category")
            if not isinstance(recovers, bool):
                raise ValueError(
                    f"event {line_number} has invalid recovery boolean"
                )
            events.append(value)
    event_ids = [
        (event["image_id"], event["gt_index"], float(event["iou_threshold"]))
        for event in events
    ]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("duplicate attribution event ID")
    return events


def _expected_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    threshold_summaries: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        rows = [
            event
            for event in events
            if float(event["iou_threshold"]) == threshold
        ]
        counts = Counter(str(event["category"]) for event in rows)
        threshold_summaries[f"{threshold:.2f}"] = {
            "denominator": len(rows),
            "category_counts": {
                category: int(counts.get(category, 0))
                for category in CATEGORIES
            },
        }
    primary = [
        event
        for event in events
        if float(event["iou_threshold"]) == 0.75
    ]
    primary_counts = Counter(str(event["category"]) for event in primary)
    denominator = len(primary)
    mixed = int(primary_counts.get("mixed_cluster_localization", 0))
    share = float(mixed) / float(denominator) if denominator else 0.0
    return {
        "primary_ap75": {
            "unique_event_key": ["image_id", "gt_index", "iou_threshold"],
            "denominator": denominator,
            "mixed_cluster_localization": mixed,
            "mechanism_share": share,
            "category_counts": {
                category: int(primary_counts.get(category, 0))
                for category in CATEGORIES
            },
        },
        "secondary_repeated_measures": {
            "note": "The ten thresholds are pooled repeated measures, not independent samples.",
            "thresholds": threshold_summaries,
        },
    }


def _verify_summary(
    events: list[dict[str, Any]], summary: Any
) -> tuple[int, int, float]:
    if not isinstance(summary, Mapping):
        raise ValueError("attribution summary is not an object")
    _assert_finite(summary, "attribution_summary")
    primary_value = summary.get("primary_ap75")
    if not isinstance(primary_value, Mapping):
        raise ValueError("summary primary AP75 block is missing")
    _strict_nonnegative_int(
        primary_value.get("denominator"), "summary AP75 denominator"
    )
    _strict_nonnegative_int(
        primary_value.get("mixed_cluster_localization"),
        "summary AP75 mixed count",
    )
    _finite_number(
        primary_value.get("mechanism_share"), "summary mechanism share"
    )
    primary_counts = primary_value.get("category_counts")
    if not isinstance(primary_counts, Mapping) or set(primary_counts) != set(
        CATEGORIES
    ):
        raise ValueError("summary AP75 category count set is invalid")
    for category in CATEGORIES:
        _strict_nonnegative_int(
            primary_counts.get(category),
            f"summary AP75 category {category}",
        )
    secondary_value = summary.get("secondary_repeated_measures")
    if not isinstance(secondary_value, Mapping):
        raise ValueError("summary secondary block is missing")
    thresholds_value = secondary_value.get("thresholds")
    expected_threshold_keys = {f"{threshold:.2f}" for threshold in THRESHOLDS}
    if not isinstance(thresholds_value, Mapping) or set(
        thresholds_value
    ) != expected_threshold_keys:
        raise ValueError("summary secondary threshold set is invalid")
    for threshold_key in sorted(expected_threshold_keys):
        row = thresholds_value[threshold_key]
        if not isinstance(row, Mapping):
            raise ValueError(
                f"summary secondary threshold {threshold_key} is invalid"
            )
        _strict_nonnegative_int(
            row.get("denominator"),
            f"summary secondary {threshold_key} denominator",
        )
        counts = row.get("category_counts")
        if not isinstance(counts, Mapping) or set(counts) != set(CATEGORIES):
            raise ValueError(
                f"summary secondary {threshold_key} categories are invalid"
            )
        for category in CATEGORIES:
            _strict_nonnegative_int(
                counts.get(category),
                f"summary secondary {threshold_key} category {category}",
            )
    expected = _expected_summary(events)
    if summary != expected:
        raise ValueError("event rows and attribution summary disagree")
    primary = expected["primary_ap75"]
    return (
        int(primary["denominator"]),
        int(primary["mixed_cluster_localization"]),
        float(primary["mechanism_share"]),
    )


def _verify_schema_and_hashes(root: Path, manifest: Any) -> None:
    if not isinstance(manifest, Mapping):
        raise ValueError("audit manifest is not an object")
    _assert_finite(manifest, "audit_manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("audit schema version mismatch")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("audit schema contract mismatch")
    expected_schema_hash = _sha256_json(SCHEMA)
    if (
        _digest(manifest.get("schema_hash"), "schema_hash")
        != expected_schema_hash
    ):
        raise ValueError("audit schema hash mismatch")
    recorded_hashes = manifest.get("deterministic_artifact_hashes")
    if not isinstance(recorded_hashes, Mapping) or set(recorded_hashes) != set(
        DETERMINISTIC_ARTIFACTS
    ):
        raise ValueError("deterministic artifact hash set mismatch")
    actual_hashes = {
        name: _sha256_file(root / name) for name in DETERMINISTIC_ARTIFACTS
    }
    normalized_hashes = {
        name: _digest(recorded_hashes.get(name), f"artifact hash {name}")
        for name in DETERMINISTIC_ARTIFACTS
    }
    if normalized_hashes != actual_hashes:
        raise ValueError("deterministic artifact hash mismatch")
    if (
        _digest(
            manifest.get("deterministic_evidence_hash"),
            "deterministic_evidence_hash",
        )
        != _sha256_json(actual_hashes)
    ):
        raise ValueError("deterministic evidence hash mismatch")
    frozen = manifest.get("frozen_constants")
    if not isinstance(frozen, Mapping):
        raise ValueError("frozen constants are missing")
    if (
        not _close(frozen.get("mechanism_share_threshold"), 0.60)
        or not _close(frozen.get("large_ap_tolerance"), -0.005)
        or not _close(frozen.get("primary_iou_threshold"), 0.75)
        or frozen.get("secondary_iou_thresholds") != list(THRESHOLDS)
    ):
        raise ValueError("frozen gate constants mismatch")


def _verify_invariants(invariants: Any, upper: Mapping[str, Any]) -> bool:
    if not isinstance(invariants, Mapping):
        raise ValueError("invariants artifact is not an object")
    _assert_finite(invariants, "invariants")
    for key in INVARIANT_KEYS:
        if not isinstance(invariants.get(key), bool):
            raise ValueError(f"invariant {key} is not boolean")
    singleton = _finite_number(
        invariants.get("singleton_preservation"),
        "invariant singleton_preservation",
    )
    if not isinstance(invariants.get("passed"), bool):
        raise ValueError("invariant passed is not boolean")
    per_image = invariants.get("per_image")
    if not isinstance(per_image, list):
        raise ValueError("per-image invariants are missing")
    if invariants.get("image_count") != len(per_image):
        raise ValueError("invariant image count disagrees")
    singleton_total = 0
    singleton_preserved = 0
    for index, row in enumerate(per_image):
        if not isinstance(row, Mapping):
            raise ValueError(f"per-image invariant {index} is not an object")
        for key in INVARIANT_KEYS + ("passed",):
            if not isinstance(row.get(key), bool):
                raise ValueError(
                    f"per-image invariant {index}.{key} is not boolean"
                )
        row_singleton = _finite_number(
            row.get("singleton_preservation"),
            f"per-image invariant {index}.singleton_preservation",
        )
        row_total = _strict_nonnegative_int(
            row.get("singleton_total"),
            f"per-image invariant {index}.singleton_total",
        )
        row_preserved = _strict_nonnegative_int(
            row.get("singleton_preserved"),
            f"per-image invariant {index}.singleton_preserved",
        )
        if row_preserved > row_total:
            raise ValueError(
                f"per-image invariant {index} singleton count is impossible"
            )
        expected_row_ratio = (
            float(row_preserved) / float(row_total) if row_total else 1.0
        )
        expected_row_pass = (
            all(row[key] is True for key in INVARIANT_KEYS)
            and expected_row_ratio == 1.0
        )
        if (
            not _close(row_singleton, expected_row_ratio)
            or row["passed"] is not expected_row_pass
        ):
            raise ValueError(
                f"per-image invariant {index} aggregate disagrees"
            )
        singleton_total += row_total
        singleton_preserved += row_preserved
    for key in INVARIANT_KEYS:
        expected_value = bool(per_image) and all(
            row[key] is True for row in per_image
        )
        if invariants[key] is not expected_value:
            raise ValueError(f"per-image invariant aggregate disagrees: {key}")
    if (
        invariants.get("singleton_total") != singleton_total
        or invariants.get("singleton_preserved") != singleton_preserved
    ):
        raise ValueError("per-image invariant singleton counts disagree")
    expected_singleton = (
        float(singleton_preserved) / float(singleton_total)
        if singleton_total
        else 1.0
    )
    expected_pass = (
        bool(per_image)
        and all(invariants[key] is True for key in INVARIANT_KEYS)
        and expected_singleton == 1.0
        and all(row["passed"] is True for row in per_image)
    )
    if not _close(singleton, expected_singleton):
        raise ValueError("per-image invariant singleton aggregate disagrees")
    if invariants["passed"] is not expected_pass:
        raise ValueError("invariant passed aggregate disagrees")
    upper_invariants = upper.get("invariants")
    if not isinstance(upper_invariants, Mapping):
        raise ValueError("upper-bound invariants are missing")
    expected_upper = {
        key: invariants[key] for key in INVARIANT_KEYS
    }
    expected_upper["singleton_preservation"] = singleton
    expected_upper["passed"] = expected_pass
    if dict(upper_invariants) != expected_upper:
        raise ValueError("invariant artifact and upper-bound invariants disagree")
    return expected_pass


def _verify_upper_bound(
    upper: Any,
    mechanism_share: float,
) -> tuple[str, str]:
    if not isinstance(upper, Mapping):
        raise ValueError("upper-bound metrics are not an object")
    _assert_finite(upper, "upper_bound")
    recorded_share = _finite_number(
        upper.get("mechanism_share_ap75"), "upper mechanism share"
    )
    if not _close(recorded_share, mechanism_share):
        raise ValueError("summary and upper-bound mechanism share disagree")
    expected_mechanism_gate = (
        "PASS" if mechanism_share >= 0.60 and mechanism_share > 0.0 else "FAIL"
    )
    if upper.get("mechanism_gate") != expected_mechanism_gate:
        raise ValueError("mechanism gate disagrees with frozen 0.60 threshold")
    a_metrics = upper.get("a_metrics")
    v2_metrics = upper.get("v2_metrics")
    delta = upper.get("v2_minus_a")
    if (
        not isinstance(a_metrics, Mapping)
        or not isinstance(v2_metrics, Mapping)
        or not isinstance(delta, Mapping)
    ):
        raise ValueError("upper-bound metric blocks are incomplete")
    a_large = _finite_number(
        a_metrics.get("AP-large-SBR"), "A AP-large-SBR"
    )
    v2_large = _finite_number(
        v2_metrics.get("AP-large-SBR"), "V2 AP-large-SBR"
    )
    recorded_delta = _finite_number(
        delta.get("AP-large-SBR"), "V2-minus-A AP-large-SBR"
    )
    if not _close(recorded_delta, v2_large - a_large):
        raise ValueError("upper-bound AP-large delta disagrees")
    expected_upper_gate = (
        "PASS" if v2_large >= a_large - 0.005 else "FAIL"
    )
    if upper.get("recoverable_upper_bound_gate") != expected_upper_gate:
        raise ValueError(
            "recoverable upper-bound gate disagrees with A AP-large - 0.005"
        )
    return expected_mechanism_gate, expected_upper_gate


def _verify_primary_gate(
    gate: Any,
    mechanism_gate: str,
    upper_gate: str,
    invariants_passed: bool,
) -> tuple[str, bool]:
    if not isinstance(gate, Mapping):
        raise ValueError("primary gate is not an object")
    if not isinstance(gate.get("invariants_passed"), bool):
        raise ValueError("primary gate invariants flag is not boolean")
    expected_status = (
        "SBR_V2_AUDIT_ELIGIBLE"
        if mechanism_gate == "PASS"
        and upper_gate == "PASS"
        and invariants_passed
        else "SBR_V2_AUDIT_STOP"
    )
    expected = {
        "status": expected_status,
        "mechanism_gate": mechanism_gate,
        "recoverable_upper_bound_gate": upper_gate,
        "invariants_passed": invariants_passed,
    }
    if dict(gate) != expected:
        raise ValueError("primary gate disagrees with independent gate inputs")
    return expected_status, True


def _capture_self_state(script_path: Path) -> dict[str, Any]:
    repo = script_path.parent.parent

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    commit = _digest(
        git("rev-parse", "HEAD"), "git commit", lengths=(40, 64)
    )
    tree = _digest(
        git("rev-parse", "HEAD^{tree}"), "git tree", lengths=(40, 64)
    )
    clean = git("status", "--porcelain", "--untracked-files=all") == ""
    return {
        "commit": commit,
        "tree": tree,
        "clean": clean,
        "script_sha256": _sha256_file(script_path),
        "repo_root": str(repo.resolve()),
    }


def _validated_clean_self_state(state: Any, name: str) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise ValueError(f"adjudicator source {name} state is missing")
    normalized = {
        "commit": _digest(
            state.get("commit"),
            f"adjudicator source {name} commit",
            lengths=(40, 64),
        ),
        "tree": _digest(
            state.get("tree"),
            f"adjudicator source {name} tree",
            lengths=(40, 64),
        ),
        "script_sha256": _digest(
            state.get("script_sha256"),
            f"adjudicator source {name} script SHA-256",
        ),
        "repo_root": str(
            Path(str(state.get("repo_root", ""))).resolve()
        ),
        "clean": state.get("clean") is True,
    }
    if not normalized["clean"]:
        raise ValueError(f"adjudicator source {name} worktree is not clean")
    return normalized


def _assert_same_self_state(
    expected: Mapping[str, Any],
    actual: Any,
    name: str,
) -> dict[str, Any]:
    normalized = _validated_clean_self_state(actual, name)
    for key in ("commit", "tree", "script_sha256", "repo_root"):
        if normalized[key] != expected[key]:
            raise ValueError(f"adjudicator source changed at {name}: {key}")
    return normalized


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as fh:
            fh.write(_canonical_json_bytes(value))
            fh.write(b"\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_checksums(root: Path) -> None:
    files = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    content = "".join(
        f"{_sha256_file(path)}  {path.name}\n" for path in files
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".checksums.", suffix=".tmp", dir=str(root)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, root / "checksums.sha256")
    finally:
        if temporary.exists():
            temporary.unlink()


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def _finish_report(report: dict[str, Any]) -> dict[str, Any]:
    report["output_hash_semantics"] = OUTPUT_HASH_SEMANTICS
    report["output_hash"] = _sha256_json(report)
    return report


def adjudicate_evidence(
    evidence: Path | str,
    expected_primary_checksums_sha256: str,
) -> dict[str, Any]:
    """Independently adjudicate a primary SBR-V2 evidence directory."""

    root = Path(evidence).resolve()
    script_path = Path(__file__).resolve()
    checksums_verified = False
    primary_anchor_verified = False
    manifest: Mapping[str, Any] = {}
    start_state: dict[str, Any] | None = None
    primary_anchor: str | None = None
    report: dict[str, Any]
    try:
        primary_anchor = _digest(
            expected_primary_checksums_sha256,
            "expected primary checksums SHA-256",
        )
        start_state = _validated_clean_self_state(
            _capture_self_state(script_path), "at start"
        )
        repo_root = Path(start_state["repo_root"])
        if root == repo_root or repo_root in root.parents:
            raise ValueError(
                "evidence output must be outside the adjudicator source repo"
            )
        if not root.is_dir():
            raise ValueError(f"evidence directory does not exist: {root}")
        primary_checksum_path = root / "checksums.sha256"
        if not primary_checksum_path.is_file():
            raise ValueError("primary checksums.sha256 is missing")
        if _sha256_file(primary_checksum_path) != primary_anchor:
            raise ValueError(
                "external primary checksum anchor does not match checksums.sha256"
            )
        primary_anchor_verified = True
        missing = [
            name for name in REQUIRED_PRIMARY if not (root / name).is_file()
        ]
        if missing:
            raise ValueError(
                "missing primary evidence files: " + ", ".join(missing)
            )
        _verify_checksums(root)
        checksums_verified = True
        manifest_value = _read_json(root / "audit_manifest.json")
        if not isinstance(manifest_value, Mapping):
            raise ValueError("audit manifest is not an object")
        manifest = manifest_value
        _verify_schema_and_hashes(root, manifest)
        input_manifest_path, input_manifest_hash, input_count = (
            _verify_input_chain(manifest, evidence_root=root)
        )
        events = _read_events(root / "attribution_events.jsonl.gz")
        summary = _read_json(root / "attribution_summary.json")
        denominator, mixed, mechanism_share = _verify_summary(events, summary)
        upper = _read_json(root / "upper_bound_metrics.json")
        mechanism_gate, upper_gate = _verify_upper_bound(
            upper, mechanism_share
        )
        invariants = _read_json(root / "invariants.json")
        invariants_passed = _verify_invariants(invariants, upper)
        primary_gate = _read_json(root / "primary_gate.json")
        independent_gate, primary_agrees = _verify_primary_gate(
            primary_gate,
            mechanism_gate,
            upper_gate,
            invariants_passed,
        )
        _assert_same_self_state(
            start_state,
            _capture_self_state(script_path),
            "before evidence write",
        )
        passed = independent_gate == "SBR_V2_AUDIT_ELIGIBLE"
        report = {
            "status": (
                "SBR_V2_AUDIT_INDEPENDENT_PASS"
                if passed
                else "SBR_V2_AUDIT_INDEPENDENT_FAIL"
            ),
            "decision": "PASS" if passed else "FAIL",
            "independent_gate": independent_gate,
            "primary_gate_status": str(primary_gate.get("status", "")),
            "primary_gate_agrees": primary_agrees,
            "primary_checksum_anchor_verified": True,
            "primary_checksums_sha256": primary_anchor,
            "checksums_verified": True,
            "checksums_regenerated": True,
            "event_count": len(events),
            "ap75_denominator": denominator,
            "mixed_cluster_localization": mixed,
            "mechanism_share": mechanism_share,
            "mechanism_share_threshold": 0.60,
            "large_ap_tolerance": -0.005,
            "invariants_passed": invariants_passed,
            "input_manifest_sha256": input_manifest_hash,
            "input_manifest_uri": str(input_manifest_path),
            "verified_input_file_count": input_count,
            "primary_audit_manifest_sha256": _sha256_file(
                root / "audit_manifest.json"
            ),
            "adjudicator_script_sha256": start_state["script_sha256"],
            "adjudicator_source": start_state,
            "source_stability_verified": True,
            "environment": _environment(),
        }
    except Exception as exc:
        input_hash = None
        if isinstance(manifest.get("input_manifest"), Mapping):
            candidate = manifest["input_manifest"].get("sha256")
            if isinstance(candidate, str):
                input_hash = candidate
        report = {
            "status": "SBR_V2_AUDIT_INDEPENDENT_FAIL",
            "decision": "FAIL",
            "independent_gate": "SBR_V2_AUDIT_STOP",
            "primary_gate_agrees": False,
            "primary_checksum_anchor_verified": primary_anchor_verified,
            "primary_checksums_sha256": primary_anchor,
            "checksums_verified": checksums_verified,
            "checksums_regenerated": checksums_verified,
            "input_manifest_sha256": input_hash,
            "primary_audit_manifest_sha256": (
                _sha256_file(root / "audit_manifest.json")
                if (root / "audit_manifest.json").is_file()
                else None
            ),
            "adjudicator_script_sha256": (
                start_state["script_sha256"]
                if start_state is not None
                else _sha256_file(script_path)
            ),
            "adjudicator_source": start_state,
            "source_stability_verified": False,
            "environment": _environment(),
            "error": str(exc),
        }
    _finish_report(report)
    _atomic_write_json(root / "independent_adjudication.json", report)
    if checksums_verified:
        _write_checksums(root)
    if report.get("decision") == "PASS":
        try:
            if start_state is None:
                raise ValueError("adjudicator source start state is missing")
            _assert_same_self_state(
                start_state,
                _capture_self_state(script_path),
                "after evidence write",
            )
        except Exception as exc:
            report = {
                "status": "SBR_V2_AUDIT_INDEPENDENT_FAIL",
                "decision": "FAIL",
                "independent_gate": "SBR_V2_AUDIT_STOP",
                "primary_gate_agrees": False,
                "primary_checksum_anchor_verified": primary_anchor_verified,
                "primary_checksums_sha256": primary_anchor,
                "checksums_verified": checksums_verified,
                "checksums_regenerated": checksums_verified,
                "input_manifest_sha256": (
                    manifest.get("input_manifest", {}).get("sha256")
                    if isinstance(manifest.get("input_manifest"), Mapping)
                    else None
                ),
                "primary_audit_manifest_sha256": _sha256_file(
                    root / "audit_manifest.json"
                ),
                "adjudicator_script_sha256": start_state[
                    "script_sha256"
                ],
                "adjudicator_source": start_state,
                "source_stability_verified": False,
                "environment": _environment(),
                "error": str(exc),
            }
            _finish_report(report)
            _atomic_write_json(
                root / "independent_adjudication.json", report
            )
            if checksums_verified:
                _write_checksums(root)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently adjudicate SBR-V2 causal-audit evidence"
    )
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument(
        "--expected-primary-checksums-sha256",
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = adjudicate_evidence(
        args.evidence, args.expected_primary_checksums_sha256
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report.get("decision") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
