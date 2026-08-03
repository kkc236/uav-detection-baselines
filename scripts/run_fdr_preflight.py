#!/usr/bin/env python3
"""Fail-closed F0-F4 preflight runner for the frozen FDR-only experiment.

This module owns orchestration and evidence schemas only.  Model-dependent gates
are loaded lazily or injected by tests/callers, so importing this file never
imports or mutates the detector implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.fdr_protocol import (
    DFINE_COMMIT,
    FDR_PROTOCOL,
    canonical_json_bytes,
    write_create_only_manifest,
)


GATE_ORDER = ("F0", "F1", "F2", "F3", "F4")
FIXED_RUNTIME = {"device": "cuda:0", "batch": 8, "imgsz": 640}
SCREEN_AUTHORITY = {"schedule_epochs": 50, "cutoff_epoch": 30}
REPORT_SCHEMA_VERSION = 1
EDGE_NAMES = ("left", "top", "right", "bottom")

GateRunner = Callable[["PreflightContext"], Mapping[str, Any]]


@dataclass(frozen=True)
class PreflightContext:
    """Authority-only inputs shared by every gate."""

    protocol_manifest: Path
    baseline_checkpoint: Path
    dataset_root: Path
    report_root: Path
    repository_root: Path

    def __post_init__(self) -> None:
        for field in (
            "protocol_manifest",
            "baseline_checkpoint",
            "dataset_root",
            "report_root",
            "repository_root",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))

    @property
    def runtime(self) -> dict[str, Any]:
        return dict(FIXED_RUNTIME)


def canonical_sha256(payload: Any) -> str:
    """Return the uppercase SHA256 of the frozen canonical JSON encoding."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()


def make_evidence_record(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap evidence with a non-recursive canonical payload hash."""

    if kind not in (*GATE_ORDER, "decision"):
        raise ValueError(f"unknown evidence kind: {kind}")
    body = dict(payload)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": kind,
        "payload": body,
        "payload_sha256": canonical_sha256(body),
    }


def write_create_only_report(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Write a single canonical report without following or replacing links."""

    return write_create_only_manifest(Path(path), record)


def decide_preflight(states: Mapping[str, str]) -> dict[str, Any]:
    """Authorize screening only when the complete ordered gate set passed."""

    eligible = set(states) == set(GATE_ORDER) and all(
        states.get(gate) == "passed" for gate in GATE_ORDER
    )
    return {
        "status": "passed" if eligible else "engineering_failed",
        "screen_eligible": eligible,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(reparse and attributes & reparse)


def _reject_link_traversal(path: Path) -> None:
    for component in (*reversed(path.parents), path):
        if _is_link_or_reparse(component):
            raise ValueError("preflight path cannot traverse a symlink or reparse point")


def _read_json_authority(path: Path, label: str) -> dict[str, Any]:
    _reject_link_traversal(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _validate_authority_inputs(context: PreflightContext) -> None:
    _read_json_authority(context.protocol_manifest, "protocol manifest")
    _reject_link_traversal(context.baseline_checkpoint)
    if not context.baseline_checkpoint.is_file():
        raise FileNotFoundError(
            f"baseline checkpoint does not exist: {context.baseline_checkpoint}"
        )
    _reject_link_traversal(context.dataset_root)
    if not context.dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {context.dataset_root}")
    _reject_link_traversal(context.repository_root)
    if not context.repository_root.is_dir():
        raise FileNotFoundError(
            f"repository root does not exist: {context.repository_root}"
        )


def _manifest_commit(manifest: Mapping[str, Any]) -> Any:
    protocol = manifest.get("protocol")
    if isinstance(protocol, Mapping) and "dfine_commit" in protocol:
        return protocol["dfine_commit"]
    return manifest.get("dfine_commit")


def _load_reference(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_fdr_preflight_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned FDR reference: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_math_golden(reference: Any) -> dict[str, bool]:
    import torch

    implementation = importlib.import_module("src.fdr_math")
    checks: dict[str, bool] = {}
    for dtype, name in (
        (torch.float32, "weighting_float32"),
        (torch.float64, "weighting_float64"),
    ):
        up = torch.tensor([0.5], dtype=dtype)
        scale = torch.tensor([4.0], dtype=dtype)
        actual = implementation.weighting_function(32, up.clone(), scale.clone())
        expected = reference.weighting_function(32, up.clone(), scale.clone())
        checks[name] = bool(torch.equal(actual, expected))

    generator = torch.Generator().manual_seed(804)
    logits = torch.randn((2, 3, 132), generator=generator, dtype=torch.float32)
    project = implementation.weighting_function(
        32, torch.tensor([0.5]), torch.tensor([4.0])
    )
    actual_integral = implementation.Integral(32)(logits, project)
    expected_integral = reference.Integral(32)(logits, project)
    checks["integral"] = bool(torch.equal(actual_integral, expected_integral))

    points = torch.tensor(
        [[[0.5, 0.5, 0.2, 0.1], [0.2, 0.7, 0.1, 0.3]]],
        dtype=torch.float32,
    )
    distances = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [-4.0, -0.01, 0.01, 4.0]]],
        dtype=torch.float32,
    )
    scale = torch.tensor([4.0], dtype=torch.float32)
    checks["distance2bbox"] = bool(
        torch.equal(
            implementation.distance2bbox(points, distances, scale),
            reference.distance2bbox(points, distances, scale),
        )
    )

    target_xyxy = torch.tensor(
        [[0.35, 0.40, 0.65, 0.60], [0.0, 0.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    flat_points = points.reshape(-1, 4)
    actual_targets = implementation.bbox2distance(
        flat_points, target_xyxy, 32, scale, torch.tensor([0.5])
    )
    expected_targets = reference.bbox2distance(
        flat_points, target_xyxy, 32, scale, torch.tensor([0.5])
    )
    checks["bbox2distance"] = all(
        torch.equal(actual, expected)
        for actual, expected in zip(actual_targets, expected_targets)
    )

    pred = torch.randn((8, 33), generator=generator, dtype=torch.float32)
    labels, weight_right, weight_left = actual_targets
    sample_weight = torch.linspace(0.2, 1.0, 8, dtype=torch.float32)
    actual_fgl = implementation.fine_grained_localization_loss(
        pred,
        labels,
        weight_right,
        weight_left,
        weight=sample_weight,
        avg_factor=2.0,
    )
    expected_fgl = reference.unimodal_distribution_focal_loss(
        pred,
        labels,
        weight_right,
        weight_left,
        weight=sample_weight,
        avg_factor=2.0,
    )
    checks["fgl"] = bool(torch.equal(actual_fgl, expected_fgl))
    if not all(checks.values()):
        failures = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"F0 pinned math golden mismatch: {failures}")
    return checks


def run_f0(context: PreflightContext) -> dict[str, Any]:
    """Verify the commit-pinned authority and execute independent CPU goldens."""

    manifest = _read_json_authority(context.protocol_manifest, "protocol manifest")
    if _manifest_commit(manifest) != DFINE_COMMIT:
        raise ValueError("protocol manifest is not bound to the pinned D-FINE commit")

    authority_path = (
        context.repository_root
        / "third_party"
        / "dfine_7fe2f888"
        / "AUTHORITY.json"
    )
    authority = _read_json_authority(authority_path, "D-FINE authority")
    if authority.get("commit") != DFINE_COMMIT:
        raise ValueError("third_party authority commit differs from the frozen commit")
    reference_path = context.repository_root / str(authority.get("vendored_reference_path", ""))
    _reject_link_traversal(reference_path)
    if not reference_path.is_file():
        raise FileNotFoundError(f"vendored FDR reference does not exist: {reference_path}")
    reference_sha256 = _sha256_file(reference_path).lower()
    if reference_sha256 != authority.get("vendored_reference_sha256"):
        raise ValueError("vendored FDR reference SHA256 differs from authority")

    math_path = context.repository_root / "src" / "fdr_math.py"
    _reject_link_traversal(math_path)
    if not math_path.is_file():
        raise FileNotFoundError(f"FDR math implementation does not exist: {math_path}")
    golden = _run_math_golden(_load_reference(reference_path))
    return {
        "status": "passed",
        "device": "cpu",
        "authority": {
            "commit": DFINE_COMMIT,
            "path": authority_path.relative_to(context.repository_root).as_posix(),
            "sha256": _sha256_file(authority_path),
            "vendored_reference_path": reference_path.relative_to(
                context.repository_root
            ).as_posix(),
            "vendored_reference_sha256": reference_sha256,
            "math_path": math_path.relative_to(context.repository_root).as_posix(),
            "math_sha256": _sha256_file(math_path),
        },
        "math_golden": golden,
    }


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_passed(evidence: Mapping[str, Any], gate: str) -> None:
    if evidence.get("status") != "passed":
        raise ValueError(f"{gate} runner did not return passed evidence")


def _validate_f1(evidence: Mapping[str, Any]) -> dict[str, Any]:
    _require_passed(evidence, "F1")
    if evidence.get("device") != "cpu":
        raise ValueError("F1 must run on CPU")
    checks = _require_mapping(evidence.get("checks"), "F1 checks")
    required = {
        "neutral_encode_decode",
        "cumulative_residual",
        "fgl_zero_stock_exact",
        "classification_stock_exact",
        "matcher_stock_exact",
        "top300_stock_exact",
        "nms_stock_exact",
    }
    if set(checks) != required or not all(checks[name] is True for name in required):
        raise ValueError("F1 isolation checks are incomplete or failed")
    return dict(evidence)


def _validate_f2(evidence: Mapping[str, Any]) -> dict[str, Any]:
    _require_passed(evidence, "F2")
    if evidence.get("device") != "cpu":
        raise ValueError("F2 must run on CPU")
    shapes = _require_mapping(evidence.get("shapes"), "F2 shapes")
    expected_shapes = {
        "corner_logits": [6, 8, 300, 132],
        "boxes": [6, 8, 300, 4],
        "scores": [6, 8, 300, 10],
    }
    if dict(shapes) != expected_shapes:
        raise ValueError("F2 production shapes differ from the frozen protocol")
    cases = _require_mapping(evidence.get("cases"), "F2 cases")
    required_cases = {
        "normal_queries",
        "dn_queries",
        "empty_gt",
        "mixed_empty_gt",
        "boundary_clipping",
        "auxiliary_layers",
        "finite_forward",
        "finite_backward",
    }
    if set(cases) != required_cases or not all(
        cases[name] is True for name in required_cases
    ):
        raise ValueError("F2 edge-case evidence is incomplete or failed")
    amp = _require_mapping(evidence.get("amp"), "F2 AMP")
    if amp != {"enabled": True, "scale": 128.0, "skipped_steps": 0}:
        raise ValueError("F2 AMP evidence differs from the frozen protocol")
    return dict(evidence)


def _equal_authority_pair(
    evidence: Mapping[str, Any], left: str, right: str, label: str
) -> None:
    first = evidence.get(left)
    second = evidence.get(right)
    if not isinstance(first, str) or len(first) != 64 or first != second:
        raise ValueError(f"F3 {label} authority changed")


def validate_4090_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one real RTX 4090, batch-8, fixed-AMP optimizer step."""

    _require_passed(evidence, "F3")
    runtime = _require_mapping(evidence.get("runtime"), "F3 runtime")
    if dict(runtime) != FIXED_RUNTIME:
        differing = sorted(
            key for key, value in FIXED_RUNTIME.items() if runtime.get(key) != value
        )
        raise ValueError(f"F3 frozen runtime differs: {', '.join(differing)}")
    hardware = _require_mapping(evidence.get("hardware"), "F3 hardware")
    if hardware.get("gpu_name") != "NVIDIA GeForce RTX 4090":
        raise ValueError("F3 requires NVIDIA GeForce RTX 4090")
    if hardware.get("device_index") != 0:
        raise ValueError("F3 requires CUDA device index 0")
    if not isinstance(hardware.get("total_memory_bytes"), int) or hardware[
        "total_memory_bytes"
    ] < 24_000_000_000:
        raise ValueError("F3 RTX 4090 memory evidence is invalid")
    if hardware.get("compute_capability") != [8, 9]:
        raise ValueError("F3 requires compute capability 8.9")
    expected_environment = {
        "driver": FDR_PROTOCOL["environment"]["driver"],
        "cuda": FDR_PROTOCOL["environment"]["cuda"],
        "torch": FDR_PROTOCOL["environment"]["torch"],
        "torchvision": FDR_PROTOCOL["environment"]["torchvision"],
    }
    for key, expected in expected_environment.items():
        if hardware.get(key) != expected:
            raise ValueError(f"F3 {key} differs from the frozen environment")

    step = _require_mapping(evidence.get("single_step"), "F3 single_step")
    for field in (
        "real_visdrone_batch",
        "forward",
        "backward",
        "loss_finite",
        "gradients_finite",
        "expected_gradient_coverage",
        "validation_postprocess",
        "checkpoint_roundtrip",
    ):
        if step.get(field) is not True:
            raise ValueError(f"F3 single-step check failed: {field}")
    if step.get("optimizer") != "MuSGD" or step.get("optimizer_steps") != 1:
        raise ValueError("F3 must execute exactly one MuSGD optimizer step")
    if step.get("unexpected_trainable_parameters") != 0:
        raise ValueError("F3 found unexpected trainable parameters")
    if step.get("excluded_components") != []:
        raise ValueError("F3 found excluded FDR components")
    if (
        step.get("amp_scale_before") != 128.0
        or step.get("amp_scale_after") != 128.0
        or step.get("amp_skipped_steps") != 0
    ):
        raise ValueError("F3 fixed AMP evidence is invalid")

    immutability = _require_mapping(evidence.get("immutability"), "F3 immutability")
    _equal_authority_pair(
        immutability, "source_before_sha256", "source_after_sha256", "source"
    )
    _equal_authority_pair(
        immutability,
        "ultralytics_before_sha256",
        "ultralytics_after_sha256",
        "Ultralytics source",
    )
    _equal_authority_pair(
        immutability,
        "baseline_public_state_sha256",
        "fdr_public_state_sha256",
        "public model",
    )
    _equal_authority_pair(
        immutability,
        "baseline_data_order_sha256",
        "fdr_data_order_sha256",
        "data-order",
    )
    return dict(evidence)


def _to_rows(values: Sequence[Sequence[Any]], name: str) -> list[list[float]]:
    try:
        rows = [[float(value) for value in row] for row in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular numeric matrix") from error
    if any(len(row) != 4 for row in rows):
        raise ValueError(f"{name} must have four values per row")
    return rows


def _safe_rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _subset_representation(
    indices: Sequence[int],
    *,
    reference: Sequence[Sequence[float]],
    reconstructed: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    widths: Sequence[float],
    heights: Sequence[float],
) -> dict[str, Any]:
    finite_differences: list[float] = []
    saturated = 0
    for row_index in indices:
        for expected, actual in zip(reference[row_index], reconstructed[row_index]):
            if math.isfinite(expected) and math.isfinite(actual):
                finite_differences.append(abs(actual - expected))
        saturated += sum(
            value <= 0 or value >= 31 for value in targets[row_index] if math.isfinite(value)
        )
    finite_widths = [widths[index] for index in indices if math.isfinite(widths[index])]
    finite_heights = [heights[index] for index in indices if math.isfinite(heights[index])]
    return {
        "count": len(indices),
        "width_mean": sum(finite_widths) / len(finite_widths) if finite_widths else 0.0,
        "height_mean": (
            sum(finite_heights) / len(finite_heights) if finite_heights else 0.0
        ),
        "reconstruction_l1": (
            sum(finite_differences) / len(finite_differences)
            if finite_differences
            else 0.0
        ),
        "reconstruction_max": max(finite_differences, default=0.0),
        "saturation_count": saturated,
        "saturation_rate": _safe_rate(saturated, 4 * len(indices)),
    }


def summarize_representation(
    *,
    reference_boxes: Sequence[Sequence[Any]],
    reconstructed_boxes: Sequence[Sequence[Any]],
    target_indices: Sequence[Sequence[Any]],
    object_widths: Sequence[Any],
    object_heights: Sequence[Any],
) -> dict[str, Any]:
    """Build the frozen F4 reconstruction, saturation, and scale report."""

    reference = _to_rows(reference_boxes, "reference_boxes")
    reconstructed = _to_rows(reconstructed_boxes, "reconstructed_boxes")
    targets = _to_rows(target_indices, "target_indices")
    widths = [float(value) for value in object_widths]
    heights = [float(value) for value in object_heights]
    count = len(reference)
    if not (
        len(reconstructed) == len(targets) == len(widths) == len(heights) == count
    ):
        raise ValueError("F4 representation inputs must have matching row counts")

    all_rows = [*reference, *reconstructed, *targets]
    nonfinite_values = sum(
        not math.isfinite(value) for row in all_rows for value in row
    ) + sum(not math.isfinite(value) for value in [*widths, *heights])
    nonfinite_rows = sum(
        any(
            not math.isfinite(value)
            for value in (
                *reference[index],
                *reconstructed[index],
                *targets[index],
                widths[index],
                heights[index],
            )
        )
        for index in range(count)
    )
    invalid_boxes = sum(
        all(math.isfinite(value) for value in reconstructed[index])
        and (reconstructed[index][2] <= 0 or reconstructed[index][3] <= 0)
        for index in range(count)
    )

    finite_differences = [
        abs(actual - expected)
        for expected_row, actual_row in zip(reference, reconstructed)
        for expected, actual in zip(expected_row, actual_row)
        if math.isfinite(expected) and math.isfinite(actual)
    ]
    per_edge: dict[str, dict[str, Any]] = {}
    total_saturated = 0
    for edge_index, edge in enumerate(EDGE_NAMES):
        edge_values = [row[edge_index] for row in targets]
        edge_count = sum(
            value <= 0 or value >= 31 for value in edge_values if math.isfinite(value)
        )
        total_saturated += edge_count
        per_edge[edge] = {"count": edge_count, "rate": _safe_rate(edge_count, count)}

    scale_groups: dict[str, list[int]] = {"tiny": [], "small": [], "other": []}
    for index, (width, height) in enumerate(zip(widths, heights)):
        area = width * height
        if math.isfinite(area) and area <= 16.0**2:
            scale_groups["tiny"].append(index)
        elif math.isfinite(area) and area <= 32.0**2:
            scale_groups["small"].append(index)
        else:
            scale_groups["other"].append(index)

    stratification = {
        name: _subset_representation(
            indices,
            reference=reference,
            reconstructed=reconstructed,
            targets=targets,
            widths=widths,
            heights=heights,
        )
        for name, indices in scale_groups.items()
    }
    finite_widths = [value for value in widths if math.isfinite(value)]
    finite_heights = [value for value in heights if math.isfinite(value)]
    return {
        "count": count,
        "reconstruction": {
            "l1": (
                sum(finite_differences) / len(finite_differences)
                if finite_differences
                else 0.0
            ),
            "max": max(finite_differences, default=0.0),
        },
        "saturation": {
            "per_edge": per_edge,
            "total": {
                "count": total_saturated,
                "rate": _safe_rate(total_saturated, 4 * count),
            },
        },
        "invalid_boxes": invalid_boxes,
        "nonfinite_rows": nonfinite_rows,
        "nonfinite_values": nonfinite_values,
        "object_size": {
            "width_mean": sum(finite_widths) / len(finite_widths)
            if finite_widths
            else 0.0,
            "height_mean": sum(finite_heights) / len(finite_heights)
            if finite_heights
            else 0.0,
        },
        "stratification": stratification,
    }


def validate_f4_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on inaccurate, invalid, or non-finite FDR representation."""

    _require_passed(evidence, "F4")
    if evidence.get("official_reference_match") is not True:
        raise ValueError("F4 does not match the pinned official reference")
    tolerance = evidence.get("reconstruction_tolerance")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("F4 reconstruction tolerance is invalid")
    representation = _require_mapping(
        evidence.get("representation"), "F4 representation"
    )
    reconstruction = _require_mapping(
        representation.get("reconstruction"), "F4 reconstruction"
    )
    for field in ("l1", "max"):
        value = reconstruction.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"F4 reconstruction {field} is non-finite")
    if reconstruction["max"] > tolerance:
        raise ValueError("F4 reconstruction error exceeds its frozen tolerance")
    if representation.get("nonfinite_rows") != 0 or representation.get(
        "nonfinite_values"
    ) != 0:
        raise ValueError("F4 representation contains non-finite values")
    if representation.get("invalid_boxes") != 0:
        raise ValueError("F4 representation contains invalid boxes")
    saturation = _require_mapping(
        representation.get("saturation"), "F4 saturation"
    )
    per_edge = _require_mapping(saturation.get("per_edge"), "F4 per-edge saturation")
    if set(per_edge) != set(EDGE_NAMES) or not isinstance(saturation.get("total"), Mapping):
        raise ValueError("F4 saturation report is incomplete")
    stratification = _require_mapping(
        representation.get("stratification"), "F4 stratification"
    )
    if not {"tiny", "small"}.issubset(stratification):
        raise ValueError("F4 tiny/small stratification is incomplete")
    _require_mapping(representation.get("object_size"), "F4 object size")
    return dict(evidence)


def _validate_gate_evidence(gate: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    if gate == "F1":
        return _validate_f1(evidence)
    if gate == "F2":
        return _validate_f2(evidence)
    if gate == "F3":
        return validate_4090_evidence(evidence)
    if gate == "F4":
        return validate_f4_evidence(evidence)
    raise ValueError(f"no injectable validator exists for {gate}")


_DEFAULT_RUNNER_SYMBOLS = {
    "F1": "run_f1_preflight",
    "F2": "run_f2_preflight",
    "F3": "run_f3_preflight",
    "F4": "run_f4_representation_preflight",
}


def _load_default_runner(gate: str) -> GateRunner:
    """Delay model import until an ordered gate actually reaches execution."""

    module = importlib.import_module("src.rtdetr_fdr")
    symbol = _DEFAULT_RUNNER_SYMBOLS[gate]
    runner = getattr(module, symbol, None)
    if not callable(runner):
        raise RuntimeError(
            f"src.rtdetr_fdr must expose callable {symbol}(PreflightContext)"
        )
    return runner


def _failure_payload(gate: str, error: BaseException) -> dict[str, Any]:
    return {
        "status": "engineering_failed",
        "gate": gate,
        "error_type": type(error).__name__,
        "reason": str(error),
    }


def run_preflight(
    context: PreflightContext,
    *,
    gate_runners: Mapping[str, GateRunner] | None = None,
) -> dict[str, Any]:
    """Run F0-F4 once, in order, and publish immutable fail-closed evidence."""

    _validate_authority_inputs(context)
    _reject_link_traversal(context.report_root)
    if context.report_root.exists():
        raise FileExistsError(f"preflight report root already exists: {context.report_root}")
    context.report_root.mkdir(parents=True, exist_ok=False)
    _reject_link_traversal(context.report_root)

    supplied = dict(gate_runners or {})
    unknown = sorted(set(supplied) - set(GATE_ORDER[1:]))
    if unknown:
        raise ValueError(f"unknown injectable preflight gates: {unknown}")
    states: dict[str, str] = {}
    report_hashes: dict[str, str] = {}
    blocked_by: str | None = None

    for gate in GATE_ORDER:
        if blocked_by is not None:
            evidence: dict[str, Any] = {
                "status": "blocked",
                "gate": gate,
                "blocked_by": blocked_by,
            }
        else:
            raw_evidence: Mapping[str, Any] | None = None
            try:
                if gate == "F0":
                    evidence = run_f0(context)
                else:
                    runner = supplied.get(gate) or _load_default_runner(gate)
                    raw = runner(context)
                    raw_evidence = _require_mapping(raw, f"{gate} runner evidence")
                    if raw_evidence.get("status") == "engineering_failed":
                        evidence = dict(raw_evidence)
                    else:
                        evidence = _validate_gate_evidence(gate, raw_evidence)
            except Exception as error:  # evidence must survive engineering failures
                evidence = _failure_payload(gate, error)
                if gate == "F4" and raw_evidence is not None:
                    representation = raw_evidence.get("representation")
                    if isinstance(representation, Mapping):
                        try:
                            canonical_json_bytes(representation)
                        except (TypeError, ValueError):
                            pass
                        else:
                            evidence["representation"] = dict(representation)

        status = str(evidence.get("status", "engineering_failed"))
        if status not in {"passed", "engineering_failed", "blocked"}:
            evidence = {
                "status": "engineering_failed",
                "gate": gate,
                "error_type": "EvidenceStatusError",
                "reason": f"invalid gate status: {status!r}",
            }
            status = "engineering_failed"
        if status != "passed" and blocked_by is None:
            blocked_by = gate
        states[gate] = status
        record = make_evidence_record(gate, evidence)
        write_create_only_report(context.report_root / f"{gate}.json", record)
        report_hashes[gate] = canonical_sha256(record)

    decision = decide_preflight(states)
    decision.update(
        {
            "gate_states": states,
            "gate_report_sha256": report_hashes,
            "fixed_runtime": dict(FIXED_RUNTIME),
            "screen_authority": dict(SCREEN_AUTHORITY),
        }
    )
    write_create_only_report(
        context.report_root / "decision.json",
        make_evidence_record("decision", decision),
    )
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run immutable F0-F4 gates for the frozen FDR-only experiment."
    )
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = PreflightContext(
        protocol_manifest=args.protocol_manifest,
        baseline_checkpoint=args.baseline_checkpoint,
        dataset_root=args.dataset_root,
        report_root=args.report_root,
        repository_root=REPOSITORY_ROOT,
    )
    decision = run_preflight(context)
    print(canonical_json_bytes(decision).decode("utf-8"))
    return 0 if decision["screen_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
