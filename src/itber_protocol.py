"""Immutable scientific authorities and Gate 0 guards for I-TBER v1.1."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from src.lpr_protocol import state_fingerprint


EXPECTED_BASELINE_SHA256 = "54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B"
EXPECTED_DATASET_SHA256 = "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
EXPECTED_SUBSET_SHA256 = "52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0"
EXPECTED_CATEGORY_SHA256 = "1455A1F5D9FA9988799815B36CF2CE6D5044B5BD7CAD7CC614D6B5E5059EF2A6"
EXPECTED_SOURCE_SHA256 = {
    "head.py": "5701116D86881827AC9E1E7462DFAA44C33937BD68E23324763459685729E06F",
    "tasks.py": "B00935C1851BB9CEA240985704C12E654E68B369F6C59DE20E45FA295CB79B92",
    "rtdetr-l.yaml": "85716F626769CB5DDF00D59FCF6CAFB5814AAD196328100BDC7C93306F650E83",
}
EXPECTED_ENVIRONMENT = {
    "gpu": "NVIDIA GeForce RTX 4090",
    "driver": "550.142",
    "python": "3.10.12",
    "torch": "2.5.1+cu121",
    "torchvision": "0.20.1+cu121",
    "cuda": "12.1",
    "ultralytics": "8.4.90",
}


class ProtocolViolation(ValueError):
    """One or more immutable I-TBER authorities were violated."""

    def __init__(self, violations: Mapping[str, Mapping[str, Any]]) -> None:
        self.violations = dict(violations)
        super().__init__("I-TBER protocol violation: " + ", ".join(sorted(self.violations)))


def _hash_violation(
    violations: dict[str, dict[str, Any]],
    name: str,
    actual: str,
    expected: str,
) -> None:
    normalized = str(actual).upper()
    if normalized != expected:
        violations[name] = {"expected": expected, "actual": normalized}


def validate_authorities(
    *,
    baseline_sha256: str,
    dataset_sha256: str,
    subset_sha256: str,
    category_sha256: str,
    source_sha256: Mapping[str, str],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every frozen artifact and runtime field in one operation."""
    violations: dict[str, dict[str, Any]] = {}
    _hash_violation(
        violations,
        "baseline_sha256",
        baseline_sha256,
        EXPECTED_BASELINE_SHA256,
    )
    _hash_violation(
        violations, "dataset_sha256", dataset_sha256, EXPECTED_DATASET_SHA256
    )
    _hash_violation(
        violations, "subset_sha256", subset_sha256, EXPECTED_SUBSET_SHA256
    )
    _hash_violation(
        violations, "category_sha256", category_sha256, EXPECTED_CATEGORY_SHA256
    )
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        actual = str(source_sha256.get(name, "")).upper()
        if actual != expected:
            violations[f"source_sha256.{name}"] = {
                "expected": expected,
                "actual": actual or None,
            }
    for name, expected in EXPECTED_ENVIRONMENT.items():
        actual = environment.get(name)
        if actual != expected:
            violations[f"environment.{name}"] = {
                "expected": expected,
                "actual": actual,
            }
    if violations:
        raise ProtocolViolation(violations)
    return {
        "status": "passed",
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "subset_sha256": EXPECTED_SUBSET_SHA256,
        "category_sha256": EXPECTED_CATEGORY_SHA256,
        "source_sha256": dict(EXPECTED_SOURCE_SHA256),
        "environment": dict(EXPECTED_ENVIRONMENT),
    }


def module_state_sha256(module: nn.Module) -> str:
    """Fingerprint parameters and persistent buffers of a module."""
    return state_fingerprint(module.state_dict())


def assert_detector_frozen(detector: nn.Module) -> None:
    """Reject train mode, trainable parameters, or stale detector gradients."""
    violations: dict[str, dict[str, Any]] = {}
    if detector.training:
        violations["detector.training"] = {"expected": False, "actual": True}
    trainable = [name for name, parameter in detector.named_parameters() if parameter.requires_grad]
    if trainable:
        violations["detector.requires_grad"] = {
            "expected": [],
            "actual": trainable[:10],
        }
    gradients = [name for name, parameter in detector.named_parameters() if parameter.grad is not None]
    if gradients:
        violations["detector.grad"] = {"expected": [], "actual": gradients[:10]}
    if violations:
        raise ProtocolViolation(violations)


def write_immutable_report(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Create a report exactly once and make it read-only on POSIX hosts."""
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise ValueError("immutable report path must end in .json")
    for parent in (destination.parent, *destination.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError("immutable report path cannot traverse a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if os.name != "nt":
        destination.chmod(0o444)
    return destination
