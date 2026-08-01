"""Immutable protocol authority for trajectory-free IBER-BE v1.0."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


DESIGN_VERSION = "iber-be-v1.0"
PROBES = frozenset(("b0", "b1", "b2", "b3"))
PROBE_EPOCHS = 12
SCREEN_EPOCHS = 30
SCREEN_TRAIN_COUNT = 647
SCREEN_VAL_COUNT = 548
PRIVATE_SEED = 10_000
PRIVATE_OPTIMIZER = {
    "name": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "betas": (0.9, 0.999),
    "clip": 10.0,
}
EXPECTED_BASELINE_SHA256 = (
    "54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B"
)
EXPECTED_DATASET_SHA256 = (
    "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
)
EXPECTED_SUBSET_SHA256 = (
    "52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0"
)

EXECUTION_ENVIRONMENT = {
    "gpu": "NVIDIA GeForce RTX 4090",
    "reported_memory_mib": 49140,
    "driver": "570.133.07",
    "python": "3.10.12",
    "torch": "2.5.1+cu121",
    "torchvision": "0.20.1+cu121",
    "cuda": "12.1",
    "ultralytics": "8.4.90",
}
RUNTIME_AMENDMENT = {
    "amendment_id": "iber-be-v1.0-runtime-driver-2026-08-01",
    "approved_on": "2026-08-01",
    "baseline_driver": "550.142",
    "execution_driver": "570.133.07",
    "allowed_differences": ["driver", "reported_memory_mib"],
    "comparison": "same-checkpoint-stock-vs-refined",
}
SCREEN_CONTRACT = {
    "seed": 0,
    "epochs": SCREEN_EPOCHS,
    "imgsz": 640,
    "batch": 8,
    "workers": 8,
    "amp_scale": 128.0,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "max_det": 300,
    "nms": False,
}


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest().upper()


RUNTIME_AMENDMENT_SHA256 = _canonical_sha256(RUNTIME_AMENDMENT)
PROTOCOL_PAYLOAD = {
    "design_version": DESIGN_VERSION,
    "probes": sorted(PROBES),
    "probe_epochs": PROBE_EPOCHS,
    "screen_epochs": SCREEN_EPOCHS,
    "screen_train_count": SCREEN_TRAIN_COUNT,
    "screen_val_count": SCREEN_VAL_COUNT,
    "private_seed": PRIVATE_SEED,
    "private_optimizer": dict(PRIVATE_OPTIMIZER),
    "expected_sha256": {
        "baseline": EXPECTED_BASELINE_SHA256,
        "dataset": EXPECTED_DATASET_SHA256,
        "subset": EXPECTED_SUBSET_SHA256,
    },
    "execution_environment": dict(EXECUTION_ENVIRONMENT),
    "runtime_amendment": dict(RUNTIME_AMENDMENT),
    "screen_contract": dict(SCREEN_CONTRACT),
}
PROTOCOL_SHA256 = _canonical_sha256(PROTOCOL_PAYLOAD)


def execution_environment() -> dict[str, Any]:
    """Return a copy of the frozen execution environment."""
    return dict(EXECUTION_ENVIRONMENT)


def validate_screen_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete screen contract without accepting overrides."""
    if not isinstance(contract, Mapping):
        return {
            "status": "engineering_invalid",
            "contract": None,
            "violations": {
                "contract": {
                    "expected": "mapping",
                    "actual": type(contract).__name__,
                }
            },
            "design_version": DESIGN_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
        }

    candidate = dict(contract)
    violations: dict[str, dict[str, Any]] = {}
    for name in sorted(set(SCREEN_CONTRACT) | set(candidate)):
        expected = SCREEN_CONTRACT.get(name)
        actual = candidate.get(name)
        if name not in SCREEN_CONTRACT or name not in candidate:
            violations[name] = {"expected": expected, "actual": actual}
        elif type(actual) is not type(expected) or actual != expected:
            violations[name] = {"expected": expected, "actual": actual}

    return {
        "status": (
            "engineering_invalid"
            if violations
            else "passed_with_runtime_amendment"
        ),
        "contract": candidate,
        "violations": violations,
        "design_version": DESIGN_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
    }


def module_state_sha256(module: nn.Module) -> str:
    """Fingerprint a module's parameters and persistent buffers."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest().upper()


def file_sha256(path: str | Path) -> str:
    """Return an uppercase streaming SHA-256 for a protocol artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_immutable_report(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Create one canonical JSON report and never replace an existing report."""
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise ValueError("immutable report path must end in .json")
    if (
        "design_version" in payload
        and payload["design_version"] != DESIGN_VERSION
    ):
        raise ValueError("immutable IBER report has a foreign design_version")
    serialized = _canonical_json(payload).decode("utf-8")
    for parent in (destination.parent, *destination.parents):
        if parent.is_symlink():
            raise ValueError("immutable report path cannot traverse a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if os.name != "nt":
        destination.chmod(0o444)
    return destination


__all__ = [
    "DESIGN_VERSION",
    "EXECUTION_ENVIRONMENT",
    "EXPECTED_BASELINE_SHA256",
    "EXPECTED_DATASET_SHA256",
    "EXPECTED_SUBSET_SHA256",
    "PRIVATE_OPTIMIZER",
    "PRIVATE_SEED",
    "PROBES",
    "PROBE_EPOCHS",
    "PROTOCOL_PAYLOAD",
    "PROTOCOL_SHA256",
    "RUNTIME_AMENDMENT",
    "RUNTIME_AMENDMENT_SHA256",
    "SCREEN_CONTRACT",
    "SCREEN_EPOCHS",
    "SCREEN_TRAIN_COUNT",
    "SCREEN_VAL_COUNT",
    "execution_environment",
    "file_sha256",
    "module_state_sha256",
    "validate_screen_contract",
    "write_immutable_report",
]
