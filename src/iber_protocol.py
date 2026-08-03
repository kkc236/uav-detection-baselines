"""Immutable protocol authority for trajectory-free IBER-BE v1.0."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import nn


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _json_compatible(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


DESIGN_VERSION = "iber-be-v1.0"
PROBES = frozenset(("b0", "b1", "b2", "b3"))
PROBE_EPOCHS = 12
SCREEN_EPOCHS = 30
SCREEN_TRAIN_COUNT = 647
SCREEN_VAL_COUNT = 548
PRIVATE_SEED = 10_000
PRIVATE_OPTIMIZER = _freeze(
    {
        "name": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "betas": (0.9, 0.999),
        "clip": 10.0,
    }
)
BOUNDARY_LOSS_CONTRACT = _freeze(
    {
        "identity": "global-balanced-counterfactual-boundary-v1",
        "enabled_arms": ("b1", "b2", "b3"),
        "direction_margin": 0.05,
        "edge_relative_margin": 0.10,
        "reference_floor_pixels": 1.0,
        "direction_weight": 1.0,
        "edge_margin_weight": 0.01,
        "bucket_balance": "fixed-cache-global-edge-counts",
        "edge_reference": "min(detached_stock,detached_boundary_off)",
        "shared_context_gradient": "detached_auxiliary_only",
    }
)
EXPECTED_BASELINE_SHA256 = (
    "54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B"
)
EXPECTED_DATASET_SHA256 = (
    "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
)
EXPECTED_SUBSET_SHA256 = (
    "52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0"
)

EXECUTION_ENVIRONMENT = _freeze(
    {
        "gpu": "NVIDIA GeForce RTX 4090",
        "reported_memory_mib": 24564,
        "driver": "550.142",
        "python": "3.10.12",
        "torch": "2.5.1+cu121",
        "torchvision": "0.20.1+cu121",
        "cuda": "12.1",
        "ultralytics": "8.4.90",
    }
)
RUNTIME_AMENDMENT = _freeze(
    {
        "amendment_id": "iber-be-v1.0-baseline-aligned-runtime-2026-08-02",
        "approved_on": "2026-08-02",
        "baseline_driver": "550.142",
        "execution_driver": "550.142",
        "allowed_differences": [],
        "comparison": "same-checkpoint-stock-vs-refined",
    }
)
SCREEN_CONTRACT = _freeze(
    {
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
)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _json_compatible(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest().upper()


RUNTIME_AMENDMENT_SHA256 = _canonical_sha256(RUNTIME_AMENDMENT)
PROTOCOL_PAYLOAD = _freeze(
    {
        "design_version": DESIGN_VERSION,
        "probes": sorted(PROBES),
        "probe_epochs": PROBE_EPOCHS,
        "screen_epochs": SCREEN_EPOCHS,
        "screen_train_count": SCREEN_TRAIN_COUNT,
        "screen_val_count": SCREEN_VAL_COUNT,
        "private_seed": PRIVATE_SEED,
        "private_optimizer": PRIVATE_OPTIMIZER,
        "boundary_loss_contract": BOUNDARY_LOSS_CONTRACT,
        "expected_sha256": {
            "baseline": EXPECTED_BASELINE_SHA256,
            "dataset": EXPECTED_DATASET_SHA256,
            "subset": EXPECTED_SUBSET_SHA256,
        },
        "execution_environment": EXECUTION_ENVIRONMENT,
        "runtime_amendment": RUNTIME_AMENDMENT,
        "screen_contract": SCREEN_CONTRACT,
    }
)
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
    violations: dict[Any, dict[str, Any]] = {}
    for name, expected in SCREEN_CONTRACT.items():
        actual = candidate.get(name)
        if name not in candidate:
            violations[name] = {"expected": expected, "actual": actual}
        elif type(actual) is not type(expected) or actual != expected:
            violations[name] = {"expected": expected, "actual": actual}
    for name, actual in candidate.items():
        if name not in SCREEN_CONTRACT:
            violations[name] = {"expected": None, "actual": actual}

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
        if value.is_quantized:
            raise TypeError(
                f"state entry {name!r} is quantized and cannot be fingerprinted"
            )
        if value.layout is not torch.strided:
            raise TypeError(
                f"state entry {name!r} has non-strided layout {value.layout}"
            )
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


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(reparse_flag and file_attributes & reparse_flag)


def _reject_link_or_reparse_traversal(path: Path) -> None:
    for component in (*path.parents, path):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if _is_link_or_reparse(metadata):
            raise ValueError(
                "immutable report path cannot traverse a symlink or reparse point"
            )


def _verify_opened_file_identity(file_descriptor: int, path: Path) -> None:
    try:
        path_metadata = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError("immutable report path identity changed") from error
    if _is_link_or_reparse(path_metadata) or not os.path.samestat(
        os.fstat(file_descriptor), path_metadata
    ):
        raise RuntimeError("immutable report path identity changed")


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
    _reject_link_or_reparse_traversal(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_link_or_reparse_traversal(destination)

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    file_descriptor = os.open(destination, flags, 0o600)
    try:
        _reject_link_or_reparse_traversal(destination)
        _verify_opened_file_identity(file_descriptor, destination)
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
            closefd=False,
        ) as stream:
            stream.write(serialized + "\n")
            stream.flush()
            os.fsync(file_descriptor)
        _reject_link_or_reparse_traversal(destination)
        _verify_opened_file_identity(file_descriptor, destination)
        if os.name == "nt":
            destination.chmod(stat.S_IREAD)
        else:
            os.fchmod(file_descriptor, 0o444)
        _verify_opened_file_identity(file_descriptor, destination)
    finally:
        os.close(file_descriptor)
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
