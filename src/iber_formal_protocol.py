"""Frozen authority for seed0 full-model IBER-BE formal training."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from src.lpr_protocol import EXPECTED_SOURCE_SHA256, state_fingerprint


FORMAL_DESIGN_VERSION = "iber-be-v1.0-signed-formal100"
FORMAL_FORMAT_VERSION = 3
FORMAL_EPOCHS = 100
FORMAL_TRAIN_COUNT = 6471
FORMAL_VAL_COUNT = 548
FORMAL_CLASS_COUNT = 10
FORMAL_INITIAL_STATE_VERSION = 1
PRIVATE_PARAMETER_MARKER = "iber_refiner."
EXPECTED_DATASET_SHA256 = (
    "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
)


FORMAL_FROZEN_PROTOCOL: dict[str, Any] = {
    "model": "rtdetr-l.yaml",
    "epochs": FORMAL_EPOCHS,
    "imgsz": 640,
    "batch": 8,
    "workers": 8,
    "device": "0",
    "fraction": 1.0,
    "pretrained": False,
    "cache": False,
    "amp": True,
    "amp_scale": 128.0,
    "seed": 0,
    "deterministic": True,
    "nbs": 64,
    "query_count": 300,
    "nms": False,
    "max_det": 300,
    "save": True,
    "save_period": 1,
    "optimizer": "MuSGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.0,
    "cos_lr": False,
    "plots": True,
    "val": True,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "mixup": 0.0,
    "scale": 0.5,
    "translate": 0.1,
    "perspective": 0.0,
    "degrees": 0.0,
    "shear": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "cutmix": 0.0,
    "copy_paste": 0.0,
}

_NON_ULTRALYTICS_FIELDS = frozenset(("amp_scale", "query_count"))


def validate_formal_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Reject any identity or full-data drift before model construction."""
    if not isinstance(manifest, Mapping):
        raise ValueError("formal manifest must be a mapping")
    candidate = dict(manifest)
    expected_identity = {
        "format_version": FORMAL_FORMAT_VERSION,
        "design_version": FORMAL_DESIGN_VERSION,
        "seed": 0,
    }
    for field, expected in expected_identity.items():
        if candidate.get(field) != expected:
            raise ValueError(
                f"formal manifest {field} mismatch: "
                f"expected={expected!r}, actual={candidate.get(field)!r}"
            )

    expected_dataset = {
        "sha256": EXPECTED_DATASET_SHA256,
        "train_images": FORMAL_TRAIN_COUNT,
        "val_images": FORMAL_VAL_COUNT,
        "classes": FORMAL_CLASS_COUNT,
    }
    if candidate.get("dataset") != expected_dataset:
        raise ValueError(
            "formal manifest dataset mismatch: "
            f"expected={expected_dataset!r}, actual={candidate.get('dataset')!r}"
        )
    formal_data = candidate.get("data", {}).get("formal", {})
    if not isinstance(formal_data, Mapping) or not isinstance(formal_data.get("path"), str):
        raise ValueError("formal manifest data.formal.path is missing")
    if not isinstance(candidate.get("dataset_root"), str):
        raise ValueError("formal manifest dataset_root is missing")
    if candidate.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("formal manifest source_sha256 mismatch")
    return candidate


def build_formal_initial_state(
    control_state: Mapping[str, torch.Tensor],
    method_state: Mapping[str, torch.Tensor],
    *,
    seed: int,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one immutable from-scratch public/private initialization artifact."""
    if seed != 0:
        raise ValueError("formal IBER-BE initial state is frozen to seed0")
    common_names = set(control_state)
    method_names = set(method_state)
    missing = common_names - method_names
    if missing:
        raise ValueError(f"IBER-BE state is missing public tensors: {sorted(missing)[:5]}")
    for name in common_names:
        if control_state[name].shape != method_state[name].shape:
            raise ValueError(f"public tensor shape mismatch: {name}")
        if not torch.equal(control_state[name], method_state[name]):
            raise ValueError(f"public tensor initialization mismatch: {name}")
    private_names = method_names - common_names
    if not private_names or any(PRIVATE_PARAMETER_MARKER not in name for name in private_names):
        raise ValueError("initial state contains missing or unapproved IBER-BE tensors")

    common = {
        name: value.detach().cpu().clone()
        for name, value in control_state.items()
    }
    private = {
        name: method_state[name].detach().cpu().clone()
        for name in sorted(private_names)
    }
    return {
        "format_version": FORMAL_INITIAL_STATE_VERSION,
        "design_version": FORMAL_DESIGN_VERSION,
        "seed": 0,
        "common_state": common,
        "iber_state": private,
        "metadata": {"pretrained": False, **dict(metadata or {})},
        "fingerprints": {
            "common": state_fingerprint(common),
            "iber": state_fingerprint(private),
        },
    }


def validate_formal_initial_state(
    artifact: Mapping[str, Any],
) -> None:
    """Validate identity, key isolation, and all tensor fingerprints."""
    if artifact.get("format_version") != FORMAL_INITIAL_STATE_VERSION:
        raise ValueError("formal IBER-BE initial-state format mismatch")
    if artifact.get("design_version") != FORMAL_DESIGN_VERSION:
        raise ValueError("formal IBER-BE initial-state design mismatch")
    if artifact.get("seed") != 0:
        raise ValueError("formal IBER-BE initial state must use seed0")
    common = artifact.get("common_state")
    private = artifact.get("iber_state")
    if not isinstance(common, Mapping) or not common:
        raise ValueError("formal IBER-BE common initial state is missing")
    if not isinstance(private, Mapping) or not private:
        raise ValueError("formal IBER-BE private initial state is missing")
    if any(PRIVATE_PARAMETER_MARKER in name for name in common):
        raise ValueError("formal common initial state contains private tensors")
    if any(PRIVATE_PARAMETER_MARKER not in name for name in private):
        raise ValueError("formal private initial-state keys are invalid")
    fingerprints = artifact.get("fingerprints", {})
    if state_fingerprint(common) != fingerprints.get("common"):
        raise ValueError("formal common initial-state fingerprint mismatch")
    if state_fingerprint(private) != fingerprints.get("iber"):
        raise ValueError("formal private initial-state fingerprint mismatch")
    if artifact.get("metadata", {}).get("pretrained") is not False:
        raise ValueError("formal initial state must declare pretrained=False")


def load_formal_initial_state(
    model: torch.nn.Module,
    artifact: Mapping[str, Any],
) -> None:
    """Strictly load the shared random public state and signed private state."""
    validate_formal_initial_state(artifact)
    expected = dict(artifact["common_state"])
    expected.update(artifact["iber_state"])
    model_names = set(model.state_dict())
    if model_names != set(expected):
        raise ValueError(
            "formal initial-state keys do not match model: "
            f"missing={sorted(model_names - set(expected))[:5]}, "
            f"unexpected={sorted(set(expected) - model_names)[:5]}"
        )
    model.load_state_dict(expected, strict=True)


def build_formal_settings(
    manifest: Mapping[str, Any],
    *,
    project: str | Path,
    name: str,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    """Build exact Ultralytics overrides with no scientific degrees of freedom."""
    authority = validate_formal_manifest(manifest)
    settings = {
        key: value
        for key, value in FORMAL_FROZEN_PROTOCOL.items()
        if key not in _NON_ULTRALYTICS_FIELDS
    }
    settings.update(
        {
            "data": authority["data"]["formal"]["path"],
            "project": str(Path(project).resolve()),
            "name": str(name),
            "exist_ok": False,
        }
    )
    if resume is not None:
        settings["resume"] = str(Path(resume).resolve())
    return settings


__all__ = [
    "EXPECTED_DATASET_SHA256",
    "FORMAL_CLASS_COUNT",
    "FORMAL_DESIGN_VERSION",
    "FORMAL_EPOCHS",
    "FORMAL_FORMAT_VERSION",
    "FORMAL_FROZEN_PROTOCOL",
    "FORMAL_INITIAL_STATE_VERSION",
    "FORMAL_TRAIN_COUNT",
    "FORMAL_VAL_COUNT",
    "build_formal_settings",
    "build_formal_initial_state",
    "load_formal_initial_state",
    "validate_formal_initial_state",
    "validate_formal_manifest",
]
