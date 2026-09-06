"""Safe weight-only transfer state for the ten-class VisDrone Full graph."""

from __future__ import annotations

import string
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from src.fdr_protocol import public_state_sha256


FORMAT_VERSION = 1
ARTIFACT_ROLE = "visdrone_full_transfer"


def _validate_source_sha256(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.upper()
        or any(character not in string.hexdigits.upper() for character in value)
    ):
        raise ValueError("source checkpoint SHA-256 must be 64 uppercase hex characters")
    return value


def _tensor_state(state: Any) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("transfer state dictionary must be a non-empty mapping")
    if any(not isinstance(name, str) or not isinstance(value, torch.Tensor) for name, value in state.items()):
        raise ValueError("transfer state dictionary must contain only named tensors")
    return {
        name: value.detach().cpu().clone()
        for name, value in sorted(state.items())
    }


def build_full_transfer_artifact(
    state: Mapping[str, torch.Tensor],
    *,
    source_checkpoint_sha256: str,
    source_checkpoint_bytes: int,
    nc: int = 10,
    channels: int = 3,
) -> dict[str, Any]:
    """Build a primitive-and-tensor-only Full warm-start artifact."""

    source_sha256 = _validate_source_sha256(source_checkpoint_sha256)
    if isinstance(source_checkpoint_bytes, bool) or not isinstance(source_checkpoint_bytes, int) or source_checkpoint_bytes <= 0:
        raise ValueError("source checkpoint bytes must be a positive integer")
    if nc != 10:
        raise ValueError("VisDrone Full transfer class count must be 10")
    if channels != 3:
        raise ValueError("VisDrone Full transfer input channels must be 3")
    tensors = _tensor_state(state)
    fingerprint = public_state_sha256(tensors)
    return {
        "format_version": FORMAT_VERSION,
        "artifact_role": ARTIFACT_ROLE,
        "metadata": {
            "nc": nc,
            "channels": channels,
            "source_checkpoint_sha256": source_sha256,
            "source_checkpoint_bytes": source_checkpoint_bytes,
            "tensor_count": len(tensors),
        },
        "state_dict": tensors,
        "fingerprints": {"state": fingerprint},
    }


def validate_full_transfer_artifact(
    artifact: Mapping[str, Any],
    *,
    target_state: Mapping[str, torch.Tensor] | None = None,
    expected_nc: int = 10,
) -> dict[str, Any]:
    """Validate schema, authority, fingerprint and optional target contract."""

    if not isinstance(artifact, Mapping):
        raise TypeError("Full transfer artifact must be a mapping")
    if artifact.get("format_version") != FORMAT_VERSION:
        raise ValueError("Full transfer format version mismatch")
    if artifact.get("artifact_role") != ARTIFACT_ROLE:
        raise ValueError("Full transfer artifact role mismatch")
    metadata = artifact.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Full transfer metadata is missing")
    if expected_nc != 10 or metadata.get("nc") != expected_nc:
        raise ValueError("Full transfer class count mismatch")
    if metadata.get("channels") != 3:
        raise ValueError("Full transfer input channel count mismatch")
    source_sha256 = _validate_source_sha256(metadata.get("source_checkpoint_sha256"))
    source_bytes = metadata.get("source_checkpoint_bytes")
    if isinstance(source_bytes, bool) or not isinstance(source_bytes, int) or source_bytes <= 0:
        raise ValueError("source checkpoint bytes must be a positive integer")
    state = _tensor_state(artifact.get("state_dict"))
    if metadata.get("tensor_count") != len(state):
        raise ValueError("Full transfer tensor count mismatch")
    fingerprints = artifact.get("fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise ValueError("Full transfer fingerprints are missing")
    state_sha256 = public_state_sha256(state)
    if fingerprints.get("state") != state_sha256:
        raise ValueError("Full transfer state fingerprint mismatch")

    if target_state is not None:
        target = _tensor_state(target_state)
        if set(state) != set(target):
            missing = sorted(set(target) - set(state))
            extra = sorted(set(state) - set(target))
            raise ValueError(
                f"Full transfer target keys mismatch: missing={missing[:5]}, extra={extra[:5]}"
            )
        for name, value in state.items():
            expected = target[name]
            if value.shape != expected.shape:
                raise ValueError(f"Full transfer target shape mismatch: {name}")
            if value.dtype != expected.dtype:
                raise ValueError(f"Full transfer target dtype mismatch: {name}")

    return {
        "tensor_count": len(state),
        "state_sha256": state_sha256,
        "source_checkpoint_sha256": source_sha256,
        "source_checkpoint_bytes": source_bytes,
    }


def load_full_transfer_file(path: str | Path) -> Mapping[str, Any]:
    """Safely deserialize and validate a Full transfer artifact."""

    requested = Path(path)
    if requested.is_symlink() or not requested.is_file():
        raise FileNotFoundError(f"Full transfer initial state not found: {requested}")
    artifact = torch.load(requested.resolve(), map_location="cpu", weights_only=True)
    if not isinstance(artifact, Mapping):
        raise TypeError("Full transfer artifact must be a mapping")
    validate_full_transfer_artifact(artifact)
    return artifact


def load_full_transfer_initial_state(
    model: nn.Module,
    artifact: Mapping[str, Any],
    *,
    expected_nc: int = 10,
) -> dict[str, Any]:
    """Strict-load a validated transfer state and prove byte equality."""

    report = validate_full_transfer_artifact(
        artifact,
        target_state=model.state_dict(),
        expected_nc=expected_nc,
    )
    state = artifact["state_dict"]
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict Full transfer loading returned incompatible keys")
    loaded = model.state_dict()
    mismatch_count = sum(
        not torch.equal(loaded[name].detach().cpu(), expected.detach().cpu())
        for name, expected in state.items()
    )
    if mismatch_count:
        raise RuntimeError(f"Full transfer tensor loading mismatch: {mismatch_count}")
    return {
        **report,
        "tensor_mismatch_count": mismatch_count,
    }


__all__ = [
    "ARTIFACT_ROLE",
    "FORMAT_VERSION",
    "build_full_transfer_artifact",
    "load_full_transfer_file",
    "load_full_transfer_initial_state",
    "validate_full_transfer_artifact",
]
