"""Frozen seed0 protocol helpers for isolated LPR-G experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from src.lpr_protocol import file_sha256, state_fingerprint

FORMAT_VERSION = 2
PRIVATE_MARKER = "lpr_g_refiner."


def build_lpr_g_initial_state(
    control_state: Mapping[str, torch.Tensor],
    lpr_g_state: Mapping[str, torch.Tensor],
    *,
    seed: int,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a common-state plus LPR-G-private-state initialization artifact."""
    if seed != 0:
        raise ValueError("LPR-G v2 initial state is frozen to seed0")
    common_names = set(control_state)
    method_names = set(lpr_g_state)
    missing = common_names - method_names
    if missing:
        raise ValueError(f"LPR-G state is missing common tensors: {sorted(missing)[:5]}")
    for name in common_names:
        if control_state[name].shape != lpr_g_state[name].shape:
            raise ValueError(f"common tensor shape mismatch: {name}")
    private_names = method_names - common_names
    if not private_names or any(PRIVATE_MARKER not in name for name in private_names):
        raise ValueError("initial state contains missing or unapproved LPR-G-private tensors")

    common = {name: value.detach().cpu().clone() for name, value in control_state.items()}
    private = {
        name: lpr_g_state[name].detach().cpu().clone() for name in sorted(private_names)
    }
    return {
        "format_version": FORMAT_VERSION,
        "seed": seed,
        "common_state": common,
        "lpr_g_state": private,
        "metadata": dict(metadata or {}),
        "fingerprints": {
            "common": state_fingerprint(common),
            "lpr_g": state_fingerprint(private),
        },
    }


def validate_lpr_g_initial_state(artifact: Mapping[str, Any], *, seed: int = 0) -> None:
    """Validate format, seed, key boundaries, and both tensor fingerprints."""
    if artifact.get("format_version") != FORMAT_VERSION:
        raise ValueError("LPR-G initial-state format is invalid")
    if seed != 0 or artifact.get("seed") != seed:
        raise ValueError("LPR-G initial state must use seed0")
    common = artifact.get("common_state", {})
    private = artifact.get("lpr_g_state", {})
    if not private or any(PRIVATE_MARKER not in name for name in private):
        raise ValueError("LPR-G private state keys are invalid")
    fingerprints = artifact.get("fingerprints", {})
    if state_fingerprint(common) != fingerprints.get("common"):
        raise ValueError("LPR-G common fingerprint mismatch")
    if state_fingerprint(private) != fingerprints.get("lpr_g"):
        raise ValueError("LPR-G private fingerprint mismatch")


def load_lpr_g_initial_state(
    model,
    artifact: Mapping[str, Any],
    *,
    variant: str,
) -> None:
    """Strictly load the common control state and optional LPR-G private state."""
    if variant not in {"control", "lprg"}:
        raise ValueError(f"unknown LPR-G paired variant: {variant}")
    validate_lpr_g_initial_state(artifact, seed=0)
    expected = dict(artifact["common_state"])
    if variant == "lprg":
        expected.update(artifact["lpr_g_state"])
    model_names = set(model.state_dict())
    if model_names != set(expected):
        missing = sorted(model_names - set(expected))
        unexpected = sorted(set(expected) - model_names)
        raise ValueError(
            f"LPR-G initial-state keys do not match model: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    model.load_state_dict(expected, strict=True)


def validate_lpr_g_initial_state_file(
    path: str | Path,
    *,
    manifest_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an immutable initial-state file against its protocol manifest."""
    state_path = Path(path).resolve()
    if state_path != Path(manifest_record.get("path", "")).resolve():
        raise ValueError("LPR-G initial-state path does not match protocol manifest")
    if not state_path.is_file():
        raise FileNotFoundError(f"missing LPR-G initial state: {state_path}")
    actual_sha = file_sha256(state_path)
    if actual_sha != manifest_record.get("sha256"):
        raise ValueError("LPR-G initial-state file SHA mismatch")
    artifact = torch.load(state_path, map_location="cpu", weights_only=False)
    validate_lpr_g_initial_state(artifact, seed=0)
    if artifact["fingerprints"] != manifest_record.get("fingerprints"):
        raise ValueError("LPR-G initial-state fingerprints do not match manifest")
    return artifact
