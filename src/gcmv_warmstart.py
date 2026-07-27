"""Warm-start authority and state isolation for the GCMV-EI diagnostic."""

from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.ascv_loc_protocol import state_fingerprint


EXPECTED_BASELINE_SHA256 = (
    "54CE60289DD34C6750B8BA5F7516EEF"
    "CF3AFEF6C174C6E4F3B1EF810C883099B"
)
PLEC_EXTRA_PREFIXES = ("plec.", "gcmv_injector.")
MODULE_ARTIFACT_SCHEMA = "gcmv-ei-calibrated-module/v1"


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _checkpoint_module(payload: dict[str, Any]) -> nn.Module:
    candidate = payload.get("ema") or payload.get("model")
    if not isinstance(candidate, nn.Module):
        raise ValueError("baseline checkpoint must contain a model module")
    return candidate.float()


def load_baseline_detector_state(
    model: nn.Module,
    payload: dict[str, Any],
) -> None:
    """Load all and only stock detector keys from a weights-only checkpoint."""

    if not isinstance(payload, dict):
        raise TypeError("baseline checkpoint payload must be a mapping")
    baseline_state = _checkpoint_module(payload).state_dict()
    model_state = model.state_dict()
    extra_names = {
        name
        for name in model_state
        if name.startswith(PLEC_EXTRA_PREFIXES)
    }
    detector_names = set(model_state) - extra_names
    if set(baseline_state) != detector_names:
        raise ValueError(
            "baseline detector keys do not match GCMV stock keys: "
            f"missing={sorted(detector_names - set(baseline_state))}, "
            f"unexpected={sorted(set(baseline_state) - detector_names)}"
        )
    for name in detector_names:
        if baseline_state[name].shape != model_state[name].shape:
            raise ValueError(f"baseline tensor shape mismatch: {name}")
    incompatible = model.load_state_dict(baseline_state, strict=False)
    if set(incompatible.missing_keys) != extra_names:
        raise RuntimeError("GCMV module initialization drift during baseline load")
    if incompatible.unexpected_keys:
        raise RuntimeError("baseline load produced unexpected keys")


def load_baseline_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    expected_sha256: str = EXPECTED_BASELINE_SHA256,
) -> dict[str, Any]:
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    observed = sha256_file(checkpoint)
    if observed != expected_sha256.upper():
        raise ValueError(
            "RTX4090 baseline checksum mismatch: "
            f"expected={expected_sha256.upper()} actual={observed}"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    load_baseline_detector_state(model, payload)
    return {
        "path": checkpoint.as_posix(),
        "sha256": observed,
        "train_args": payload.get("train_args", {}),
    }


def _module_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in model.state_dict().items()
        if name.startswith(PLEC_EXTRA_PREFIXES)
    }


def build_module_artifact(model: nn.Module) -> dict[str, Any]:
    state = _module_state(model)
    if not state:
        raise ValueError("model has no GCMV module state")
    return {
        "schema_version": MODULE_ARTIFACT_SCHEMA,
        "module_state": state,
        "fingerprint": state_fingerprint(state),
    }


def load_module_artifact(
    model: nn.Module,
    artifact: dict[str, Any],
) -> None:
    if not isinstance(artifact, dict):
        raise TypeError("module artifact must be a mapping")
    if artifact.get("schema_version") != MODULE_ARTIFACT_SCHEMA:
        raise ValueError("unsupported calibrated-module schema")
    state = artifact.get("module_state")
    if not isinstance(state, dict) or not state:
        raise ValueError("calibrated artifact has no module state")
    if any(
        not isinstance(name, str)
        or not name.startswith(PLEC_EXTRA_PREFIXES)
        for name in state
    ):
        raise ValueError("calibrated artifact must be module-only")
    if state_fingerprint(state) != artifact.get("fingerprint"):
        raise ValueError("calibrated module fingerprint mismatch")
    expected = set(_module_state(model))
    if set(state) != expected:
        raise ValueError(
            "calibrated module keys do not match: "
            f"missing={sorted(expected - set(state))}, "
            f"unexpected={sorted(set(state) - expected)}"
        )
    detector_names = {
        name
        for name in model.state_dict()
        if not name.startswith(PLEC_EXTRA_PREFIXES)
    }
    detector_before = {
        name: model.state_dict()[name].detach().clone()
        for name in detector_names
    }
    incompatible = model.load_state_dict(state, strict=False)
    if not set(incompatible.missing_keys).issubset(detector_names):
        raise RuntimeError("module state was not completely loaded")
    if incompatible.unexpected_keys:
        raise RuntimeError("module artifact produced unexpected keys")
    if any(
        not torch.equal(model.state_dict()[name], value)
        for name, value in detector_before.items()
    ):
        raise RuntimeError("detector state changed during module-only load")


def open_residual_scalar(model: nn.Module, *, gamma: float) -> None:
    if not math.isfinite(gamma) or not 0.0 < gamma < 1.0:
        raise ValueError("diagnostic gamma must be finite and in (0,1)")
    rho = getattr(
        getattr(getattr(model, "gcmv_injector", None), "peg", None),
        "rho",
        None,
    )
    if not isinstance(rho, nn.Parameter) or rho.numel() != 1:
        raise ValueError("model has no scalar PEG rho parameter")
    value = math.atanh(gamma)
    with torch.no_grad():
        rho.fill_(value)


def split_warmstart_optimizer_groups(
    optimizer: torch.optim.Optimizer,
    *,
    model: nn.Module,
    detector_lr: float,
    module_lr: float,
    rho_lr: float,
    include_module: bool,
    include_detector: bool = True,
    include_rho: bool = True,
) -> None:
    """Split existing optimizer groups without changing optimizer semantics."""

    for value, label in (
        (detector_lr, "detector_lr"),
        (module_lr, "module_lr"),
        (rho_lr, "rho_lr"),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be finite and positive")
    named = dict(model.named_parameters())
    rho_ids = {
        id(parameter)
        for name, parameter in named.items()
        if name == "gcmv_injector.peg.rho"
    }
    module_ids = {
        id(parameter)
        for name, parameter in named.items()
        if name.startswith(PLEC_EXTRA_PREFIXES)
    } - rho_ids
    detector_ids = {
        id(parameter)
        for name, parameter in named.items()
        if not name.startswith(PLEC_EXTRA_PREFIXES)
    }
    roles = (
        ("detector", detector_ids, detector_lr, include_detector),
        ("module", module_ids, module_lr, include_module),
        ("rho", rho_ids, rho_lr, include_module and include_rho),
    )
    split_groups: list[dict[str, Any]] = []
    assigned: set[int] = set()
    for original in optimizer.param_groups:
        template = {
            key: value
            for key, value in original.items()
            if key not in {"params", "lr", "initial_lr", "gcmv_role"}
        }
        original_parameters = list(original["params"])
        for role, allowed_ids, learning_rate, enabled in roles:
            if not enabled:
                continue
            selected = [
                parameter
                for parameter in original_parameters
                if id(parameter) in allowed_ids
            ]
            if not selected:
                continue
            selected_ids = {id(parameter) for parameter in selected}
            if assigned & selected_ids:
                raise RuntimeError("optimizer parameter assigned twice")
            assigned.update(selected_ids)
            group = {
                **template,
                "params": selected,
                "lr": learning_rate,
                "gcmv_role": role,
            }
            if role == "rho":
                group["weight_decay"] = 0.0
            split_groups.append(group)
    expected_ids = set()
    if include_detector:
        expected_ids.update(detector_ids)
    if include_module:
        expected_ids.update(module_ids)
        if include_rho:
            expected_ids.update(rho_ids)
    if assigned != expected_ids:
        raise RuntimeError(
            "optimizer split did not cover the requested parameters"
        )
    optimizer.param_groups[:] = split_groups
