"""Common training-state fingerprints for strict LPR-G pairing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn

PRIVATE_MARKER = "lpr_g_refiner."


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _update_tensor(digest, name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    digest.update(b"\n")


def common_model_fingerprint(model: nn.Module) -> str:
    """Hash every non-private model tensor by stable state-dict name."""
    digest = hashlib.sha256()
    for name, value in sorted(_unwrap_model(model).state_dict().items()):
        if PRIVATE_MARKER not in name:
            _update_tensor(digest, name, value)
    return digest.hexdigest().upper()


def common_optimizer_fingerprint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> str:
    """Hash optimizer state associated with non-private named parameters."""
    named = {id(parameter): name for name, parameter in _unwrap_model(model).named_parameters()}
    digest = hashlib.sha256()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = named.get(id(parameter))
            if name is None:
                raise ValueError("optimizer parameter has no model name")
            if PRIVATE_MARKER in name:
                continue
            digest.update(name.encode("utf-8") + b"\0")
            state = optimizer.state.get(parameter, {})
            for key in sorted(state):
                value = state[key]
                if isinstance(value, torch.Tensor):
                    _update_tensor(digest, str(key), value)
                else:
                    digest.update(f"{key}={value!r}\n".encode("utf-8"))
    return digest.hexdigest().upper()


def write_epoch_audit(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict:
    """Atomically append one common model/optimizer fingerprint record."""
    record = {
        "epoch": int(epoch),
        "common_model_sha256": common_model_fingerprint(model),
        "common_optimizer_sha256": common_optimizer_fingerprint(model, optimizer),
    }
    path = Path(path)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        existing + json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return record
