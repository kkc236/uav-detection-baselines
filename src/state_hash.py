"""Stable tensor-state fingerprints used by training evidence."""

from __future__ import annotations

import hashlib
from typing import Mapping

import torch


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor keys, dtypes, shapes, and exact contiguous bytes."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state entry {name!r} is not a tensor")
        if value.is_quantized or value.layout is not torch.strided:
            raise TypeError(f"state entry {name!r} cannot be byte-fingerprinted")
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


__all__ = ["state_sha256"]
