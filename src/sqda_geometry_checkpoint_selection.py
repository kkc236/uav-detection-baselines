from __future__ import annotations

import hashlib
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"geometry checkpoint must be a dictionary: {path}")
    return payload


def _adapter_state(path: Path) -> dict[str, torch.Tensor]:
    payload = _load_checkpoint(path)
    source = next(
        (
            payload[key]
            for key in ("ema", "model")
            if isinstance(payload.get(key), nn.Module)
        ),
        None,
    )
    if source is None:
        raise TypeError(f"geometry checkpoint lacks an EMA/model module: {path}")
    adapter = getattr(source, "sqda_sgc", None)
    if not isinstance(adapter, nn.Module):
        raise TypeError(f"geometry checkpoint lacks an SQDA adapter: {path}")
    state = adapter.state_dict()
    selected = {
        key: value.detach().cpu().contiguous()
        for key, value in state.items()
        if key.startswith("geometry_trust.")
    }
    if not selected:
        raise ValueError(f"geometry checkpoint lacks SMGT state: {path}")
    return selected


def geometry_state_fingerprint(path: str | Path) -> str:
    """Fingerprint the complete trainable SMGT state in one checkpoint payload."""
    digest = hashlib.sha256()
    for key, value in sorted(_adapter_state(Path(path)).items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest().upper()


def _epoch_number(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("epoch") or not stem[5:].isdigit():
        raise ValueError(f"checkpoint is not an epoch checkpoint: {path}")
    return int(stem[5:])


def select_trainable_candidates(weights: str | Path) -> list[Path]:
    """Return updated epoch/last checkpoints, never inferring selection from `best.pt`."""
    weights_path = Path(weights).expanduser().resolve()
    initial = weights_path / "epoch0.pt"
    if not initial.is_file():
        raise FileNotFoundError(f"missing initial G1 checkpoint: {initial}")
    initial_fingerprint = geometry_state_fingerprint(initial)
    epoch_paths = sorted(
        (path for path in weights_path.glob("epoch*.pt") if _epoch_number(path) > 0),
        key=_epoch_number,
    )
    last = weights_path / "last.pt"
    ordered = epoch_paths + ([last] if last.is_file() else [])
    selected: list[Path] = []
    seen_fingerprints: set[str] = set()
    for path in ordered:
        fingerprint = geometry_state_fingerprint(path)
        if fingerprint == initial_fingerprint or fingerprint in seen_fingerprints:
            continue
        selected.append(path)
        seen_fingerprints.add(fingerprint)
    return selected


def select_earliest_passing_candidate(
    evaluations: Sequence[tuple[Path, Mapping[str, Any]]],
) -> Path | None:
    """Return the first strictly passing candidate from chronological evaluation records."""
    for path, decision in evaluations:
        if decision.get("passed") is True:
            return path
    return None


def select_earliest_feasible_candidate(
    evaluations: Sequence[tuple[Path, Mapping[str, Any]]],
) -> Path | None:
    """Return the earliest candidate allowed into the independent G2 check."""
    for path, decision in evaluations:
        feasibility = decision.get("g2_feasibility")
        if isinstance(feasibility, Mapping) and feasibility.get("eligible") is True:
            return path
    return None
