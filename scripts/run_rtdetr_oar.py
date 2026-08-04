"""Run the frozen all-pair Objective-Aligned Reranker offline gate."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.oar_protocol import OAR_GAIN_RECOVERY, OAR_K_GRID  # noqa: E402
from src.rtdetr_oar import select_candidate_k  # noqa: E402


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _atomic_create_bytes(path: Path, payload: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest().upper()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable report drift: {path}") from None
        return digest
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return digest


def _metric(value: Any) -> Decimal:
    metric = Decimal(str(value))
    if not metric.is_finite():
        raise ValueError("D0 metrics must be finite")
    return metric


def _sparse_d0_reports(
    *,
    decomposition: Mapping[str, Mapping[str, Any]],
    restricted_map: Mapping[int, Any],
) -> dict[str, dict[str, Any]]:
    """Build the immutable sparse-D0 failure and all-pair handoff reports."""
    required = ("stock", "presence", "query_iou", "same_class")
    if tuple(decomposition) != required:
        raise ValueError(f"decomposition must have ordered families {required}")
    for family in required:
        if "map" not in decomposition[family] or "ap75" not in decomposition[family]:
            raise ValueError(f"{family} metrics must contain map and ap75")
        for value in decomposition[family].values():
            _metric(value)

    stock_map = decomposition["stock"]["map"]
    full_map = decomposition["same_class"]["map"]
    selection = select_candidate_k(
        stock_map=stock_map,
        full_map=full_map,
        restricted_map=restricted_map,
    )
    stock = _metric(stock_map)
    full_gain = _metric(full_map) - stock
    if full_gain <= 0:
        raise ValueError("same-class D0 oracle must improve over stock")

    coverage: dict[str, dict[str, Any]] = {}
    for k in OAR_K_GRID:
        candidate_map = _metric(restricted_map[k])
        coverage[str(k)] = {
            "map": str(candidate_map),
            "map_delta": str(candidate_map - stock),
            "recovered": str((candidate_map - stock) / full_gain),
        }

    decomposition_report = {
        "format_version": 1,
        "identity": "oar-sparse-d0-oracle-decomposition",
        "metrics": {
            family: {name: str(_metric(value)) for name, value in metrics.items()}
            for family, metrics in decomposition.items()
        },
    }
    coverage_report = {
        "format_version": 1,
        "identity": "oar-sparse-d0-k-coverage",
        "recovery_threshold": str(OAR_GAIN_RECOVERY),
        "coverage": coverage,
    }
    decision_report = {
        "format_version": 1,
        "identity": "oar-sparse-d0-decision",
        "status": selection["status"],
        "selected_k": selection["selected_k"],
        "frozen_k_grid": list(OAR_K_GRID),
        "grid_extended": False,
        "next_authority": "oar-all-pair-amendment",
    }
    return {
        "d0-oracle-decomposition.json": decomposition_report,
        "d0-k-coverage.json": coverage_report,
        "sparse-d0-decision.json": decision_report,
    }


def _write_sparse_d0_reports(
    root: Path,
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    expected = (
        "d0-oracle-decomposition.json",
        "d0-k-coverage.json",
        "sparse-d0-decision.json",
    )
    if set(reports) != set(expected):
        raise ValueError(f"reports must contain exactly {expected}")
    return {
        name: _atomic_create_bytes(Path(root) / name, _canonical_json_bytes(reports[name]))
        for name in expected
    }
