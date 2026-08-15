"""Run the frozen all-pair Objective-Aligned Reranker offline gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.oar_protocol import OAR_GAIN_RECOVERY, OAR_K_GRID  # noqa: E402
from src.rtdetr_oar import select_candidate_k  # noqa: E402
from src.rtdetr_quality_probe import c1_features  # noqa: E402


EPOCHS = 20
BATCH_SIZE = 8
NUM_CLASSES = 10
METRIC_TOLERANCE = Decimal("1e-12")
FROZEN_D0_DECOMPOSITION: dict[str, dict[str, Decimal]] = {
    "stock": {
        "map": Decimal("0.28628865801344866"),
        "ap75": Decimal("0.292364074762"),
        "ap50": Decimal("0.476388925325"),
    },
    "presence": {
        "map": Decimal("0.294608200682"),
        "ap75": Decimal("0.300115294639"),
    },
    "query_iou": {
        "map": Decimal("0.324185693386"),
        "ap75": Decimal("0.344708985960"),
    },
    "same_class": {
        "map": Decimal("0.409733588907"),
        "ap75": Decimal("0.413238330995"),
    },
}
FROZEN_D0_RESTRICTED_MAP: dict[int, Decimal] = {
    20: Decimal("0.304549967436"),
    40: Decimal("0.335016751522"),
    60: Decimal("0.358000690130"),
    100: Decimal("0.385568106152"),
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--historical-report-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser.parse_args(argv)


def _oar_r2_features(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(hidden, torch.Tensor):
        raise TypeError("hidden must be a tensor")
    if hidden.ndim != 3 or hidden.shape != (*boxes.shape[:2], 256):
        raise ValueError("hidden must have shape [B,Q,256]")
    if hidden.device != boxes.device or logits.device != boxes.device:
        raise ValueError("boxes, logits, and hidden must share a device")
    hidden = hidden.detach().float()
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError("hidden must contain only finite values")
    control = c1_features(boxes, logits, num_classes=NUM_CLASSES)
    expanded = hidden.unsqueeze(2).expand(-1, -1, NUM_CLASSES, -1)
    result = torch.cat((control, expanded), dim=-1).contiguous().detach()
    if result.shape[-1] != 276:
        raise RuntimeError("OAR-R2 feature dimension drift")
    return result


def _select_checkpoint(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not history:
        raise ValueError("checkpoint history is empty")
    checked: list[dict[str, Any]] = []
    for item in history:
        epoch = item.get("epoch")
        metrics = item.get("metrics")
        if type(epoch) is not int or epoch < 1 or not isinstance(metrics, Mapping):
            raise ValueError("checkpoint history is invalid")
        try:
            values = tuple(float(metrics[name]) for name in ("map", "ap75", "ap50"))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("checkpoint metrics are invalid") from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError("checkpoint metrics are invalid")
        checked.append(dict(item))
    return max(
        checked,
        key=lambda item: (
            float(item["metrics"]["map"]),
            float(item["metrics"]["ap75"]),
            float(item["metrics"]["ap50"]),
            -int(item["epoch"]),
        ),
    )


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


def _require_frozen_metric(value: Any, expected: Decimal, label: str) -> Decimal:
    actual = _metric(value)
    if abs(actual - expected) > METRIC_TOLERANCE:
        raise RuntimeError(f"sparse D0 metric drift: {label}")
    return actual


def _sparse_d0_reports(
    *,
    decomposition: Mapping[str, Mapping[str, Any]],
    restricted_map: Mapping[int, Any],
) -> dict[str, dict[str, Any]]:
    """Build the immutable sparse-D0 failure and all-pair handoff reports."""
    required = tuple(FROZEN_D0_DECOMPOSITION)
    if set(decomposition) != set(required):
        raise ValueError(f"decomposition must contain exactly {required}")
    normalized: dict[str, dict[str, Decimal]] = {}
    for family in required:
        expected_metrics = FROZEN_D0_DECOMPOSITION[family]
        actual_metrics = decomposition[family]
        if set(actual_metrics) != set(expected_metrics):
            raise ValueError(
                f"{family} metrics must contain exactly {tuple(expected_metrics)}"
            )
        normalized[family] = {
            name: _require_frozen_metric(
                actual_metrics[name], expected, f"{family}.{name}"
            )
            for name, expected in expected_metrics.items()
        }
    if set(restricted_map) != set(FROZEN_D0_RESTRICTED_MAP):
        raise ValueError(f"restricted_map must contain exactly {OAR_K_GRID}")
    normalized_restricted = {
        k: _require_frozen_metric(
            restricted_map[k], FROZEN_D0_RESTRICTED_MAP[k], f"restricted_map[{k}]"
        )
        for k in OAR_K_GRID
    }

    stock_map = normalized["stock"]["map"]
    full_map = normalized["same_class"]["map"]
    selection = select_candidate_k(
        stock_map=stock_map,
        full_map=full_map,
        restricted_map=normalized_restricted,
    )
    stock = _metric(stock_map)
    full_gain = _metric(full_map) - stock
    if full_gain <= 0:
        raise ValueError("same-class D0 oracle must improve over stock")

    coverage: dict[str, dict[str, Any]] = {}
    for k in OAR_K_GRID:
        candidate_map = normalized_restricted[k]
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
            for family, metrics in normalized.items()
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
    if selection != {"status": "scientific_failed", "selected_k": None}:
        raise RuntimeError("frozen sparse D0 must remain scientific_failed")
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
