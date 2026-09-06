"""Shared per-epoch diagnostics for current VisDrone and UAVDT runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.rtdetr_lrs_system import SYSTEM_REVISION


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)


class RuntimeEvidenceRecorder:
    """Aggregate detached training diagnostics without entering model state."""

    def __init__(self) -> None:
        self.reset(None)

    def reset(self, trainer: Any) -> None:
        del trainer
        self.batches = 0
        self.bpdd_stable_sum = 0.0
        self.bpdd_active_sum = 0.0
        self.bpdd_reliability_sum = 0.0
        self.bpdd_loss_sum = 0.0
        self.bpdd_observations = 0
        self.candidate_source_matches = 0
        self.stable_source_matches = 0
        self.geometry_total = 0
        self.geometry_horizontal = 0
        self.geometry_vertical = 0
        self.geometry_minimum_horizontal: float | None = None
        self.geometry_minimum_vertical: float | None = None
        self.minimum_extent: float | None = None
        self.decoded_minima: dict[str, float] = {}
        self.nonfinite_geometry_observations = 0

    def capture(self, trainer: Any) -> None:
        model = _unwrap_model(trainer.model)
        bpdd = getattr(model, "last_bpdd_statistics", {})
        losses = getattr(model, "last_fdr_losses", {})
        stable = _number(bpdd.get("stable_match_ratio"))
        active = _number(bpdd.get("active_edge_ratio"))
        reliability = _number(bpdd.get("mean_reliability"))
        loss = _number(losses.get("loss_bpdd"))
        if None not in (stable, active, reliability, loss):
            self.bpdd_stable_sum += float(stable)
            self.bpdd_active_sum += float(active)
            self.bpdd_reliability_sum += float(reliability)
            self.bpdd_loss_sum += float(loss)
            self.bpdd_observations += 1
        self.candidate_source_matches += int(
            _number(bpdd.get("candidate_source_matches")) or 0
        )
        self.stable_source_matches += int(
            _number(bpdd.get("stable_source_matches")) or 0
        )

        fdr = getattr(model, "fdr", None)
        geometry = getattr(fdr, "last_geometry_statistics", {})
        self.geometry_total += int(_number(geometry.get("total")) or 0)
        self.geometry_horizontal += int(
            _number(geometry.get("horizontal_infeasible")) or 0
        )
        self.geometry_vertical += int(
            _number(geometry.get("vertical_infeasible")) or 0
        )
        horizontal = _number(geometry.get("minimum_raw_horizontal"))
        vertical = _number(geometry.get("minimum_raw_vertical"))
        if horizontal is not None:
            self.geometry_minimum_horizontal = (
                horizontal
                if self.geometry_minimum_horizontal is None
                else min(self.geometry_minimum_horizontal, horizontal)
            )
        if vertical is not None:
            self.geometry_minimum_vertical = (
                vertical
                if self.geometry_minimum_vertical is None
                else min(self.geometry_minimum_vertical, vertical)
            )
        extent = _number(geometry.get("minimum_extent"))
        if extent is not None:
            self.minimum_extent = extent
        for key in ("minimum_decoded_width", "minimum_decoded_height"):
            value = _number(geometry.get(key))
            if value is not None:
                self.decoded_minima[key] = min(self.decoded_minima.get(key, value), value)
            elif geometry.get(key) is not None:
                self.nonfinite_geometry_observations += 1
        self.batches += 1

    def write(self, trainer: Any) -> dict[str, Any]:
        observations = max(self.bpdd_observations, 1)
        norms = getattr(trainer, "last_gradient_norms", {})
        record = {
            "method_revision": SYSTEM_REVISION,
            "completed_epoch": int(trainer.epoch) + 1,
            "batches": self.batches,
            "bpdd_observations": self.bpdd_observations,
            "bpdd_stable_match_ratio_mean": (
                self.bpdd_stable_sum / observations if self.bpdd_observations else None
            ),
            "bpdd_active_edge_ratio_mean": self.bpdd_active_sum / observations if self.bpdd_observations else None,
            "bpdd_mean_reliability": self.bpdd_reliability_sum / observations if self.bpdd_observations else None,
            "loss_bpdd_mean": self.bpdd_loss_sum / observations if self.bpdd_observations else None,
            "bpdd_candidate_source_matches": self.candidate_source_matches,
            "bpdd_stable_source_matches": self.stable_source_matches,
            "geometry_total": self.geometry_total,
            "geometry_horizontal_infeasible": self.geometry_horizontal,
            "geometry_vertical_infeasible": self.geometry_vertical,
            "geometry_minimum_raw_horizontal": self.geometry_minimum_horizontal,
            "geometry_minimum_raw_vertical": self.geometry_minimum_vertical,
            "geometry_minimum_extent": self.minimum_extent,
            "gradients_finite": norms.get("gradients_finite"),
            "geometry_minimum_decoded_width": self.decoded_minima.get("minimum_decoded_width"),
            "geometry_minimum_decoded_height": self.decoded_minima.get("minimum_decoded_height"),
            "nonfinite_geometry_observations": self.nonfinite_geometry_observations,
        }
        path = Path(trainer.save_dir).resolve() / "full-runtime.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        return record

