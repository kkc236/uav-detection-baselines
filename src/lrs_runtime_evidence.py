"""Shared per-epoch diagnostics for current VisDrone and UAVDT runs."""

from __future__ import annotations

import json
import math
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
        self.capacity_observations = 0
        self.capacity_active_sum = 0.0
        self.capacity_class_active_sum = 0.0
        self.capacity_loc_loss_sum = 0.0
        self.capacity_cls_loss_sum = 0.0
        self.capacity_loc_layers_sum = 0.0
        self.capacity_cls_layers_sum = 0.0
        self.capacity_loc_advantage_sum = 0.0
        self.capacity_cls_advantage_sum = 0.0
        self.capacity_candidate_matches = 0
        self.capacity_consistent_matches = 0
        self.capacity_active_edges = 0
        self.capacity_active_classes = 0
        self.geometry_total = 0
        self.geometry_horizontal = 0
        self.geometry_vertical = 0
        self.geometry_minimum_horizontal: float | None = None
        self.geometry_minimum_vertical: float | None = None
        self.minimum_extent: float | None = None
        self.decoded_minima: dict[str, float] = {}
        self.nonfinite_geometry_observations = 0
        self.expert_geometry_total = 0
        self.expert_geometry_horizontal = 0
        self.expert_geometry_vertical = 0
        self.expert_decoded_minima: dict[str, float] = {}

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

        capacity = getattr(model, "last_capacity_statistics", {})
        if capacity:
            self.capacity_observations += 1
            self.capacity_active_sum += float(
                _number(capacity.get("active_edge_ratio")) or 0.0
            )
            self.capacity_class_active_sum += float(
                _number(capacity.get("active_class_ratio")) or 0.0
            )
            self.capacity_loc_loss_sum += float(
                _number(capacity.get("localization_loss")) or 0.0
            )
            self.capacity_cls_loss_sum += float(
                _number(capacity.get("classification_loss")) or 0.0
            )
            self.capacity_loc_layers_sum += float(
                _number(capacity.get("localization_active_layers")) or 0.0
            )
            self.capacity_cls_layers_sum += float(
                _number(capacity.get("classification_active_layers")) or 0.0
            )
            self.capacity_loc_advantage_sum += float(
                _number(capacity.get("mean_localization_advantage")) or 0.0
            )
            self.capacity_cls_advantage_sum += float(
                _number(capacity.get("mean_classification_advantage")) or 0.0
            )
            self.capacity_candidate_matches += int(
                _number(capacity.get("candidate_source_matches")) or 0
            )
            self.capacity_consistent_matches += int(
                _number(capacity.get("identity_consistent_matches")) or 0
            )
            self.capacity_active_edges += int(
                _number(capacity.get("active_edges")) or 0
            )
            self.capacity_active_classes += int(
                _number(capacity.get("active_classes")) or 0
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

        expert_geometry = getattr(fdr, "last_expert_geometry_statistics", {})
        self.expert_geometry_total += int(_number(expert_geometry.get("total")) or 0)
        self.expert_geometry_horizontal += int(
            _number(expert_geometry.get("horizontal_infeasible")) or 0
        )
        self.expert_geometry_vertical += int(
            _number(expert_geometry.get("vertical_infeasible")) or 0
        )
        for key in ("minimum_decoded_width", "minimum_decoded_height"):
            value = _number(expert_geometry.get(key))
            if value is not None:
                self.expert_decoded_minima[key] = min(
                    self.expert_decoded_minima.get(key, value), value
                )
        self.batches += 1

    def write(self, trainer: Any) -> dict[str, Any]:
        observations = max(self.bpdd_observations, 1)
        capacity_observations = max(self.capacity_observations, 1)
        norms = getattr(trainer, "last_gradient_norms", {})
        norm_values = [_number(value) for value in norms.values()]
        gradients_finite = (
            all(value is not None and math.isfinite(value) for value in norm_values)
            if norm_values
            else None
        )
        model = _unwrap_model(trainer.model)
        record = {
            "method_revision": getattr(model, "capacity_method_revision", SYSTEM_REVISION),
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
            "gradients_finite": gradients_finite,
            "gradient_norm": _number(norms.get("gradient_norm")),
            "fdr_gradient_norm": _number(norms.get("fdr_gradient_norm")),
            "expert_gradient_norm": _number(norms.get("expert_gradient_norm")),
            "capacity_observations": self.capacity_observations,
            "capacity_active_edge_ratio_mean": (
                self.capacity_active_sum / capacity_observations
                if self.capacity_observations else None
            ),
            "capacity_active_class_ratio_mean": (
                self.capacity_class_active_sum / capacity_observations
                if self.capacity_observations else None
            ),
            "capacity_localization_loss_mean": (
                self.capacity_loc_loss_sum / capacity_observations
                if self.capacity_observations else None
            ),
            "capacity_classification_loss_mean": (
                self.capacity_cls_loss_sum / capacity_observations
                if self.capacity_observations else None
            ),
            "capacity_localization_active_layers_mean": (
                self.capacity_loc_layers_sum / capacity_observations
                if self.capacity_observations else None
            ),
            "capacity_classification_active_layers_mean": (
                self.capacity_cls_layers_sum / capacity_observations
                if self.capacity_observations else None
            ),
            "capacity_mean_localization_advantage": (
                self.capacity_loc_advantage_sum / capacity_observations
                if self.capacity_observations else None
            ),
            "capacity_mean_classification_advantage": (
                self.capacity_cls_advantage_sum / capacity_observations
                if self.capacity_observations else None
            ),
            "capacity_candidate_source_matches": self.capacity_candidate_matches,
            "capacity_identity_consistent_matches": self.capacity_consistent_matches,
            "capacity_active_edges": self.capacity_active_edges,
            "capacity_active_classes": self.capacity_active_classes,
            "expert_geometry_total": self.expert_geometry_total,
            "expert_geometry_horizontal_infeasible": self.expert_geometry_horizontal,
            "expert_geometry_vertical_infeasible": self.expert_geometry_vertical,
            "expert_geometry_minimum_decoded_width": self.expert_decoded_minima.get(
                "minimum_decoded_width"
            ),
            "expert_geometry_minimum_decoded_height": self.expert_decoded_minima.get(
                "minimum_decoded_height"
            ),
            "geometry_minimum_decoded_width": self.decoded_minima.get("minimum_decoded_width"),
            "geometry_minimum_decoded_height": self.decoded_minima.get("minimum_decoded_height"),
            "nonfinite_geometry_observations": self.nonfinite_geometry_observations,
        }
        path = Path(trainer.save_dir).resolve() / "full-runtime.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        return record

