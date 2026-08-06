from __future__ import annotations

import pytest
import torch

from src.scads import build_support_projects
from src.scads_evaluation import (
    area_buckets,
    gate_decision,
    reconstruction_summary,
    summarize_representation,
)


def test_area_buckets_use_registered_16_and_32_pixel_boundaries() -> None:
    boxes = torch.tensor(
        [
            [0.5, 0.5, 15.0 / 640, 15.0 / 640],
            [0.5, 0.5, 16.0 / 640, 16.0 / 640],
            [0.5, 0.5, 32.0 / 640, 32.0 / 640],
        ]
    )
    assert area_buckets(boxes).tolist() == [0, 1, 2]


def test_unsaturated_soft_labels_reconstruct_offsets() -> None:
    bank = build_support_projects((0.25, 0.5, 1.0))
    offsets = torch.tensor([[-0.1, -0.05, 0.05, 0.1]])
    result = reconstruction_summary(offsets, bank[1])
    assert result["unsaturated_edges"] == 4
    assert result["l1_mean"] <= 1e-6
    assert result["max_error"] <= 1e-6


def test_representation_reports_saturation_reduction_and_route_scale_relation() -> None:
    bank = build_support_projects((0.25, 0.5, 1.0))
    target_boxes = torch.tensor(
        [
            [0.5, 0.5, 0.02, 0.02],
            [0.5, 0.5, 0.04, 0.04],
            [0.5, 0.5, 0.10, 0.10],
        ]
    )
    offsets = torch.tensor(
        [
            [-0.8, -0.7, 0.7, 0.8],
            [-0.4, -0.3, 0.3, 0.4],
            [-0.1, -0.1, 0.1, 0.1],
        ]
    )
    weights = torch.tensor(
        [[0.02, 0.03, 0.95], [0.05, 0.90, 0.05], [0.95, 0.03, 0.02]]
    )
    projects = torch.einsum("nk,kb->nb", weights, bank)
    result = summarize_representation(
        fdr_offsets=offsets,
        scads_offsets=offsets,
        target_boxes=target_boxes,
        base_project=bank[1],
        scads_projects=projects,
        route_weights=weights,
        project_bank=bank,
        support_ups=(0.25, 0.5, 1.0),
        margin_ratio=0.02,
    )
    assert result["matched_objects"] == 3
    assert result["route"]["active_predicted_routes_ge_5pct"] == 3
    assert result["route"]["effective_up_scale_range"] > 0.01
    assert result["scads_adaptive"]["overall"]["edge_saturation_rate"] <= result[
        "fdr_fixed_base"
    ]["overall"]["edge_saturation_rate"]


def test_gate_requires_every_preregistered_scientific_condition() -> None:
    representation = {
        "tiny_saturation_relative_reduction": 0.6,
        "scads_adaptive": {"reconstruction": {"l1_mean": 1e-7}},
        "route": {
            "active_oracle_routes_ge_5pct": 2,
            "active_predicted_routes_ge_5pct": 2,
            "effective_up_scale_range": 0.02,
        },
    }
    positive = {"map": 0.01}
    exact = {"ap75": 0.01, "ap_tiny": 0.01}
    result = gate_decision(
        final_delta=positive,
        tail3_delta=positive,
        exact_delta=exact,
        representation=representation,
        engineering_complete=True,
    )
    assert result["passed"] is True
    exact["ap_tiny"] = -0.01
    assert gate_decision(
        final_delta=positive,
        tail3_delta=positive,
        exact_delta=exact,
        representation=representation,
        engineering_complete=True,
    )["passed"] is False
