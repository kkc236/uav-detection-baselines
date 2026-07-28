from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.acr_eg_integration import (
    ACREGConfig,
    ACREGIntegratedRTDETR,
    build_integrated_artifact,
    load_acr_eg_config,
)
from src.gcqf import GCQF
from src.gcte_types import QueryEvidence, ViewGeometry


ROOT = Path(__file__).resolve().parents[1]


class TinyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


def _evidence(batch: int, queries: int, dim: int, classes: int) -> QueryEvidence:
    return QueryEvidence(
        queries=torch.randn(batch, queries, dim),
        logits=torch.randn(batch, queries, classes),
        boxes=torch.full((batch, queries, 4), 0.25),
        quality=torch.full((batch, queries, 1), 0.5),
    )


def _geometry(batch: int, queries: int) -> ViewGeometry:
    return ViewGeometry(
        homography=torch.eye(3).reshape(1, 1, 3, 3).repeat(batch, queries, 1, 1),
        crop_metadata=torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]]]
        ).repeat(batch, queries, 1),
        view_index=torch.zeros(batch, queries, dtype=torch.long),
        valid_mask=torch.ones(batch, queries, dtype=torch.bool),
    )


def test_yaml_declares_acr_eg_as_forward_network_module() -> None:
    config = load_acr_eg_config(ROOT / "configs" / "rtdetr-l-gcte.yaml")

    assert config.enabled is True
    assert config.forward_integration is True
    assert config.num_views == 4
    assert config.query_dim == 256
    assert config.residual_eta == 0.2


def test_integrated_wrapper_registers_detector_and_acr_eg_parameters() -> None:
    config = ACREGConfig(
        enabled=True,
        forward_integration=True,
        query_dim=32,
        num_classes=3,
        num_heads=4,
        num_views=4,
        residual_eta=0.2,
    )
    wrapper = ACREGIntegratedRTDETR(TinyDetector(), config)
    state_keys = set(wrapper.state_dict())
    optimizer_keys = {
        id(parameter)
        for parameter in torch.optim.AdamW(wrapper.parameters()).param_groups[0]["params"]
    }

    assert any(key.startswith("detector.") for key in state_keys)
    assert any(key.startswith("acr_eg.") for key in state_keys)
    assert any(
        id(parameter) in optimizer_keys
        for parameter in wrapper.acr_eg.parameters()
    )


def test_forward_calls_acr_eg_and_disabled_mode_restores_global_evidence() -> None:
    config = ACREGConfig(
        enabled=True,
        forward_integration=True,
        query_dim=32,
        num_classes=3,
        num_heads=4,
        num_views=4,
        residual_eta=0.2,
    )
    wrapper = ACREGIntegratedRTDETR(TinyDetector(), config)
    global_evidence = _evidence(1, 3, 32, 3)
    local_evidence = _evidence(1, 2, 32, 3)
    output = wrapper(
        global_evidence=global_evidence,
        local_evidence=local_evidence,
        geometry=_geometry(1, 2),
        anchor_mask=torch.ones(1, 2, 1, dtype=torch.bool),
        residual_enabled=True,
    )
    assert output.module_output.global_evidence is global_evidence
    assert output.module_output.adjusted_local_scores.shape == (1, 2, 1)

    wrapper.config = ACREGConfig(
        enabled=False,
        forward_integration=True,
        query_dim=32,
        num_classes=3,
        num_heads=4,
        num_views=4,
        residual_eta=0.2,
    )
    disabled = wrapper(
        global_evidence=global_evidence,
        local_evidence=local_evidence,
        geometry=_geometry(1, 2),
        anchor_mask=torch.ones(1, 2, 1, dtype=torch.bool),
        residual_enabled=True,
    )
    assert disabled.module_output is None
    assert disabled.global_evidence is global_evidence


def test_integrated_artifact_contains_detector_and_acr_eg_state() -> None:
    config = ACREGConfig(
        enabled=True,
        forward_integration=True,
        query_dim=32,
        num_classes=3,
        num_heads=4,
        num_views=4,
        residual_eta=0.2,
    )
    wrapper = ACREGIntegratedRTDETR(TinyDetector(), config)
    artifact = build_integrated_artifact(
        wrapper,
        baseline_sha256="A" * 64,
        module_sha256="B" * 64,
        source_commit="c" * 40,
    )

    assert artifact["schema_version"] == "gcte-acr-eg-integrated/v1"
    assert artifact["baseline_sha256"] == "A" * 64
    assert artifact["module_sha256"] == "B" * 64
    assert artifact["config"]["forward_integration"] is True
    assert any(
        key.startswith("detector.")
        for key in artifact["wrapper_state"]
    )
    assert any(
        key.startswith("acr_eg.")
        for key in artifact["wrapper_state"]
    )
