from __future__ import annotations

from pathlib import Path

import torch
import yaml

from src.fdr_math import decode_feasible_fdr_boxes
from src.rtdetr_lrs_system import LRS_GFDR_CONFIG, LRSGFDRDetectionModel


ROOT = Path(__file__).resolve().parents[1]


def test_lrs_gfdr_config_is_explicit_and_excludes_other_adapters() -> None:
    payload = yaml.safe_load(LRS_GFDR_CONFIG.read_text(encoding="utf-8"))
    options = payload["head"][-1][3][-1]

    assert LRS_GFDR_CONFIG == ROOT / "configs" / "rtdetr-l-lrs-gfdr.yaml"
    assert options["feasible_geometry"] is True
    assert options["preliminary_box"] is False
    assert payload["fdr_loss"]["reliability_shrinkage_alpha"] == 0.25
    assert "bpdd_loss" not in payload
    assert all(layer[2] != "FIA" for layer in payload["head"])


def test_lrs_gfdr_model_resolves_joint_contract_without_bpdd_or_fia() -> None:
    model = LRSGFDRDetectionModel(nc=10, verbose=False)
    decoder = model.model[-1].decoder

    assert decoder.feasible_geometry is True
    assert model.model[-1].fdr_options["feasible_geometry"] is True
    assert not hasattr(model, "bpdd_options")
    assert not any(type(layer).__name__ == "FIA" for layer in model.model)
    assert model.init_criterion().reliability_shrinkage_alpha == 0.25


def test_lrs_gfdr_projection_returns_finite_positive_boxes() -> None:
    points = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
    raw_distance = torch.tensor([[-8.0, -7.0, -8.0, -7.0]])
    boxes, stats = decode_feasible_fdr_boxes(points, raw_distance, 4.0)

    assert torch.isfinite(boxes).all()
    assert torch.all(boxes[..., 2:] > 0)
    assert stats["horizontal_infeasible"].item() == 1
    assert stats["vertical_infeasible"].item() == 1
