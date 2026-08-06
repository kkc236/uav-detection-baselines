from __future__ import annotations

from pathlib import Path

import torch
import ultralytics
import yaml

from src.rtdetr_fdr import FDRRTDETRDetectionModel
from src.rtdetr_scads import (
    SCADS_MODEL_CFG,
    SCADSFDRRTDETRDetectionModel,
    SCADSTrainingEvidence,
)
from src.scads_head import SCADSFDRRTDETRDecoder
from src.scads_loss import SCADSFDRDetectionLoss


def _targets(batch_size: int, empty: bool = False) -> dict:
    if empty:
        return {
            "cls": torch.empty(0, dtype=torch.long),
            "bboxes": torch.empty(0, 4),
            "batch_idx": torch.empty(0, dtype=torch.long),
            "gt_groups": [0] * batch_size,
        }
    return {
        "cls": torch.arange(batch_size, dtype=torch.long) % 3,
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]] * batch_size),
        "batch_idx": torch.arange(batch_size, dtype=torch.long),
        "gt_groups": [1] * batch_size,
    }


def _batch(batch_size: int = 2) -> dict:
    targets = _targets(batch_size)
    return {
        "img": torch.zeros(batch_size, 3, 128, 128),
        "cls": targets["cls"].view(-1, 1).float(),
        "bboxes": targets["bboxes"],
        "batch_idx": targets["batch_idx"].float(),
    }


def _model() -> SCADSFDRRTDETRDetectionModel:
    return SCADSFDRRTDETRDetectionModel(
        SCADS_MODEL_CFG,
        nc=10,
        verbose=False,
        private_seed=10_000,
        support_private_seed=20_000,
    )


def test_scads_yaml_preserves_stock_graph_and_declares_visible_module() -> None:
    stock_path = (
        Path(ultralytics.__file__).resolve().parent
        / "cfg"
        / "models"
        / "rt-detr"
        / "rtdetr-l.yaml"
    )
    stock = yaml.safe_load(stock_path.read_text(encoding="utf-8"))
    scads = yaml.safe_load(Path(SCADS_MODEL_CFG).read_text(encoding="utf-8"))
    assert scads["backbone"] + scads["head"][:-1] == stock["backbone"] + stock["head"][:-1]
    assert scads["head"][-1][2] == "SCADSFDRRTDETRDecoder"
    options = scads["head"][-1][3][-1]
    assert options["support_up_values"] == [0.25, 0.5, 1.0]
    assert scads["fdr_loss"]["scads_route_weight"] == 0.05


def test_scads_model_builds_visible_head_and_keeps_output_contract() -> None:
    model = _model()
    assert isinstance(model.model[-1], SCADSFDRRTDETRDecoder)
    model.eval()
    with torch.inference_mode():
        output = model.predict(torch.zeros(1, 3, 128, 128))
    assert isinstance(output, tuple) and len(output) == 2
    predictions = output[0]
    assert predictions.shape == (1, 300, 6)
    assert torch.isfinite(predictions).all()


def test_scads_and_fdr_common_state_is_byte_exact_under_shared_seed() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        fdr = FDRRTDETRDetectionModel(
            "configs/rtdetr-l-fdr.yaml",
            nc=10,
            verbose=False,
            private_seed=10_000,
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        scads = _model()
    fdr_state = fdr.state_dict()
    scads_state = scads.state_dict()
    assert set(fdr_state).issubset(scads_state)
    for name, expected in fdr_state.items():
        torch.testing.assert_close(scads_state[name], expected, rtol=0, atol=0)
    private = set(scads_state) - set(fdr_state)
    assert private
    assert all(
        ".decoder.support_router." in name
        or ".decoder.adaptive_integral.projects" in name
        for name in private
    )


def test_training_forward_splits_normal_and_dn_support_evidence() -> None:
    model = _model()
    model.train()
    outputs = model.predict(torch.zeros(2, 3, 128, 128), batch=_targets(2))
    dn_meta = outputs[-1]
    assert dn_meta is not None
    evidence = model.last_fdr_evidence
    assert isinstance(evidence, SCADSTrainingEvidence)
    assert evidence.corner_logits.shape == (6, 2, 300, 132)
    assert evidence.support_logits.shape == (2, 300, 3)
    assert evidence.support_weights.shape == (2, 300, 3)
    assert evidence.support_projects.shape == (2, 300, 33)
    assert evidence.dn_support_logits is not None
    assert evidence.dn_support_projects is not None
    torch.testing.assert_close(
        evidence.support_weights.sum(-1),
        torch.ones_like(evidence.support_weights[..., 0]),
    )


def test_real_batch_loss_is_finite_and_reaches_scads_router() -> None:
    model = _model()
    model.train()
    assert isinstance(model.init_criterion(), SCADSFDRDetectionLoss)
    total, displayed = model.loss(_batch())
    assert torch.isfinite(total)
    assert displayed.shape == (3,) and torch.isfinite(displayed).all()
    assert "loss_scads_route" in model.last_fdr_losses
    total.backward()
    router = model.fdr.support_router
    assert router.output_layer.weight.grad is not None
    assert router.output_layer.bias.grad is not None
    assert torch.isfinite(router.output_layer.weight.grad).all()
    assert torch.isfinite(router.output_layer.bias.grad).all()
    assert router.output_layer.bias.grad.abs().sum() > 0
    assert model.criterion.fgl_extra_match_calls == 0
    assert model.criterion.stock_match_calls == 7
