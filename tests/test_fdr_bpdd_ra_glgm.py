from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from src.bpdd_loss import BPDDDetectionLoss, BPDDOptions
from src.rtdetr_fdr_bpdd_ra_glgm import (
    _BPDDEpochStatistics,
    FDR_BPDD_RA_GLGM_MODEL_CFG,
    FDRBPDDRAGLGMDetectionModel,
    FDRBPDDRAGLGMTrainer,
    _parse_bpdd_options,
)
from src.rtdetr_ra_glgm import RA_GLGM_MODEL_CFG, RAGLGMDetectionModel, RAGLGMTrainer


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _batch() -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(1, 3, 128, 128),
        "cls": torch.tensor([[1.0], [-1.0]]),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]]
        ),
        "batch_idx": torch.tensor([0.0, 0.0]),
    }


def test_combo_yaml_adds_only_the_locked_bpdd_options() -> None:
    expected = deepcopy(_yaml(RA_GLGM_MODEL_CFG))
    expected["bpdd_loss"] = {
        "enabled": True,
        "weight": 0.5,
        "temperature": 0.5,
        "margin": 0.02,
        "eps": 1.0e-6,
        "matched_layer": "final",
        "include_dn": False,
    }
    assert _yaml(FDR_BPDD_RA_GLGM_MODEL_CFG) == expected


def test_combo_options_are_frozen_and_reject_contract_drift() -> None:
    options = _parse_bpdd_options(_yaml(FDR_BPDD_RA_GLGM_MODEL_CFG)["bpdd_loss"])
    assert options == BPDDOptions()
    with pytest.raises(ValueError, match="unknown BPDD loss options"):
        _parse_bpdd_options({"unexpected": True})
    with pytest.raises(ValueError, match="final stock assignment"):
        _parse_bpdd_options({"matched_layer": "first"})
    with pytest.raises(ValueError, match="excludes denoising"):
        _parse_bpdd_options({"include_dn": True})


def test_combo_has_ra_state_and_parameter_contract_and_bpdd_criterion() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        ra = RAGLGMDetectionModel(nc=10, verbose=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        combo = FDRBPDDRAGLGMDetectionModel(nc=10, verbose=False)

    assert set(combo.state_dict()) == set(ra.state_dict())
    for name, value in ra.state_dict().items():
        assert torch.equal(value, combo.state_dict()[name]), name
    assert sum(parameter.numel() for parameter in combo.parameters()) == sum(
        parameter.numel() for parameter in ra.parameters()
    )
    criterion = combo.init_criterion()
    assert isinstance(criterion, BPDDDetectionLoss)
    assert criterion.bpdd_options == BPDDOptions()
    assert issubclass(FDRBPDDRAGLGMTrainer, RAGLGMTrainer)


def test_combo_eval_prediction_is_bit_exact_to_ra_and_bpdd_is_inactive() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        ra = RAGLGMDetectionModel(nc=3, verbose=False).eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        combo = FDRBPDDRAGLGMDetectionModel(nc=3, verbose=False).eval()
    combo.load_state_dict(ra.state_dict(), strict=True)

    image = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        ra_output = ra.predict(image)
        combo_output = combo.predict(image)
    assert torch.equal(ra_output[0], combo_output[0])

    combo.loss(_batch(), preds=combo_output)
    assert combo.criterion.bpdd_runtime_enabled is False
    assert combo.last_bpdd_statistics == {}
    assert "loss_bpdd" not in combo.last_fdr_losses


def test_combo_training_reuses_stock_matches_and_opens_both_modules() -> None:
    torch.manual_seed(0)
    model = FDRBPDDRAGLGMDetectionModel(nc=3, verbose=False).train()

    loss, displayed = model.loss(_batch())
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(displayed).all()
    assert isinstance(model.criterion, BPDDDetectionLoss)
    assert model.criterion.stock_match_calls == 7
    assert model.criterion.fgl_extra_match_calls == 0
    assert model.criterion.last_normal_decoder_assignment is not None
    snapshot = model.criterion.normal_assignment_snapshot()
    assert len(snapshot) == 7
    final = model.criterion.last_normal_decoder_assignment
    for (snapshot_source, snapshot_target), (final_source, final_target) in zip(
        snapshot[-1], final, strict=True
    ):
        assert torch.equal(snapshot_source, final_source)
        assert torch.equal(snapshot_target, final_target)
    assert "loss_bpdd" in model.last_fdr_losses
    assert torch.isfinite(model.last_fdr_losses["loss_bpdd"])
    assert model.last_bpdd_statistics["matched_queries"] >= 0
    assert model.last_ra_glgm_losses["loss_ra_support"] > 0
    assert model.last_ra_glgm_losses["loss_ra_scale"] > 0
    assert model.ra_glgm.alpha.grad is not None
    assert model.ra_glgm.alpha.grad.abs().sum() > 0
    assert model.ra_glgm.support_head.weight.grad is not None
    assert model.ra_glgm.support_head.weight.grad.abs().sum() > 0


def test_bpdd_epoch_statistics_use_eligible_edge_weighting() -> None:
    accumulator = _BPDDEpochStatistics()
    first = {
        "active_edge_ratio": torch.tensor(0.25),
        "mean_reliability": torch.tensor(0.1),
        "mean_teacher_improvement": torch.tensor(0.2),
        "mixture_beats_final_ratio": torch.tensor(0.3),
        "mean_mixture_advantage_over_final": torch.tensor(0.4),
        "matched_queries": torch.tensor(2),
        "eligible_edges": torch.tensor(10),
    }
    second = {
        **first,
        "active_edge_ratio": torch.tensor(0.75),
        "matched_queries": torch.tensor(3),
        "eligible_edges": torch.tensor(30),
    }
    accumulator.update(first)
    accumulator.update(second)

    values = accumulator.values()

    assert values["active_edge_ratio"] == pytest.approx(0.625)
    assert values["mean_reliability"] == pytest.approx(0.1)
    assert values["matched_queries"] == 5
    assert values["eligible_edges"] == 40


def test_combo_trainer_strictly_rejects_non_module_resume_weights() -> None:
    trainer = object.__new__(FDRBPDDRAGLGMTrainer)
    trainer.data = {"nc": 3, "channels": 3}
    trainer.experiment_seed = 0

    with pytest.raises(TypeError, match="loaded checkpoint model"):
        trainer.get_model(weights="untrusted.pt", verbose=False)
