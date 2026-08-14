from __future__ import annotations

import argparse
import hashlib
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from src.bpdd_loss import BPDDDetectionLoss, BPDDOptions
from src.fdr_bpdd_bridge_protocol import (
    BRIDGE_PROTOCOL,
    BRIDGE_PROTOCOL_SHA256,
    EXPECTED_GPU_UUID,
    build_bridge_run_identity,
)
from src.fdr_protocol import canonical_json_bytes
from src.rtdetr_fdr_bpdd_bridge import (
    FDR_BPDD_BRIDGE_MODEL_CFG,
    FDRBPDDBridgeDetectionModel,
    FDRBPDDBridgeTrainer,
)
from src.rtdetr_ra_glgm import (
    RA_GLGM_CONTROL_CFG,
    RAGLGMControlDetectionModel,
    RAGLGMControlTrainer,
)


def _yaml(path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _batch() -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(1, 3, 128, 128),
        "cls": torch.tensor([[1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "batch_idx": torch.tensor([0.0]),
    }


def test_bridge_yaml_adds_only_locked_bpdd_to_a_graph() -> None:
    expected = deepcopy(_yaml(RA_GLGM_CONTROL_CFG))
    expected["bpdd_loss"] = {
        "enabled": True,
        "weight": 0.5,
        "temperature": 0.5,
        "margin": 0.02,
        "eps": 1.0e-6,
        "matched_layer": "final",
        "include_dn": False,
    }
    assert _yaml(FDR_BPDD_BRIDGE_MODEL_CFG) == expected


def test_bridge_protocol_is_canonical_and_bound_to_the_historical_gpu() -> None:
    assert (
        BRIDGE_PROTOCOL_SHA256
        == hashlib.sha256(canonical_json_bytes(BRIDGE_PROTOCOL)).hexdigest().upper()
    )
    source = {"git_commit": "a" * 40, "tree_sha256": "B" * 64}
    smoke = build_bridge_run_identity(source, stage="smoke", gpu_uuid=EXPECTED_GPU_UUID)
    formal = build_bridge_run_identity(
        source, stage="formal", gpu_uuid=EXPECTED_GPU_UUID
    )
    assert smoke["run_id"] != formal["run_id"]
    with pytest.raises(ValueError, match="A/C/D physical GPU"):
        build_bridge_run_identity(source, stage="formal", gpu_uuid="GPU-foreign")


def test_bridge_training_settings_are_the_frozen_full100_contract(
    tmp_path: Path,
) -> None:
    from scripts.train_fdr_bpdd_bridge import build_settings

    data = tmp_path / "data.yaml"
    data.write_text("path: .\n", encoding="utf-8")
    args = argparse.Namespace(
        stage="formal",
        output_root=tmp_path / "runs",
        name=None,
        resume=None,
    )
    settings = build_settings(args, data)
    assert settings["epochs"] == 100
    assert settings["batch"] == 8
    assert settings["workers"] == 8
    assert settings["nbs"] == 64
    assert settings["seed"] == 0
    assert settings["pretrained"] is False
    assert settings["deterministic"] is True
    assert settings["amp"] is True
    assert settings["save_period"] == -1


def test_bridge_initial_graph_is_tensor_identical_to_a() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        baseline = RAGLGMControlDetectionModel(nc=3, verbose=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        bridge = FDRBPDDBridgeDetectionModel(nc=3, verbose=False)

    assert set(bridge.state_dict()) == set(baseline.state_dict())
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, bridge.state_dict()[name]), name
    assert sum(parameter.numel() for parameter in bridge.parameters()) == sum(
        parameter.numel() for parameter in baseline.parameters()
    )
    assert isinstance(bridge.init_criterion(), BPDDDetectionLoss)
    assert bridge.bpdd_options == BPDDOptions()
    assert issubclass(FDRBPDDBridgeTrainer, RAGLGMControlTrainer)


def test_bridge_eval_is_bit_exact_and_bpdd_is_absent() -> None:
    torch.manual_seed(23)
    baseline = RAGLGMControlDetectionModel(nc=3, verbose=False).eval()
    torch.manual_seed(23)
    bridge = FDRBPDDBridgeDetectionModel(nc=3, verbose=False).eval()
    bridge.load_state_dict(baseline.state_dict(), strict=True)

    image = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        baseline_output = baseline.predict(image)
        bridge_output = bridge.predict(image)
    assert torch.equal(baseline_output[0], bridge_output[0])

    bridge.loss(_batch(), preds=bridge_output)
    assert bridge.criterion.bpdd_runtime_enabled is False
    assert bridge.last_bpdd_statistics == {}
    assert "loss_bpdd" not in bridge.last_fdr_losses


def test_bridge_training_adds_only_bpdd_loss() -> None:
    torch.manual_seed(29)
    model = FDRBPDDBridgeDetectionModel(nc=3, verbose=False).train()
    loss, displayed = model.loss(_batch())
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(displayed).all()
    assert "loss_bpdd" in model.last_fdr_losses
    assert torch.isfinite(model.last_fdr_losses["loss_bpdd"])
    assert model.last_bpdd_statistics["matched_queries"] >= 0
    assert not hasattr(model, "ra_glgm")


def test_bridge_trainer_rejects_untrusted_resume_path() -> None:
    trainer = object.__new__(FDRBPDDBridgeTrainer)
    trainer.data = {"nc": 3, "channels": 3}
    trainer.experiment_seed = 0
    with pytest.raises(TypeError, match="loaded model"):
        trainer.get_model(weights="untrusted.pt", verbose=False)
