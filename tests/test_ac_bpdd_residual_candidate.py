from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import yaml

from scripts import train_lrs_gfdr_ac_bpdd_residual as launcher
from scripts.train_rtdetr_fdr import FORMAL_EPOCHS, FROZEN_SETTINGS
from src.rtdetr_fdr_bpdd import FDRBPDDDetectionModel
from src.rtdetr_lrs_system import LRSFDRBPDDResidualTrainer


ROOT = Path(__file__).resolve().parents[1]
G_CONFIG = ROOT / "configs" / "rtdetr-l-lrs-fdr-bpdd.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _effective_decoder_options(payload: dict) -> dict:
    options = dict(payload["head"][-1][3][-1])
    options.setdefault("feasible_geometry", True)
    return options


def _effective_bpdd_options(payload: dict) -> dict:
    options = dict(payload["bpdd_loss"])
    options.setdefault("decoded_iou_gate", False)
    options.setdefault("iou_margin", 0.0)
    options.setdefault("residual_gradient_only", False)
    options.setdefault("distribution_objective", "kl")
    return options


def test_candidate_is_a_single_factor_change_from_formal_g() -> None:
    base = yaml.safe_load(G_CONFIG.read_text(encoding="utf-8"))
    candidate = yaml.safe_load(launcher.CANDIDATE_CONFIG.read_text(encoding="utf-8"))

    assert _effective_decoder_options(candidate)["feasible_geometry"] is True
    assert candidate["fdr_loss"] == base["fdr_loss"]
    assert candidate["backbone"] == base["backbone"]
    assert candidate["head"][:-1] == base["head"][:-1]
    assert _effective_decoder_options(candidate) == _effective_decoder_options(base)

    base_bpdd = _effective_bpdd_options(base)
    candidate_bpdd = _effective_bpdd_options(candidate)
    assert candidate_bpdd.pop("residual_gradient_only") is True
    assert base_bpdd.pop("residual_gradient_only") is False
    assert candidate_bpdd == base_bpdd
    assert candidate_bpdd["decoded_iou_gate"] is False
    assert candidate_bpdd["distribution_objective"] == "kl"
    assert candidate["fdr_loss"]["reliability_shrinkage_alpha"] == 0.25


def test_candidate_changes_no_initialized_model_tensor() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(909)
        base = FDRBPDDDetectionModel(G_CONFIG, nc=10, verbose=False)
        torch.manual_seed(909)
        candidate = FDRBPDDDetectionModel(
            launcher.CANDIDATE_CONFIG,
            nc=10,
            verbose=False,
        )

    assert candidate.bpdd_options.residual_gradient_only is True
    assert candidate.bpdd_options.decoded_iou_gate is False
    assert candidate.bpdd_options.distribution_objective == "kl"
    assert candidate.init_criterion().reliability_shrinkage_alpha == 0.25
    assert candidate.model[-1].decoder.feasible_geometry is True
    assert base.state_dict().keys() == candidate.state_dict().keys()
    for key, expected in base.state_dict().items():
        torch.testing.assert_close(
            candidate.state_dict()[key], expected, rtol=0, atol=0
        )


def test_residual_launcher_trainer_forces_its_residual_yaml() -> None:
    """The recorded residual config must be the config used to build the model."""

    trainer = object.__new__(LRSFDRBPDDResidualTrainer)
    trainer.data = {"nc": 10, "channels": 3}
    trainer.experiment_seed = 0
    trainer.initial_state_path = None

    model = trainer.get_model(cfg=str(G_CONFIG), verbose=False)

    assert model.bpdd_options.residual_gradient_only is True


def test_launcher_freezes_formal100_and_candidate_identity(tmp_path: Path) -> None:
    data_yaml = tmp_path / "formal.yaml"
    data_yaml.write_text("{}\n", encoding="utf-8")
    settings = launcher.build_settings(data_yaml, tmp_path / "runs")

    for key, value in FROZEN_SETTINGS.items():
        if key not in {"model", "save_period"}:
            assert settings[key] == value
    assert settings["model"] == str(launcher.CANDIDATE_CONFIG.resolve())
    assert settings["epochs"] == FORMAL_EPOCHS == 100
    assert settings["seed"] == 0
    assert settings["save_period"] == -1
    assert settings["name"] == "formal-seed0-lrs_gfdr_ac_bpdd_residual-v1"
    assert settings["exist_ok"] is False
    assert "resume" not in settings


def test_launch_record_is_source_and_artifact_bound(tmp_path: Path) -> None:
    config = tmp_path / "candidate.yaml"
    initial = tmp_path / "initial-state.pt"
    config.write_bytes(b"candidate: true\n")
    initial.write_bytes(b"same-start")
    settings = {"model": str(config), "epochs": 100, "seed": 0}

    record = launcher.build_launch_record(
        source_identity={"git_commit": "a" * 40, "tree_sha256": "B" * 64},
        config_path=config,
        initial_state_path=initial,
        dataset={"sha256": "C" * 64},
        settings=settings,
    )

    assert record["format_version"] == 1
    assert record["method_revision"] == "ac-bpdd-residual-v1"
    assert record["method"] == "lrs_gfdr_ac_bpdd_residual"
    assert record["config"] == {
        "path": str(config.resolve()),
        "sha256": _sha256(config),
    }
    assert record["initial_state"]["sha256"] == _sha256(initial)
    assert record["settings"] == settings
