from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts.train_rtdetr_gcmv_warmstart import (
    build_parser,
    build_settings,
    trainer_class,
)
from src.rtdetr_gcmv_warmstart import (
    GCMVWarmStartCalibrationTrainer,
    GCMVWarmStartControlTrainer,
    GCMVWarmStartTrainer,
)


def _parse(tmp_path, stage: str):
    return build_parser().parse_args(
        [
            "--stage",
            stage,
            "--baseline",
            "baseline.pt",
            "--module-artifact",
            str(tmp_path / "module.pt"),
            "--data",
            "visdrone-full.yaml",
            "--project",
            str(tmp_path / "runs"),
            "--name",
            stage,
            "--source-commit",
            "a" * 40,
        ]
    )


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("calibration", GCMVWarmStartCalibrationTrainer),
        ("control", GCMVWarmStartControlTrainer),
        ("method", GCMVWarmStartTrainer),
    ],
)
def test_stage_selects_the_intended_trainer(tmp_path, stage, expected):
    assert trainer_class(_parse(tmp_path, stage)) is expected


def test_finetune_settings_are_matched_and_low_lr(tmp_path):
    control = build_settings(_parse(tmp_path, "control"))
    method = build_settings(_parse(tmp_path, "method"))

    assert {
        key: value for key, value in control.items() if key != "name"
    } == {
        key: value for key, value in method.items() if key != "name"
    }
    assert control["epochs"] == 10
    assert control["fraction"] == 1.0
    assert control["batch"] == 8
    assert control["workers"] == 8
    assert control["imgsz"] == 640
    assert control["optimizer"] == "MuSGD"
    assert control["lr0"] == 1e-4
    assert control["lrf"] == 1.0
    assert control["warmup_epochs"] == 0.0
    assert control["nbs"] == 64
    assert control["amp"] is True
    assert control["deterministic"] is True
    assert control["mosaic"] == 1.0
    assert control["close_mosaic"] == 10


def test_calibration_is_one_epoch_and_uses_the_same_data_contract(tmp_path):
    settings = build_settings(_parse(tmp_path, "calibration"))

    assert settings["epochs"] == 1
    assert settings["batch"] == 8
    assert settings["workers"] == 8
    assert settings["imgsz"] == 640
    assert settings["lr0"] == 1e-4
    assert settings["save"] is False


def test_calibration_parameter_policy_excludes_detector_and_rho():
    trainer = object.__new__(GCMVWarmStartCalibrationTrainer)
    trainer.model = torch.nn.Module()
    trainer.model.detector = torch.nn.Linear(2, 2)
    trainer.model.plec = torch.nn.Linear(2, 2)
    trainer.model.gcmv_injector = torch.nn.Module()
    trainer.model.gcmv_injector.gglf = torch.nn.Linear(2, 2)
    trainer.model.gcmv_injector.peg = torch.nn.Module()
    trainer.model.gcmv_injector.peg.rho = torch.nn.Parameter(
        torch.zeros(())
    )
    for parameter in trainer.model.parameters():
        parameter.requires_grad_(True)

    trainer._apply_parameter_policy()

    states = {
        name: parameter.requires_grad
        for name, parameter in trainer.model.named_parameters()
    }
    assert states["detector.weight"] is False
    assert states["plec.weight"] is True
    assert states["gcmv_injector.gglf.weight"] is True
    assert states["gcmv_injector.peg.rho"] is False


def test_parser_requires_an_explicit_stage():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
