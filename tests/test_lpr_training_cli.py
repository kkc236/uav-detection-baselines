from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.train_rtdetr_lpr import (
    build_parser,
    build_settings,
    capture_lpr_epoch_state,
    write_lpr_diagnostics,
)
from src.lpr_head import LocalizationPriorRefiner
from src.rtdetr_lpr import LPRTrainer


def test_default_settings_match_frozen_screen_protocol(tmp_path) -> None:
    args = build_parser().parse_args(["--project", str(tmp_path)])

    settings = build_settings(args)

    assert settings["epochs"] == 10
    assert settings["pretrained"] is False
    assert settings["seed"] == 0
    assert settings["deterministic"] is True
    assert settings["batch"] == 8
    assert settings["imgsz"] == 640
    assert settings["optimizer"] == "auto"
    assert settings["lr0"] == 0.01
    assert settings["lrf"] == 0.01
    assert settings["mosaic"] == 1.0
    assert settings["fraction"] == 1.0


@pytest.mark.parametrize(
    "arguments",
    (
        ["--batch", "4"],
        ["--imgsz", "800"],
        ["--seed", "1"],
        ["--optimizer", "AdamW"],
        ["--mosaic", "0"],
        ["--amp", "false"],
        ["--fraction", "0.5"],
    ),
)
def test_parser_rejects_scientific_protocol_drift(arguments) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)


def test_resume_accepts_only_total_epoch_100(tmp_path) -> None:
    checkpoint = tmp_path / "last.pt"
    checkpoint.touch()
    args = build_parser().parse_args(["--epochs", "100", "--resume", str(checkpoint)])

    settings = build_settings(args)

    assert settings["epochs"] == 100
    assert settings["resume"] == str(checkpoint.resolve())
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--epochs", "50"])


def test_help_renders_literal_smoke_percentage() -> None:
    help_text = build_parser().format_help()

    assert "1%" in help_text


def test_resume_overrides_checkpoint_total_epochs(monkeypatch) -> None:
    trainer = object.__new__(LPRTrainer)
    trainer.args = SimpleNamespace(epochs=10)
    trainer.resume = True
    monkeypatch.setattr(
        "src.rtdetr_lpr.RTDETRTrainer.check_resume",
        lambda self, overrides: None,
    )

    trainer.check_resume({"epochs": 100})

    assert trainer.args.epochs == 100


def test_epoch_diagnostics_write_one_complete_jsonl_record(tmp_path) -> None:
    refiner = LocalizationPriorRefiner(hidden_dim=8, seed=3407)
    refiner.alpha.data.fill_(0.2)
    refiner.last_residual_mean = torch.tensor(0.12)
    refiner.last_residual_max = torch.tensor(0.45)
    model = nn.Sequential(refiner)
    trainer = SimpleNamespace(
        model=model,
        epoch=0,
        batch_size=8,
        amp=True,
        save_dir=tmp_path,
        validator=SimpleNamespace(metrics=SimpleNamespace(box=SimpleNamespace(map75=0.031))),
        _lpr_grad_sq=0.25,
    )

    capture_lpr_epoch_state(trainer)
    write_lpr_diagnostics(trainer)

    records = (tmp_path / "lpr_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    record = json.loads(records[0])
    assert record["epoch"] == 1
    assert record["map75"] == pytest.approx(0.031)
    assert len(record["gates"]) == 1
    assert record["residual_mean"] == pytest.approx(0.12)
    assert record["residual_max"] == pytest.approx(0.45)
    assert record["lpr_grad_norm"] == pytest.approx(0.5)
    assert "cuda_peak_mib" in record
