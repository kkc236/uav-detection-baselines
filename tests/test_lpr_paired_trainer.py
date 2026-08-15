from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from src.rtdetr_lpr import LPRTrainer


def test_paired_optimizer_requires_exact_musgd_contract(monkeypatch) -> None:
    captured = {}

    def fake_build(self, model, name, lr, momentum, decay, iterations):
        captured.update(name=name, lr=lr, momentum=momentum, decay=decay, iterations=iterations)
        return "optimizer"

    monkeypatch.setattr("src.rtdetr_lpr.RTDETRTrainer.build_optimizer", fake_build)
    trainer = object.__new__(LPRTrainer)

    result = trainer.build_optimizer(nn.Linear(2, 2), "MuSGD", 0.01, 0.937, 0.0005, 1000)

    assert result == "optimizer"
    assert captured == {
        "name": "MuSGD",
        "lr": 0.01,
        "momentum": 0.937,
        "decay": 0.0005,
        "iterations": 1000,
    }
    with pytest.raises(ValueError, match="MuSGD"):
        trainer.build_optimizer(nn.Linear(2, 2), "auto", 0.01, 0.937, 0.0005, 1000)


def test_setup_installs_fixed_amp128_scaler(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeSetupScaler:
        def __init__(self, device, enabled, init_scale, growth_interval):
            captured.update(
                device=device,
                enabled=enabled,
                init_scale=init_scale,
                growth_interval=growth_interval,
            )

        def get_scale(self):
            return captured["init_scale"]

    monkeypatch.setattr("src.rtdetr_lpr.RTDETRTrainer._setup_train", lambda self: None)
    monkeypatch.setattr("src.rtdetr_lpr.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("src.rtdetr_lpr.torch.amp.GradScaler", FakeSetupScaler)
    trainer = object.__new__(LPRTrainer)
    trainer.amp = True
    trainer.save_dir = tmp_path
    trainer.resume = False

    trainer._setup_train()

    assert captured == {
        "device": "cuda",
        "enabled": True,
        "init_scale": 128.0,
        "growth_interval": 2**31 - 1,
    }


def test_resume_continues_valid_optimizer_evidence_sequence(monkeypatch, tmp_path) -> None:
    records = [
        {
            "optimizer_attempt": attempt,
            "amp_scale_before": 128.0,
            "amp_scale_after": 128.0,
            "amp_step_skipped": False,
            "gradient_norm_finite": True,
        }
        for attempt in (1, 2)
    ]
    evidence = tmp_path / "optimizer-evidence.jsonl"
    evidence.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    monkeypatch.setattr("src.rtdetr_lpr.RTDETRTrainer._setup_train", lambda self: None)
    monkeypatch.setattr("src.rtdetr_lpr.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("src.rtdetr_lpr.torch.amp.GradScaler", lambda *args, **kwargs: _FakeScaler())
    trainer = object.__new__(LPRTrainer)
    trainer.amp = True
    trainer.save_dir = tmp_path
    trainer.resume = True

    trainer._setup_train()

    assert trainer.optimizer_attempt == 2


def test_resume_rejects_invalid_optimizer_evidence(monkeypatch, tmp_path) -> None:
    (tmp_path / "optimizer-evidence.jsonl").write_text(
        json.dumps(
            {
                "optimizer_attempt": 1,
                "amp_scale_before": 128.0,
                "amp_scale_after": 64.0,
                "amp_step_skipped": True,
                "gradient_norm_finite": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.rtdetr_lpr.RTDETRTrainer._setup_train", lambda self: None)
    monkeypatch.setattr("src.rtdetr_lpr.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("src.rtdetr_lpr.torch.amp.GradScaler", lambda *args, **kwargs: _FakeScaler())
    trainer = object.__new__(LPRTrainer)
    trainer.amp = True
    trainer.save_dir = tmp_path
    trainer.resume = True

    with pytest.raises(ValueError, match="optimizer evidence"):
        trainer._setup_train()


class _FakeScaler:
    def __init__(self, after: float = 128.0) -> None:
        self.scale = 128.0
        self.after = after

    def get_scale(self) -> float:
        return self.scale

    def unscale_(self, optimizer) -> None:
        del optimizer

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        self.scale = self.after


def _optimizer_trainer(tmp_path: Path, *, scale_after: float = 128.0):
    trainer = object.__new__(LPRTrainer)
    trainer.model = nn.Linear(2, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scaler = _FakeScaler(after=scale_after)
    trainer.ema = None
    trainer.optimizer_attempt = 0
    trainer.optimizer_evidence_path = tmp_path / "optimizer-evidence.jsonl"
    output = trainer.model(torch.ones(1, 2)).sum()
    output.backward()
    return trainer


def test_fixed_amp_optimizer_step_records_constant_scale(tmp_path) -> None:
    trainer = _optimizer_trainer(tmp_path)

    trainer.optimizer_step()

    record = json.loads(trainer.optimizer_evidence_path.read_text(encoding="utf-8"))
    assert record["optimizer_attempt"] == 1
    assert record["amp_scale_before"] == 128.0
    assert record["amp_scale_after"] == 128.0
    assert record["amp_step_skipped"] is False
    assert record["gradient_norm_finite"] is True


def test_fixed_amp_optimizer_step_aborts_on_scale_change(tmp_path) -> None:
    trainer = _optimizer_trainer(tmp_path, scale_after=64.0)

    with pytest.raises(RuntimeError, match="scale changed"):
        trainer.optimizer_step()

    record = json.loads(trainer.optimizer_evidence_path.read_text(encoding="utf-8"))
    assert record["amp_step_skipped"] is True
