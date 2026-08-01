from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from src.rtdetr_lpr_g import LPRGTrainer


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stock = nn.Linear(2, 2)
        self.lpr_g_refiner = nn.Linear(2, 1)


class FakeScaler:
    def get_scale(self) -> float:
        return 128.0

    def unscale_(self, optimizer) -> None:
        del optimizer

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


def test_lpr_g_trainer_returns_disjoint_stock_and_private_groups() -> None:
    trainer = object.__new__(LPRGTrainer)
    trainer.model = Tiny()

    groups = trainer.gradient_parameter_groups()

    assert set(groups) == {"gradient_norm", "lpr_g_gradient_norm"}
    assert set(map(id, groups["gradient_norm"])).isdisjoint(
        map(id, groups["lpr_g_gradient_norm"])
    )
    assert len(groups["gradient_norm"]) == 2
    assert len(groups["lpr_g_gradient_norm"]) == 2


def test_optimizer_step_records_separate_norms(tmp_path: Path) -> None:
    trainer = object.__new__(LPRGTrainer)
    trainer.model = Tiny()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scaler = FakeScaler()
    trainer.ema = None
    trainer.optimizer_attempt = 0
    trainer.optimizer_evidence_path = tmp_path / "optimizer-evidence.jsonl"
    for name, parameter in trainer.model.named_parameters():
        parameter.grad = torch.full_like(parameter, 100.0 if "lpr_g_refiner" in name else 1.0)

    trainer.optimizer_step()

    record = json.loads(trainer.optimizer_evidence_path.read_text(encoding="utf-8"))
    assert record["gradient_norm"] is not None
    assert record["lpr_g_gradient_norm"] is not None
    assert record["lpr_g_gradient_norm"] > record["gradient_norm"]
    assert record["gradient_norm_finite"] is True
