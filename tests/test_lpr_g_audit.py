from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from src.lpr_g_audit import (
    common_model_fingerprint,
    common_optimizer_fingerprint,
    write_epoch_audit,
)


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stock = nn.Linear(2, 2)
        self.lpr_g_refiner = nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.stock(inputs).sum() + self.lpr_g_refiner(inputs).sum()


def _stepped_pair() -> tuple[Tiny, torch.optim.Optimizer, Tiny, torch.optim.Optimizer]:
    first = Tiny()
    second = Tiny()
    second.stock.load_state_dict(first.stock.state_dict())
    optimizer_a = torch.optim.SGD(first.parameters(), lr=0.1, momentum=0.937)
    optimizer_b = torch.optim.SGD(second.parameters(), lr=0.1, momentum=0.937)
    for model, optimizer in ((first, optimizer_a), (second, optimizer_b)):
        model(torch.ones(1, 2)).backward()
        optimizer.step()
    return first, optimizer_a, second, optimizer_b


def test_common_fingerprints_ignore_private_parameters_and_state() -> None:
    first, optimizer_a, second, optimizer_b = _stepped_pair()
    with torch.no_grad():
        second.lpr_g_refiner.weight.add_(5)
    optimizer_b.state[second.lpr_g_refiner.weight]["momentum_buffer"].add_(7)

    assert common_model_fingerprint(first) == common_model_fingerprint(second)
    assert common_optimizer_fingerprint(first, optimizer_a) == common_optimizer_fingerprint(
        second,
        optimizer_b,
    )


def test_common_fingerprint_changes_with_stock_parameter() -> None:
    first, _, second, _ = _stepped_pair()
    with torch.no_grad():
        second.stock.weight.add_(1)

    assert common_model_fingerprint(first) != common_model_fingerprint(second)


def test_write_epoch_audit_appends_machine_readable_record(tmp_path: Path) -> None:
    model = Tiny()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.937)
    model(torch.ones(1, 2)).backward()
    optimizer.step()
    path = tmp_path / "common_state_audit.jsonl"

    record = write_epoch_audit(path, epoch=1, model=model, optimizer=optimizer)

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == record
    assert stored["epoch"] == 1
    assert len(stored["common_model_sha256"]) == 64
    assert len(stored["common_optimizer_sha256"]) == 64
