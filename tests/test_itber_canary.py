from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from scripts.run_itber_canary import run_private_step_canary


class _TinyFrozenDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2)
        self.requires_grad_(False)
        self.eval()


class _TinyAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.detector = _TinyFrozenDetector()
        self.refiner = nn.Linear(2, 4)
        nn.init.zeros_(self.refiner.weight)
        nn.init.zeros_(self.refiner.bias)

    def forward_evidence(self, image: torch.Tensor):
        stock = torch.full((image.shape[0], 1, 4), 0.5)
        delta = self.refiner(torch.ones(image.shape[0], 1, 2))
        return SimpleNamespace(stock_boxes=stock, refined_boxes=stock + delta)

    def training_step(self, batch: dict[str, torch.Tensor]):
        output = self.forward_evidence(batch["img"])
        target = torch.full_like(output.refined_boxes, 0.6)
        total = (output.refined_boxes - target).abs().mean()
        return SimpleNamespace(total=total, box=total, gate=total * 0, noop=total * 0)


def test_canary_proves_identity_private_update_and_detector_invariance() -> None:
    adapter = _TinyAdapter()
    batch = {"img": torch.zeros(1, 3, 8, 8)}

    report = run_private_step_canary(adapter, batch, use_amp=False)

    assert report["status"] == "passed"
    assert report["checks"] == {
        "zero_init_identity": True,
        "finite_private_loss": True,
        "private_gradient_present": True,
        "detector_gradient_absent": True,
        "detector_state_unchanged": True,
        "private_state_changed": True,
        "checkpoint_roundtrip": True,
    }


def test_canary_rejects_trainable_detector() -> None:
    adapter = _TinyAdapter()
    adapter.detector.projection.weight.requires_grad_(True)

    try:
        run_private_step_canary(adapter, {"img": torch.zeros(1, 3, 8, 8)}, use_amp=False)
    except ValueError as error:
        assert "requires_grad" in str(error)
    else:
        raise AssertionError("trainable detector passed Gate 0")


def test_canary_source_uses_approved_execution_environment_and_status() -> None:
    source = __import__("pathlib").Path("scripts/run_itber_canary.py").read_text(
        encoding="utf-8"
    )
    assert "current_execution_environment" in source
    assert "passed_with_runtime_amendment" in source
    assert "ACCEPTED_GATE_STATUSES" in source
