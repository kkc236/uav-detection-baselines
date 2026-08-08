from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "glgm_experiment.py"
SPEC = importlib.util.spec_from_file_location("glgm_experiment", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_epoch_finite_guard_accepts_finite_values() -> None:
    trainer = SimpleNamespace(
        epoch=4,
        metrics={"metrics/mAP50-95(B)": 0.25, "val/giou_loss": 0.71},
        tloss={"giou_loss": torch.tensor(0.4)},
        loss_items={"cls_loss": torch.tensor(0.2)},
        loss=torch.tensor(0.6),
        fitness=0.25,
    )
    MODULE.make_epoch_finite_guard(torch)(trainer)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_epoch_finite_guard_rejects_non_finite_metrics(bad_value: float) -> None:
    trainer = SimpleNamespace(
        epoch=4,
        metrics={"val/giou_loss": bad_value},
        tloss={},
        loss_items={},
        loss=torch.tensor(0.6),
        fitness=0.25,
    )
    with pytest.raises(FloatingPointError, match=r"epoch 5: metrics\.val/giou_loss"):
        MODULE.make_epoch_finite_guard(torch)(trainer)


def test_epoch_finite_guard_rejects_non_finite_tensor() -> None:
    trainer = SimpleNamespace(
        epoch=0,
        metrics={},
        tloss={"giou_loss": torch.tensor([0.3, float("nan")])},
        loss_items={},
        loss=torch.tensor(0.6),
        fitness=0.25,
    )
    with pytest.raises(FloatingPointError, match=r"tloss\.giou_loss"):
        MODULE.make_epoch_finite_guard(torch)(trainer)


def strict_trainer() -> SimpleNamespace:
    model = torch.nn.Linear(2, 1)
    ema = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return SimpleNamespace(
        epoch=2,
        metrics={"metrics/mAP50-95(B)": 0.2},
        tloss={"giou_loss": torch.tensor(0.4)},
        loss_items={"cls_loss": torch.tensor(0.2)},
        loss=torch.tensor(0.6),
        fitness=0.2,
        nan_recovery_attempts=0,
        model=model,
        ema=SimpleNamespace(ema=ema),
        optimizer=optimizer,
        scaler=scaler,
    )


def test_strict_nan_handler_disables_internal_recovery() -> None:
    trainer = strict_trainer()
    handler = MODULE.make_strict_nan_recovery_handler(torch)
    assert handler(trainer, trainer.epoch) is False
    assert trainer._glgm_strict_finite_epoch == trainer.epoch
    assert handler._glgm_strict_nan_policy is True


def test_strict_nan_handler_fails_before_recovery_on_non_finite_loss() -> None:
    trainer = strict_trainer()
    trainer.loss = torch.tensor(float("nan"))
    with pytest.raises(FloatingPointError, match=r"loss"):
        MODULE.make_strict_nan_recovery_handler(torch)(trainer, trainer.epoch)
    assert not hasattr(trainer, "_glgm_strict_finite_epoch")


@pytest.mark.parametrize("state_name", ["model", "ema", "optimizer"])
def test_strict_nan_handler_rejects_non_finite_training_state(state_name: str) -> None:
    trainer = strict_trainer()
    if state_name == "model":
        trainer.model.weight.data[0, 0] = float("nan")
    elif state_name == "ema":
        trainer.ema.ema.bias.data[0] = float("inf")
    else:
        parameter = next(trainer.model.parameters())
        trainer.optimizer.state[parameter]["momentum_buffer"] = torch.tensor(float("nan"))
    with pytest.raises(FloatingPointError, match=state_name):
        MODULE.make_strict_nan_recovery_handler(torch)(trainer, trainer.epoch)


def test_strict_nan_handler_rejects_existing_recovery_history() -> None:
    trainer = strict_trainer()
    trainer.nan_recovery_attempts = 1
    with pytest.raises(RuntimeError, match=r"1 NaN recovery attempt"):
        MODULE.make_strict_nan_recovery_handler(torch)(trainer, trainer.epoch)


def test_engine_source_contract_accepts_only_audited_hashes(monkeypatch, tmp_path: Path) -> None:
    engine = tmp_path / "ultralytics" / "engine"
    engine.mkdir(parents=True)
    for name in MODULE.EXPECTED_ENGINE_SHA256:
        (engine / name).write_text(name, encoding="ascii")

    monkeypatch.setattr(
        MODULE,
        "sha256_file",
        lambda path: MODULE.EXPECTED_ENGINE_SHA256[path.name],
    )
    MODULE.verify_engine_source(tmp_path)

    monkeypatch.setattr(MODULE, "sha256_file", lambda _path: "0" * 64)
    with pytest.raises(RuntimeError, match="not the audited strict-training version"):
        MODULE.verify_engine_source(tmp_path)
