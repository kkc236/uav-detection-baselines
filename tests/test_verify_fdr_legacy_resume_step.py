from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts import verify_fdr_legacy_resume_step as verifier


class MuSGD(torch.optim.SGD):
    """Tiny optimizer whose name matches the frozen production contract."""


class TinyEMA:
    def __init__(self, model: nn.Module, updates: int = 0) -> None:
        self.ema = model
        self.updates = updates

    def update(self, model: nn.Module) -> None:
        del model
        self.updates += 1


class InfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        del ctx
        return value.clone()

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        del ctx
        return (torch.full_like(gradient, float("inf")),)


class TinyModel(nn.Module):
    def __init__(self, *, loss_mode: str = "finite") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.loss_mode = loss_mode

    def loss(self, batch: dict[str, object]):
        assert tuple(batch["img"].shape) == (1, 3, 16, 16)
        if self.loss_mode == "nan":
            total = self.weight * torch.tensor(float("nan"))
        elif self.loss_mode == "inf_gradient":
            total = InfiniteGradient.apply(self.weight)
        else:
            total = self.weight.square()
        return total, torch.zeros(3)


class TinyTrainer:
    def __init__(self, *, loss_mode: str = "finite") -> None:
        self.model = TinyModel(loss_mode=loss_mode)
        self.args = SimpleNamespace(close_mosaic=9, model="legacy.pt")
        self.resume = True
        self.build_calls: list[dict[str, float | str]] = []

    def build_optimizer(self, model, *, name, lr, momentum, decay, iterations):
        del iterations
        self.build_calls.append(
            {
                "name": name,
                "lr": lr,
                "momentum": momentum,
                "decay": decay,
            }
        )
        return MuSGD(model.parameters(), lr=lr, momentum=momentum)

    def resume_training(self, checkpoint: dict[str, object]) -> None:
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.ema = verifier._new_model_ema(self.model)
        self.ema.updates = int(checkpoint["updates"])


def _tiny_resume_case(*, loss_mode: str = "finite"):
    trainer = TinyTrainer(loss_mode=loss_mode)
    optimizer = MuSGD(trainer.model.parameters(), lr=0.01, momentum=0.937)
    trainer.model.weight.square().backward()
    optimizer.step()
    checkpoint = {
        "epoch": 4,
        "optimizer": optimizer.state_dict(),
        "scaler": verifier._new_cpu_scaler().state_dict(),
        "updates": 17,
    }
    return verifier.LegacyResumeContext(
        trainer=trainer,
        checkpoint=checkpoint,
        saved_yaml_head="RTDETRDecoder",
        normalized_yaml_head="FDRRTDETRDecoder",
        state_tensor_count=1,
    )


def test_verify_resume_step_writes_atomic_success_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"legacy-fdr-checkpoint")
    output = tmp_path / "reports" / "resume.json"
    context = _tiny_resume_case()

    monkeypatch.setattr(
        verifier,
        "prepare_legacy_resume",
        lambda **_kwargs: context,
    )
    monkeypatch.setattr(
        verifier,
        "_new_model_ema",
        lambda model: TinyEMA(model),
    )

    report = verifier.verify_legacy_resume_step(
        checkpoint=checkpoint,
        output=output,
        nc=7,
        imgsz=16,
    )

    assert report["checkpoint"] == str(checkpoint.resolve())
    assert report["saved_yaml_head"] == "RTDETRDecoder"
    assert report["normalized_yaml_head"] == "FDRRTDETRDecoder"
    assert report["state_tensor_count"] == 1
    assert report["checkpoint_epoch"] == 4
    assert report["resume_start_epoch"] == 5
    assert report["optimizer_param_groups"] == 1
    assert report["optimizer_state_entries"] == 1
    assert report["amp_scale_before_step"] == 128.0
    assert report["amp_scale_after_step"] == 128.0
    assert report["ema_updates_before_step"] == 17
    assert report["ema_updates_after_step"] == 18
    assert report["synthetic_resume_step_input"] == [1, 3, 16, 16]
    assert report["finite_loss"] is True
    assert report["finite_gradients"] is True
    assert report["optimizer_step"] is True
    assert report["resume_step_verified"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert list(output.parent.glob("*.tmp")) == []
    assert context.trainer.args.close_mosaic == 0
    assert context.trainer.epochs == 6
    assert context.trainer.build_calls == [
        {
            "name": "MuSGD",
            "lr": 0.01,
            "momentum": 0.937,
            "decay": 0.0005,
        }
    ]


@pytest.mark.parametrize(
    ("loss_mode", "message"),
    [("nan", "non-finite loss"), ("inf_gradient", "non-finite gradients")],
)
def test_nonfinite_resume_step_fails_without_publishing_success(
    loss_mode: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"legacy-fdr-checkpoint")
    output = tmp_path / "resume.json"
    monkeypatch.setattr(
        verifier,
        "prepare_legacy_resume",
        lambda **_kwargs: _tiny_resume_case(loss_mode=loss_mode),
    )
    monkeypatch.setattr(
        verifier,
        "_new_model_ema",
        lambda model: TinyEMA(model),
    )

    with pytest.raises(RuntimeError, match=message):
        verifier.verify_legacy_resume_step(
            checkpoint=checkpoint,
            output=output,
            imgsz=16,
        )

    assert not output.exists()


def test_atomic_writer_preserves_old_report_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "resume.json"
    output.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr(verifier.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        verifier.write_json_atomic(output, {"new": True})

    assert json.loads(output.read_text(encoding="utf-8")) == {"old": True}
    assert list(tmp_path.glob("*.tmp")) == []


def test_cli_parses_all_arguments_and_prints_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    output = tmp_path / "resume.json"
    expected = {"resume_step_verified": True}
    calls: list[dict[str, object]] = []

    def fake_verify(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(verifier, "verify_legacy_resume_step", fake_verify)
    exit_code = verifier.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--nc",
            "9",
            "--imgsz",
            "16",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "checkpoint": checkpoint,
            "output": output,
            "nc": 9,
            "imgsz": 16,
        }
    ]
    assert json.loads(capsys.readouterr().out) == expected


def test_prepare_failure_never_creates_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "missing.pt"
    output = tmp_path / "resume.json"

    def fail_prepare(**_kwargs):
        raise ValueError("legacy checkpoint rejected")

    monkeypatch.setattr(verifier, "prepare_legacy_resume", fail_prepare)
    with pytest.raises(ValueError, match="legacy checkpoint rejected"):
        verifier.verify_legacy_resume_step(
            checkpoint=checkpoint,
            output=output,
        )

    assert not output.exists()
