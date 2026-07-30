from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.train_rtdetr_sqda_sgc import (
    build_parser,
    build_settings,
    record_stage_status,
)
from src.rtdetr_sqda_sgc import (
    MATCHED_AMP_GROWTH_INTERVAL,
    MATCHED_AMP_SCALE,
    SQDAGeometryTrustTrainer,
    SQDASGCTrainer,
    assert_geometry_trust_contract,
    build_geometry_trust_optimizer,
    build_sqda_optimizer,
    freeze_inherited_sqda,
    freeze_stock_model,
)
from src.sqda_sgc import SQDASGCAdapter


class _ToyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 8),
            nn.BatchNorm1d(8),
            nn.Linear(8, 4),
        )
        self.sqda_sgc = SQDASGCAdapter()


class _FixedTestScaler:
    def get_scale(self) -> float:
        return MATCHED_AMP_SCALE

    def unscale_(self, _optimizer) -> None:
        return None

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


def _parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def test_freeze_contract_and_module_only_optimizer() -> None:
    detector = _ToyDetector()
    freeze_stock_model(detector)
    optimizer = build_sqda_optimizer(detector)

    assert all(not parameter.requires_grad for parameter in detector.model.parameters())
    assert all(parameter.requires_grad for parameter in detector.sqda_sgc.parameters())
    assert _parameter_ids(optimizer) == {id(p) for p in detector.sqda_sgc.parameters()}
    assert not (_parameter_ids(optimizer) & {id(p) for p in detector.model.parameters()})
    assert isinstance(optimizer, torch.optim.AdamW)
    assert all(group["lr"] == pytest.approx(1e-4) for group in optimizer.param_groups)
    assert all(group["betas"] == (0.9, 0.999) for group in optimizer.param_groups)


def test_geometry_trust_step_changes_only_the_new_gate() -> None:
    torch.manual_seed(23)
    detector = _ToyDetector()
    freeze_inherited_sqda(detector)
    assert_geometry_trust_contract(detector)
    optimizer = build_geometry_trust_optimizer(detector)
    stock_before = {
        key: value.detach().clone()
        for key, value in detector.model.state_dict().items()
    }
    adapter_before = {
        key: value.detach().clone()
        for key, value in detector.sqda_sgc.state_dict().items()
    }

    queries = torch.randn(1, 12, 256)
    boxes = torch.rand(1, 12, 4)
    c2 = torch.randn(1, 128, 16, 16)
    enhanced, _ = detector.sqda_sgc(queries, boxes, c2)
    enhanced.square().mean().backward()
    optimizer.step()

    geometry_names = {
        name for name, parameter in detector.sqda_sgc.named_parameters() if parameter.requires_grad
    }
    assert geometry_names
    assert all(name.startswith("geometry_trust.") for name in geometry_names)
    assert _parameter_ids(optimizer) == {
        id(parameter)
        for name, parameter in detector.sqda_sgc.named_parameters()
        if name.startswith("geometry_trust.")
    }
    assert all(
        torch.equal(value, stock_before[key])
        for key, value in detector.model.state_dict().items()
    )
    assert all(
        torch.equal(value, adapter_before[key])
        for key, value in detector.sqda_sgc.state_dict().items()
        if not key.startswith("geometry_trust.")
    )
    assert any(
        not torch.equal(value, adapter_before[key])
        for key, value in detector.sqda_sgc.state_dict().items()
        if key.startswith("geometry_trust.")
    )
    for name in geometry_names:
        gradient = dict(detector.sqda_sgc.named_parameters())[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient), name


def test_optimizer_decay_groups_match_parameter_roles() -> None:
    detector = _ToyDetector()
    freeze_stock_model(detector)
    optimizer = build_sqda_optimizer(detector)
    groups = {group["param_group"]: group for group in optimizer.param_groups}

    assert set(groups) == {"matrix", "no_decay"}
    assert groups["matrix"]["weight_decay"] == pytest.approx(1e-4)
    assert groups["no_decay"]["weight_decay"] == 0.0

    matrix_ids = {id(p) for p in groups["matrix"]["params"]}
    no_decay_ids = {id(p) for p in groups["no_decay"]["params"]}
    for module_name, module in detector.sqda_sgc.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            if isinstance(module, nn.Linear) and parameter_name == "weight":
                assert id(parameter) in matrix_ids, full_name
            else:
                assert id(parameter) in no_decay_ids, full_name


def test_one_step_changes_adapter_but_not_stock_parameters_or_buffers() -> None:
    torch.manual_seed(17)
    detector = _ToyDetector()
    freeze_stock_model(detector)
    optimizer = build_sqda_optimizer(detector)
    stock_before = {
        key: value.detach().clone()
        for key, value in detector.model.state_dict().items()
    }
    adapter_before = {
        key: value.detach().clone()
        for key, value in detector.sqda_sgc.state_dict().items()
    }

    queries = torch.randn(1, 12, 256)
    boxes = torch.rand(1, 12, 4)
    c2 = torch.randn(1, 128, 16, 16)
    enhanced, _ = detector.sqda_sgc(queries, boxes, c2)
    enhanced.square().mean().backward()
    torch.nn.utils.clip_grad_norm_(detector.sqda_sgc.parameters(), max_norm=0.1)
    optimizer.step()

    assert all(
        torch.equal(value, stock_before[key])
        for key, value in detector.model.state_dict().items()
    )
    assert any(
        not torch.equal(value, adapter_before[key])
        for key, value in detector.sqda_sgc.state_dict().items()
    )
    representative_branches = {
        "point_offset_heads.0.weight",
        "value_projector.0.weight",
        "point_query.weight",
        "edge_query.weight",
        "reliability_projection.weight",
        "gate.0.weight",
        "fusion.weight",
        "geometry_trust.0.weight",
        "geometry_trust.2.weight",
        "context_logit",
        "layer_scale_logit",
    }
    named_parameters = dict(detector.sqda_sgc.named_parameters())
    for name in representative_branches:
        gradient = named_parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient), name


def test_trainer_optimizer_step_clips_only_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = _ToyDetector()
    freeze_stock_model(detector)
    optimizer = build_sqda_optimizer(detector)
    for parameter in detector.sqda_sgc.parameters():
        parameter.grad = torch.ones_like(parameter)

    trainer = SQDASGCTrainer.__new__(SQDASGCTrainer)
    trainer.model = detector
    trainer.optimizer = optimizer
    trainer.scaler = _FixedTestScaler()
    trainer.ema = None
    recorded: dict[str, object] = {}
    original_clip = torch.nn.utils.clip_grad_norm_

    def recording_clip(parameters, max_norm, *args, **kwargs):
        materialized = list(parameters)
        recorded["ids"] = {id(parameter) for parameter in materialized}
        recorded["max_norm"] = max_norm
        return original_clip(materialized, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)
    trainer.optimizer_step()

    assert recorded["ids"] == {id(p) for p in detector.sqda_sgc.parameters()}
    assert recorded["max_norm"] == pytest.approx(0.1)


def test_geometry_trust_trainer_clips_only_new_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = _ToyDetector()
    freeze_inherited_sqda(detector)
    optimizer = build_geometry_trust_optimizer(detector)
    for parameter in detector.sqda_sgc.geometry_trust.parameters():
        parameter.grad = torch.ones_like(parameter)

    trainer = SQDAGeometryTrustTrainer.__new__(SQDAGeometryTrustTrainer)
    trainer.model = detector
    trainer.optimizer = optimizer
    trainer.scaler = _FixedTestScaler()
    trainer.ema = None
    recorded: dict[str, object] = {}
    original_clip = torch.nn.utils.clip_grad_norm_

    def recording_clip(parameters, max_norm, *args, **kwargs):
        materialized = list(parameters)
        recorded["ids"] = {id(parameter) for parameter in materialized}
        recorded["max_norm"] = max_norm
        return original_clip(materialized, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)
    trainer.optimizer_step()

    assert recorded["ids"] == {id(p) for p in detector.sqda_sgc.geometry_trust.parameters()}
    assert recorded["max_norm"] == pytest.approx(0.1)


@pytest.mark.parametrize("gate,epochs", [("g1", 3), ("g1r", 3), ("g2", 10)])
def test_formal_settings_are_frozen(
    tmp_path: Path,
    gate: str,
    epochs: int,
) -> None:
    checkpoint = tmp_path / "baseline.pt"
    data = tmp_path / "VisDrone.yaml"
    args = build_parser().parse_args(
        [
            "--gate",
            gate,
            "--checkpoint",
            str(checkpoint),
            "--data",
            str(data),
            "--project",
            str(tmp_path / "runs"),
        ]
    )
    settings = build_settings(args)

    assert settings["epochs"] == epochs
    assert settings["seed"] == 0
    assert settings["deterministic"] is True
    assert settings["imgsz"] == 640
    assert settings["batch"] == 8
    assert settings["optimizer"] == "AdamW"
    assert settings["lr0"] == pytest.approx(1e-4)
    assert settings["lrf"] == 1.0
    assert settings["momentum"] == 0.9
    assert settings["weight_decay"] == pytest.approx(1e-4)
    assert settings["warmup_epochs"] == 0.5
    assert settings["warmup_momentum"] == 0.8
    assert settings["warmup_bias_lr"] == 0.0
    assert settings["cos_lr"] is False
    assert settings["resume"] is False
    assert settings["nms"] is False
    assert settings["max_det"] == 300
    assert settings["freeze"] == list(range(29))
    assert settings["pretrained"] is False
    assert settings["model"] == "rtdetr-l.yaml"
    assert settings["close_mosaic"] == 10
    assert settings["degrees"] == 0.0
    assert settings["shear"] == 0.0
    assert settings["perspective"] == 0.0
    assert settings["flipud"] == 0.0
    assert settings["fliplr"] == 0.5
    assert settings["hsv_h"] == 0.015
    assert settings["hsv_s"] == 0.7
    assert settings["hsv_v"] == 0.4
    assert settings["cutmix"] == 0.0
    assert settings["copy_paste"] == 0.0
    assert MATCHED_AMP_SCALE == 128.0
    assert MATCHED_AMP_GROWTH_INTERVAL == 2**31 - 1


def test_resume_and_target_epoch_controls_are_explicit_and_relocatable(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "baseline.pt"
    data = tmp_path / "VisDrone.yaml"
    resume = tmp_path / "runs" / "sqda-sgc-g2" / "weights" / "epoch4.pt"
    args = build_parser().parse_args(
        [
            "--gate",
            "formal",
            "--checkpoint",
            str(checkpoint),
            "--data",
            str(data),
            "--project",
            str(tmp_path / "runs"),
            "--resume-from",
            str(resume),
            "--target-epochs",
            "100",
        ]
    )
    settings = build_settings(args)

    assert settings["epochs"] == 100
    assert settings["resume"] == str(resume.resolve())
    assert settings["name"] == "sqda-sgc-formal-seed0-100ep"


def test_cli_does_not_expose_protocol_mutations() -> None:
    options = {action.dest for action in build_parser()._actions}
    assert not {
        "epochs",
        "seed",
        "imgsz",
        "batch",
        "optimizer",
        "lr0",
        "resume",
        "amp",
        "max_det",
    }.intersection(options)


def test_stage_status_never_exceeds_the_requested_epoch_budget(tmp_path: Path) -> None:
    trainer = SimpleNamespace(
        sqda_gate="g2",
        epoch=10,
        epochs=10,
        metrics={"metrics/mAP50-95(B)": 0.24},
        fitness=0.24,
        best_fitness=0.24,
        save_dir=tmp_path,
    )

    record_stage_status(trainer)

    status = json.loads((tmp_path / "stage-status.json").read_text(encoding="utf-8"))
    assert status["completed_epoch"] == 10
    assert status["target_epochs"] == 10
