from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.train_rtdetr_vsf_rmr import (
    ROOT,
    build_parser,
    build_settings,
    run_name_for_variant,
    trainer_class_for_variant,
    update_adaptive_state_after_save,
)
from src.gpu_adaptive_batch import AdaptiveTrainingState, save_adaptive_state
from src.rtdetr_vsf_rmr import MatchedBaselineTrainer, VSFRMRTrainer, apply_resume_runtime_overrides
from ultralytics.models.rtdetr.train import RTDETRTrainer


def test_training_defaults_are_the_frozen_fair_protocol():
    args = build_parser().parse_args([])
    settings = build_settings(args)

    assert args.variant == "vsf-rmr"
    assert settings["model"] == "rtdetr-l.yaml"
    assert settings["data"] == "VisDrone.yaml"
    assert settings["epochs"] == 100
    assert settings["imgsz"] == 640
    assert settings["batch"] == 8
    assert settings["workers"] == 8
    assert settings["pretrained"] is False
    assert settings["amp"] is True
    assert settings["seed"] == 0
    assert settings["deterministic"] is True
    assert settings["optimizer"] == "auto"
    assert settings["lr0"] == 0.01
    assert settings["lrf"] == 0.01
    assert settings["momentum"] == 0.937
    assert settings["weight_decay"] == 0.0005
    assert settings["warmup_epochs"] == 3.0
    assert settings["nbs"] == 64
    assert settings["mosaic"] == 1.0
    assert settings["mixup"] == 0.0
    assert settings["scale"] == 0.5
    assert settings["translate"] == 0.1
    assert settings["perspective"] == 0.0
    assert settings["max_det"] == 300
    assert settings["save_period"] == 1
    assert settings["project"] == str(ROOT / "runs" / "vsf-rmr")


def test_baseline_and_vsf_use_different_models_but_same_settings():
    vsf_args = build_parser().parse_args(["--variant", "vsf-rmr"])
    baseline_args = build_parser().parse_args(["--variant", "baseline"])

    vsf_settings = build_settings(vsf_args)
    baseline_settings = build_settings(baseline_args)

    assert trainer_class_for_variant("vsf-rmr") is VSFRMRTrainer
    assert trainer_class_for_variant("baseline") is MatchedBaselineTrainer
    assert issubclass(MatchedBaselineTrainer, RTDETRTrainer)
    assert run_name_for_variant("vsf-rmr") != run_name_for_variant("baseline")
    differing = {key for key in vsf_settings if vsf_settings[key] != baseline_settings[key]}
    assert differing == {"name"}


def test_resume_keeps_frozen_protocol_and_allows_relocating_project(tmp_path: Path):
    checkpoint = tmp_path / "last.pt"
    project = tmp_path / "runs"
    args = build_parser().parse_args(
        [
            "--batch",
            "8",
            "--amp",
            "true",
            "--workers",
            "8",
            "--resume",
            str(checkpoint),
            "--project",
            str(project),
        ]
    )

    settings = build_settings(args)

    assert settings["batch"] == 8
    assert settings["amp"] is True
    assert settings["workers"] == 8
    assert settings["resume"] == str(checkpoint.resolve())
    assert settings["project"] == str(project.resolve())


@pytest.mark.parametrize(
    "arguments",
    [
        ["--batch", "6"],
        ["--amp", "false"],
        ["--workers", "12"],
        ["--optimizer", "AdamW"],
        ["--lr0", "0.000714"],
        ["--momentum", "0.9"],
        ["--imgsz", "800"],
        ["--epochs", "50"],
        ["--fraction", "0.5"],
    ],
)
def test_frozen_protocol_rejects_parameter_drift(arguments: list[str]):
    with pytest.raises(ValueError, match="frozen matched protocol"):
        build_settings(build_parser().parse_args(arguments))


def test_lambda_vsf_is_kept_out_of_ultralytics_overrides():
    args = build_parser().parse_args(["--lambda-vsf", "0.2"])
    settings = build_settings(args)

    assert args.lambda_vsf == 0.2
    assert "lambda_vsf" not in settings


def test_resume_reapplies_every_frozen_runtime_setting(tmp_path: Path):
    runtime_args = SimpleNamespace(project="old", name="old", save_dir="old")
    overrides = build_settings(
        build_parser().parse_args(["--project", str(tmp_path / "runs"), "--name", "fixed-run"])
    )

    apply_resume_runtime_overrides(runtime_args, overrides)

    for key in (
        "batch",
        "workers",
        "amp",
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "seed",
        "deterministic",
        "nbs",
        "mosaic",
        "mixup",
        "scale",
        "translate",
    ):
        assert getattr(runtime_args, key) == overrides[key]
    assert runtime_args.save_dir == str((tmp_path / "runs" / "fixed-run").resolve())


def test_smoke_mode_changes_only_duration_fraction_and_name():
    normal = build_settings(build_parser().parse_args([]))
    smoke = build_settings(build_parser().parse_args(["--smoke"]))

    changed = {key for key in normal if normal[key] != smoke[key]}
    assert changed == {"epochs", "fraction", "name"}
    assert smoke["epochs"] == 1
    assert smoke["fraction"] == 0.01


def test_checkpoint_callback_mirrors_adaptive_state_and_history_into_run_dir(tmp_path: Path):
    state_path = tmp_path / "state" / "vsf_state.json"
    run_dir = tmp_path / "runs" / "vsf"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    save_adaptive_state(
        state_path,
        AdaptiveTrainingState(levels=(1,), current_batch=1),
    )
    trainer = SimpleNamespace(
        last=weights / "last.pt",
        epoch=0,
        save_dir=run_dir,
    )

    update_adaptive_state_after_save(trainer, state_path)

    assert (run_dir / "adaptive_state.json").is_file()
    assert (run_dir / "batch_history.jsonl").is_file()
