from __future__ import annotations

from pathlib import Path

from scripts.train_rtdetr_vsf_rmr import (
    ROOT,
    build_parser,
    build_settings,
    run_name_for_variant,
    trainer_class_for_variant,
)
from src.rtdetr_vsf_rmr import VSFRMRTrainer
from ultralytics.models.rtdetr.train import RTDETRTrainer


def test_training_defaults_are_the_frozen_fair_protocol():
    args = build_parser().parse_args([])
    settings = build_settings(args)

    assert args.variant == "vsf-rmr"
    assert settings["model"] == "rtdetr-l.yaml"
    assert settings["data"] == "VisDrone.yaml"
    assert settings["epochs"] == 100
    assert settings["imgsz"] == 640
    assert settings["pretrained"] is False
    assert settings["seed"] == 0
    assert settings["deterministic"] is True
    assert settings["mosaic"] == 0.0
    assert settings["mixup"] == 0.0
    assert settings["scale"] == 0.5
    assert settings["perspective"] == 0.0
    assert settings["project"] == str(ROOT / "runs" / "vsf-rmr")


def test_baseline_and_vsf_use_different_models_but_same_settings():
    vsf_args = build_parser().parse_args(["--variant", "vsf-rmr"])
    baseline_args = build_parser().parse_args(["--variant", "baseline"])

    vsf_settings = build_settings(vsf_args)
    baseline_settings = build_settings(baseline_args)

    assert trainer_class_for_variant("vsf-rmr") is VSFRMRTrainer
    assert trainer_class_for_variant("baseline") is RTDETRTrainer
    assert run_name_for_variant("vsf-rmr") != run_name_for_variant("baseline")
    differing = {key for key in vsf_settings if vsf_settings[key] != baseline_settings[key]}
    assert differing == {"name"}


def test_resume_batch_amp_and_persistent_project_are_configurable(tmp_path: Path):
    checkpoint = tmp_path / "last.pt"
    project = tmp_path / "runs"
    args = build_parser().parse_args(
        [
            "--batch",
            "8",
            "--amp",
            "false",
            "--workers",
            "12",
            "--resume",
            str(checkpoint),
            "--project",
            str(project),
        ]
    )

    settings = build_settings(args)

    assert settings["batch"] == 8
    assert settings["amp"] is False
    assert settings["workers"] == 12
    assert settings["resume"] == str(checkpoint.resolve())
    assert settings["project"] == str(project.resolve())


def test_lambda_vsf_is_kept_out_of_ultralytics_overrides():
    args = build_parser().parse_args(["--lambda-vsf", "0.2"])
    settings = build_settings(args)

    assert args.lambda_vsf == 0.2
    assert "lambda_vsf" not in settings


def test_smoke_mode_changes_only_duration_fraction_and_name():
    normal = build_settings(build_parser().parse_args([]))
    smoke = build_settings(build_parser().parse_args(["--smoke"]))

    changed = {key for key in normal if normal[key] != smoke[key]}
    assert changed == {"epochs", "fraction", "name"}
    assert smoke["epochs"] == 1
    assert smoke["fraction"] == 0.01

