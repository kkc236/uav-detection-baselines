from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.train_rtdetr_ioqc_sa import ROOT, build_parser, build_settings
from src.rtdetr_ioqc_sa import apply_resume_runtime_overrides


def test_training_defaults_match_scratch_rtdetr_l_baseline():
    args = build_parser().parse_args([])
    settings = build_settings(args)

    assert settings["model"] == "rtdetr-l.yaml"
    assert settings["data"] == "VisDrone.yaml"
    assert settings["epochs"] == 100
    assert settings["imgsz"] == 640
    assert settings["pretrained"] is False
    assert settings["deterministic"] is True
    assert settings["seed"] == 0
    assert settings["nms"] is False
    assert settings["save_period"] == 1
    assert settings["optimizer"] == "AdamW"
    assert settings["nbs"] == 64
    assert settings["project"] == str(ROOT / "runs" / "ioqc-sa")


def test_method_weights_are_not_forwarded_as_ultralytics_overrides():
    args = build_parser().parse_args(
        [
            "--lambda-competition",
            "0.2",
            "--lambda-alignment",
            "0.3",
            "--density-threshold",
            "1.2",
            "--duplicate-threshold",
            "0.15",
        ]
    )
    settings = build_settings(args)

    assert args.lambda_competition == 0.2
    assert args.lambda_alignment == 0.3
    assert args.density_threshold == 1.2
    assert args.duplicate_threshold == 0.15
    assert "lambda_competition" not in settings
    assert "lambda_alignment" not in settings
    assert "density_threshold" not in settings
    assert "duplicate_threshold" not in settings


def test_amp_batch_resume_and_persistent_project_are_configurable(tmp_path: Path):
    checkpoint = tmp_path / "last.pt"
    project = tmp_path / "runs"
    args = build_parser().parse_args(
        [
            "--amp",
            "false",
            "--batch",
            "6",
            "--workers",
            "8",
            "--save-period",
            "5",
            "--resume",
            str(checkpoint),
            "--project",
            str(project),
        ]
    )
    settings = build_settings(args)

    assert settings["amp"] is False
    assert settings["batch"] == 6
    assert settings["workers"] == 8
    assert settings["save_period"] == 5
    assert settings["resume"] == str(checkpoint.resolve())
    assert settings["project"] == str(project.resolve())


def test_smoke_mode_limits_epoch_and_fraction_only():
    args = build_parser().parse_args(["--smoke"])
    settings = build_settings(args)

    assert settings["epochs"] == 1
    assert settings["fraction"] == 0.01
    assert settings["imgsz"] == 640


def test_resume_runtime_override_can_permanently_disable_amp():
    runtime_args = SimpleNamespace(
        amp=True,
        project="old/project",
        name="old-run",
        optimizer="auto",
        save_dir="/old/server/run",
    )

    apply_resume_runtime_overrides(
        runtime_args,
        {
            "amp": False,
            "project": "/new/persistent/runs",
            "name": "ioqc-new-server",
            "optimizer": "AdamW",
        },
    )

    assert runtime_args.amp is False
    assert runtime_args.project == "/new/persistent/runs"
    assert runtime_args.name == "ioqc-new-server"
    assert runtime_args.optimizer == "AdamW"
    assert runtime_args.save_dir == str(Path("/new/persistent/runs/ioqc-new-server").resolve())
