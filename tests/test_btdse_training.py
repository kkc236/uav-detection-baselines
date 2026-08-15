from pathlib import Path

from scripts.train_rtdetr_btdse import ROOT, build_parser, build_settings


def test_btdse_training_defaults_are_scratch_and_local_gpu_safe():
    args = build_parser().parse_args([])
    settings = build_settings(args)

    assert settings["model"] == "configs/rtdetr-l-btdse.yaml"
    assert settings["data"] == "VisDrone.yaml"
    assert settings["pretrained"] is False
    assert settings["imgsz"] == 640
    assert settings["batch"] == 1
    assert settings["workers"] == 2
    assert settings["cache"] is False
    assert settings["project"] == str(ROOT / "runs" / "btdse")
    assert settings["exist_ok"] is True
    assert settings["save_period"] == 1


def test_smoke_mode_limits_data_and_epochs_without_changing_image_size():
    args = build_parser().parse_args(["--smoke"])
    settings = build_settings(args)

    assert settings["epochs"] == 1
    assert settings["fraction"] == 0.01
    assert settings["imgsz"] == 640


def test_loss_weights_are_cli_configurable_but_not_ultralytics_overrides():
    args = build_parser().parse_args(["--lambda-background", "0.2", "--lambda-saliency", "0.3"])
    settings = build_settings(args)

    assert args.lambda_background == 0.2
    assert args.lambda_saliency == 0.3
    assert "lambda_background" not in settings
    assert "lambda_saliency" not in settings


def test_resume_checkpoint_is_forwarded_to_ultralytics(tmp_path: Path):
    checkpoint = tmp_path / "last.pt"
    args = build_parser().parse_args(["--resume", str(checkpoint)])
    settings = build_settings(args)

    assert settings["resume"] == str(checkpoint.resolve())


def test_server_output_directory_and_workers_are_configurable(tmp_path: Path):
    project = tmp_path / "persistent-runs"
    args = build_parser().parse_args(
        ["--project", str(project), "--workers", "8", "--batch", "8"]
    )
    settings = build_settings(args)

    assert settings["project"] == str(project.resolve())
    assert settings["workers"] == 8
    assert settings["batch"] == 8
