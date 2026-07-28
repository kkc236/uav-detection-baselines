from __future__ import annotations

from scripts.train_gcte_formal import build_parser, build_settings


def test_formal_parser_defaults_to_frozen_seed0_protocol() -> None:
    args = build_parser().parse_args([])
    settings = build_settings(args)

    assert args.epochs == 100
    assert args.imgsz == 640
    assert args.batch == 8
    assert args.workers == 8
    assert args.device == "0"
    assert args.seed == 0
    assert settings["pretrained"] is False
    assert settings["amp"] is True
    assert settings["deterministic"] is True
    assert settings["optimizer"] == "MuSGD"
    assert settings["lr0"] == 0.01
    assert settings["lrf"] == 0.01
    assert settings["momentum"] == 0.937
    assert settings["weight_decay"] == 0.0005
    assert settings["warmup_epochs"] == 3.0
    assert settings["nbs"] == 64
    assert settings["workers"] == 8
    assert settings["max_det"] == 300
    assert settings["nms"] is False
    assert settings["amp_scale"] == 128.0


def test_formal_settings_keep_frozen_augmentation() -> None:
    args = build_parser().parse_args([])
    settings = build_settings(args)

    assert settings["mosaic"] == 1.0
    assert settings["close_mosaic"] == 10
    assert settings["mixup"] == 0.0
    assert settings["scale"] == 0.5
    assert settings["translate"] == 0.1
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


def test_formal_entry_uses_new_output_and_explicit_protocol_manifest(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--project",
            str(tmp_path),
            "--name",
            "acr-eg-formal-100",
            "--data",
            "/mnt/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml",
        ]
    )
    settings = build_settings(args)

    assert settings["project"] == str(tmp_path.resolve())
    assert settings["name"] == "acr-eg-formal-100"
    assert settings["data"].endswith("source-VisDrone-full.yaml")
    assert settings["exist_ok"] is False
    assert settings["resume"] is False
    assert settings["cache"] is False
