from __future__ import annotations

from scripts.train_rtdetr_gcmv_plec import build_parser, build_settings


def test_training_settings_are_bounded_and_explicit(tmp_path):
    args = build_parser().parse_args(
        [
            "--pretrained-weights",
            "baseline.pt",
            "--data",
            "visdrone.yaml",
            "--project",
            str(tmp_path),
            "--name",
            "plec-screen",
        ]
    )

    settings = build_settings(args)

    assert settings["model"].endswith("rtdetr-l-gcmv-plec.yaml")
    assert settings["pretrained"] == "baseline.pt"
    assert settings["data"] == "visdrone.yaml"
    assert settings["epochs"] == 3
    assert settings["fraction"] == 0.03
    assert settings["batch"] == 1
    assert settings["imgsz"] == 640
    assert settings["val"] is False
    assert settings["mosaic"] == 0.0
    assert settings["mixup"] == 0.0
    assert settings["cutmix"] == 0.0
    assert settings["copy_paste"] == 0.0

