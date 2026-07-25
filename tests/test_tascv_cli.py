from __future__ import annotations

from argparse import Namespace

from src.tascv_cli import build_parser, build_settings
from src.tascv_stage import TASCVStage


def test_cli_exposes_no_scientific_tuning_switches() -> None:
    parser = build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    forbidden = {
        "--epochs",
        "--batch",
        "--imgsz",
        "--optimizer",
        "--lr0",
        "--lambda",
        "--tiny-threshold",
        "--crop-size",
        "--amp-scale",
        "--workers",
        "--resume",
        "--pretrained",
        "--val",
        "--test",
    }
    assert not options.intersection(forbidden)


def test_build_settings_exactly_matches_frozen_baseline() -> None:
    args = Namespace(
        stage=TASCVStage.PREFLIGHT_1,
        data=__file__,
        project="runs",
        name="preflight-control",
        device="0",
        seed=0,
    )

    settings = build_settings(args)

    assert settings["epochs"] == 100
    assert settings["imgsz"] == 640
    assert settings["batch"] == 8
    assert settings["workers"] == 8
    assert settings["optimizer"] == "MuSGD"
    assert settings["lr0"] == 0.01
    assert settings["momentum"] == 0.937
    assert settings["pretrained"] is False
    assert settings["resume"] is False
    assert settings["amp"] is True
    assert settings["save_period"] == -1
    assert settings["val"] is False
    assert settings["nms"] is False
    assert settings["max_det"] == 300
