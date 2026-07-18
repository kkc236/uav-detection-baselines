from pathlib import Path


def test_matched_baseline_settings_match_btdse_protocol(tmp_path: Path):
    from scripts.train_rtdetr_matched_baseline import build_parser, build_settings

    args = build_parser().parse_args(["--project", str(tmp_path)])
    settings = build_settings(args)

    assert settings == {
        "model": "rtdetr-l.yaml",
        "data": "VisDrone.yaml",
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "device": "0",
        "project": str(tmp_path.resolve()),
        "name": "scratch-rtdetr-l-btdse-matched-baseline-100ep",
        "exist_ok": True,
        "pretrained": False,
        "cache": False,
        "amp": True,
        "deterministic": True,
        "seed": 0,
        "nbs": 64,
        "nms": False,
        "max_det": 300,
        "save": True,
        "save_period": 1,
        "optimizer": "auto",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "scale": 0.5,
        "translate": 0.1,
        "plots": True,
        "val": True,
    }


def test_matched_baseline_resume_keeps_the_same_protocol(tmp_path: Path):
    from scripts.train_rtdetr_matched_baseline import build_parser, build_settings

    checkpoint = tmp_path / "last.pt"
    args = build_parser().parse_args(["--project", str(tmp_path), "--resume", str(checkpoint)])
    settings = build_settings(args)

    assert settings["resume"] == str(checkpoint.resolve())
    assert settings["batch"] == 8
    assert settings["amp"] is True
    assert settings["optimizer"] == "auto"
    assert settings["lr0"] == 0.01


def test_matched_baseline_rejects_protocol_changes():
    from scripts.train_rtdetr_matched_baseline import build_parser

    options = {action.dest for action in build_parser()._actions}
    assert "batch" not in options
    assert "amp" not in options
    assert "optimizer" not in options
    assert "lr0" not in options


def test_matched_baseline_help_is_renderable():
    from scripts.train_rtdetr_matched_baseline import build_parser

    help_text = build_parser().format_help()

    assert "fixed-protocol" in help_text
