from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.train_rtdetr_gcmv_plec import (
    build_parser,
    build_settings,
    trainer_class,
)
from src.rtdetr_gcmv_plec import (
    GCMVPLECControlTrainer,
    GCMVPLECTrainer,
)


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


def test_control_flag_selects_the_matched_stock_arm():
    method = build_parser().parse_args(
        [
            "--pretrained-weights",
            "baseline.pt",
            "--data",
            "visdrone.yaml",
            "--project",
            "runs",
            "--name",
            "method",
        ]
    )
    control = build_parser().parse_args(
        [
            "--pretrained-weights",
            "baseline.pt",
            "--data",
            "visdrone.yaml",
            "--project",
            "runs",
            "--name",
            "control",
            "--control",
        ]
    )

    assert trainer_class(method) is GCMVPLECTrainer
    assert trainer_class(control) is GCMVPLECControlTrainer


def test_control_preprocessing_drops_local_inputs_and_normalizes_global():
    trainer = object.__new__(GCMVPLECControlTrainer)
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(multi_scale=0.0)
    batch = {
        "img": torch.full((1, 3, 2, 2), 255, dtype=torch.uint8),
        "local_views": torch.zeros((1, 4, 3, 2, 2), dtype=torch.uint8),
        "source_shape": torch.tensor([[2, 2]]),
    }

    processed = trainer.preprocess_batch(batch)

    assert set(processed) == {"img"}
    assert processed["img"].dtype == torch.float32
    assert torch.equal(processed["img"], torch.ones_like(processed["img"]))
