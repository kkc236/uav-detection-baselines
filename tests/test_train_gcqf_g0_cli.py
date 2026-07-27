import torch
import pytest

from scripts.train_gcqf_g0 import (
    MODULE_ARTIFACT_SCHEMA,
    build_module_artifact,
    build_parser,
    schedule_values,
    validate_training_protocol,
)
from src.gcqf import GCQF


def test_train_cli_freezes_seed0_only_screen_defaults():
    args = build_parser().parse_args(
        [
            "--train-cache",
            "train/manifest.json",
            "--output",
            "run",
            "--seed",
            "0",
            "--source-commit",
            "A" * 40,
        ]
    )

    assert args.epochs == 10
    assert args.batch == 8
    assert args.amp_scale == 128.0
    assert args.optimizer == "MuSGD"
    assert args.device == "0"
    assert args.source_commit == "A" * 40
    validate_training_protocol(args)


def test_train_cli_rejects_seed1_and_protocol_drift():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--train-cache",
                "train/manifest.json",
                "--output",
                "run",
                "--seed",
                "1",
                "--source-commit",
                "A" * 40,
            ]
        )
    args = build_parser().parse_args(
        [
            "--train-cache",
            "train/manifest.json",
            "--output",
            "run",
            "--seed",
            "0",
            "--epochs",
            "9",
            "--source-commit",
            "A" * 40,
        ]
    )
    with pytest.raises(ValueError, match="protocol drift"):
        validate_training_protocol(args)


def test_schedule_reaches_frozen_warmup_and_final_values():
    start_lr, start_momentum = schedule_values(
        step=0,
        total_steps=100,
        warmup_steps=30,
    )
    warm_lr, warm_momentum = schedule_values(
        step=30,
        total_steps=100,
        warmup_steps=30,
    )
    final_lr, final_momentum = schedule_values(
        step=99,
        total_steps=100,
        warmup_steps=30,
    )

    assert start_lr == 0.0
    assert start_momentum == 0.8
    assert warm_lr == 0.01
    assert warm_momentum == 0.937
    assert abs(final_lr - 0.0001) < 1e-12
    assert final_momentum == 0.937


def test_artifact_is_module_only_and_cpu_backed():
    module = GCQF(
        query_dim=8,
        num_classes=3,
        num_heads=2,
        num_views=4,
    )

    artifact = build_module_artifact(
        module,
        seed=0,
        epoch=3,
        train_cache_sha256="A" * 64,
        source_commit="B" * 40,
        train_image_ids=("a.jpg",),
        calibration_image_ids=("b.jpg",),
        positive_weights={"tiny": 2.0, "risk": 3.0, "retain": 4.0},
    )

    assert artifact["schema_version"] == MODULE_ARTIFACT_SCHEMA
    assert set(artifact["module_state"]) == set(module.state_dict())
    assert all(
        value.device.type == "cpu"
        for value in artifact["module_state"].values()
    )
    assert artifact["source_commit"] == "b" * 40
    assert artifact["train_image_ids"] == ["a.jpg"]
    assert artifact["calibration_image_ids"] == ["b.jpg"]
    assert all(
        isinstance(value, torch.Tensor)
        for value in artifact["module_state"].values()
    )
