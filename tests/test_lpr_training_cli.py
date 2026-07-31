from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.train_rtdetr_lpr import (
    build_parser,
    build_settings,
    capture_lpr_epoch_state,
    validate_launch_authority,
    write_lpr_diagnostics,
)
from src.lpr_head import LocalizationPriorRefiner
from src.lpr_protocol import EXPECTED_DATASET_SHA256, EXPECTED_ENVIRONMENT, EXPECTED_SUBSET_SHA256


def _manifest(tmp_path, *, seed=0):
    initial_state = tmp_path / f"initial-state-seed{seed}.pt"
    initial_state.touch()
    return {
        "seed": seed,
        "dataset": {"file_count": 14038, "sha256": EXPECTED_DATASET_SHA256},
        "subset": {"count": 647, "fraction": 0.10, "sha256": EXPECTED_SUBSET_SHA256},
        "data": {
            "screen": {"path": str(tmp_path / "VisDrone-screen.yaml")},
            "formal": {"path": str(tmp_path / "VisDrone-full.yaml")},
        },
        "initial_state": {"path": str(initial_state)},
    }


def _args(tmp_path, *extra):
    manifest_path = tmp_path / "protocol.json"
    manifest = _manifest(tmp_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return build_parser().parse_args(
        [
            "--variant",
            "lpr",
            "--stage",
            "screen",
            "--seed",
            "0",
            "--protocol-manifest",
            str(manifest_path),
            "--initial-state",
            manifest["initial_state"]["path"],
            "--project",
            str(tmp_path / "runs"),
            *extra,
        ]
    ), manifest


def test_screen_settings_match_strict_paired_protocol(tmp_path) -> None:
    args, manifest = _args(tmp_path)

    settings = build_settings(args, manifest)

    assert settings["data"] == manifest["data"]["screen"]["path"]
    assert settings["epochs"] == 10
    assert settings["pretrained"] is False
    assert settings["seed"] == 0
    assert settings["deterministic"] is True
    assert settings["batch"] == 8
    assert settings["workers"] == 8
    assert settings["imgsz"] == 640
    assert settings["optimizer"] == "MuSGD"
    assert settings["lr0"] == 0.01
    assert settings["momentum"] == 0.937
    assert settings["warmup_bias_lr"] == 0.0
    assert settings["mosaic"] == 1.0
    assert settings["close_mosaic"] == 10
    assert settings["fraction"] == 1.0


def test_formal_is_fresh_100_epoch_full_data(tmp_path) -> None:
    args, manifest = _args(tmp_path)
    args.stage = "formal"

    settings = build_settings(args, manifest)

    assert settings["epochs"] == 100
    assert settings["data"] == manifest["data"]["formal"]["path"]
    assert "resume" not in settings


@pytest.mark.parametrize(
    "arguments",
    (
        ["--batch", "4"],
        ["--imgsz", "800"],
        ["--optimizer", "AdamW"],
        ["--mosaic", "0"],
        ["--amp", "false"],
        ["--workers", "2"],
        ["--warmup-bias-lr", "0.1"],
    ),
)
def test_parser_rejects_scientific_protocol_drift(arguments) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)


def test_launch_authority_rejects_environment_dataset_or_seed_drift(tmp_path) -> None:
    args, manifest = _args(tmp_path)
    actual_environment = dict(EXPECTED_ENVIRONMENT)
    actual_environment["driver"] = "595.84"

    with pytest.raises(ValueError, match="environment"):
        validate_launch_authority(args, manifest, actual_environment, manifest["dataset"])

    actual_environment = dict(EXPECTED_ENVIRONMENT)
    with pytest.raises(ValueError, match="dataset"):
        validate_launch_authority(
            args,
            manifest,
            actual_environment,
            {"file_count": 14038, "sha256": "BAD"},
        )

    manifest["seed"] = 1
    with pytest.raises(ValueError, match="seed"):
        validate_launch_authority(args, manifest, actual_environment, manifest["dataset"])


def test_help_renders_paired_options() -> None:
    help_text = build_parser().format_help()

    assert "--variant" in help_text
    assert "--stage" in help_text
    assert "--protocol-manifest" in help_text


def test_epoch_diagnostics_write_one_complete_jsonl_record(tmp_path) -> None:
    refiner = LocalizationPriorRefiner(hidden_dim=8, seed=3407)
    refiner.alpha.data.fill_(0.2)
    refiner.last_residual_mean = torch.tensor(0.12)
    refiner.last_residual_max = torch.tensor(0.45)
    model = nn.Sequential(refiner)
    trainer = SimpleNamespace(
        model=model,
        epoch=0,
        batch_size=8,
        amp=True,
        save_dir=tmp_path,
        validator=SimpleNamespace(metrics=SimpleNamespace(box=SimpleNamespace(map75=0.031))),
        _lpr_grad_sq=0.25,
    )

    capture_lpr_epoch_state(trainer)
    write_lpr_diagnostics(trainer)

    record = json.loads((tmp_path / "lpr_diagnostics.jsonl").read_text(encoding="utf-8"))
    assert record["epoch"] == 1
    assert record["map75"] == pytest.approx(0.031)
    assert len(record["gates"]) == 1
    assert record["residual_mean"] == pytest.approx(0.12)
    assert record["residual_max"] == pytest.approx(0.45)
    assert record["lpr_grad_norm"] == pytest.approx(0.5)
    assert "cuda_peak_mib" in record
