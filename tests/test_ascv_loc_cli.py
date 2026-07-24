from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch

from src.ascv_loc import ASCVLocLossResult
from src.ascv_loc_cli import build_parser, build_settings, sha256_file, validate_protocol_inputs
from src.ascv_loc_diagnostics import ASCVMechanismAccumulator
from src.ascv_loc_stage import ASCVStage


def _args_and_manifest(tmp_path: Path, stage: ASCVStage = ASCVStage.MECHANISM_500):
    checkpoint = tmp_path / "mature.pt"
    checkpoint.write_bytes(b"checkpoint")
    subset_data = tmp_path / "train_only.yaml"
    subset_data.write_text("train: subset.txt\nval: subset.txt\n")
    full_data = tmp_path / "train_full_only.yaml"
    full_data.write_text("train: full.txt\nval: full.txt\n")
    manifest = {
        "schema_version": "ascv-loc-protocol/v1",
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "ultralytics_version": "8.4.90",
        "checkpoint": {"path": checkpoint.as_posix(), "sha256": sha256_file(checkpoint)},
        "train_only_yaml": {"path": subset_data.as_posix(), "sha256": sha256_file(subset_data)},
        "full_train_only_yaml": {"path": full_data.as_posix(), "sha256": sha256_file(full_data)},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    data = subset_data if stage in {ASCVStage.MECHANISM_500, ASCVStage.SCREEN_6} else full_data
    model = "rtdetr-l.yaml" if stage.name.startswith("SEED") else str(checkpoint)
    args = build_parser().parse_args(
        [
            "--stage",
            stage.value,
            "--model",
            model,
            "--data",
            str(data),
            "--protocol-manifest",
            str(manifest_path),
            "--project",
            str(tmp_path / "runs"),
            "--name",
            "frozen",
            "--batch",
            "4",
        ]
    )
    return args


def test_cli_exposes_no_scientific_tuning_switches() -> None:
    option_strings = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }

    assert "--lambda" not in option_strings
    assert "--tile-ratio" not in option_strings
    assert "--tiny-boundary" not in option_strings
    assert "--warmup" not in option_strings
    assert "--direction" not in option_strings


def test_mechanism_settings_are_train_only_and_fixed(tmp_path: Path) -> None:
    args = _args_and_manifest(tmp_path)
    validate_protocol_inputs(args)
    settings = build_settings(args)

    assert settings["epochs"] == 100
    assert settings["val"] is False
    assert settings["fraction"] == 1.0
    assert settings["pretrained"] is False
    assert settings["resume"] is False
    assert settings["deterministic"] is True
    assert settings["optimizer"] == "AdamW"
    assert settings["lr0"] == 0.000714
    assert settings["nbs"] == 64


def test_stage_rejects_wrong_data_and_test_dev(tmp_path: Path) -> None:
    args = _args_and_manifest(tmp_path)
    wrong = tmp_path / "wrong.yaml"
    wrong.write_text("train: wrong\nval: wrong\n")
    args.data = wrong
    with pytest.raises(ValueError, match="does not match"):
        validate_protocol_inputs(args)

    args.data = Path(tmp_path / "test-dev" / "data.yaml")
    with pytest.raises(ValueError, match="test-dev is forbidden"):
        validate_protocol_inputs(args)


def test_stage_rejects_source_commit_drift(tmp_path: Path) -> None:
    args = _args_and_manifest(tmp_path)
    manifest = json.loads(args.protocol_manifest.read_text())
    manifest["source_commit"] = "0" * 40
    args.protocol_manifest.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="source commit does not match"):
        validate_protocol_inputs(args)


def _result(tiny_advantage: float, non_tiny_advantage: float) -> ASCVLocLossResult:
    return ASCVLocLossResult(
        loss=torch.tensor(1.0),
        pair_count=2,
        tiny_pair_count=1,
        non_tiny_pair_count=1,
        tiny_teacher_advantage_sum=torch.tensor(tiny_advantage),
        tiny_teacher_win_count=int(tiny_advantage > 0),
        non_tiny_teacher_advantage_sum=torch.tensor(non_tiny_advantage),
        non_tiny_teacher_win_count=int(non_tiny_advantage > 0),
    )


def test_mechanism_gate_requires_exact_500_and_teacher_advantage() -> None:
    accumulator = ASCVMechanismAccumulator()
    for _ in range(500):
        accumulator.record(_result(0.1, 0.2))

    passed, failures = accumulator.mechanism_gate()

    assert passed is True
    assert failures == []


def test_mechanism_gate_stops_when_teacher_direction_is_not_supported() -> None:
    accumulator = ASCVMechanismAccumulator()
    for _ in range(500):
        accumulator.record(_result(-0.1, 0.2))

    passed, failures = accumulator.mechanism_gate()

    assert passed is False
    assert "tiny_teacher_advantage_mean<=0" in failures
    assert "tiny_teacher_win_rate<=0.5" in failures
