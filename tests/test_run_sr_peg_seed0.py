import json
from pathlib import Path

import pytest

from scripts.run_sr_peg_seed0 import (
    EXPECTED_RUNTIME,
    STAGES,
    build_parser,
    build_stage_commands,
    finalize_pipeline,
    validate_runtime_environment,
)


def _args(tmp_path: Path):
    return build_parser().parse_args(
        [
            "--source",
            str(tmp_path / "source-abcdef12"),
            "--source-commit",
            "abcdef12" + "0" * 32,
            "--output",
            str(tmp_path / "output"),
            "--checkpoint",
            str(tmp_path / "baseline.pt"),
            "--data",
            str(tmp_path / "visdrone.yaml"),
            "--train-images",
            str(tmp_path / "train10.txt"),
            "--val-cache",
            str(tmp_path / "val-cache" / "manifest.json"),
            "--anchor-reference",
            str(tmp_path / "anchor-reference.json"),
            "--seed",
            "0",
        ]
    )


def test_runner_has_only_four_seed0_stages_and_rejects_other_seeds(tmp_path):
    assert STAGES == ("TRAIN_CACHE", "TRAIN_SEED0", "CALIBRATE", "EVALUATE")
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--source",
                "source",
                "--source-commit",
                "a" * 40,
                "--output",
                "output",
                "--checkpoint",
                "baseline.pt",
                "--data",
                "data.yaml",
                "--train-images",
                "train.txt",
                "--val-cache",
                "val/manifest.json",
                "--anchor-reference",
                "anchor.json",
                "--seed",
                "1",
            ]
        )


def test_runtime_preflight_validates_every_frozen_environment_field():
    validate_runtime_environment(dict(EXPECTED_RUNTIME))
    for key in EXPECTED_RUNTIME:
        drifted = dict(EXPECTED_RUNTIME)
        drifted[key] = "wrong"
        with pytest.raises(ValueError, match=key):
            validate_runtime_environment(drifted)


def test_stage_commands_never_regenerate_val_or_request_extra_seeds(tmp_path):
    args = _args(tmp_path)
    commands = build_stage_commands(args, python="/venv/python")
    joined = "\n".join(" ".join(command) for command in commands.values())

    assert set(commands) == set(STAGES)
    assert "--seed 0" in joined
    assert "--seed 1" not in joined
    assert "--seed 2" not in joined
    assert "cache_gcqf_evidence" in " ".join(commands["TRAIN_CACHE"])
    assert str(args.val_cache) not in " ".join(commands["TRAIN_CACHE"])
    assert str(args.val_cache) in " ".join(commands["EVALUATE"])
    assert not any(
        token in joined.lower()
        for token in ("shutdown", "reboot", "poweroff", "kill")
    )


def test_pipeline_complete_is_written_only_for_passing_evaluation(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    evaluation = output / "seed0-evaluation.json"
    evaluation.write_text(
        json.dumps({"per_seed_gate": {"passed": False}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="hard gate"):
        finalize_pipeline(output, evaluation)
    assert not (output / "PIPELINE_COMPLETE").exists()

    evaluation.write_text(
        json.dumps({"per_seed_gate": {"passed": True}}),
        encoding="utf-8",
    )
    finalize_pipeline(output, evaluation)
    assert (output / "PIPELINE_COMPLETE").is_file()
