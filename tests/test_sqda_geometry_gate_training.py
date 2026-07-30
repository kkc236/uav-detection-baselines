from __future__ import annotations

from pathlib import Path

import pytest

from scripts.train_rtdetr_sqda_geometry_gate import RUN_NAMES, build_parser, build_settings


def test_smgt_runs_use_a_fresh_namespace() -> None:
    assert RUN_NAMES == {
        "g1": "sqda-geometry-smgt-g1-seed0-3ep",
        "g2": "sqda-geometry-smgt-g2-seed0-10ep",
    }


def test_smgt_runner_requires_the_inventory_selected_checkpoint_for_g2() -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_sqda_geometry_gate_server.sh"
    ).read_text(encoding="utf-8")

    assert "sqda-geometry-smgt-" in runner
    assert "candidate-inventory.json" in runner
    assert "selected_checkpoint" in runner


@pytest.mark.parametrize(("gate", "epochs"), [("g1", 3), ("g2", 10)])
def test_geometry_gate_settings_are_fixed_and_require_inherited_adapter(
    tmp_path,
    gate: str,
    epochs: int,
) -> None:
    args = build_parser().parse_args(
        [
            "--gate",
            gate,
            "--checkpoint",
            str(tmp_path / "baseline.pt"),
            "--adapter-checkpoint",
            str(tmp_path / "g2.pt"),
            "--data",
            str(tmp_path / "VisDrone.yaml"),
            "--project",
            str(tmp_path / "runs"),
        ]
    )

    settings = build_settings(args)

    assert settings["epochs"] == epochs
    assert settings["name"] == RUN_NAMES[gate]
    assert settings["imgsz"] == 640
    assert settings["batch"] == 8
    assert settings["workers"] == 8
    assert settings["seed"] == 0
    assert settings["deterministic"] is True
    assert settings["amp"] is True
    assert settings["nms"] is False
    assert settings["max_det"] == 300
    assert settings["optimizer"] == "AdamW"
    assert settings["lr0"] == pytest.approx(1e-4)
    assert settings["weight_decay"] == pytest.approx(1e-4)
    assert settings["freeze"] == list(range(29))


def test_geometry_gate_cli_excludes_training_protocol_mutations() -> None:
    options = {action.dest for action in build_parser()._actions}
    assert not {
        "epochs",
        "seed",
        "imgsz",
        "batch",
        "optimizer",
        "lr0",
        "amp",
        "max_det",
    }.intersection(options)


def test_post_g1_evaluation_cli_is_fixed_to_the_frozen_protocol() -> None:
    from scripts.evaluate_sqda_geometry_gate import build_parser

    options = {action.dest for action in build_parser()._actions}
    assert {
        "checkpoint",
        "candidate_checkpoint",
        "diagnosis",
        "data",
        "images",
        "labels",
        "output",
    }.issubset(options)
    assert not {"epochs", "optimizer", "lr0", "mode", "threshold"}.intersection(options)


def test_post_g1_evaluation_cli_accepts_an_updated_checkpoint_inventory(tmp_path) -> None:
    from scripts.evaluate_sqda_geometry_gate import build_parser

    args = build_parser().parse_args(
        [
            "--checkpoint",
            str(tmp_path / "baseline.pt"),
            "--weights-dir",
            str(tmp_path / "weights"),
            "--diagnosis",
            str(tmp_path / "diagnosis.json"),
            "--data",
            str(tmp_path / "VisDrone.yaml"),
            "--images",
            str(tmp_path / "images"),
            "--labels",
            str(tmp_path / "labels"),
            "--output",
            str(tmp_path / "evaluation"),
        ]
    )

    assert args.candidate_checkpoint is None
    assert args.weights_dir == tmp_path / "weights"
