from __future__ import annotations

from pathlib import Path

import pytest

from scripts.train_rtdetr_sqda_geometry_gate import RUN_NAMES, build_parser, build_settings


def test_smgt_runs_use_a_fresh_namespace() -> None:
    assert RUN_NAMES == {
        "g1": "sqda-geometry-smgt-g1-seed0-3ep",
        "g2": "sqda-geometry-smgt-g2-seed0-10ep",
        "g2r1": "sqda-geometry-smgt-g2r1-seed0-10ep",
        "formal": "sqda-geometry-smgt-formal-seed0-100ep",
    }


def test_smgt_runner_requires_the_inventory_g2_feasibility_checkpoint() -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_sqda_geometry_gate_server.sh"
    ).read_text(encoding="utf-8")

    assert "sqda-geometry-smgt-" in runner
    assert "candidate-inventory.json" in runner
    assert "g2_eligible_checkpoint" in runner
    assert "smgt-${gate}-github-sync-status.json" in runner
    assert "sqda-smgt-${gate}-live" in runner


def test_smgt_runner_requires_strict_g2_selection_before_formal_training() -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_sqda_geometry_gate_server.sh"
    ).read_text(encoding="utf-8")

    formal_guard = runner.split('if [[ "$gate" == "formal" ]]')[1].split(
        "if pgrep"
    )[0]
    assert (
        'g2_inventory="$project/sqda-geometry-smgt-g2r1-seed0-10ep/'
        'evaluation-inventory/candidate-inventory.json"'
    ) in formal_guard
    assert 'get("selected_checkpoint")' in formal_guard
    assert "g2_eligible_checkpoint" not in formal_guard
    assert "smgt-${gate}-github-sync-status.json" in runner
    assert "sqda-smgt-${gate}-live" in runner


def test_smgt_g2_sync_retains_every_epoch_snapshot_for_inventory() -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_sqda_geometry_gate_server.sh"
    ).read_text(encoding="utf-8")

    assert (
        'g2) run_name="sqda-geometry-smgt-g2-seed0-10ep"; sync_retain=10 ;;'
    ) in runner
    assert '--retain "$sync_retain"' in runner
    assert (
        'g2r1) run_name="sqda-geometry-smgt-g2r1-seed0-10ep"; sync_retain=10 ;;'
    ) in runner


@pytest.mark.parametrize(
    ("gate", "epochs"), [("g1", 3), ("g2", 10), ("g2r1", 10), ("formal", 100)]
)
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
