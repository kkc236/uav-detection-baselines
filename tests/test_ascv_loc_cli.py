from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch

from src.ascv_loc import ASCVLocLossResult
import src.ascv_loc_cli as cli_module
from src.ascv_loc_cli import build_parser, build_settings, sha256_file, validate_protocol_inputs
from src.ascv_loc_diagnostics import ASCVMechanismAccumulator
from src.ascv_loc_diagnostics import validate_local_checkpoint_runtime
from src.ascv_loc_protocol import source_bundle_sha256
from src.ascv_loc_protocol import (
    FROZEN_CROP_CONTRACT,
    FROZEN_FORMAL_THRESHOLDS,
    FROZEN_MECHANISM_GATE,
    FROZEN_SCREEN_GATE,
    FROZEN_STATE_MACHINE,
)
from src.ascv_loc_stage import ASCVStage


def _args_and_manifest(
    tmp_path: Path,
    monkeypatch,
    stage: ASCVStage = ASCVStage.MECHANISM_500,
    *,
    seed: int = 0,
):
    initial_state = tmp_path / "initial.pt"
    initial_state.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1}, initial_state)
    subset = tmp_path / "subset.txt"
    subset.write_text("image.jpg\n")
    subset_data = tmp_path / "train_only.yaml"
    subset_data.write_text(f"train: {subset.as_posix()}\nval: {subset.as_posix()}\n")
    full_train = tmp_path / "images" / "train"
    full_train.mkdir(parents=True)
    full_data = tmp_path / "train_full_only.yaml"
    full_data.write_text(
        f"path: {tmp_path.as_posix()}\ntrain: {full_train.as_posix()}\nval: {full_train.as_posix()}\n"
    )
    manifest = {
        "schema_version": "ascv-loc-matched/v2",
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "environment": {"ultralytics": "8.4.90"},
        "initial_states": {
            str(seed): {"path": initial_state.as_posix(), "sha256": sha256_file(initial_state).upper()}
        },
        "subset": {
            "path": subset.as_posix(),
            "count": 1,
            "file_sha256": sha256_file(subset).upper(),
            "semantic_sha256": "SEMANTIC",
        },
        "dataset": {"root": tmp_path.as_posix(), "file_count": 14038, "sha256": "DATASET"},
        "parent_lineage": {
            str(seed): {
                "parent_protocol": (tmp_path / "parent.json").as_posix(),
                "parent_protocol_sha256": "PARENT",
            }
        },
        "source": {
            "repo_files": {"src/ascv_loc_cli.py": "SOURCE"},
            "repo_bundle_sha256": source_bundle_sha256({"src/ascv_loc_cli.py": "SOURCE"}),
            "upstream": cli_module.EXPECTED_UPSTREAM_SOURCE_SHA256,
        },
        "scientific_contract": {
            "state_machine": list(FROZEN_STATE_MACHINE),
            "crop": FROZEN_CROP_CONTRACT,
            "mechanism_gate": FROZEN_MECHANISM_GATE,
            "screen_gate": FROZEN_SCREEN_GATE,
            "formal_thresholds": FROZEN_FORMAL_THRESHOLDS,
        },
        "train_only_yaml": {"path": subset_data.as_posix(), "sha256": sha256_file(subset_data)},
        "full_train_only_yaml": {"path": full_data.as_posix(), "sha256": sha256_file(full_data)},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    predecessor_decisions = {
        ASCVStage.MECHANISM_500: "PREFLIGHT_GO",
        ASCVStage.SCREEN_10: "GO",
        ASCVStage.SEED0_100: "SCREEN_GO",
        ASCVStage.SEED1_100: "FORMAL_SEED0_GO",
        ASCVStage.SEED2_100: "FORMAL_SEED0_GO",
    }
    predecessor = tmp_path / "predecessor.json"
    predecessor.write_text(
        json.dumps(
            {
                "decision": predecessor_decisions.get(stage),
                "protocol_manifest_sha256": sha256_file(manifest_path),
                "protocol_source_commit": manifest["source_commit"],
            }
        )
    )
    data = subset_data if stage in {ASCVStage.PREFLIGHT_1, ASCVStage.MECHANISM_500, ASCVStage.SCREEN_10} else full_data
    args = build_parser().parse_args(
        [
            "--stage",
            stage.value,
            "--arm",
            "ascv",
            "--initial-state",
            str(initial_state),
            "--data",
            str(data),
            "--protocol-manifest",
            str(manifest_path),
            "--project",
            str(tmp_path / "runs"),
            "--name",
            "frozen",
            "--seed",
            str(seed),
        ]
        + (
            []
            if stage is ASCVStage.PREFLIGHT_1
            else ["--predecessor-evidence", str(predecessor)]
        )
    )
    monkeypatch.setattr(cli_module, "require_clean_repo", lambda _root: None)
    monkeypatch.setattr(cli_module, "repo_source_hashes", lambda _root: {"src/ascv_loc_cli.py": "SOURCE"})
    monkeypatch.setattr(cli_module, "validate_parent_attestation", lambda _manifest, _seed: None)
    monkeypatch.setattr(cli_module, "current_environment", lambda: cli_module.EXPECTED_ENVIRONMENT)
    monkeypatch.setattr(
        cli_module,
        "current_upstream_source_hashes",
        lambda: cli_module.EXPECTED_UPSTREAM_SOURCE_SHA256,
    )
    monkeypatch.setattr(cli_module, "EXPECTED_SUBSET_COUNT", 1)
    monkeypatch.setattr(cli_module, "EXPECTED_SUBSET_FILE_SHA256", sha256_file(subset))
    monkeypatch.setattr(cli_module, "EXPECTED_SUBSET_SHA256", "SEMANTIC")
    monkeypatch.setattr(cli_module, "EXPECTED_INITIAL_STATE_SHA256", {seed: sha256_file(initial_state)})
    monkeypatch.setattr(cli_module, "validate_initial_state_artifact", lambda _artifact, seed: None)
    monkeypatch.setattr(
        cli_module,
        "subset_signature",
        lambda _path, root: {"count": 1, "sha256": "SEMANTIC"},
    )
    monkeypatch.setattr(
        cli_module,
        "replay_preflight_gate",
        lambda gate: {
            "decision": gate["decision"],
            "protocol": {
                "manifest_sha256": sha256_file(manifest_path),
                "source_commit": manifest["source_commit"],
            },
        },
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
    assert "--batch" not in option_strings
    assert "--workers" not in option_strings
    assert "--amp" not in option_strings


def test_runtime_rejects_any_device_other_than_single_gpu_zero(tmp_path: Path, monkeypatch) -> None:
    args = _args_and_manifest(tmp_path, monkeypatch)
    args.device = "0,1"
    with pytest.raises(ValueError, match="single GPU device 0"):
        validate_protocol_inputs(args)


def test_non_preflight_stage_requires_valid_predecessor_evidence(tmp_path: Path, monkeypatch) -> None:
    args = _args_and_manifest(tmp_path, monkeypatch)
    args.predecessor_evidence = None
    with pytest.raises(ValueError, match="requires predecessor evidence"):
        validate_protocol_inputs(args)

    args = _args_and_manifest(tmp_path / "wrong", monkeypatch)
    predecessor = json.loads(args.predecessor_evidence.read_text())
    predecessor["decision"] = "ASCV_LOC_STOP"
    args.predecessor_evidence.write_text(json.dumps(predecessor))
    with pytest.raises(ValueError, match="does not authorize"):
        validate_protocol_inputs(args)


def test_mechanism_settings_are_train_only_and_fixed(tmp_path: Path, monkeypatch) -> None:
    args = _args_and_manifest(tmp_path, monkeypatch)
    validate_protocol_inputs(args)
    settings = build_settings(args)

    assert settings["epochs"] == 100
    assert settings["val"] is False
    assert settings["fraction"] == 1.0
    assert settings["pretrained"] is False
    assert settings["resume"] is False
    assert settings["save"] is True
    assert settings["save_period"] == -1
    assert settings["deterministic"] is True
    assert settings["batch"] == 8
    assert settings["workers"] == 8
    assert settings["amp"] is True
    assert settings["optimizer"] == "MuSGD"
    assert settings["lr0"] == 0.01
    assert settings["momentum"] == 0.937
    assert settings["warmup_bias_lr"] == 0.0
    assert settings["nbs"] == 64


def test_stage_rejects_wrong_data_and_test_dev(tmp_path: Path, monkeypatch) -> None:
    args = _args_and_manifest(tmp_path, monkeypatch)
    wrong = tmp_path / "wrong.yaml"
    wrong.write_text("train: wrong\nval: wrong\n")
    args.data = wrong
    with pytest.raises(ValueError, match="does not match"):
        validate_protocol_inputs(args)

    args.data = Path(tmp_path / "test-dev" / "data.yaml")
    with pytest.raises(ValueError, match="test-dev is forbidden"):
        validate_protocol_inputs(args)


def test_stage_rejects_source_commit_drift(tmp_path: Path, monkeypatch) -> None:
    args = _args_and_manifest(tmp_path, monkeypatch, ASCVStage.PREFLIGHT_1)
    manifest = json.loads(args.protocol_manifest.read_text())
    manifest["source_commit"] = "0" * 40
    args.protocol_manifest.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="source commit does not match"):
        validate_protocol_inputs(args)


def test_stage_rejects_scientific_contract_drift(tmp_path: Path, monkeypatch) -> None:
    args = _args_and_manifest(tmp_path, monkeypatch)
    manifest = json.loads(args.protocol_manifest.read_text())
    manifest["scientific_contract"]["formal_thresholds"]["AP-large-SBR"] = -1.0
    args.protocol_manifest.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="scientific contract"):
        validate_protocol_inputs(args)


def test_runtime_rehashes_subset_inputs(tmp_path: Path, monkeypatch) -> None:
    args = _args_and_manifest(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "subset_signature",
        lambda _path, root: {"count": 1, "sha256": "CHANGED"},
    )
    with pytest.raises(ValueError, match="subset semantic signature"):
        validate_protocol_inputs(args)


def test_runtime_rejects_repo_source_drift(tmp_path: Path, monkeypatch) -> None:
    args = _args_and_manifest(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "repo_source_hashes",
        lambda _root: {"src/ascv_loc_cli.py": "CHANGED"},
    )
    with pytest.raises(ValueError, match="source file checksum"):
        validate_protocol_inputs(args)


@pytest.mark.parametrize(
    ("stage", "seed", "allowed"),
    [
        (ASCVStage.PREFLIGHT_1, 0, True),
        (ASCVStage.PREFLIGHT_1, 1, False),
        (ASCVStage.MECHANISM_500, 0, True),
        (ASCVStage.MECHANISM_500, 1, False),
        (ASCVStage.SCREEN_10, 0, True),
        (ASCVStage.SCREEN_10, 1, True),
        (ASCVStage.SCREEN_10, 2, True),
        (ASCVStage.SEED0_100, 0, True),
        (ASCVStage.SEED0_100, 1, False),
        (ASCVStage.SEED1_100, 1, True),
        (ASCVStage.SEED1_100, 0, False),
        (ASCVStage.SEED2_100, 2, True),
        (ASCVStage.SEED2_100, 0, False),
    ],
)
def test_stage_seed_contract_is_fail_closed(
    tmp_path: Path, monkeypatch, stage: ASCVStage, seed: int, allowed: bool
) -> None:
    args = _args_and_manifest(tmp_path, monkeypatch, stage, seed=seed)
    if allowed:
        validate_protocol_inputs(args)
    else:
        with pytest.raises(ValueError, match="stage/seed mismatch"):
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


def test_mechanism_scientific_direction_gate_uses_frozen_tail_401_to_500() -> None:
    accumulator = ASCVMechanismAccumulator()
    for _ in range(400):
        accumulator.record(_result(-10.0, -10.0))
    for _ in range(100):
        accumulator.record(_result(0.1, 0.2))

    passed, failures = accumulator.mechanism_gate()
    summary = accumulator.summary()

    assert passed is True
    assert failures == []
    assert summary["tail_window"] == [401, 500]
    assert summary["tail"]["tiny_teacher_advantage_mean"] > 0
    assert summary["all"]["tiny_teacher_advantage_mean"] < 0


def test_mechanism_tail_requires_each_direction_in_at_least_80_batches_and_100_pairs() -> None:
    accumulator = ASCVMechanismAccumulator()
    for _ in range(400):
        accumulator.record(_result(0.1, 0.2))
    for index in range(100):
        result = _result(0.1, 0.2)
        if index >= 79:
            result = ASCVLocLossResult(
                loss=result.loss,
                pair_count=1,
                tiny_pair_count=0,
                non_tiny_pair_count=1,
                tiny_teacher_advantage_sum=torch.tensor(0.0),
                tiny_teacher_win_count=0,
                non_tiny_teacher_advantage_sum=result.non_tiny_teacher_advantage_sum,
                non_tiny_teacher_win_count=result.non_tiny_teacher_win_count,
            )
        accumulator.record(result)

    passed, failures = accumulator.mechanism_gate()

    assert passed is False
    assert "tail_tiny_batches_with_pairs=79, required>=80" in failures
    assert "tail_tiny_pairs=79, required>=100" in failures


def test_mechanism_accumulator_retains_only_detached_python_scalars() -> None:
    accumulator = ASCVMechanismAccumulator()
    loss = torch.tensor(1.0, requires_grad=True) * 2
    result = _result(0.1, 0.2)
    result = ASCVLocLossResult(
        loss=loss,
        pair_count=result.pair_count,
        tiny_pair_count=result.tiny_pair_count,
        non_tiny_pair_count=result.non_tiny_pair_count,
        tiny_teacher_advantage_sum=result.tiny_teacher_advantage_sum,
        tiny_teacher_win_count=result.tiny_teacher_win_count,
        non_tiny_teacher_advantage_sum=result.non_tiny_teacher_advantage_sum,
        non_tiny_teacher_win_count=result.non_tiny_teacher_win_count,
    )
    accumulator.record(result)

    assert all(
        isinstance(value, (int, float))
        for value in accumulator._results[0].values()
    )


def test_local_checkpoint_recompute_is_required_only_for_preflight() -> None:
    with pytest.raises(RuntimeError, match="CHECKPOINT_RECOMPUTE_INVALID"):
        validate_local_checkpoint_runtime(
            stage=ASCVStage.PREFLIGHT_1,
            calls=1,
            batchnorm_preserved=True,
            non_tiny_pair_count=0,
        )

    validate_local_checkpoint_runtime(
        stage=ASCVStage.MECHANISM_500,
        calls=1,
        batchnorm_preserved=True,
        non_tiny_pair_count=0,
    )
    validate_local_checkpoint_runtime(
        stage=ASCVStage.MECHANISM_500,
        calls=2,
        batchnorm_preserved=True,
        non_tiny_pair_count=1,
    )
    with pytest.raises(RuntimeError, match="CHECKPOINT_RECOMPUTE_INVALID"):
        validate_local_checkpoint_runtime(
            stage=ASCVStage.MECHANISM_500,
            calls=1,
            batchnorm_preserved=True,
            non_tiny_pair_count=1,
        )


def test_local_checkpoint_runtime_rejects_impossible_calls_and_bn_drift() -> None:
    with pytest.raises(RuntimeError, match="CHECKPOINT_RECOMPUTE_INVALID"):
        validate_local_checkpoint_runtime(
            stage=ASCVStage.MECHANISM_500,
            calls=0,
            batchnorm_preserved=True,
            non_tiny_pair_count=0,
        )
    with pytest.raises(RuntimeError, match="BATCHNORM"):
        validate_local_checkpoint_runtime(
            stage=ASCVStage.MECHANISM_500,
            calls=1,
            batchnorm_preserved=False,
            non_tiny_pair_count=0,
        )
