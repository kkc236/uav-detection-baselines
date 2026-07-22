import json
from hashlib import sha256
from pathlib import Path

import pytest

from scripts.train_rtdetr_ebc_qp import (
    CompactDiagnosticsWriter,
    build_ebc_config,
    build_parser,
    build_settings,
    validate_protocol,
)
from src.rtdetr_ebc_qp import resolve_protocol_optimizer


def test_d2_defaults_are_frozen_ten_epoch_ten_percent_scratch_settings():
    args = build_parser().parse_args(["--stage", "d2", "--arm", "a2", "--initial-state", "init.pt"])

    settings = build_settings(args)

    assert Path(settings["model"]).name == "rtdetr-l-ebc-qp.yaml"
    assert settings["epochs"] == 10
    assert settings["fraction"] == 1.0
    assert settings["pretrained"] is False
    assert settings["seed"] == 0
    assert settings["imgsz"] == 640
    assert settings["batch"] == 8
    assert settings["workers"] == 8
    assert settings["max_det"] == 300
    assert settings["optimizer"] == "auto"
    assert settings["lr0"] == 0.01
    assert settings["lrf"] == 0.01
    assert settings["momentum"] == 0.937
    assert settings["weight_decay"] == 0.0005
    assert settings["warmup_epochs"] == 3.0
    assert settings["warmup_momentum"] == 0.8
    assert settings["warmup_bias_lr"] == 0.0
    assert settings["nbs"] == 64
    assert settings["cos_lr"] is False
    assert settings["mosaic"] == 1.0
    assert settings["close_mosaic"] == 10
    assert settings["mixup"] == 0.0
    assert settings["scale"] == 0.5
    assert settings["translate"] == 0.1
    assert settings["degrees"] == 0.0
    assert settings["shear"] == 0.0
    assert settings["perspective"] == 0.0
    assert settings["flipud"] == 0.0
    assert settings["fliplr"] == 0.5
    assert settings["hsv_h"] == 0.015
    assert settings["hsv_s"] == 0.7
    assert settings["hsv_v"] == 0.4
    assert settings["cutmix"] == 0.0
    assert settings["copy_paste"] == 0.0
    assert settings["resume"] is False
    assert settings["exist_ok"] is False


def test_fixed_batch_and_workers_cannot_be_overridden_from_cli():
    options = {action.dest for action in build_parser()._actions}

    assert "batch" not in options
    assert "workers" not in options


def test_quality_weighted_ebc_is_an_explicit_opt_in():
    args = build_parser().parse_args(
        ["--stage", "d2", "--arm", "a2", "--initial-state", "init.pt", "--quality-weighted-ebc"]
    )

    assert args.quality_weighted_ebc is True


def test_fusion_gamma_is_explicit_and_mutually_exclusive_with_quality_ebc():
    args = build_parser().parse_args(
        ["--stage", "d2", "--arm", "a2", "--initial-state", "init.pt", "--learnable-fusion-gamma"]
    )
    assert args.learnable_fusion_gamma is True

    conflicting = build_parser().parse_args(
        [
            "--stage",
            "d2",
            "--arm",
            "a2",
            "--initial-state",
            "init.pt",
            "--quality-weighted-ebc",
            "--learnable-fusion-gamma",
        ]
    )
    with pytest.raises(SystemExit, match="mutually exclusive"):
        validate_protocol(conflicting)


def test_d2_a1_copies_gamma_a2_settings_and_only_disables_ebc():
    common = [
        "--stage",
        "d2",
        "--initial-state",
        "init.pt",
        "--name",
        "paired",
        "--learnable-fusion-gamma",
    ]
    a1 = build_parser().parse_args([*common, "--arm", "a1"])
    a2 = build_parser().parse_args([*common, "--arm", "a2"])

    validate_protocol(a1)
    validate_protocol(a2)
    assert build_settings(a1) == build_settings(a2)

    a1_config = build_ebc_config(a1)
    a2_config = build_ebc_config(a2)
    a1_values = a1_config.as_dict()
    a2_values = a2_config.as_dict()
    assert a1_values.pop("lambda_ebc") == 0.0
    assert a2_values.pop("lambda_ebc") == 0.05
    assert a1_values == a2_values
    assert a1_config.learnable_fusion_gamma


def test_d2_a1_rejects_quality_weighted_ebc():
    args = build_parser().parse_args(
        ["--stage", "d2", "--arm", "a1", "--initial-state", "init.pt", "--quality-weighted-ebc"]
    )

    with pytest.raises(SystemExit, match="only valid for the A2 arm"):
        validate_protocol(args)


def test_d2_a1_no_injection_only_changes_the_query_injection_flag():
    common = [
        "--stage",
        "d2",
        "--arm",
        "a1",
        "--initial-state",
        "init.pt",
        "--learnable-fusion-gamma",
    ]
    injected = build_parser().parse_args(common)
    isolated = build_parser().parse_args([*common, "--disable-query-injection"])

    validate_protocol(isolated)
    injected_values = build_ebc_config(injected).as_dict()
    isolated_values = build_ebc_config(isolated).as_dict()
    assert injected_values.pop("query_injection_enabled") is True
    assert isolated_values.pop("query_injection_enabled") is False
    assert injected_values == isolated_values


def test_no_injection_flag_is_restricted_to_d2_a1():
    args = build_parser().parse_args(
        ["--stage", "d2", "--arm", "a2", "--initial-state", "init.pt", "--disable-query-injection"]
    )

    with pytest.raises(SystemExit, match="only valid for the D2 A1 arm"):
        validate_protocol(args)


def test_qg_p2_is_a_frozen_d2_arm_with_zero_ebc_and_fusion_gamma():
    args = build_parser().parse_args(
        ["--stage", "d2", "--arm", "qg-p2", "--initial-state", "init.pt", "--name", "paired"]
    )

    validate_protocol(args)
    config = build_ebc_config(args)

    assert config.quality_gated_p2 is True
    assert config.lambda_ebc == 0.0
    assert config.learnable_fusion_gamma is True
    assert config.query_injection_enabled is True
    assert build_settings(args)["epochs"] == 10


def test_qg_p2_rejects_other_quality_or_injection_switches():
    quality_ebc = build_parser().parse_args(
        [
            "--stage",
            "d2",
            "--arm",
            "qg-p2",
            "--initial-state",
            "init.pt",
            "--quality-weighted-ebc",
        ]
    )
    no_injection = build_parser().parse_args(
        [
            "--stage",
            "d2",
            "--arm",
            "qg-p2",
            "--initial-state",
            "init.pt",
            "--disable-query-injection",
        ]
    )

    with pytest.raises(SystemExit, match="only valid for the A2 arm"):
        validate_protocol(quality_ebc)
    with pytest.raises(SystemExit, match="only valid for the D2 A1 arm"):
        validate_protocol(no_injection)


def test_auto_optimizer_is_locked_to_musgd_without_rewriting_lr_or_momentum():
    assert resolve_protocol_optimizer("auto", lr=0.01, momentum=0.937) == ("MuSGD", 0.01, 0.937)
    assert resolve_protocol_optimizer("SGD", lr=0.02, momentum=0.8) == ("SGD", 0.02, 0.8)


def test_d2_control_uses_stock_yaml_and_the_same_initial_state():
    args = build_parser().parse_args(["--stage", "d2", "--arm", "control", "--initial-state", "init.pt"])

    settings = build_settings(args)

    assert settings["model"] == "rtdetr-l.yaml"
    assert settings["epochs"] == 10
    assert settings["fraction"] == 1.0
    assert settings["seed"] == 0


def test_d3_forces_zero_ebc_and_requires_passing_d2_manifest(tmp_path: Path):
    manifest = tmp_path / "d2.json"
    manifest.write_text(json.dumps({"gate": {"passed": False}}), encoding="utf-8")
    args = build_parser().parse_args(["--stage", "d3", "--d2-manifest", str(manifest)])

    with pytest.raises(SystemExit, match="passing D2 manifest"):
        validate_protocol(args)


def test_formal_a1_requires_completed_a2_seed0_and_exact_initial_state(tmp_path: Path):
    initial_state = tmp_path / "initial.pt"
    initial_state.write_bytes(b"frozen-state")
    manifest = tmp_path / "a2-seed0.json"
    manifest.write_text(
        json.dumps(
            {
                "formal_a2_seed0": {
                    "complete": True,
                    "initial_state": str(initial_state.resolve()),
                    "initial_state_sha256": sha256(initial_state.read_bytes()).hexdigest().upper(),
                }
            }
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--stage",
            "formal",
            "--arm",
            "a1",
            "--initial-state",
            str(initial_state),
            "--a2-manifest",
            str(manifest),
        ]
    )

    validate_protocol(args)
    initial_state.write_bytes(b"changed")
    with pytest.raises(SystemExit, match="exact A2 seed-0 initial state"):
        validate_protocol(args)


def test_seed_one_rejects_changed_frozen_signature(tmp_path: Path):
    frozen = tmp_path / "frozen.json"
    current = tmp_path / "current.json"
    frozen.write_text(json.dumps({"signature": {"git": "abc", "dataset": "same"}}), encoding="utf-8")
    current.write_text(json.dumps({"git": "changed", "dataset": "same"}), encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--stage",
            "d2",
            "--arm",
            "a2",
            "--initial-state",
            "init.pt",
            "--seed",
            "1",
            "--frozen-manifest",
            str(frozen),
            "--signature-file",
            str(current),
        ]
    )

    with pytest.raises(SystemExit, match="frozen experiment signature"):
        validate_protocol(args)


def test_compact_diagnostics_writer_rejects_candidate_maps(tmp_path: Path):
    path = tmp_path / "diagnostics.jsonl"
    writer = CompactDiagnosticsWriter(path)

    writer.append({"epoch": 1, "ap_tiny": 0.1, "p2_entry_count": 4})
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record == {"epoch": 1, "ap_tiny": 0.1, "p2_entry_count": 4}

    with pytest.raises(ValueError, match="unsupported diagnostic fields"):
        writer.append({"epoch": 2, "p2_map": [[1, 2]]})
