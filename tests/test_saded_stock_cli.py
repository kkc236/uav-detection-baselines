from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch

import src.saded_stock_cli as cli
from src.tascv_protocol import FROZEN_TRAINING_CONTRACT


def _args(tmp_path: Path) -> Namespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = tmp_path / "train_only.yaml"
    train = (
        Path(cli.EXPECTED_DATASET_ROOT).resolve() / "images" / "train"
    ).as_posix()
    data.write_text(
        f"path: {Path(cli.EXPECTED_DATASET_ROOT).resolve().as_posix()}\n"
        f"train: {train}\n"
        f"val: {train}\n"
        "names:\n"
        + "".join(
            f"  {index}: {name}\n"
            for index, name in cli.EXPECTED_NAMES.items()
        )
    )
    initial = tmp_path / "initial-state-seed0.pt"
    torch.save({"format_version": 1}, initial)
    project = tmp_path / "runs"
    manifest = {
        "schema_version": cli.PROTOCOL_SCHEMA,
        "run_id": "final-saded-fresh100-deadbeef",
        "stage": "FORMAL_100",
        "arm": "stock_control",
        "fresh_start": True,
        "predecessor_required": False,
        "checkpoint_reuse": "forbidden",
        "environment": cli.EXPECTED_ENVIRONMENT,
        "training_contract": FROZEN_TRAINING_CONTRACT,
        "dataset": {
            "root": cli.EXPECTED_DATASET_ROOT,
            "sha256": cli.EXPECTED_DATASET_SHA256,
            "file_count": cli.EXPECTED_DATASET_FILE_COUNT,
            "train_images": 6471,
            "val_images": 548,
            "classes": 10,
        },
        "runtime_source": {
            "commit": "a" * 40,
            "repo_files": {"src/saded_stock_cli.py": "SOURCE"},
            "repo_bundle_sha256": cli.source_bundle_sha256(
                {"src/saded_stock_cli.py": "SOURCE"}
            ),
            "upstream": cli.EXPECTED_UPSTREAM_SOURCE_SHA256,
            "upstream_bundle_sha256": cli.source_bundle_sha256(
                cli.EXPECTED_UPSTREAM_SOURCE_SHA256
            ),
        },
        "initial_state": {
            "path": initial.resolve().as_posix(),
            "sha256": cli.sha256_file(initial),
            "common_fingerprint": cli.EXPECTED_COMMON_FINGERPRINTS[0],
        },
        "data": {
            "path": data.resolve().as_posix(),
            "sha256": cli.sha256_file(data),
        },
        "endpoint": {
            "project": project.resolve().as_posix(),
            "name": "seed0",
            "target_dir": (project / "seed0").resolve().as_posix(),
        },
    }
    manifest_path = tmp_path / "protocol_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return Namespace(
        protocol_manifest=manifest_path,
        initial_state=initial,
        data=data,
        project=project,
        name="seed0",
        device="0",
        seed=0,
    )


def _patch_authorities(monkeypatch, args: Namespace) -> None:
    monkeypatch.setattr(cli, "require_clean_repo", lambda _root: None)
    monkeypatch.setattr(
        cli, "current_environment", lambda: cli.EXPECTED_ENVIRONMENT
    )
    monkeypatch.setattr(
        cli,
        "current_upstream_source_hashes",
        lambda: cli.EXPECTED_UPSTREAM_SOURCE_SHA256,
    )
    monkeypatch.setattr(
        cli,
        "source_closure",
        lambda _root: {"src/saded_stock_cli.py": "SOURCE"},
    )
    monkeypatch.setattr(cli, "current_commit", lambda _root: "a" * 40)
    monkeypatch.setattr(
        cli,
        "EXPECTED_INITIAL_STATE_SHA256",
        {0: cli.sha256_file(args.initial_state)},
    )
    monkeypatch.setattr(
        cli, "validate_initial_state_artifact", lambda _artifact, seed: None
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 0})(),
    )


def test_cli_exposes_no_scientific_tuning_switches() -> None:
    options = {
        option
        for action in cli.build_parser()._actions
        for option in action.option_strings
    }
    forbidden = {
        "--epochs",
        "--batch",
        "--imgsz",
        "--optimizer",
        "--lr0",
        "--momentum",
        "--amp-scale",
        "--workers",
        "--resume",
        "--pretrained",
        "--val",
        "--test",
    }
    assert not options.intersection(forbidden)


def test_settings_exactly_match_frozen_formal_baseline(
    tmp_path: Path,
) -> None:
    settings = cli.build_settings(_args(tmp_path))
    assert settings["epochs"] == 100
    assert settings["imgsz"] == 640
    assert settings["batch"] == 8
    assert settings["workers"] == 8
    assert settings["optimizer"] == "MuSGD"
    assert settings["lr0"] == 0.01
    assert settings["lrf"] == 0.01
    assert settings["momentum"] == 0.937
    assert settings["pretrained"] is False
    assert settings["resume"] is False
    assert settings["amp"] is True
    assert settings["val"] is False
    assert settings["nms"] is False
    assert settings["max_det"] == 300


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("device", "0,1", "single GPU device 0"),
        ("seed", 1, "seed 0"),
        ("name", "test-dev", "test-dev is forbidden"),
    ],
)
def test_validation_rejects_scope_drift(
    tmp_path: Path,
    monkeypatch,
    field: str,
    value: object,
    message: str,
) -> None:
    args = _args(tmp_path)
    _patch_authorities(monkeypatch, args)
    setattr(args, field, value)
    with pytest.raises(ValueError, match=message):
        cli.validate_protocol_inputs(args)


def test_validation_accepts_exact_fresh_stock_endpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = _args(tmp_path)
    _patch_authorities(monkeypatch, args)
    protocol = cli.validate_protocol_inputs(args)
    assert protocol["stage"] == "FORMAL_100"
    assert protocol["checkpoint_reuse"] == "forbidden"


def test_validation_rejects_existing_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = _args(tmp_path)
    _patch_authorities(monkeypatch, args)
    (args.project / args.name).mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        cli.validate_protocol_inputs(args)


def test_completed_endpoint_validation_allows_bound_existing_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = _args(tmp_path)
    _patch_authorities(monkeypatch, args)
    (args.project / args.name).mkdir(parents=True)
    protocol = cli.validate_protocol_inputs(
        args,
        repo_root=Path(__file__).resolve().parents[1],
        require_fresh_target=False,
    )
    assert protocol["arm"] == "stock_control"


def test_validation_rejects_source_or_data_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = _args(tmp_path)
    _patch_authorities(monkeypatch, args)
    manifest = json.loads(args.protocol_manifest.read_text())
    manifest["runtime_source"]["commit"] = "b" * 40
    args.protocol_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source closure drift"):
        cli.validate_protocol_inputs(args)

    args = _args(tmp_path / "data-drift")
    _patch_authorities(monkeypatch, args)
    args.data.write_text("changed")
    with pytest.raises(ValueError, match="data binding drift"):
        cli.validate_protocol_inputs(args)


def test_train_only_yaml_rejects_test_key_and_class_mapping_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = _args(tmp_path)
    _patch_authorities(monkeypatch, args)
    payload = args.data.read_text()
    args.data.write_text(payload + "test: images/test\n")
    manifest = json.loads(args.protocol_manifest.read_text())
    manifest["data"]["sha256"] = cli.sha256_file(args.data)
    args.protocol_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="must be a mapping"):
        cli.validate_protocol_inputs(args)

    args = _args(tmp_path / "names")
    _patch_authorities(monkeypatch, args)
    args.data.write_text(args.data.read_text().replace("0: pedestrian", "0: person"))
    manifest = json.loads(args.protocol_manifest.read_text())
    manifest["data"]["sha256"] = cli.sha256_file(args.data)
    args.protocol_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="class mapping drift"):
        cli.validate_protocol_inputs(args)
