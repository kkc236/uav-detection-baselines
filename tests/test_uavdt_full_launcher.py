from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_uavdt_full.py"


def _load_module():
    assert SCRIPT.is_file(), "UAVDT Full launcher has not been implemented"
    return importlib.import_module("scripts.train_uavdt_full")


def test_help_exposes_only_full_cross_server_inputs() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for allowed in (
        "--data-yaml",
        "--baseline-args",
        "--initial-state",
        "--output-root",
        "--name",
        "--dry-run",
    ):
        assert allowed in result.stdout
    for forbidden in ("--arm", "--epochs", "--seed", "--batch", "--resume"):
        assert forbidden not in result.stdout


def test_build_settings_preserves_baseline_except_declared_identity_fields(
    tmp_path: Path,
) -> None:
    module = _load_module()
    baseline = {
        "model": "baseline.yaml",
        "data": "old-data.yaml",
        "project": "old-runs",
        "name": "baseline",
        "save_dir": "old-runs/baseline",
        "resume": True,
        "epochs": 100,
        "seed": 7,
        "batch": 16,
        "imgsz": 640,
        "optimizer": "MuSGD",
        "lr0": 0.01,
        "workers": 8,
    }

    settings = module.build_settings(
        baseline,
        data_yaml=tmp_path / "uavdt.yaml",
        output_root=tmp_path / "runs",
        name="uavdt-ac-full",
    )

    for key in ("epochs", "seed", "batch", "imgsz", "optimizer", "lr0", "workers"):
        assert settings[key] == baseline[key]
    assert settings["model"] == str(module.CONFIG.resolve())
    assert settings["data"] == str((tmp_path / "uavdt.yaml").resolve())
    assert settings["project"] == str((tmp_path / "runs").resolve())
    assert settings["name"] == "uavdt-ac-full"
    assert settings["exist_ok"] is False
    assert "resume" not in settings
    assert "save_dir" not in settings


def test_data_yaml_derives_nc_and_rejects_conflict(tmp_path: Path) -> None:
    module = _load_module()
    valid = tmp_path / "valid.yaml"
    valid.write_text(
        yaml.safe_dump(
            {
                "path": "/datasets/UAVDT",
                "train": "images/train",
                "val": "images/val",
                "names": {0: "car", 1: "truck", 2: "bus"},
                "nc": 3,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    record = module.validate_data_yaml(valid)

    assert record["nc"] == 3
    assert record["names"] == ["car", "truck", "bus"]

    conflicting = tmp_path / "conflicting.yaml"
    conflicting.write_text(
        yaml.safe_dump(
            {
                "train": "images/train",
                "val": "images/val",
                "names": ["car", "truck", "bus"],
                "nc": 4,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="nc"):
        module.validate_data_yaml(conflicting)


@pytest.mark.parametrize(
    "names",
    [{1: "truck"}, {0: "car", 2: "bus"}, [], {}],
)
def test_data_yaml_rejects_noncontiguous_or_empty_names(
    tmp_path: Path, names: object
) -> None:
    module = _load_module()
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {"train": "images/train", "val": "images/val", "names": names}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="names"):
        module.validate_data_yaml(path)


def test_dry_run_maps_only_revised_full_and_never_constructs_trainer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    baseline_args = tmp_path / "args.yaml"
    baseline_args.write_text(
        yaml.safe_dump(
            {
                "model": "baseline.yaml",
                "data": "old.yaml",
                "project": "old",
                "name": "baseline",
                "epochs": 100,
                "seed": 0,
                "batch": 8,
                "imgsz": 640,
            }
        ),
        encoding="utf-8",
    )
    data_yaml = tmp_path / "uavdt.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "train": "images/train",
                "val": "images/val",
                "names": ["car", "truck", "bus"],
            }
        ),
        encoding="utf-8",
    )
    initial_state = tmp_path / "initial-state.pt"
    initial_state.write_bytes(b"validated-by-test-double")
    output_root = tmp_path / "runs"

    monkeypatch.setattr(module, "require_clean_tracked_worktree", lambda: None)
    monkeypatch.setattr(
        module, "validate_initial_state_file", lambda path: Path(path).resolve()
    )
    monkeypatch.setattr(
        module,
        "current_source_identity",
        lambda: {"git_commit": "a" * 40, "tree_sha256": "B" * 64},
    )

    class ForbiddenTrainer:
        def __init__(self, **kwargs) -> None:
            raise AssertionError(f"trainer constructed during dry-run: {kwargs}")

    monkeypatch.setattr(module, "TRAINER", ForbiddenTrainer)

    result = module.main(
        [
            "--data-yaml",
            str(data_yaml),
            "--baseline-args",
            str(baseline_args),
            "--initial-state",
            str(initial_state),
            "--output-root",
            str(output_root),
            "--dry-run",
        ]
    )

    assert result == 0
    authority = output_root / "authority" / "uavdt-formal100-seed0-lrs_fdr_ac_bpdd_fia-v1.json"
    record = json.loads(authority.read_text(encoding="utf-8"))
    assert record["method"] == "lrs_fdr_ac_bpdd_fia"
    assert record["arm"] == "i"
    assert record["config"]["path"] == str(module.CONFIG.resolve())
    assert record["dataset"]["nc"] == 3
    assert record["settings"]["epochs"] == 100
    printed = json.loads(capsys.readouterr().out)
    assert printed == record
