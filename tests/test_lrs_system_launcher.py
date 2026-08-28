from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import train_lrs_fdr
from scripts.train_rtdetr_fdr import FORMAL_EPOCHS, FROZEN_SETTINGS
from src.rtdetr_lrs_system import ARM_CONFIGS, TRAINER_TYPES


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_visdrone_lrs_system.py"


def _load_module():
    assert SCRIPT.is_file(), "unified LRS system launcher has not been implemented"
    return importlib.import_module("scripts.train_visdrone_lrs_system")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    module,
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    dataset_root = (tmp_path / "VisDrone").resolve()
    initial_state = (tmp_path / "initial.pt").resolve()
    output_root = (tmp_path / "runs").resolve()
    data_yaml = (output_root / "authority" / "data" / "formal-data.yaml").resolve()
    dataset_root.mkdir()
    initial_state.write_bytes(b"structurally-valid-state")
    data_yaml.parent.mkdir(parents=True)
    data_yaml.write_text("path: VisDrone\n", encoding="utf-8")

    monkeypatch.setattr(module, "require_clean_tracked_worktree", lambda: None)
    monkeypatch.setattr(
        module,
        "prepare_data_yaml",
        lambda root, stage, authority: data_yaml,
    )
    monkeypatch.setattr(
        module,
        "validate_initial_state_file",
        lambda path: Path(path).resolve(),
    )
    monkeypatch.setattr(
        module,
        "dataset_signature",
        lambda root: {"sha256": "D" * 64, "train_images": 6471, "val_images": 548},
    )
    monkeypatch.setattr(
        module,
        "current_source_identity",
        lambda: {"git_commit": "a" * 40, "tree_sha256": "B" * 64},
    )
    return dataset_root, initial_state, output_root


def test_help_exposes_only_the_unified_frozen_launcher_contract() -> None:
    assert SCRIPT.is_file(), "unified LRS system launcher has not been implemented"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for allowed in (
        "--arm",
        "--dataset-root",
        "--initial-state",
        "--output-root",
        "--name",
        "--dry-run",
    ):
        assert allowed in result.stdout
    for arm in ("g", "h", "i"):
        assert arm in result.stdout
    for forbidden in (
        "--epochs",
        "--seed",
        "--batch",
        "--workers",
        "--imgsz",
        "--device",
        "--lr0",
        "--optimizer",
        "--bpdd-weight",
        "--bpdd-temperature",
        "--fia-seed",
        "--resume",
    ):
        assert forbidden not in result.stdout


def test_arm_maps_are_exact_and_build_settings_rejects_unknown_arm(
    tmp_path: Path,
) -> None:
    module = _load_module()

    assert module.ARM_METHODS == {
        "g": "lrs_fdr_bpdd",
        "h": "lrs_fdr_fia",
        "i": "lrs_fdr_bpdd_fia",
    }
    assert module.ARM_CONFIGS == ARM_CONFIGS
    assert module.TRAINER_TYPES == TRAINER_TYPES
    with pytest.raises(ValueError, match="arm"):
        module.build_settings(
            "x",
            tmp_path / "formal.yaml",
            tmp_path / "runs",
            None,
        )


def test_arm_settings_differ_only_by_model_and_name_and_freeze_formal100(
    tmp_path: Path,
) -> None:
    module = _load_module()
    data_yaml = tmp_path / "formal.yaml"
    output_root = tmp_path / "runs"
    settings = {
        arm: module.build_settings(arm, data_yaml, output_root)
        for arm in ("g", "h", "i")
    }
    reference = train_lrs_fdr.build_settings(
        data_yaml=data_yaml,
        output_root=output_root,
    )

    for arm, actual in settings.items():
        assert {k: v for k, v in actual.items() if k not in {"model", "name"}} == {
            k: v for k, v in reference.items() if k not in {"model", "name"}
        }
        assert actual["model"] == str(ARM_CONFIGS[arm].resolve())
        assert actual["name"] == f"formal-seed0-{module.ARM_METHODS[arm]}-v1"
        assert actual["epochs"] == FORMAL_EPOCHS == 100
        assert actual["seed"] == 0
        assert actual["imgsz"] == FROZEN_SETTINGS["imgsz"] == 640
        assert actual["save_period"] == -1
        assert actual["data"] == str(data_yaml.resolve())
        assert actual["project"] == str(output_root.resolve())
        assert actual["exist_ok"] is False
        assert "resume" not in actual

    for left, right in (("g", "h"), ("g", "i"), ("h", "i")):
        differing = {
            key
            for key in settings[left].keys() | settings[right].keys()
            if settings[left].get(key) != settings[right].get(key)
        }
        assert differing == {"model", "name"}


def test_authority_record_is_deterministic_hashed_and_conflict_safe(
    tmp_path: Path,
) -> None:
    module = _load_module()
    config = tmp_path / "arm.yaml"
    state = tmp_path / "initial.pt"
    config.write_bytes(b"model: arm-g\n")
    state.write_bytes(b"initial-state")
    source = {"git_commit": "a" * 40, "tree_sha256": "B" * 64}
    dataset = {"sha256": "C" * 64, "train_images": 6471, "val_images": 548}
    settings = {"model": str(config.resolve()), "epochs": 100, "seed": 0}

    record = module.build_launch_record(
        arm="g",
        source_identity=source,
        config_path=config,
        initial_state_path=state,
        dataset=dataset,
        settings=settings,
    )
    repeated = module.build_launch_record(
        arm="g",
        source_identity=source,
        config_path=config,
        initial_state_path=state,
        dataset=dataset,
        settings=settings,
    )

    assert repeated == record
    assert record == {
        "format_version": 1,
        "arm": "g",
        "method": "lrs_fdr_bpdd",
        "source": source,
        "config": {"path": str(config.resolve()), "sha256": _sha256(config)},
        "initial_state": {"path": str(state.resolve()), "sha256": _sha256(state)},
        "dataset": dataset,
        "settings": settings,
    }

    authority = tmp_path / "authority" / "run.json"
    module.write_authority(authority, record)
    first_bytes = authority.read_bytes()
    module.write_authority(authority, repeated)
    assert authority.read_bytes() == first_bytes
    assert json.loads(first_bytes) == record
    with pytest.raises(ValueError, match="different"):
        module.write_authority(authority, {**record, "arm": "h"})


def test_dry_run_writes_and_prints_authority_without_constructing_trainer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    dataset_root, initial_state, output_root = _patch_runtime(
        monkeypatch, module, tmp_path
    )

    def forbidden_trainer(**kwargs):
        raise AssertionError(f"trainer constructed during dry run: {kwargs}")

    monkeypatch.setattr(
        module,
        "TRAINER_TYPES",
        {arm: forbidden_trainer for arm in ("g", "h", "i")},
    )
    result = module.main(
        [
            "--arm",
            "g",
            "--dataset-root",
            str(dataset_root),
            "--initial-state",
            str(initial_state),
            "--output-root",
            str(output_root),
            "--dry-run",
        ]
    )

    assert result == 0
    authority = output_root / "authority" / "formal-seed0-lrs_fdr_bpdd-v1.json"
    record = json.loads(authority.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == record
    assert record["arm"] == "g"
    assert record["method"] == "lrs_fdr_bpdd"


@pytest.mark.parametrize("arm", ["g", "h", "i"])
def test_non_dry_run_dispatches_the_selected_trainer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arm: str,
) -> None:
    module = _load_module()
    dataset_root, initial_state, output_root = _patch_runtime(
        monkeypatch, module, tmp_path
    )
    calls: list[tuple[str, object]] = []

    class SpyTrainer:
        def __init__(self, **kwargs) -> None:
            calls.append(("init", kwargs))

        def train(self) -> None:
            calls.append(("train", None))

    trainers = {key: type(f"Unused{key}", (), {}) for key in ("g", "h", "i")}
    trainers[arm] = SpyTrainer
    monkeypatch.setattr(module, "TRAINER_TYPES", trainers)

    assert module.main(
        [
            "--arm",
            arm,
            "--dataset-root",
            str(dataset_root),
            "--initial-state",
            str(initial_state),
            "--output-root",
            str(output_root),
        ]
    ) == 0

    assert calls[0][0] == "init"
    assert calls[0][1] == {
        "overrides": module.build_settings(
            arm,
            output_root / "authority" / "data" / "formal-data.yaml",
            output_root,
            None,
        ),
        "initial_state_path": initial_state,
        "experiment_seed": 0,
    }
    assert calls[1] == ("train", None)


def test_invalid_arm_is_rejected_by_argparse_and_build_settings(
    tmp_path: Path,
) -> None:
    module = _load_module()

    with pytest.raises(SystemExit) as exc_info:
        module.build_parser().parse_args(
            [
                "--arm",
                "x",
                "--dataset-root",
                str(tmp_path),
                "--initial-state",
                str(tmp_path / "initial.pt"),
                "--output-root",
                str(tmp_path / "runs"),
            ]
        )
    assert exc_info.value.code == 2
    with pytest.raises(ValueError, match="unknown LRS system arm"):
        module.build_settings("x", tmp_path / "data.yaml", tmp_path / "runs", None)


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "../escape", "..\\escape", "/absolute", "C:\\absolute"],
)
def test_run_name_rejects_empty_absolute_or_non_component_values(
    tmp_path: Path,
    name: str,
) -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="name"):
        module.build_settings(
            "g",
            tmp_path / "data.yaml",
            tmp_path / "runs",
            name,
        )


def test_run_name_accepts_a_safe_single_component(tmp_path: Path) -> None:
    module = _load_module()

    settings = module.build_settings(
        "g",
        tmp_path / "data.yaml",
        tmp_path / "runs",
        "formal-seed0_custom-run_1",
    )

    assert settings["name"] == "formal-seed0_custom-run_1"
