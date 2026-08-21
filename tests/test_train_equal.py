from __future__ import annotations

import hashlib
from pathlib import Path

from scripts import train_equal as launcher
from scripts.train_rtdetr_fdr import FORMAL_EPOCHS, FROZEN_SETTINGS


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_launcher_exposes_one_equal_method() -> None:
    assert launcher.EQUAL_CONFIG == ROOT / "configs" / "rtdetr-l-equal.yaml"
    args = launcher.build_parser().parse_args(
        [
            "--dataset-root",
            "dataset",
            "--initial-state",
            "initial.pt",
            "--output-root",
            "runs",
            "--dry-run",
        ]
    )

    assert not hasattr(args, "variant")
    assert args.dry_run is True


def test_equal_formal_settings_reuse_the_frozen_fdr_protocol(tmp_path: Path) -> None:
    data_yaml = tmp_path / "formal-data.yaml"
    data_yaml.write_text("{}\n", encoding="utf-8")

    settings = launcher.build_settings(
        data_yaml=data_yaml,
        output_root=tmp_path / "runs",
    )

    for key, value in FROZEN_SETTINGS.items():
        if key not in {"model", "save_period"}:
            assert settings[key] == value
    assert settings["model"] == str(launcher.EQUAL_CONFIG.resolve())
    assert settings["epochs"] == FORMAL_EPOCHS == 100
    assert settings["seed"] == 0
    assert settings["save_period"] == -1
    assert settings["name"] == "formal-seed0-equal-v1"


def test_equal_launch_record_can_bind_the_immutable_ace_runtime_alias(
    tmp_path: Path,
) -> None:
    config = tmp_path / "equal.yaml"
    initial_state = tmp_path / "initial.pt"
    config.write_bytes(b"model: equal\n")
    initial_state.write_bytes(b"frozen-state")
    source = {"git_commit": "a" * 40, "tree_sha256": "B" * 64}
    dataset = {"sha256": "C" * 64, "train_images": 6471, "val_images": 548}
    settings = {"model": str(config), "epochs": 100, "seed": 0}

    record = launcher.build_launch_record(
        source_identity=source,
        config_path=config,
        initial_state_path=initial_state,
        dataset=dataset,
        settings=settings,
        runtime_alias="ace_fdr",
    )

    assert record == {
        "format_version": 1,
        "method": "equal",
        "runtime_alias": "ace_fdr",
        "source": source,
        "config": {"path": str(config.resolve()), "sha256": _sha256(config)},
        "initial_state": {
            "path": str(initial_state.resolve()),
            "sha256": _sha256(initial_state),
        },
        "dataset": dataset,
        "settings": settings,
    }
