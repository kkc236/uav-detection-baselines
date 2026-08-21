from __future__ import annotations

import hashlib
from pathlib import Path

from scripts import train_ace_fdr as launcher
from scripts.train_rtdetr_fdr import FORMAL_EPOCHS, FROZEN_SETTINGS


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_launcher_exposes_one_integrated_ace_fdr_method() -> None:
    assert launcher.ACE_FDR_CONFIG == ROOT / "configs" / "rtdetr-l-ace-fdr.yaml"
    parser = launcher.build_parser()
    args = parser.parse_args(
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


def test_formal_settings_reuse_the_frozen_fdr_protocol(tmp_path: Path) -> None:
    data_yaml = tmp_path / "formal-data.yaml"
    data_yaml.write_text("{}\n", encoding="utf-8")

    settings = launcher.build_settings(
        data_yaml=data_yaml,
        output_root=tmp_path / "runs",
    )

    for key, value in FROZEN_SETTINGS.items():
        if key not in {"model", "save_period"}:
            assert settings[key] == value
    assert settings["model"] == str(launcher.ACE_FDR_CONFIG.resolve())
    assert settings["data"] == str(data_yaml.resolve())
    assert settings["epochs"] == FORMAL_EPOCHS == 100
    assert settings["seed"] == 0
    assert settings["save_period"] == -1
    assert settings["name"] == "formal-seed0-ace-fdr-v1"
    assert settings["exist_ok"] is False


def test_launch_record_binds_the_whole_ace_fdr_run(tmp_path: Path) -> None:
    config = tmp_path / "ace.yaml"
    initial_state = tmp_path / "initial.pt"
    config.write_bytes(b"model: ace-fdr\n")
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
    )

    assert record == {
        "format_version": 1,
        "method": "ace_fdr",
        "source": source,
        "config": {"path": str(config.resolve()), "sha256": _sha256(config)},
        "initial_state": {
            "path": str(initial_state.resolve()),
            "sha256": _sha256(initial_state),
        },
        "dataset": dataset,
        "settings": settings,
    }
