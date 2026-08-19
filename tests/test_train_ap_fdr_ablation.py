from __future__ import annotations

import hashlib
from pathlib import Path

from scripts import train_ap_fdr_ablation as launcher
from scripts.train_rtdetr_fdr import FORMAL_EPOCHS, FROZEN_SETTINGS


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_variant_configs_are_the_two_high_value_ap_fdr_ablations() -> None:
    assert launcher.VARIANT_CONFIGS == {
        "no_preliminary_reference": (
            ROOT / "configs" / "rtdetr-l-fdr-no-prebox.yaml"
        ),
        "no_dn_fdr": ROOT / "configs" / "rtdetr-l-fdr-no-dn.yaml",
    }


def test_formal_settings_reuse_the_frozen_fdr_protocol(tmp_path: Path) -> None:
    data_yaml = tmp_path / "formal-data.yaml"
    data_yaml.write_text("{}\n", encoding="utf-8")

    settings = launcher.build_settings(
        "no_dn_fdr",
        data_yaml=data_yaml,
        output_root=tmp_path / "runs",
        name="paired-no-dn",
    )

    for key, value in FROZEN_SETTINGS.items():
        if key != "model":
            assert settings[key] == value
    assert settings["model"] == str(
        (ROOT / "configs" / "rtdetr-l-fdr-no-dn.yaml").resolve()
    )
    assert settings["data"] == str(data_yaml.resolve())
    assert settings["epochs"] == FORMAL_EPOCHS == 100
    assert settings["seed"] == 0
    assert settings["project"] == str((tmp_path / "runs").resolve())
    assert settings["name"] == "paired-no-dn"
    assert settings["exist_ok"] is False


def test_launch_record_binds_source_config_initial_state_data_and_settings(
    tmp_path: Path,
) -> None:
    config = tmp_path / "model.yaml"
    initial_state = tmp_path / "initial.pt"
    config.write_bytes(b"model: ap-fdr\n")
    initial_state.write_bytes(b"frozen-state")
    settings = {"model": str(config), "epochs": 100, "seed": 0}
    source = {"git_commit": "a" * 40, "tree_sha256": "B" * 64}
    dataset = {"sha256": "C" * 64, "train_images": 6471, "val_images": 548}

    record = launcher.build_launch_record(
        variant="no_dn_fdr",
        source_identity=source,
        config_path=config,
        initial_state_path=initial_state,
        dataset=dataset,
        settings=settings,
    )

    assert record == {
        "format_version": 1,
        "variant": "no_dn_fdr",
        "source": source,
        "config": {
            "path": str(config.resolve()),
            "sha256": _sha256(config),
        },
        "initial_state": {
            "path": str(initial_state.resolve()),
            "sha256": _sha256(initial_state),
        },
        "dataset": dataset,
        "settings": settings,
    }


def test_parser_restricts_variant_to_declared_ablations() -> None:
    parser = launcher.build_parser()
    args = parser.parse_args(
        [
            "--variant",
            "no_preliminary_reference",
            "--dataset-root",
            "dataset",
            "--initial-state",
            "initial.pt",
            "--output-root",
            "runs",
            "--dry-run",
        ]
    )
    assert args.variant == "no_preliminary_reference"
    assert args.dry_run is True

