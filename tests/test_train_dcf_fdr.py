from pathlib import Path

from scripts import train_dcf_fdr as launcher
from scripts.train_rtdetr_fdr import FORMAL_EPOCHS, FROZEN_SETTINGS


def test_dcf_launcher_binds_one_frozen_formal_method(tmp_path: Path) -> None:
    settings = launcher.build_settings(
        data_yaml=tmp_path / "data.yaml",
        output_root=tmp_path / "runs",
    )
    assert launcher.DCF_FDR_CONFIG.name == "rtdetr-l-dcf-fdr.yaml"
    assert settings["model"] == str(launcher.DCF_FDR_CONFIG.resolve())
    assert settings["name"] == "formal-seed0-dcf-fdr-v1"
    assert settings["epochs"] == FORMAL_EPOCHS
    assert settings["seed"] == 0
    for key, value in FROZEN_SETTINGS.items():
        if key not in {"model", "epochs", "seed", "save_period"}:
            assert settings[key] == value

    clean = launcher.build_settings(
        data_yaml=tmp_path / "data.yaml",
        output_root=tmp_path / "runs",
        arm="clean",
    )
    assert clean["model"] == str(launcher.CLEAN_FDR_CONFIG.resolve())
    assert clean["name"] == "formal-seed0-clean-fdr-v1"


def test_dcf_launch_record_has_distinct_method_identity(tmp_path: Path) -> None:
    initial = tmp_path / "initial.pt"
    initial.write_bytes(b"state")
    record = launcher.build_launch_record(
        source_identity={"commit": "abc"},
        initial_state_path=initial,
        dataset={"sha256": "dataset"},
        settings={"seed": 0, "model": str(launcher.DCF_FDR_CONFIG)},
    )
    assert record["method"] == "dcf_fdr"
    assert record["config"]["path"] == str(launcher.DCF_FDR_CONFIG.resolve())
