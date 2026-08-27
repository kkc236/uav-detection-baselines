from __future__ import annotations

import hashlib
from pathlib import Path

from scripts import train_lrs_fdr as launcher
from scripts.audit_lrs_fgl_gate0 import decide_gate0
from scripts.train_rtdetr_fdr import FORMAL_EPOCHS, FROZEN_SETTINGS


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _passing_layers() -> list[dict[str, float | int]]:
    return [
        {
            "layer_index": index,
            "matches": 100,
            "low_matches": 30 if index < 3 else 20,
            "quality_sum": 50.0,
            "low_quality_sum": 5.0,
            "low_edges": 120 if index < 3 else 80,
            "saturated_low_edges": 24 if index < 3 else 16,
            "max_sum_error": 1e-7,
        }
        for index in range(5)
    ]


def test_lrs_launcher_reuses_every_frozen_formal100_setting(tmp_path: Path) -> None:
    data_yaml = tmp_path / "formal-data.yaml"
    data_yaml.write_text("{}\n", encoding="utf-8")

    settings = launcher.build_settings(
        data_yaml=data_yaml,
        output_root=tmp_path / "runs",
    )

    for key, value in FROZEN_SETTINGS.items():
        if key not in {"model", "save_period"}:
            assert settings[key] == value
    assert launcher.LRS_FDR_CONFIG == ROOT / "configs" / "rtdetr-l-lrs-fdr.yaml"
    assert settings["model"] == str(launcher.LRS_FDR_CONFIG.resolve())
    assert settings["epochs"] == FORMAL_EPOCHS == 100
    assert settings["seed"] == 0
    assert settings["save_period"] == -1
    assert settings["name"] == "formal-seed0-lrs-fdr-v1"
    assert settings["exist_ok"] is False
    assert "resume" not in settings


def test_lrs_launch_record_freezes_the_single_module_and_alpha(tmp_path: Path) -> None:
    config = tmp_path / "lrs.yaml"
    initial_state = tmp_path / "initial.pt"
    config.write_bytes(b"model: lrs-fdr\n")
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

    assert record["method"] == "lrs_fdr"
    assert record["config"] == {
        "path": str(config.resolve()),
        "sha256": _sha256(config),
    }
    assert record["initial_state"]["sha256"] == _sha256(initial_state)
    assert record["lrs_fgl"] == {
        "kind": "layerwise_reliability_shrinkage_v1",
        "alpha0": 0.25,
        "schedule": [0.25, 0.20, 0.15, 0.10, 0.05, 0.0],
        "scope": "normal_decoder_fgl_only",
        "grouping": "same_image_same_layer",
        "resume_policy": "restart_from_epoch_0",
    }


def test_gate0_accepts_three_supported_shallow_layers() -> None:
    decision = decide_gate0(_passing_layers())

    assert decision["passed"] is True
    assert decision["checks"] == {
        "three_shallow_layers_have_low_q_support": True,
        "supported_layers_are_underweighted": True,
        "low_q_edge_saturation_below_half": True,
        "per_image_sum_conserved": True,
    }
    assert decision["eligible_layer_indices"] == [0, 1, 2]


def test_gate0_rejects_saturation_or_conservation_failure() -> None:
    saturated = _passing_layers()
    saturated[0]["saturated_low_edges"] = 120
    saturated[1]["saturated_low_edges"] = 120
    saturated[2]["saturated_low_edges"] = 120
    assert decide_gate0(saturated)["passed"] is False

    drifted = _passing_layers()
    drifted[2]["max_sum_error"] = 3e-6
    assert decide_gate0(drifted)["passed"] is False


def test_gate0_rejects_fewer_than_three_low_q_layers() -> None:
    layers = _passing_layers()
    layers[1]["low_matches"] = 10
    layers[2]["low_matches"] = 10

    decision = decide_gate0(layers)

    assert decision["passed"] is False
    assert decision["eligible_layer_indices"] == [0]
