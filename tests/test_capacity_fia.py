from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

import src.rtdetr_bpdd_capacity_fia as capacity_fia
from src.rtdetr_bpdd_capacity_fia import (
    CAPACITY_BPDD_FIA_CFG,
    FIA_MODEL_INDEX,
    remap_capacity_fia_shared_key,
)


ROOT = Path(__file__).resolve().parents[1]


def test_capacity_fia_yaml_preserves_four_scale_decoder_and_p3_bypass() -> None:
    payload = yaml.safe_load(Path(CAPACITY_BPDD_FIA_CFG).read_text(encoding="utf-8"))
    head = payload["head"]
    assert head[12][2] == "FIA"
    assert head[12][0] == 21
    assert head[13][0] == 21  # stock P4 bypasses FIA
    assert head[-1][0] == [1, 22, 25, 28]
    assert head[-1][2] == "FDRRTDETRDecoder"
    assert head[-1][3][1] == [128, 256, 256, 256]
    assert head[-1][3][2]["local_expert_preserve_base"] is True


def test_capacity_fia_state_remap_shifts_inserted_suffix_only() -> None:
    assert FIA_MODEL_INDEX == 22
    assert remap_capacity_fia_shared_key("model.21.foo") == "model.21.foo"
    assert remap_capacity_fia_shared_key("model.22.foo") == "model.23.foo"
    assert remap_capacity_fia_shared_key("model.28.decoder.layers.0.self_attn.weight") == (
        "model.29.decoder.layers.0.self_attn.weight"
    )


def test_safe_capacity_fia_candidate_changes_only_bpdd_training_controls() -> None:
    safe_path = getattr(capacity_fia, "CAPACITY_BPDD_FIA_SAFE_CFG", ROOT / "missing.yaml")
    assert safe_path.is_file(), "safe Capacity-BPDD/FIA YAML has not been implemented"
    historical = yaml.safe_load(Path(CAPACITY_BPDD_FIA_CFG).read_text(encoding="utf-8"))
    safe = yaml.safe_load(Path(safe_path).read_text(encoding="utf-8"))

    assert safe["backbone"] == historical["backbone"]
    assert safe["head"] == historical["head"]
    assert safe["fdr_loss"] == historical["fdr_loss"]
    assert safe["head"][-1][3][2]["feasible_geometry"] is True
    assert safe["head"][-1][3][2]["local_expert_preserve_base"] is True
    assert safe["capacity_bpdd_loss"] == {
        "enabled": True,
        "loc_weight": 0.15,
        "cls_weight": 0.0,
        "loc_margin": 0.02,
        "cls_margin": 0.02,
        "loc_tau": 0.10,
        "cls_tau": 0.10,
        "warmup_start": 10,
        "warmup_end": 20,
        "decay_start": 80,
        "decay_end": 100,
    }


def test_safe_capacity_fia_launcher_keeps_formal100_settings(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "train_lrs_gfdr_capacity_v3_safe_fia.py"
    assert script.is_file(), "safe Capacity-BPDD/FIA launcher has not been implemented"
    launcher = importlib.import_module("scripts.train_lrs_gfdr_capacity_v3_safe_fia")
    data_yaml = tmp_path / "formal.yaml"
    output_root = tmp_path / "runs"
    settings = launcher.build_settings(data_yaml, output_root, "formal-safe")

    assert settings["model"] == str(capacity_fia.CAPACITY_BPDD_FIA_SAFE_CFG.resolve())
    assert settings["epochs"] == 100
    assert settings["seed"] == 0
    assert settings["imgsz"] == 640
    assert settings["name"] == "formal-safe"
    assert settings["save_period"] == -1
    assert settings["exist_ok"] is False
    assert "resume" not in settings


def test_safe_model_cannot_be_mislabeled_with_historical_bpdd_controls() -> None:
    safe_model = capacity_fia.CapacityBPDDFIASafeDetectionModel(nc=3, verbose=False)
    assert safe_model.capacity_method_revision == "v3-safe-loc-kd-decay80-100-no-cls-kd"
    assert safe_model.capacity_options.cls_weight == 0.0
    assert safe_model.capacity_options.decay_start == 80
    assert safe_model.capacity_options.decay_end == 100

    with pytest.raises(ValueError, match="conflict-safe"):
        capacity_fia.CapacityBPDDFIASafeDetectionModel(
            CAPACITY_BPDD_FIA_CFG, nc=3, verbose=False
        )
