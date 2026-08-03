from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.iber_formal_protocol import (
    FORMAL_EPOCHS,
    FORMAL_FROZEN_PROTOCOL,
    FORMAL_TRAIN_COUNT,
    FORMAL_VAL_COUNT,
    build_formal_initial_state,
    build_formal_settings,
    load_formal_initial_state,
    validate_formal_initial_state,
    validate_formal_manifest,
)
from src.lpr_protocol import EXPECTED_SOURCE_SHA256


def _manifest(tmp_path: Path) -> dict:
    return {
        "format_version": 3,
        "design_version": "iber-be-v1.0-signed-formal100",
        "seed": 0,
        "dataset_root": str(tmp_path / "VisDrone"),
        "dataset": {
            "sha256": "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB",
            "train_images": 6471,
            "val_images": 548,
            "classes": 10,
        },
        "data": {"formal": {"path": str(tmp_path / "formal.yaml")}},
        "source_sha256": EXPECTED_SOURCE_SHA256,
    }


def test_formal_protocol_exactly_matches_frozen_seed0_baseline() -> None:
    assert FORMAL_EPOCHS == 100
    assert FORMAL_TRAIN_COUNT == 6471
    assert FORMAL_VAL_COUNT == 548
    assert FORMAL_FROZEN_PROTOCOL == {
        "model": "rtdetr-l.yaml",
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "device": "0",
        "fraction": 1.0,
        "pretrained": False,
        "cache": False,
        "amp": True,
        "amp_scale": 128.0,
        "seed": 0,
        "deterministic": True,
        "nbs": 64,
        "query_count": 300,
        "nms": False,
        "max_det": 300,
        "save": True,
        "save_period": 1,
        "optimizer": "MuSGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "cos_lr": False,
        "plots": True,
        "val": True,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "scale": 0.5,
        "translate": 0.1,
        "perspective": 0.0,
        "degrees": 0.0,
        "shear": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "cutmix": 0.0,
        "copy_paste": 0.0,
    }


def test_formal_settings_strip_non_ultralytics_authority_fields(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    settings = build_formal_settings(
        manifest,
        project=tmp_path / "runs",
        name="formal-seed0-iber-be-signed",
    )

    assert settings["data"] == manifest["data"]["formal"]["path"]
    assert settings["epochs"] == 100
    assert settings["pretrained"] is False
    assert settings["exist_ok"] is False
    assert settings["project"] == str((tmp_path / "runs").resolve())
    assert settings["name"] == "formal-seed0-iber-be-signed"
    assert "amp_scale" not in settings
    assert "query_count" not in settings


def test_formal_settings_resume_only_changes_checkpoint_pointer(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "weights" / "last.pt"
    settings = build_formal_settings(
        _manifest(tmp_path),
        project=tmp_path / "runs",
        name="formal",
        resume=checkpoint,
    )

    assert settings["resume"] == str(checkpoint.resolve())
    assert settings["epochs"] == 100
    assert settings["seed"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("seed", 1),
        ("format_version", 2),
        ("design_version", "iber-be-v1.0"),
    ),
)
def test_manifest_rejects_identity_drift(tmp_path: Path, field: str, value: object) -> None:
    manifest = _manifest(tmp_path)
    manifest[field] = value
    with pytest.raises(ValueError, match=field.replace("_", "[-_ ]?")):
        validate_formal_manifest(manifest)


def test_manifest_rejects_data_count_or_hash_drift(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["dataset"]["train_images"] = 6470
    with pytest.raises(ValueError, match="dataset"):
        validate_formal_manifest(manifest)


def test_random_initial_state_artifact_separates_public_and_private_tensors() -> None:
    common = {"model.0.weight": torch.tensor([1.0])}
    method = {
        **common,
        "iber_refiner.boundary.weight": torch.tensor([2.0]),
    }

    artifact = build_formal_initial_state(common, method, seed=0)
    validate_formal_initial_state(artifact)

    assert set(artifact["common_state"]) == {"model.0.weight"}
    assert set(artifact["iber_state"]) == {"iber_refiner.boundary.weight"}
    assert artifact["metadata"]["pretrained"] is False


def test_initial_state_load_is_strict_and_refuses_public_drift() -> None:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.public = torch.nn.Linear(1, 1, bias=False)
            self.iber_refiner = torch.nn.Linear(1, 1, bias=False)

    common = {"public.weight": torch.tensor([[1.0]])}
    method = {**common, "iber_refiner.weight": torch.tensor([[2.0]])}
    artifact = build_formal_initial_state(common, method, seed=0)
    model = Tiny()

    load_formal_initial_state(model, artifact)
    torch.testing.assert_close(model.public.weight, torch.tensor([[1.0]]))
    torch.testing.assert_close(model.iber_refiner.weight, torch.tensor([[2.0]]))

    changed = dict(artifact)
    changed["common_state"] = {"public.weight": torch.tensor([[9.0]])}
    with pytest.raises(ValueError, match="fingerprint"):
        load_formal_initial_state(Tiny(), changed)
