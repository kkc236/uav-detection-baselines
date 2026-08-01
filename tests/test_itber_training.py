from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.train_itber import (
    AUGMENTATION,
    TRAINING_CONSTANTS,
    atomic_save_private_checkpoint,
    build_private_optimizer,
    stage_protocol,
    validate_resume_checkpoint,
)
from src.itber_protocol import (
    BASELINE_TRAINING_CONTRACT_SHA256,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    RUNTIME_AMENDMENT_SHA256,
)


def test_stage_protocol_locks_screen_and_formal_without_shared_checkpoint() -> None:
    screen = stage_protocol("screen")
    formal = stage_protocol("formal")

    assert (screen.train_images, screen.val_images, screen.epochs, screen.use_subset) == (647, 548, 12, True)
    assert (formal.train_images, formal.val_images, formal.epochs, formal.use_subset) == (6471, 548, 30, False)
    assert screen.fresh_private_initialization is True
    assert formal.fresh_private_initialization is True
    assert screen.allow_prior_stage_checkpoint is False
    assert formal.allow_prior_stage_checkpoint is False


def test_training_constants_and_augmentation_match_frozen_contract() -> None:
    assert TRAINING_CONSTANTS == {
        "seed": 0,
        "private_seed": 10000,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "amp": True,
        "amp_scale": 128.0,
        "save_period": 1,
        "optimizer": "AdamW",
        "lr": 0.001,
        "weight_decay": 0.0001,
        "betas": (0.9, 0.999),
        "clip_grad_norm": 10.0,
        "on_the_fly_evidence": True,
    }
    assert AUGMENTATION == {
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "scale": 0.5,
        "translate": 0.1,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "cutmix": 0.0,
        "copy_paste": 0.0,
    }


def test_private_optimizer_is_exact_adamw() -> None:
    model = nn.Linear(2, 2)
    optimizer = build_private_optimizer(model)
    group = optimizer.param_groups[0]

    assert isinstance(optimizer, torch.optim.AdamW)
    assert group["lr"] == 0.001
    assert group["weight_decay"] == 0.0001
    assert group["betas"] == (0.9, 0.999)


def test_resume_validation_rejects_cross_stage_or_authority() -> None:
    cache_sha = "C" * 64
    artifact = {
        "format_version": 1,
        "design_version": "itber-v1.1",
        "stage": "screen",
        "probe": "p3",
        "seed": 0,
        "epoch": 4,
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "cache_manifest_sha256": cache_sha,
        "baseline_training_contract_sha256": BASELINE_TRAINING_CONTRACT_SHA256,
        "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
    }
    validate_resume_checkpoint(artifact, stage="screen", cache_manifest_sha256=cache_sha)
    for key, value in (
        ("stage", "formal"),
        ("probe", "p2"),
        ("seed", 1),
        ("baseline_sha256", "BAD"),
        ("dataset_sha256", "BAD"),
        ("runtime_amendment_sha256", "F" * 64),
    ):
        changed = dict(artifact, **{key: value})
        with pytest.raises(ValueError, match=key):
            validate_resume_checkpoint(changed, stage="screen", cache_manifest_sha256=cache_sha)
    with pytest.raises(ValueError, match="cache_manifest_sha256"):
        validate_resume_checkpoint(
            artifact,
            stage="screen",
            cache_manifest_sha256="D" * 64,
        )


def test_atomic_checkpoint_roundtrip_contains_private_state_only(tmp_path) -> None:
    model = nn.Linear(2, 2)
    path = tmp_path / "epoch-0001.pt"
    artifact = {
        "format_version": 1,
        "design_version": "itber-v1.1",
        "stage": "screen",
        "probe": "p3",
        "seed": 0,
        "epoch": 1,
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "baseline_training_contract_sha256": BASELINE_TRAINING_CONTRACT_SHA256,
        "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
        "refiner": model.state_dict(),
    }

    atomic_save_private_checkpoint(path, artifact)
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    assert loaded["epoch"] == 1
    assert "detector" not in loaded
    assert not path.with_suffix(".pt.tmp").exists()


def test_cli_exposes_only_operational_paths_stage_device_and_resume() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/train_itber.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    help_text = result.stdout
    for allowed in ("--stage", "--baseline-checkpoint", "--dataset-root", "--gate1-cache-manifest", "--publication-config", "--output-root", "--device", "--resume-checkpoint"):
        assert allowed in help_text
    for forbidden in ("--epochs", "--seed", "--batch", "--workers", "--imgsz", "--lr"):
        assert forbidden not in help_text


def test_source_forces_frozen_on_the_fly_detector_and_epoch_checkpoints() -> None:
    source = Path("scripts/train_itber.py").read_text(encoding="utf-8")
    for marker in (
        "FrozenITBERAdapter",
        "detector.eval()",
        "requires_grad_(False)",
        "training_step",
        "close_mosaic",
        "amp_scale",
        "epoch-{epoch:04d}.pt",
        "detector_sha_before",
        "detector_sha_after",
        "matched_correction_rms",
        "unmatched_correction_rms",
        "runtime_amendment_sha256",
        "evaluate_itber.py",
        "publish_itber_epoch.py",
        "subprocess.run",
        "ITBER epoch publication did not verify",
        "generator.manual_seed",
        "loader.reset()",
    ):
        assert marker in source


def test_publication_cli_is_operational_only() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/publish_itber_epoch.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for allowed in ("--run-dir", "--checkpoint", "--config"):
        assert allowed in result.stdout
    for forbidden in ("--stage", "--seed", "--probe", "--retain", "--repo", "--tag"):
        assert forbidden not in result.stdout
