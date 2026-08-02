from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from scripts.train_iber import (
    AUGMENTATION,
    REQUIRED_CHECKPOINT_KEYS,
    TRAINING_CONSTANTS,
    atomic_save_private_checkpoint,
    build_private_optimizer,
    highest_contiguous_verified_epoch,
    _restore_rng,
    _rng_state,
    validate_resume_checkpoint,
)
from src.iber_protocol import (
    DESIGN_VERSION,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    PROTOCOL_SHA256,
    RUNTIME_AMENDMENT_SHA256,
)


SOURCE_COMMIT = "a" * 40
CATEGORY_SHA = "B" * 64
GATE1_SHA = "C" * 64


def _checkpoint(epoch: int = 4) -> dict:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return {
        "format_version": 1,
        "design_version": DESIGN_VERSION,
        "stage": "screen",
        "probe": "b3",
        "seed": 0,
        "epoch": epoch,
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "subset_sha256": EXPECTED_SUBSET_SHA256,
        "source_commit": SOURCE_COMMIT,
        "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "refiner": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": {"scale": 128.0},
        "rng": {"python": (), "numpy": (), "torch": torch.get_rng_state(), "cuda": None},
    }


def _publication_record(epoch: int, *, checkpoint_sha: str | None = None) -> dict:
    digest = checkpoint_sha or f"{epoch:064x}"
    return {
        "format_version": 1,
        "design_version": DESIGN_VERSION,
        "stage": "screen",
        "probe": "b3",
        "seed": 0,
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "subset_sha256": EXPECTED_SUBSET_SHA256,
        "category_sha256": CATEGORY_SHA,
        "protocol_sha256": PROTOCOL_SHA256,
        "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
        "gate1_decision_sha256": GATE1_SHA,
        "source_commit": SOURCE_COMMIT,
        "completed_epoch": epoch,
        "checkpoint": {"bytes": 10, "sha256": digest},
        "remote_verification": {
            "checkpoint": {"bytes": 10, "sha256": digest},
            "manifest": {"bytes": 20, "sha256": "d" * 64},
        },
        "result_commit_sha": "e" * 40,
        "result_commit_verified": True,
        "verified": True,
    }


def test_training_constants_and_augmentation_are_frozen_screen_contract() -> None:
    assert TRAINING_CONSTANTS == {
        "stage": "screen",
        "probe": "b3",
        "seed": 0,
        "private_seed": 10_000,
        "epochs": 30,
        "train_images": 647,
        "val_images": 548,
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
        "max_det": 300,
        "nms": False,
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
    optimizer = build_private_optimizer(nn.Linear(2, 2))
    group = optimizer.param_groups[0]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert group["lr"] == 0.001
    assert group["weight_decay"] == 0.0001
    assert group["betas"] == (0.9, 0.999)


def test_checkpoint_schema_and_resume_require_highest_verified_epoch() -> None:
    artifact = _checkpoint()
    assert REQUIRED_CHECKPOINT_KEYS == {
        "format_version",
        "design_version",
        "stage",
        "probe",
        "seed",
        "epoch",
        "baseline_sha256",
        "dataset_sha256",
        "subset_sha256",
        "source_commit",
        "runtime_amendment_sha256",
        "protocol_sha256",
        "refiner",
        "optimizer",
        "scaler",
        "rng",
    }
    validate_resume_checkpoint(
        artifact,
        source_commit=SOURCE_COMMIT,
        highest_verified_epoch=4,
        publication_record=_publication_record(4),
    )
    for key, value in (
        ("design_version", "itber-v1.1"),
        ("stage", "formal"),
        ("probe", "b2"),
        ("seed", 1),
        ("baseline_sha256", "BAD"),
        ("dataset_sha256", "BAD"),
        ("subset_sha256", "BAD"),
        ("source_commit", "b" * 40),
        ("runtime_amendment_sha256", "F" * 64),
        ("protocol_sha256", "F" * 64),
    ):
        changed = copy.deepcopy(artifact)
        changed[key] = value
        with pytest.raises(ValueError, match=key):
            validate_resume_checkpoint(
                changed,
                source_commit=SOURCE_COMMIT,
                highest_verified_epoch=4,
                publication_record=_publication_record(4),
            )
    with pytest.raises(ValueError, match="highest_verified_epoch"):
        validate_resume_checkpoint(
            artifact,
            source_commit=SOURCE_COMMIT,
            highest_verified_epoch=3,
            publication_record=_publication_record(4),
        )

    for forbidden in ("detector", "model", "ema"):
        changed = copy.deepcopy(artifact)
        changed[forbidden] = {"forbidden": torch.ones(1)}
        with pytest.raises(ValueError, match=forbidden):
            validate_resume_checkpoint(
                changed,
                source_commit=SOURCE_COMMIT,
                highest_verified_epoch=4,
                publication_record=_publication_record(4),
            )

    foreign_tip = _publication_record(4)
    foreign_tip["source_commit"] = "f" * 40
    with pytest.raises(ValueError, match="publication_record.source_commit"):
        validate_resume_checkpoint(
            artifact,
            source_commit=SOURCE_COMMIT,
            highest_verified_epoch=4,
            publication_record=foreign_tip,
        )


def test_verified_epoch_ledger_must_be_contiguous_and_exact() -> None:
    rows = [_publication_record(epoch) for epoch in range(1, 4)]
    assert highest_contiguous_verified_epoch(rows) == 3
    with pytest.raises(ValueError, match="contiguous"):
        highest_contiguous_verified_epoch([rows[0], rows[2]])
    with pytest.raises(ValueError, match="verified"):
        highest_contiguous_verified_epoch(
            [rows[0], {**rows[1], "verified": False}]
        )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda row: row.update(result_commit_verified=False), "result commit"),
        (
            lambda row: row["remote_verification"]["checkpoint"].update(
                sha256="f" * 64
            ),
            "remote checkpoint",
        ),
        (
            lambda row: row["remote_verification"].pop("manifest"),
            "remote manifest",
        ),
    ],
)
def test_verified_epoch_ledger_requires_task10_remote_receipts(
    mutation, message: str
) -> None:
    row = _publication_record(1)
    mutation(row)
    with pytest.raises(ValueError, match=message):
        highest_contiguous_verified_epoch([row])


def test_atomic_checkpoint_roundtrip_contains_private_state_only(tmp_path: Path) -> None:
    artifact = _checkpoint(epoch=1)
    path = tmp_path / "epoch-0001.pt"
    atomic_save_private_checkpoint(path, artifact)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert REQUIRED_CHECKPOINT_KEYS <= set(loaded)
    assert loaded["epoch"] == 1
    assert "detector" not in loaded
    assert not path.with_suffix(".pt.tmp").exists()


def test_safe_rng_payload_restores_numpy_sequence() -> None:
    np.random.seed(123)
    state = _rng_state()
    expected = np.random.random(5)
    np.random.seed(999)
    _restore_rng({"rng": state})
    np.testing.assert_array_equal(np.random.random(5), expected)


def test_rng_restore_validates_every_state_before_mutating_globals() -> None:
    random.seed(123)
    np.random.seed(123)
    bad_state = _rng_state()
    bad_state["numpy"]["position"] = 625

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    with pytest.raises(ValueError, match="NumPy RNG"):
        _restore_rng({"rng": bad_state})

    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
