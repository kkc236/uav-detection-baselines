from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.restore_itber_checkpoint import (
    select_latest_pair,
    verify_downloaded_checkpoint,
)
from src.itber_protocol import (
    BASELINE_TRAINING_CONTRACT_SHA256,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    RUNTIME_AMENDMENT_SHA256,
)


CACHE_SHA = "C" * 64
PREFIX = "itber-v1.1-screen-seed0-p3"


def _checkpoint(path: Path, epoch: int, *, stage: str = "screen") -> None:
    torch.save(
        {
            "format_version": 1,
            "design_version": "itber-v1.1",
            "stage": stage,
            "probe": "p3",
            "seed": 0,
            "epoch": epoch,
            "baseline_sha256": EXPECTED_BASELINE_SHA256,
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "cache_manifest_sha256": CACHE_SHA,
            "baseline_training_contract_sha256": BASELINE_TRAINING_CONTRACT_SHA256,
            "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
            "refiner": {"weight": torch.ones(1)},
            "optimizer": {"state": {}, "param_groups": []},
            "scaler": {"scale": 128.0},
            "rng": {"torch": torch.get_rng_state()},
        },
        path,
    )


def test_latest_pair_uses_exact_prefix_and_ignores_incomplete_or_cross_stage() -> None:
    assets = [
        {"id": 1, "name": f"{PREFIX}-epoch-0010.pt"},
        {"id": 2, "name": f"{PREFIX}-epoch-0010.json"},
        {"id": 3, "name": f"{PREFIX}-epoch-0011.pt"},
        {"id": 4, "name": "itber-v1.1-formal-seed0-p3-epoch-0030.pt"},
        {"id": 5, "name": "itber-v1.1-formal-seed0-p3-epoch-0030.json"},
    ]

    checkpoint, manifest, epoch = select_latest_pair(assets, prefix=PREFIX)
    assert (checkpoint["id"], manifest["id"], epoch) == (1, 2, 10)


def test_download_verification_checks_all_scientific_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "download.pt.tmp"
    _checkpoint(checkpoint, 7)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "format_version": 1,
        "design_version": "itber-v1.1",
        "stage": "screen",
        "probe": "p3",
        "seed": 0,
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "cache_manifest_sha256": CACHE_SHA,
        "baseline_training_contract_sha256": BASELINE_TRAINING_CONTRACT_SHA256,
        "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
        "completed_epoch": 7,
        "checkpoint": {
            "asset_name": f"{PREFIX}-epoch-0007.pt",
            "bytes": checkpoint.stat().st_size,
            "sha256": digest,
        },
    }

    metadata = verify_downloaded_checkpoint(
        checkpoint,
        manifest,
        expected_epoch=7,
        expected_stage="screen",
        expected_probe="p3",
        expected_prefix=PREFIX,
        expected_cache_sha256=CACHE_SHA,
    )
    assert metadata.completed_epoch == 7

    for field, value in (
        ("stage", "formal"),
        ("probe", "p2"),
        ("seed", 1),
        ("baseline_sha256", "B" * 64),
        ("dataset_sha256", "D" * 64),
        ("cache_manifest_sha256", "E" * 64),
        ("runtime_amendment_sha256", "F" * 64),
    ):
        with pytest.raises(RuntimeError, match=field.replace("_sha256", "")):
            verify_downloaded_checkpoint(
                checkpoint,
                {**manifest, field: value},
                expected_epoch=7,
                expected_stage="screen",
                expected_probe="p3",
                expected_prefix=PREFIX,
                expected_cache_sha256=CACHE_SHA,
            )


def test_download_verification_rejects_corrupted_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "download.pt.tmp"
    _checkpoint(checkpoint, 1)
    manifest = {
        "format_version": 1,
        "design_version": "itber-v1.1",
        "stage": "screen",
        "probe": "p3",
        "seed": 0,
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "cache_manifest_sha256": CACHE_SHA,
        "baseline_training_contract_sha256": BASELINE_TRAINING_CONTRACT_SHA256,
        "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
        "completed_epoch": 1,
        "checkpoint": {
            "asset_name": f"{PREFIX}-epoch-0001.pt",
            "bytes": checkpoint.stat().st_size,
            "sha256": "0" * 64,
        },
    }
    with pytest.raises(RuntimeError, match="SHA-256"):
        verify_downloaded_checkpoint(
            checkpoint,
            manifest,
            expected_epoch=1,
            expected_stage="screen",
            expected_probe="p3",
            expected_prefix=PREFIX,
            expected_cache_sha256=CACHE_SHA,
        )


def test_restore_cli_runs_directly() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/restore_itber_checkpoint.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--cache-manifest" in result.stdout
    assert "--asset-prefix" in result.stdout
