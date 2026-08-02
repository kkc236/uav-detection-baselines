from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.restore_iber_checkpoint import (
    install_verified_checkpoint,
    select_latest_pair,
    verify_downloaded_checkpoint,
)
from src.iber_protocol import (
    DESIGN_VERSION,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    PROTOCOL_SHA256,
    RUNTIME_AMENDMENT_SHA256,
)
from src.iber_publication import ASSET_PREFIX, PublicationIdentity


CATEGORY_SHA = "A" * 64
GATE1_SHA = "B" * 64
SOURCE_COMMIT = "c" * 40


def _identity() -> PublicationIdentity:
    return PublicationIdentity(
        design_version=DESIGN_VERSION,
        stage="screen",
        probe="b3",
        seed=0,
        baseline_sha256=EXPECTED_BASELINE_SHA256,
        dataset_sha256=EXPECTED_DATASET_SHA256,
        subset_sha256=EXPECTED_SUBSET_SHA256,
        category_sha256=CATEGORY_SHA,
        protocol_sha256=PROTOCOL_SHA256,
        runtime_amendment_sha256=RUNTIME_AMENDMENT_SHA256,
        gate1_decision_sha256=GATE1_SHA,
        source_commit=SOURCE_COMMIT,
    )


def _checkpoint(path: Path, epoch: int) -> None:
    torch.save(
        {
            "format_version": 1,
            **_identity().as_dict(),
            "private_seed": 10_000,
            "epoch": epoch,
            "detector_sha_before": "D" * 64,
            "detector_sha_after": "D" * 64,
            "refiner": {"weight": torch.ones(1)},
            "optimizer": {"state": {}, "param_groups": []},
            "scaler": {"scale": 128.0},
            "rng": {"torch": torch.get_rng_state()},
        },
        path,
    )


def _manifest(path: Path, epoch: int) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "format_version": 1,
        **_identity().as_dict(),
        "completed_epoch": epoch,
        "checkpoint": {
            "asset_name": f"{ASSET_PREFIX}-epoch-{epoch:04d}.pt",
            "bytes": path.stat().st_size,
            "sha256": digest,
        },
        "remote_verification": {
            "checkpoint": {"bytes": path.stat().st_size, "sha256": digest},
            "manifest": {"bytes": 20, "sha256": "a" * 64},
        },
        "result_commit_sha": "e" * 40,
        "result_commit_verified": True,
        "verified": True,
    }


def test_latest_pair_selects_highest_complete_exact_screen_pair() -> None:
    assets = [
        {"id": 1, "name": f"{ASSET_PREFIX}-epoch-0001.pt"},
        {"id": 2, "name": f"{ASSET_PREFIX}-epoch-0001.json"},
        {"id": 3, "name": f"{ASSET_PREFIX}-epoch-0002.pt"},
        {"id": 4, "name": "itber-v1.1-screen-seed0-p3-epoch-0030.pt"},
        {"id": 5, "name": "itber-v1.1-screen-seed0-p3-epoch-0030.json"},
    ]
    checkpoint, manifest, epoch = select_latest_pair(assets, prefix=ASSET_PREFIX)
    assert (checkpoint["id"], manifest["id"], epoch) == (1, 2, 1)

    with pytest.raises(ValueError, match="1..30"):
        select_latest_pair(
            [
                {"id": 6, "name": f"{ASSET_PREFIX}-epoch-0031.pt"},
                {"id": 7, "name": f"{ASSET_PREFIX}-epoch-0031.json"},
            ],
            prefix=ASSET_PREFIX,
        )


def test_download_verification_requires_identity_bytes_sha_and_remote_receipt(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "download.pt.tmp"
    _checkpoint(checkpoint, 7)
    manifest = _manifest(checkpoint, 7)
    metadata = verify_downloaded_checkpoint(
        checkpoint,
        manifest,
        expected_epoch=7,
        identity=_identity(),
        expected_prefix=ASSET_PREFIX,
    )
    assert metadata.completed_epoch == 7

    for field, value in (
        ("probe", "b2"),
        ("seed", 1),
        ("source_commit", "f" * 40),
        ("result_commit_verified", False),
    ):
        with pytest.raises(RuntimeError, match=field.replace("_", " ")):
            verify_downloaded_checkpoint(
                checkpoint,
                {**manifest, field: value},
                expected_epoch=7,
                identity=_identity(),
                expected_prefix=ASSET_PREFIX,
            )

    corrupted = {**manifest, "checkpoint": {**manifest["checkpoint"], "sha256": "0" * 64}}
    with pytest.raises(RuntimeError, match="SHA-256"):
        verify_downloaded_checkpoint(
            checkpoint,
            corrupted,
            expected_epoch=7,
            identity=_identity(),
            expected_prefix=ASSET_PREFIX,
        )


def test_atomic_install_is_idempotent_and_never_overwrites_changed_destination(
    tmp_path: Path,
) -> None:
    downloaded = tmp_path / ".download.pt.tmp"
    destination = tmp_path / "epoch-0004.pt"
    _checkpoint(downloaded, 4)
    digest = hashlib.sha256(downloaded.read_bytes()).hexdigest()

    result = install_verified_checkpoint(downloaded, destination, expected_sha256=digest)
    assert result == destination.resolve()
    assert destination.is_file()
    assert not downloaded.exists()

    duplicate = tmp_path / ".duplicate.pt.tmp"
    duplicate.write_bytes(destination.read_bytes())
    assert install_verified_checkpoint(duplicate, destination, expected_sha256=digest) == destination.resolve()
    assert not duplicate.exists()

    changed = tmp_path / ".changed.pt.tmp"
    changed.write_bytes(b"changed")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        install_verified_checkpoint(
            changed,
            destination,
            expected_sha256=hashlib.sha256(changed.read_bytes()).hexdigest(),
        )
    assert destination.read_bytes() != b"changed"
    assert not changed.exists()


def test_atomic_install_keeps_existing_destination_when_replace_fails(
    monkeypatch, tmp_path: Path
) -> None:
    downloaded = tmp_path / ".download.pt.tmp"
    destination = tmp_path / "epoch-0001.pt"
    downloaded.write_bytes(b"new")
    digest = hashlib.sha256(b"new").hexdigest()

    def fail_replace(*_args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        install_verified_checkpoint(downloaded, destination, expected_sha256=digest)
    assert not destination.exists()
    assert not downloaded.exists()


def test_restore_cli_runs_directly_with_frozen_namespace() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/restore_iber_checkpoint.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--token-file" in result.stdout
    assert "--run-dir" in result.stdout
    assert "itber" not in result.stdout.lower()
