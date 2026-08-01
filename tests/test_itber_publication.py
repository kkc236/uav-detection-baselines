from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import src.itber_publication as publication
from src.itber_protocol import EXPECTED_BASELINE_SHA256, EXPECTED_DATASET_SHA256
from src.itber_publication import (
    PublicationIdentity,
    PublicationLedger,
    pending_epoch_checkpoints,
    private_checkpoint_metadata,
    publish_with_retry,
    read_token_file,
)


CACHE_SHA = "C" * 64


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
            "refiner": {"weight": torch.ones(1)},
            "optimizer": {"state": {}, "param_groups": []},
            "scaler": {"scale": 128.0},
            "rng": {"torch": torch.get_rng_state()},
        },
        path,
    )


def _identity(stage: str = "screen") -> PublicationIdentity:
    return PublicationIdentity(
        design_version="itber-v1.1",
        stage=stage,
        probe="p3",
        seed=0,
        baseline_sha256=EXPECTED_BASELINE_SHA256,
        dataset_sha256=EXPECTED_DATASET_SHA256,
        cache_manifest_sha256=CACHE_SHA,
    )


def _record(epoch: int, identity: PublicationIdentity | None = None) -> dict:
    return {
        **(identity or _identity()).as_dict(),
        "completed_epoch": epoch,
        "checkpoint": {"bytes": 10, "sha256": f"{epoch:064x}"},
        "verified": True,
    }


def test_private_checkpoint_metadata_uses_private_one_based_epoch(tmp_path: Path) -> None:
    path = tmp_path / "epoch-0001.pt"
    _checkpoint(path, 1)

    metadata = private_checkpoint_metadata(path, identity=_identity())

    assert metadata.completed_epoch == 1
    assert metadata.bytes == path.stat().st_size
    assert len(metadata.sha256) == 64


@pytest.mark.parametrize("field", ["refiner", "optimizer", "scaler", "rng"])
def test_private_checkpoint_rejects_stripped_recovery_state(tmp_path: Path, field: str) -> None:
    path = tmp_path / "bad.pt"
    _checkpoint(path, 1)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    artifact.pop(field)
    torch.save(artifact, path)

    with pytest.raises(ValueError, match=field):
        private_checkpoint_metadata(path, identity=_identity())


def test_ledger_requires_contiguous_verified_exact_identity(tmp_path: Path) -> None:
    ledger = PublicationLedger(tmp_path / "ledger.jsonl", _identity())
    ledger.append_verified(_record(1))
    ledger.append_verified(_record(1))

    with pytest.raises(ValueError, match="gap"):
        ledger.append_verified(_record(3))
    with pytest.raises(ValueError, match="identity"):
        ledger.append_verified(_record(2, _identity("formal")))
    with pytest.raises(ValueError, match="verified"):
        ledger.append_verified({**_record(2), "verified": False})
    assert ledger.last_completed_epoch == 1


def test_pending_private_checkpoints_are_contiguous(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    _checkpoint(checkpoint_root / "epoch-0001.pt", 1)
    _checkpoint(checkpoint_root / "epoch-0002.pt", 2)
    ledger = PublicationLedger(tmp_path / "ledger.jsonl", _identity())

    pending = pending_epoch_checkpoints(checkpoint_root, ledger, identity=_identity())
    assert [epoch for epoch, _ in pending] == [1, 2]

    (checkpoint_root / "epoch-0002.pt").rename(checkpoint_root / "epoch-0003.pt")
    artifact = torch.load(checkpoint_root / "epoch-0003.pt", map_location="cpu", weights_only=False)
    artifact["epoch"] = 3
    torch.save(artifact, checkpoint_root / "epoch-0003.pt")
    with pytest.raises(RuntimeError, match="missing completed epoch 2"):
        pending_epoch_checkpoints(checkpoint_root, ledger, identity=_identity())


def test_token_file_requires_mode_600_without_echoing_token(tmp_path: Path) -> None:
    token_file = tmp_path / "github-token"
    secret = "not-for-error-output"
    token_file.write_text(secret, encoding="utf-8")
    if os.name != "nt":
        token_file.chmod(0o644)
        with pytest.raises(PermissionError) as error:
            read_token_file(token_file)
        assert secret not in str(error.value)
        token_file.chmod(0o600)
    assert read_token_file(token_file) == secret


def test_retry_exhaustion_does_not_include_credentials(monkeypatch, tmp_path: Path) -> None:
    attempts = []

    def fail(*_args, **_kwargs):
        attempts.append(1)
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(publication, "publish_exact_epoch", fail)
    with pytest.raises(RuntimeError, match="failed after 3 attempts") as error:
        publish_with_retry(tmp_path, tmp_path / "checkpoint.pt", object(), attempts=3, delay=0)
    assert len(attempts) == 3
    assert "token" not in str(error.value).lower()
