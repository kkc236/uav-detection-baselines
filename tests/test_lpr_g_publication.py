from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.lpr_g_publication import PublicationLedger, pending_epoch_checkpoints


def _checkpoint(path: Path, completed_epoch: int) -> None:
    torch.save(
        {"epoch": completed_epoch - 1, "optimizer": {}, "ema": {"x": torch.ones(1)}},
        path,
    )


def test_pending_checkpoints_are_contiguous_and_zero_based_filenames_are_not_authority(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    _checkpoint(weights / "epoch0.pt", 1)
    _checkpoint(weights / "epoch1.pt", 2)
    ledger = PublicationLedger(tmp_path / "publication-ledger.jsonl")

    assert [epoch for epoch, _ in pending_epoch_checkpoints(weights, ledger)] == [1, 2]


def test_gap_is_rejected_before_later_epoch_can_publish(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    _checkpoint(weights / "epoch0.pt", 1)
    _checkpoint(weights / "epoch2.pt", 3)
    ledger = PublicationLedger(tmp_path / "publication-ledger.jsonl")

    with pytest.raises(RuntimeError, match="missing completed epoch 2"):
        pending_epoch_checkpoints(weights, ledger)


def test_ledger_is_idempotent_and_rejects_changed_sha(tmp_path: Path) -> None:
    ledger = PublicationLedger(tmp_path / "publication-ledger.jsonl")
    record = {
        "completed_epoch": 1,
        "checkpoint": {"sha256": "a" * 64},
        "verified": True,
    }
    ledger.append_verified(record)
    ledger.append_verified(record)

    with pytest.raises(ValueError, match="changed publication"):
        ledger.append_verified(
            {**record, "checkpoint": {"sha256": "b" * 64}}
        )
    assert len(ledger.records()) == 1


def test_ledger_rejects_unverified_and_noncontiguous_records(tmp_path: Path) -> None:
    ledger = PublicationLedger(tmp_path / "publication-ledger.jsonl")
    with pytest.raises(ValueError, match="not remotely verified"):
        ledger.append_verified({"completed_epoch": 1, "verified": False})
    with pytest.raises(ValueError, match="gap"):
        ledger.append_verified({"completed_epoch": 2, "verified": True})
