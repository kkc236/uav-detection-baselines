from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.restore_vsf_rmr_checkpoint import select_latest_pair, verify_downloaded_checkpoint
from scripts.sync_experiment_checkpoint import prune_local_epoch_checkpoints


def test_latest_pair_matches_prefix_epoch_and_ignores_other_innovations():
    assets = [
        {"name": "btdse-last-epoch-0099.pt", "id": 1},
        {"name": "ioqc-sa-last-epoch-0099.pt", "id": 2},
        {"name": "vsf-rmr-last-epoch-0007.pt", "id": 3},
        {"name": "vsf-rmr-last-epoch-0007.json", "id": 4},
        {"name": "vsf-rmr-last-epoch-0012.pt", "id": 5},
        {"name": "vsf-rmr-last-epoch-0012.json", "id": 6},
    ]

    checkpoint, manifest, epoch = select_latest_pair(assets, prefix="vsf-rmr-last")

    assert checkpoint["id"] == 5
    assert manifest["id"] == 6
    assert epoch == 12


def test_latest_pair_rejects_checkpoint_without_matching_manifest():
    assets = [{"name": "vsf-rmr-last-epoch-0003.pt", "id": 1}]

    with pytest.raises(FileNotFoundError, match="matching checkpoint and manifest"):
        select_latest_pair(assets, prefix="vsf-rmr-last")


def _save_checkpoint(path: Path, epoch: int) -> None:
    torch.save(
        {"epoch": epoch, "optimizer": {"state": {}, "param_groups": []}, "ema": {"weights": torch.ones(1)}},
        path,
    )


def test_download_validation_checks_epoch_size_and_sha256(tmp_path: Path):
    checkpoint = tmp_path / "download.pt.tmp"
    _save_checkpoint(checkpoint, epoch=4)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "completed_epoch": 5,
        "checkpoint": {"bytes": checkpoint.stat().st_size, "sha256": digest},
    }

    metadata = verify_downloaded_checkpoint(checkpoint, manifest, expected_epoch=5)

    assert metadata.completed_epoch == 5
    bad = json.loads(json.dumps(manifest))
    bad["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="SHA-256"):
        verify_downloaded_checkpoint(checkpoint, bad, expected_epoch=5)


def test_local_pruning_keeps_latest_three_epoch_files_and_last_best(tmp_path: Path):
    weights = tmp_path / "weights"
    weights.mkdir()
    for epoch in range(1, 7):
        _save_checkpoint(weights / f"epoch{epoch}.pt", epoch=epoch - 1)
    _save_checkpoint(weights / "last.pt", epoch=5)
    _save_checkpoint(weights / "best.pt", epoch=4)

    removed = prune_local_epoch_checkpoints(weights, retain=3)

    assert {path.name for path in removed} == {"epoch1.pt", "epoch2.pt", "epoch3.pt"}
    assert {path.name for path in weights.glob("*.pt")} == {
        "epoch4.pt",
        "epoch5.pt",
        "epoch6.pt",
        "last.pt",
        "best.pt",
    }

