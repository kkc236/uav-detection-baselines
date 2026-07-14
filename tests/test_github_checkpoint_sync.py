from pathlib import Path

import torch

from src.github_checkpoint_sync import (
    assets_to_delete,
    build_manifest,
    checkpoint_asset_name,
    checkpoint_metadata,
    matching_checkpoint_assets,
)


def save_resumable_checkpoint(path: Path, raw_epoch: int) -> None:
    torch.save(
        {
            "epoch": raw_epoch,
            "optimizer": {"state": {}, "param_groups": []},
            "ema": {"weights": torch.ones(1)},
        },
        path,
    )


def test_checkpoint_metadata_uses_human_completed_epoch_and_sha256(tmp_path: Path):
    checkpoint = tmp_path / "last.pt"
    save_resumable_checkpoint(checkpoint, raw_epoch=0)

    metadata = checkpoint_metadata(checkpoint)

    assert metadata.completed_epoch == 1
    assert metadata.bytes == checkpoint.stat().st_size
    assert len(metadata.sha256) == 64
    assert metadata.source == checkpoint.resolve()
    assert checkpoint_asset_name(metadata.completed_epoch) == "btdse-last-epoch-0001.pt"


def test_matching_assets_are_sorted_by_completed_epoch():
    assets = [
        {"id": 9, "name": "notes.txt", "size": 1},
        {"id": 3, "name": "btdse-last-epoch-0003.pt", "size": 30},
        {"id": 1, "name": "btdse-last-epoch-0001.pt", "size": 10},
        {"id": 2, "name": "btdse-last-epoch-0002.pt", "size": 20},
    ]

    matched = matching_checkpoint_assets(assets)

    assert [asset["id"] for asset in matched] == [1, 2, 3]


def test_retention_keeps_newest_three_checkpoint_assets():
    assets = [
        {"id": epoch, "name": checkpoint_asset_name(epoch), "size": epoch * 10}
        for epoch in range(1, 6)
    ]

    expired = assets_to_delete(assets, retain=3)

    assert [asset["id"] for asset in expired] == [1, 2]


def test_manifest_records_remote_asset_and_local_integrity(tmp_path: Path):
    checkpoint = tmp_path / "epoch4.pt"
    save_resumable_checkpoint(checkpoint, raw_epoch=4)
    metadata = checkpoint_metadata(checkpoint)

    manifest = build_manifest(
        metadata,
        asset={"id": 55, "name": checkpoint_asset_name(5), "size": metadata.bytes},
        release_url="https://github.com/example/repo/releases/tag/live",
    )

    assert manifest["completed_epoch"] == 5
    assert manifest["checkpoint"]["asset_name"] == "btdse-last-epoch-0005.pt"
    assert manifest["checkpoint"]["sha256"] == metadata.sha256
    assert manifest["checkpoint"]["bytes"] == metadata.bytes
    assert manifest["release_url"].endswith("/live")
