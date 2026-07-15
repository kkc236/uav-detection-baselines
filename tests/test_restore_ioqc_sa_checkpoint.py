from __future__ import annotations

from scripts.restore_ioqc_sa_checkpoint import select_latest_asset


def test_select_latest_asset_uses_ioqc_prefix_and_highest_epoch():
    assets = [
        {"name": "btdse-last-epoch-0099.pt", "id": 1},
        {"name": "ioqc-sa-last-epoch-0007.pt", "id": 2},
        {"name": "ioqc-sa-last-epoch-0012.pt", "id": 3},
        {"name": "notes.txt", "id": 4},
    ]

    selected, epoch = select_latest_asset(assets, prefix="ioqc-sa-last")

    assert selected["id"] == 3
    assert epoch == 12
