from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.restore_lpr_g_checkpoint import (
    select_latest_pair,
    verify_downloaded_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def _checkpoint(path: Path, completed_epoch: int) -> None:
    torch.save(
        {
            "epoch": completed_epoch - 1,
            "optimizer": {"state": {}, "param_groups": []},
            "ema": {"weights": torch.ones(1)},
        },
        path,
    )


def test_latest_pair_is_exact_prefix_and_ignores_other_arm() -> None:
    assets = [
        {"name": "screen-seed0-lprg-epoch-0030.pt", "id": 1},
        {"name": "screen-seed0-lprg-epoch-0030.json", "id": 2},
        {"name": "screen-seed0-control-epoch-0030.pt", "id": 3},
        {"name": "screen-seed0-control-epoch-0030.json", "id": 4},
        {"name": "screen-seed0-lprg-epoch-0029.pt", "id": 5},
        {"name": "screen-seed0-lprg-epoch-0029.json", "id": 6},
    ]

    checkpoint, manifest, epoch = select_latest_pair(
        assets, prefix="screen-seed0-control"
    )

    assert checkpoint["id"] == 3
    assert manifest["id"] == 4
    assert epoch == 30


def test_download_validation_checks_integrity_and_protocol(tmp_path: Path) -> None:
    checkpoint = tmp_path / "download.pt.tmp"
    _checkpoint(checkpoint, completed_epoch=7)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "completed_epoch": 7,
        "design_version": "lpr-g-v2",
        "variant": "lprg",
        "stage": "screen",
        "seed": 0,
        "checkpoint": {
            "asset_name": "screen-seed0-lprg-epoch-0007.pt",
            "bytes": checkpoint.stat().st_size,
            "sha256": digest,
        },
    }

    metadata = verify_downloaded_checkpoint(
        checkpoint,
        manifest,
        expected_epoch=7,
        expected_variant="lprg",
        expected_stage="screen",
        expected_prefix="screen-seed0-lprg",
    )

    assert metadata.completed_epoch == 7
    with pytest.raises(RuntimeError, match="variant"):
        verify_downloaded_checkpoint(
            checkpoint,
            {**manifest, "variant": "control"},
            expected_epoch=7,
            expected_variant="lprg",
            expected_stage="screen",
            expected_prefix="screen-seed0-lprg",
        )


def test_restore_script_runs_as_direct_cli() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "restore_lpr_g_checkpoint.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--variant" in result.stdout
    assert "--asset-prefix" in result.stdout
