from pathlib import Path

import pytest
import torch

from scripts.watch_persistent_dcf_checkpoints import (
    checkpoint_summary,
    preserve_milestone,
)


def _checkpoint(path: Path, *, zero_based_epoch: int) -> None:
    torch.save(
        {
            "epoch": zero_based_epoch,
            "optimizer": {"state": {}, "param_groups": []},
            "scaler": {"scale": 128.0},
            "ema": {"weight": torch.tensor([1.0])},
        },
        path,
    )


def test_checkpoint_summary_maps_zero_based_epoch_to_paper_epoch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "last.pt"
    _checkpoint(source, zero_based_epoch=65)
    assert checkpoint_summary(source)["paper_epoch"] == 66


@pytest.mark.parametrize("missing", ["optimizer", "scaler", "ema"])
def test_checkpoint_summary_requires_resumable_state(
    tmp_path: Path, missing: str
) -> None:
    source = tmp_path / "last.pt"
    payload = {
        "epoch": 65,
        "optimizer": {},
        "scaler": {},
        "ema": {},
    }
    payload[missing] = None
    torch.save(payload, source)
    with pytest.raises(RuntimeError, match=missing):
        checkpoint_summary(source)


def test_preserve_milestone_is_atomic_and_rejects_stale_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "last.pt"
    target = tmp_path / "milestones"
    _checkpoint(source, zero_based_epoch=65)
    saved = preserve_milestone(source, target, expected_paper_epoch=66)
    assert saved.name == "epoch0066.pt"
    assert checkpoint_summary(saved)["paper_epoch"] == 66
    assert not list(target.glob("*.tmp"))

    _checkpoint(source, zero_based_epoch=66)
    with pytest.raises(RuntimeError, match="expected paper epoch 66"):
        preserve_milestone(source, target / "other", expected_paper_epoch=66)
