from pathlib import Path

import torch

from src.checkpoint_recovery import find_resume_checkpoint, validate_checkpoint


def save_checkpoint(path: Path, epoch: int) -> None:
    torch.save(
        {
            "epoch": epoch,
            "optimizer": {"state": {}, "param_groups": []},
            "ema": {"weights": torch.ones(1)},
        },
        path,
    )


def test_valid_checkpoint_can_be_resumed(tmp_path: Path):
    checkpoint = tmp_path / "last.pt"
    save_checkpoint(checkpoint, epoch=3)

    valid, reason = validate_checkpoint(checkpoint)

    assert valid is True
    assert reason == "epoch=3"


def test_corrupt_last_checkpoint_falls_back_to_latest_epoch_snapshot(tmp_path: Path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "last.pt").write_bytes(b"interrupted checkpoint")
    save_checkpoint(weights / "epoch3.pt", epoch=3)
    save_checkpoint(weights / "epoch5.pt", epoch=5)

    selected = find_resume_checkpoint(tmp_path)

    assert selected == (weights / "epoch5.pt").resolve()


def test_completed_stripped_checkpoint_is_not_selected(tmp_path: Path):
    weights = tmp_path / "weights"
    weights.mkdir()
    torch.save({"epoch": 99, "optimizer": None, "ema": None, "model": {}}, weights / "last.pt")

    assert find_resume_checkpoint(tmp_path) is None
