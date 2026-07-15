from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from scripts.supervise_ioqc_sa import (
    EXIT_PLANNED_RESTART,
    acquire_pid_lock,
    build_child_command,
    classify_child_exit,
    release_pid_lock,
    select_resume_checkpoint,
)


@pytest.mark.parametrize(
    ("returncode", "output", "expected"),
    [
        (EXIT_PLANNED_RESTART, "epoch checkpoint saved", "planned_restart"),
        (1, "torch.OutOfMemoryError: CUDA out of memory", "oom"),
        (1, "FloatingPointError: NONFINITE_LOSS total", "numeric_failure"),
        (0, "100 epochs completed", "success"),
        (2, "network filesystem unavailable", "failure"),
    ],
)
def test_child_exit_classification(returncode: int, output: str, expected: str):
    assert classify_child_exit(returncode, output) == expected


def test_child_command_carries_batch_amp_paths_and_optional_resume(tmp_path: Path):
    command = build_child_command(
        python_executable="python",
        train_script=Path("scripts/train_rtdetr_ioqc_sa.py"),
        project=tmp_path / "runs",
        run_name="ioqc-run",
        state_path=tmp_path / "state.json",
        batch=6,
        amp_enabled=False,
        epochs=100,
        workers=8,
        device="0",
        resume=tmp_path / "last.pt",
    )

    assert command[0] == "python"
    assert Path(command[1]) == Path("scripts/train_rtdetr_ioqc_sa.py")
    assert command[command.index("--batch") + 1] == "6"
    assert command[command.index("--amp") + 1] == "false"
    assert command[command.index("--state") + 1].endswith("state.json")
    assert command[command.index("--resume") + 1].endswith("last.pt")


def test_stale_pid_lock_is_replaced_but_live_lock_is_rejected(tmp_path: Path):
    lock = tmp_path / "supervisor.pid"
    lock.write_text("99999999", encoding="ascii")

    acquire_pid_lock(lock)
    assert int(lock.read_text(encoding="ascii")) == os.getpid()
    with pytest.raises(RuntimeError, match="already running"):
        acquire_pid_lock(lock)
    release_pid_lock(lock)
    assert not lock.exists()


def _save_checkpoint(path: Path, epoch: int) -> None:
    torch.save(
        {"epoch": epoch, "optimizer": {"state": {}, "param_groups": []}, "ema": {"weights": torch.ones(1)}},
        path,
    )


def test_resume_selection_falls_back_from_corrupt_last_to_latest_epoch(tmp_path: Path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "last.pt").write_bytes(b"corrupt")
    _save_checkpoint(weights / "epoch2.pt", 2)
    _save_checkpoint(weights / "epoch4.pt", 4)

    selected = select_resume_checkpoint(tmp_path)

    assert selected == (weights / "epoch4.pt").resolve()
