import os
import time
from pathlib import Path

from scripts.sync_acr_eg_checkpoints import (
    build_parser,
    checkpoint_is_stable,
    checkpoint_path_for_epoch,
    observe_checkpoint,
)


def test_sync_cli_freezes_formal_epoch_range_and_repository(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--token-file",
            str(tmp_path / "token"),
            "--source-commit",
            "a" * 40,
        ]
    )

    assert args.repo == "kkc236/uav-detection-baselines"
    assert args.source_branch == "codex/gcte-rtdetr-g0"
    assert args.results_branch == "training-results"
    assert args.start_epoch == 10
    assert args.end_epoch == 100
    assert args.interval == 60


def test_human_epoch_maps_to_ultralytics_epoch_checkpoint(tmp_path: Path) -> None:
    assert checkpoint_path_for_epoch(tmp_path, 10) == tmp_path / "weights" / "epoch9.pt"
    assert checkpoint_path_for_epoch(tmp_path, 100) == tmp_path / "weights" / "epoch99.pt"


def test_checkpoint_requires_unchanged_observation_and_minimum_age(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch9.pt"
    checkpoint.write_bytes(b"complete checkpoint")
    old = time.time() - 120
    os.utime(checkpoint, (old, old))

    first = observe_checkpoint(checkpoint)
    stable, second = checkpoint_is_stable(
        checkpoint,
        previous=first,
        stable_seconds=30,
    )

    assert stable is True
    assert second == first

    checkpoint.write_bytes(b"checkpoint is still changing")
    stable, changed = checkpoint_is_stable(
        checkpoint,
        previous=first,
        stable_seconds=0,
    )
    assert stable is False
    assert changed != first
