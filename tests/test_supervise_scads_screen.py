from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.supervise_scads_screen import (
    EXPECTED_EPOCHS,
    arm_evidence,
    scads_command,
    verified_epochs,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_arm_evidence_requires_continuous_authorized_epochs(tmp_path: Path) -> None:
    run = tmp_path / "run"
    rows = [
        {
            "completed_epoch": epoch,
            "variant": "fdr",
            "stage": "screen",
            "run_id": "fdr-screen",
        }
        for epoch in range(1, EXPECTED_EPOCHS + 1)
    ]
    _write_jsonl(run / "scads-epochs.jsonl", rows)
    assert arm_evidence(run, "fdr") == rows
    rows[4]["completed_epoch"] = 9
    _write_jsonl(run / "scads-epochs.jsonl", rows)
    with pytest.raises(ValueError, match="not continuous"):
        arm_evidence(run, "fdr")


def test_verified_epochs_filters_run_and_status_and_rejects_duplicates(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(
        ledger,
        [
            {"run_id": "run", "completed_epoch": 1, "status": "published-verified"},
            {"run_id": "other", "completed_epoch": 1, "status": "published-verified"},
            {"run_id": "run", "completed_epoch": 2, "status": "pending"},
        ],
    )
    assert verified_epochs(ledger, run_id="run") == [1]
    rows = json.loads(ledger.read_text().splitlines()[0])
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(rows) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        verified_epochs(ledger, run_id="run")


def test_scads_command_exposes_no_training_hyperparameter_override(tmp_path: Path) -> None:
    args = argparse.Namespace(
        training_root=tmp_path / "source",
        experiment_root=tmp_path / "experiment",
        dataset_root=tmp_path / "VisDrone",
        python=tmp_path / "venv" / "python",
    )
    command = scads_command(args)
    assert command[2].endswith("train_rtdetr_scads.py")
    assert command[command.index("--variant") + 1] == "scads"
    assert command[command.index("--stage") + 1] == "screen"
    for forbidden in ("--epochs", "--batch", "--imgsz", "--seed", "--lr0"):
        assert forbidden not in command
