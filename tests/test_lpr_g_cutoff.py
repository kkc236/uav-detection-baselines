from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.lpr_g_cutoff import materialize_cutoff_view


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _overshot_run(path: Path, *, ledger_sha: str | None = None) -> Path:
    path.mkdir()
    weights = path / "weights"
    weights.mkdir()
    checkpoint = weights / "epoch2.pt"
    torch.save(
        {
            "epoch": 2,
            "updates": 3,
            "optimizer": {"state": {}, "param_groups": []},
            "ema": {"weights": torch.ones(1)},
        },
        checkpoint,
    )
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with (path / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("epoch", "metrics/mAP50-95(B)", "metrics/mAP50(B)"),
        )
        writer.writeheader()
        for epoch in range(1, 5):
            writer.writerow(
                {
                    "epoch": epoch,
                    "metrics/mAP50-95(B)": epoch / 100,
                    "metrics/mAP50(B)": epoch / 50,
                }
            )
    _jsonl(
        path / "lpr_g_diagnostics.jsonl",
        [{"epoch": epoch, "map75": epoch / 200} for epoch in range(1, 5)],
    )
    _jsonl(
        path / "common_state_audit.jsonl",
        [
            {
                "epoch": epoch,
                "common_model_sha256": f"{epoch:064x}",
                "common_optimizer_sha256": f"{epoch + 100:064x}",
            }
            for epoch in range(1, 5)
        ],
    )
    ledger = []
    for epoch in range(1, 5):
        sha = checkpoint_sha if epoch == 3 else f"{epoch:064x}"
        ledger.append(
            {
                "completed_epoch": epoch,
                "verified": True,
                "checkpoint": {"sha256": ledger_sha or sha},
            }
        )
    _jsonl(path / "publication-ledger.jsonl", ledger)
    _jsonl(
        path / "optimizer-evidence.jsonl",
        [
            {
                "optimizer_attempt": attempt,
                "amp_scale_before": 128.0,
                "amp_scale_after": 128.0,
                "amp_step_skipped": False,
                "gradient_norm_finite": True,
            }
            for attempt in range(1, 5)
        ],
    )
    (path / "lpr_g_protocol.json").write_text(
        json.dumps({"variant": "control", "stage": "screen", "seed": 0, "epochs": 50}),
        encoding="utf-8",
    )
    return path


def _epochs(path: Path, field: str) -> list[int]:
    return [
        int(json.loads(line)[field])
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_materializes_exact_cutoff_without_changing_overshot_source(tmp_path: Path) -> None:
    source = _overshot_run(tmp_path / "control-overshoot")
    destination = tmp_path / "control"

    manifest = materialize_cutoff_view(source, destination, cutoff_epoch=3)

    assert _epochs(source / "publication-ledger.jsonl", "completed_epoch") == [1, 2, 3, 4]
    assert _epochs(destination / "publication-ledger.jsonl", "completed_epoch") == [1, 2, 3]
    assert _epochs(destination / "lpr_g_diagnostics.jsonl", "epoch") == [1, 2, 3]
    assert _epochs(destination / "common_state_audit.jsonl", "epoch") == [1, 2, 3]
    assert _epochs(destination / "optimizer-evidence.jsonl", "optimizer_attempt") == [1, 2, 3]
    with (destination / "results.csv").open(newline="", encoding="utf-8") as stream:
        assert [int(row["epoch"]) for row in csv.DictReader(stream)] == [1, 2, 3]
    assert (destination / "weights" / "epoch2.pt").is_file()
    assert manifest["source_ledger_epochs"] == [1, 2, 3, 4]
    assert manifest["cutoff_epoch"] == 3
    assert manifest["checkpoint"]["optimizer_updates"] == 3


def test_cutoff_view_rejects_changed_existing_destination(tmp_path: Path) -> None:
    source = _overshot_run(tmp_path / "control-overshoot")
    destination = tmp_path / "control"
    materialize_cutoff_view(source, destination, cutoff_epoch=3)
    (destination / "results.csv").write_text("changed\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="changed cutoff view"):
        materialize_cutoff_view(source, destination, cutoff_epoch=3)


def test_cutoff_view_rejects_checkpoint_not_matching_verified_ledger(tmp_path: Path) -> None:
    source = _overshot_run(tmp_path / "control-overshoot", ledger_sha="f" * 64)

    with pytest.raises(ValueError, match="checkpoint SHA256"):
        materialize_cutoff_view(source, tmp_path / "control", cutoff_epoch=3)
