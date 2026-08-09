from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.evaluate_bpdd_gate import evaluate_gate


def test_cli_can_start_outside_repository_root(tmp_path: Path) -> None:
    """The supervisor invokes the script by path from an operations directory."""

    script = Path(__file__).parents[1] / "scripts" / "evaluate_bpdd_gate.py"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _manifest(variant: str) -> dict:
    return {
        "format_version": 1,
        "protocol_sha256": "P" * 64,
        "source": {"git_commit": "a" * 40, "tree_sha256": "T" * 64},
        "run_identity": {
            "source_sha256": "S" * 64,
            "protocol_sha256": "P" * 64,
            "fdr_protocol_sha256": "F" * 64,
            "initial_state_sha256": "I" * 64,
            "run_id": f"{variant}-screen-seed0-authority",
            "stage": "screen",
            "variant": variant,
            "seed": 0,
        },
        "initial_state": {"path": "/authority/initial-state.pt", "sha256": "I" * 64},
        "data": "/authority/screen-data.yaml",
        "screen_cutoff_epoch": 30,
        "publication_queue": f"/runs/{variant}/publication-queue.jsonl",
    }


def _write_arm(root: Path, variant: str, *, delta: float, ap75_delta: float) -> Path:
    run = root / variant
    run.mkdir()
    evidence = []
    result_rows = []
    for epoch in range(1, 31):
        map_value = 0.10 + epoch * 0.001 + delta
        map75 = 0.06 + epoch * 0.0008 + ap75_delta
        row = {
            "completed_epoch": epoch,
            "variant": variant,
            "stage": "screen",
            "run_id": f"{variant}-screen-seed0-authority",
            "precision": 0.30 + epoch * 0.001,
            "recall": 0.40 + epoch * 0.001,
            "map50": map_value + 0.1,
            "map": map_value,
            "map75": map75,
            "loss_giou": 2.0 - epoch * 0.02,
            "loss_class": 1.5 - epoch * 0.01,
            "loss_bbox": 1.0 - epoch * 0.005,
            "loss_fgl": 0.5 - epoch * 0.003,
            "loss_fgl_aux": 0.3 - epoch * 0.002,
            "loss_bbox_pre": 1.1 - epoch * 0.004,
            "loss_giou_pre": 1.7 - epoch * 0.008,
            "loss_bpdd": 0.02 if variant == "fdr_bpdd" else None,
            "bpdd_active_edge_ratio": 0.25 if variant == "fdr_bpdd" else None,
            "bpdd_mean_reliability": 0.08 if variant == "fdr_bpdd" else None,
            "bpdd_mean_teacher_improvement": 0.03 if variant == "fdr_bpdd" else None,
            "bpdd_mixture_beats_final_ratio": 0.15 if variant == "fdr_bpdd" else None,
            "bpdd_mean_mixture_advantage_over_final": 0.01 if variant == "fdr_bpdd" else None,
            "gradient_norm": 3.0,
            "fdr_gradient_norm": 2.0,
            "gradients_finite": True,
            "cuda_peak_mib": 1000.0,
        }
        evidence.append(row)
        result_rows.append(
            {
                "epoch": epoch,
                "metrics/precision(B)": row["precision"],
                "metrics/recall(B)": row["recall"],
                "metrics/mAP50(B)": row["map50"],
                "metrics/mAP50-95(B)": row["map"],
                "val/giou_loss": row["loss_giou"],
                "val/cls_loss": row["loss_class"],
                "val/l1_loss": row["loss_bbox"],
            }
        )
    (run / "bpdd-epochs.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in evidence),
        encoding="utf-8",
    )
    with (run / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    (run / "bpdd-run.json").write_text(
        json.dumps(_manifest(variant), allow_nan=False), encoding="utf-8"
    )
    return run


def test_positive_paired_screen_with_live_bpdd_evidence_passes(tmp_path: Path) -> None:
    fdr = _write_arm(tmp_path, "fdr", delta=0.0, ap75_delta=0.0)
    bpdd = _write_arm(tmp_path, "fdr_bpdd", delta=0.004, ap75_delta=0.002)

    report = evaluate_gate(fdr, bpdd)

    assert report["engineering"]["complete"] is True
    assert report["engineering"]["checks"]["bpdd_signal_live"] is True
    assert report["metrics"]["final"]["delta"]["map"] == pytest.approx(0.004)
    assert set(report["metrics"]["final"]) == {"epochs", "fdr", "fdr_bpdd", "delta"}
    assert report["gate"]["passed"] is True
    assert report["formal_eligible"] is True


def test_zero_or_missing_bpdd_activity_fails_closed(tmp_path: Path) -> None:
    fdr = _write_arm(tmp_path, "fdr", delta=0.0, ap75_delta=0.0)
    bpdd = _write_arm(tmp_path, "fdr_bpdd", delta=0.004, ap75_delta=0.002)
    path = bpdd / "bpdd-epochs.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["bpdd_active_edge_ratio"] = 0.0
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = evaluate_gate(fdr, bpdd)

    assert report["engineering"]["checks"]["bpdd_signal_live"] is False
    assert report["formal_eligible"] is False
