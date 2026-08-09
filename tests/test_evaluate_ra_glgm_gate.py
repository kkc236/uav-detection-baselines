from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_ra_glgm_gate import evaluate_gate, write_create_only_report
from src.ra_experiment_protocol import (
    BASELINE_PARAMETERS,
    RA_EXPERIMENT_PROTOCOL_SHA256,
    file_sha256,
)


def _manifest(variant: str, parameters: int) -> dict:
    return {
        "format_version": 1,
        "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
        "source": {"git_commit": "a" * 40, "tree_sha256": "B" * 64},
        "run_identity": {
            "source_sha256": "S" * 64,
            "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
            "stage": "screen",
            "variant": variant,
            "seed": 0,
            "pair_id": "paired-screen-seed0",
            "run_id": f"{variant}-screen-seed0",
        },
        "initial_state": {"path": "/authority/initial.pt", "sha256": "I" * 64},
        "data": "/authority/screen.yaml",
        "dataset_authority": {
            "root": "/authority/VisDrone",
            "positive": {"sha256": "D" * 64},
            "ignore": {"sha256": "G" * 64},
        },
        "learnability_report_sha256": "L" * 64,
        "gpu_uuid": "GPU-fixed",
        "schedule_epochs": 50,
        "cutoff_epoch": 30,
        "locked_evaluator_sha256": "E" * 64,
        "model_parameters": parameters,
    }


def _write_arm(
    root: Path,
    variant: str,
    *,
    delta: float = 0.0,
    class_wins: int = 10,
    parameters: int = BASELINE_PARAMETERS,
) -> Path:
    run = root / variant
    weights = run / "weights"
    weights.mkdir(parents=True)
    manifest = _manifest(variant, parameters)
    (run / "ra-run.json").write_text(json.dumps(manifest), encoding="utf-8")
    evidence = []
    queue = []
    for epoch in range(1, 31):
        checkpoint = weights / f"epoch{epoch - 1}.pt"
        checkpoint.write_bytes(f"checkpoint-{variant}-{epoch}".encode())
        evidence.append(
            {
                "completed_epoch": epoch,
                "variant": variant,
                "stage": "screen",
                "run_id": manifest["run_identity"]["run_id"],
                "map": 0.10 + epoch / 1000 + delta,
                "map50": 0.20 + epoch / 1000 + delta,
                "map75": 0.08 + epoch / 1000 + delta,
                "precision": 0.30 + epoch / 1000 + delta,
                "recall": 0.40 + epoch / 1000 + delta,
                "cuda_peak_mib": 15_000 + epoch,
            }
        )
        queue.append(
            {
                "run_id": manifest["run_identity"]["run_id"],
                "variant": variant,
                "stage": "screen",
                "completed_epoch": epoch,
                "status": "pending",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint),
            }
        )
    (run / "ra-epochs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in evidence), encoding="utf-8"
    )
    (run / "publication-queue.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in queue), encoding="utf-8"
    )
    detailed = []
    for epoch in (28, 29, 30):
        detailed.append(
            {
                "completed_epoch": epoch,
                "evaluator_sha256": "E" * 64,
                "map": 0.20 + epoch / 1000 + delta,
                "map50": 0.30 + epoch / 1000 + delta,
                "map75": 0.15 + epoch / 1000 + delta,
                "precision": 0.35 + epoch / 1000 + delta,
                "recall": 0.45 + epoch / 1000 + delta,
                "ap_tiny": 0.05 + epoch / 1000 + delta,
                "ap_small": 0.10 + epoch / 1000 + delta,
                "class_ap": [
                    0.10 + index / 100 + (delta if index < class_wins else -delta)
                    for index in range(10)
                ],
            }
        )
    (run / "locked-evaluation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in detailed), encoding="utf-8"
    )
    return run


def test_complete_positive_screen_pair_passes_all_frozen_gates(tmp_path: Path) -> None:
    baseline = _write_arm(tmp_path, "baseline")
    method = _write_arm(
        tmp_path,
        "ra_glgm",
        delta=0.006,
        class_wins=8,
        parameters=BASELINE_PARAMETERS + 890_193,
    )

    report = evaluate_gate(baseline, method)

    assert report["engineering"]["complete"] is True
    assert report["formal_eligible"] is True
    assert all(report["gate"]["checks"].values())
    assert report["metrics"]["final_delta"]["map"] == pytest.approx(0.006)
    assert report["metrics"]["class_wins"] == 8


def test_0_5_pp_boundary_passes_but_missing_small_or_class_wins_fails(tmp_path: Path) -> None:
    baseline = _write_arm(tmp_path, "baseline")
    method = _write_arm(
        tmp_path,
        "ra_glgm",
        delta=0.005,
        class_wins=6,
        parameters=BASELINE_PARAMETERS + 890_193,
    )
    rows = [json.loads(line) for line in (method / "locked-evaluation.jsonl").read_text().splitlines()]
    rows[-1]["ap_small"] -= 0.005
    (method / "locked-evaluation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = evaluate_gate(baseline, method)

    assert report["gate"]["checks"]["epoch30_map_delta_at_least_0_005"] is True
    assert report["gate"]["checks"]["epoch30_ap_small_delta_positive"] is False
    assert report["gate"]["checks"]["at_least_7_of_10_classes_improve"] is False
    assert report["formal_eligible"] is False


def test_queue_or_checkpoint_drift_is_engineering_failure(tmp_path: Path) -> None:
    baseline = _write_arm(tmp_path, "baseline")
    method = _write_arm(tmp_path, "ra_glgm", delta=0.006, parameters=BASELINE_PARAMETERS + 890_193)
    queue_path = method / "publication-queue.jsonl"
    rows = [json.loads(line) for line in queue_path.read_text().splitlines()]
    rows[10]["checkpoint_sha256"] = "0" * 64
    queue_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = evaluate_gate(baseline, method)

    assert report["engineering"]["complete"] is False
    assert report["formal_eligible"] is False
    assert any("SHA256 mismatch" in error for error in report["engineering"]["errors"])


def test_gate_report_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    assert write_create_only_report(output, {"formal_eligible": False}) == output.resolve()
    with pytest.raises(FileExistsError):
        write_create_only_report(output, {"formal_eligible": True})
