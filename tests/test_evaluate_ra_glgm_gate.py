from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_ra_glgm_gate import (
    evaluate_formal_report,
    evaluate_gate,
    validate_evaluated_arm,
    validate_formal_report,
    validate_screen_gate_report,
    write_create_only_report,
)
from src.ra_experiment_protocol import (
    BASELINE_PARAMETERS,
    RA_EXPERIMENT_PROTOCOL,
    RA_EXPERIMENT_PROTOCOL_SHA256,
    file_sha256,
)


def _manifest(variant: str, parameters: int, *, stage: str = "screen") -> dict:
    return {
        "format_version": 1,
        "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
        "source": {"git_commit": "a" * 40, "tree_sha256": "B" * 64},
        "run_identity": {
            "source_sha256": "S" * 64,
            "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
            "stage": stage,
            "variant": variant,
            "seed": 0,
            "pair_id": f"paired-{stage}-seed0",
            "run_id": f"{variant}-{stage}-seed0",
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
        "schedule_epochs": 50 if stage == "screen" else 100,
        "cutoff_epoch": 30 if stage == "screen" else None,
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
    stage: str = "screen",
) -> Path:
    run = root / variant
    weights = run / "weights"
    weights.mkdir(parents=True)
    manifest = _manifest(variant, parameters, stage=stage)
    (run / "ra-run.json").write_text(json.dumps(manifest), encoding="utf-8")
    evidence = []
    optimizer = []
    queue = []
    completed_epochs = 30 if stage == "screen" else 100
    tail_epochs = (28, 29, 30) if stage == "screen" else (98, 99, 100)
    for epoch in range(1, completed_epochs + 1):
        checkpoint = weights / f"epoch{epoch - 1}.pt"
        checkpoint.write_bytes(f"checkpoint-{variant}-{epoch}".encode())
        evidence.append(
            {
                "completed_epoch": epoch,
                "variant": variant,
                "stage": stage,
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
                "stage": stage,
                "completed_epoch": epoch,
                "status": "pending",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint),
            }
        )
        optimizer.append(
            {
                "optimizer_attempt": epoch,
                "completed_epoch": epoch,
                "run_id": manifest["run_identity"]["run_id"],
                "variant": variant,
                "stage": stage,
                "recovery_generation": 0,
                "amp_scale_before": 128.0,
                "amp_scale_after": 128.0,
                "amp_step_skipped": False,
                "gradient_norm_finite": True,
                "gradient_norm": 1.0,
                "fdr_gradient_norm": 1.0,
                "ra_glgm_gradient_norm": 1.0 if variant == "ra_glgm" else None,
            }
        )
    (run / "ra-epochs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in evidence), encoding="utf-8"
    )
    (run / "publication-queue.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in queue), encoding="utf-8"
    )
    (run / "optimizer-evidence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in optimizer), encoding="utf-8"
    )
    detailed = []
    for epoch in tail_epochs:
        detailed.append(
            {
                "completed_epoch": epoch,
                "variant": variant,
                "stage": stage,
                "run_id": manifest["run_identity"]["run_id"],
                "evaluator_sha256": "E" * 64,
                "checkpoint": queue[epoch - 1]["checkpoint"],
                "checkpoint_sha256": queue[epoch - 1]["checkpoint_sha256"],
                "model_parameters": parameters,
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
        parameters=BASELINE_PARAMETERS
        + int(RA_EXPERIMENT_PROTOCOL["module"]["private_parameters"]),
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
        parameters=BASELINE_PARAMETERS
        + int(RA_EXPERIMENT_PROTOCOL["module"]["private_parameters"]),
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
    method = _write_arm(
        tmp_path,
        "ra_glgm",
        delta=0.006,
        parameters=BASELINE_PARAMETERS
        + int(RA_EXPERIMENT_PROTOCOL["module"]["private_parameters"]),
    )
    queue_path = method / "publication-queue.jsonl"
    rows = [json.loads(line) for line in queue_path.read_text().splitlines()]
    rows[10]["checkpoint_sha256"] = "0" * 64
    queue_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = evaluate_gate(baseline, method)

    assert report["engineering"]["complete"] is False
    assert report["formal_eligible"] is False
    assert any("SHA256 mismatch" in error for error in report["engineering"]["errors"])


def test_locked_evaluation_cannot_be_copied_across_arms(tmp_path: Path) -> None:
    baseline = _write_arm(tmp_path, "baseline")
    method = _write_arm(
        tmp_path,
        "ra_glgm",
        delta=0.006,
        parameters=BASELINE_PARAMETERS
        + int(RA_EXPERIMENT_PROTOCOL["module"]["private_parameters"]),
    )
    (method / "locked-evaluation.jsonl").write_bytes(
        (baseline / "locked-evaluation.jsonl").read_bytes()
    )

    report = evaluate_gate(baseline, method)

    assert report["engineering"]["complete"] is False
    assert any(
        "run authority mismatch" in error
        for error in report["engineering"]["errors"]
    )


def test_wrong_private_parameter_delta_fails_the_frozen_gate(tmp_path: Path) -> None:
    baseline = _write_arm(tmp_path, "baseline")
    method = _write_arm(
        tmp_path,
        "ra_glgm",
        delta=0.006,
        parameters=BASELINE_PARAMETERS + 890_193,
    )

    report = evaluate_gate(baseline, method)

    assert report["engineering"]["complete"] is True
    assert report["gate"]["checks"]["parameter_increase_at_most_10_percent"] is False
    assert report["formal_eligible"] is False


def test_formal100_report_uses_epoch100_and_tail_three_primary_evidence(
    tmp_path: Path,
) -> None:
    baseline = _write_arm(tmp_path, "baseline", stage="formal")
    method = _write_arm(
        tmp_path,
        "ra_glgm",
        stage="formal",
        delta=0.006,
        class_wins=8,
        parameters=BASELINE_PARAMETERS
        + int(RA_EXPERIMENT_PROTOCOL["module"]["private_parameters"]),
    )

    report = evaluate_formal_report(baseline, method)

    assert report["engineering"]["complete"] is True
    assert report["primary_evidence"] == ["epoch100", "tail3_mean"]
    assert report["metrics"]["epoch100_delta"]["map"] == pytest.approx(0.006)
    assert report["formal_success"] is True


def test_formal_launch_recomputes_and_rejects_a_tampered_screen_gate(
    tmp_path: Path,
) -> None:
    baseline = _write_arm(tmp_path, "baseline")
    method = _write_arm(
        tmp_path,
        "ra_glgm",
        delta=0.006,
        class_wins=8,
        parameters=BASELINE_PARAMETERS
        + int(RA_EXPERIMENT_PROTOCOL["module"]["private_parameters"]),
    )
    output = tmp_path / "screen-gate.json"
    report = evaluate_gate(baseline, method)
    output.write_text(json.dumps(report), encoding="utf-8")
    assert validate_screen_gate_report(
        output, baseline_run=baseline, method_run=method
    )["formal_eligible"] is True

    report["metrics"]["final_delta"]["map"] = 1.0
    output.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="recomputation"):
        validate_screen_gate_report(output, baseline_run=baseline, method_run=method)


def test_existing_evaluation_and_formal_report_are_reused_only_after_recomputation(
    tmp_path: Path,
) -> None:
    baseline = _write_arm(tmp_path, "baseline", stage="formal")
    method = _write_arm(
        tmp_path,
        "ra_glgm",
        stage="formal",
        delta=0.006,
        class_wins=8,
        parameters=BASELINE_PARAMETERS
        + int(RA_EXPERIMENT_PROTOCOL["module"]["private_parameters"]),
    )
    assert validate_evaluated_arm(
        baseline, variant="baseline", stage="formal"
    )["checks"]["locked_evaluation"] is True
    output = tmp_path / "formal-report.json"
    output.write_text(
        json.dumps(evaluate_formal_report(baseline, method)), encoding="utf-8"
    )
    assert validate_formal_report(
        output, baseline_run=baseline, method_run=method
    )["engineering"]["complete"] is True

    report = json.loads(output.read_text(encoding="utf-8"))
    report["formal_success"] = not report["formal_success"]
    output.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="recomputation"):
        validate_formal_report(output, baseline_run=baseline, method_run=method)


def test_gate_report_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    assert write_create_only_report(output, {"formal_eligible": False}) == output.resolve()
    with pytest.raises(FileExistsError):
        write_create_only_report(output, {"formal_eligible": True})
