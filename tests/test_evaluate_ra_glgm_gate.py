from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import scripts.evaluate_ra_glgm_gate as gate_module

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
    build_ra_run_identity,
    current_source_identity,
)
from src.fdr_protocol import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
REAL_BUILD_PREDICTION_CONTEXT = gate_module._build_prediction_metric_context


def test_gradient_p99_excludes_recovery_lineage_discarded_attempts() -> None:
    rows = [
        {"optimizer_attempt": 1, "gradient_norm": 1.0},
        {"optimizer_attempt": 2, "gradient_norm": 1000.0},
        {"optimizer_attempt": 3, "gradient_norm": 2.0},
    ]
    optimizer = {"discarded_optimizer_attempt_numbers": [2]}

    assert gate_module._active_gradient_p99(rows, optimizer) == pytest.approx(1.99)


@pytest.fixture(autouse=True)
def _stub_expensive_prediction_rederivation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate_module, "_build_prediction_metric_context", lambda *_a, **_k: ())
    monkeypatch.setattr(
        gate_module,
        "_recompute_prediction_metrics",
        lambda artifact, _context: json.loads(artifact.read_text(encoding="utf-8"))["metrics"],
    )
    monkeypatch.setattr(
        gate_module,
        "_recompute_upstream_report",
        lambda root, *, stage: json.loads(
            (root / "RA_GLGM_SCREEN30_GATE.json").read_text(encoding="utf-8")
        ),
    )


def test_screen30_metric_rederivation_uses_official_val(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluate_ra_glgm_checkpoints as evaluator_module

    dataset_root = tmp_path / "VisDrone"
    observed: dict[str, object] = {}

    def fake_dataset(_data: Path, *, expected_images: int):
        observed["expected_images"] = expected_images
        return dataset_root.resolve(), [], [], (dataset_root / "images" / "val").resolve()

    monkeypatch.setattr(evaluator_module, "_dataset", fake_dataset)
    monkeypatch.setattr(
        evaluator_module,
        "_coco_ground_truth",
        lambda _images, _names, *, expected_objects: (
            observed.setdefault("expected_objects", expected_objects) or {},
            {},
            {},
            {},
        ),
    )
    manifest = {
        "data": str(tmp_path / "screen-data.yaml"),
        "dataset_authority": {
            "root": str(dataset_root.resolve()),
        },
    }

    REAL_BUILD_PREDICTION_CONTEXT(manifest, stage="screen")

    assert observed == {"expected_images": 548, "expected_objects": 38_759}


def _manifest(
    variant: str,
    parameters: int,
    *,
    initial_state: Path,
    upstream_sha256: str | None,
    stage: str = "screen",
) -> dict:
    source = current_source_identity(ROOT)
    pair_id = f"paired-{stage}-seed0"
    return {
        "format_version": 1,
        "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
        "source": source,
        "run_identity": build_ra_run_identity(
            source, stage=stage, variant=variant, seed=0, pair_id=pair_id
        ),
        "initial_state": {
            "path": str(initial_state.resolve()),
            "sha256": file_sha256(initial_state),
        },
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
        "locked_evaluator_sha256": file_sha256(
            ROOT / "scripts" / "evaluate_ra_glgm_checkpoints.py"
        ),
        "initialization_mode": "fresh_paired_scratch",
        "parent_checkpoint": None,
        "screen_gate_sha256": upstream_sha256 if stage == "formal" else None,
        "model_parameters": parameters,
    }


def _upstream_gate(root: Path, stage: str) -> str | None:
    if stage == "formal":
        path = root / "RA_GLGM_SCREEN30_GATE.json"
        payload = {
            "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
            "gate_name": "RA-GLGM-Screen30-v1.2",
            "formal_eligible": True,
            "formal_instruction": "start_fresh_from_paired_scratch_initial_state",
            "engineering": {"complete": True},
            "gate": {"passed": True},
        }
    else:
        return None
    if not path.exists():
        path.write_text(json.dumps(payload), encoding="utf-8")
    return file_sha256(path)


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
    initial_state = root / "initial.pt"
    if not initial_state.exists():
        initial_state.write_bytes(b"paired-initial-state")
    manifest = _manifest(
        variant,
        parameters,
        stage=stage,
        initial_state=initial_state,
        upstream_sha256=_upstream_gate(root, stage),
    )
    (run / "ra-run.json").write_text(json.dumps(manifest), encoding="utf-8")
    evidence = []
    optimizer = []
    queue = []
    completed_epochs = 30 if stage == "screen" else 100
    tail_epochs = (26, 27, 28, 29, 30) if stage == "screen" else (98, 99, 100)
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
                "scale_instances": 100.0,
                "scale_mae": 0.10,
                "scale_rmse": 0.12,
                "scale_prediction_mean": 0.50,
                "scale_prediction_std": 0.20,
                "scale_target_mean": 0.50,
                "scale_target_std": 0.25,
                "scale_pearson": 0.60,
                "scale_spearman": 0.60,
                "route_entropy": 0.69,
                "route_global_mean": 0.50,
                "route_global_std": 0.10,
                "route_load_min": 0.25,
                "route_load_max": 0.75,
                "scale_route_correlation_mean_abs": 0.20,
                "scale_route_correlation_max_abs": 0.30,
                "scale_slope_rms": 0.10,
                "scale_slope_max_abs": 0.20,
                "scale_modulation_route_delta_mean": 0.01,
                "scale_modulation_route_delta_max": 0.03,
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
    previous_row_sha = "0" * 64
    for epoch in tail_epochs:
        prediction = run / "locked-evaluator" / f"epoch{epoch:04d}" / "predictions.json"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        row = {
                "completed_epoch": epoch,
                "variant": variant,
                "stage": stage,
                "run_id": manifest["run_identity"]["run_id"],
                "evaluator_sha256": manifest["locked_evaluator_sha256"],
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
                    0.10
                    + index / 100
                    + (delta if index < class_wins else -min(delta, 0.002))
                    for index in range(10)
                ],
            }
        artifact_metrics = {
            name: row[name]
            for name in ("map", "map50", "map75", "precision", "recall", "ap_tiny", "ap_small", "class_ap")
        }
        prediction.write_text(json.dumps({"metrics": artifact_metrics}), encoding="utf-8")
        row["predictions_artifact"] = {
            "path": str(prediction.resolve()),
            "sha256": file_sha256(prediction),
        }
        row["previous_evaluation_row_sha256"] = previous_row_sha
        row["evaluation_row_sha256"] = hashlib.sha256(
            canonical_json_bytes(row)
        ).hexdigest().upper()
        previous_row_sha = row["evaluation_row_sha256"]
        detailed.append(row)
    (run / "locked-evaluation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in detailed), encoding="utf-8"
    )
    return run


def _rewrite_evaluation_with_valid_chain(
    path: Path, rows: list[dict], *, sync_prediction_artifact: bool = True
) -> None:
    previous = "0" * 64
    for row in rows:
        if sync_prediction_artifact:
            artifact = Path(row["predictions_artifact"]["path"])
            metrics = {
                name: row[name]
                for name in ("map", "map50", "map75", "precision", "recall", "ap_tiny", "ap_small", "class_ap")
            }
            artifact.write_text(json.dumps({"metrics": metrics}), encoding="utf-8")
            row["predictions_artifact"]["sha256"] = file_sha256(artifact)
        row["previous_evaluation_row_sha256"] = previous
        row.pop("evaluation_row_sha256", None)
        row["evaluation_row_sha256"] = hashlib.sha256(
            canonical_json_bytes(row)
        ).hexdigest().upper()
        previous = row["evaluation_row_sha256"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


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
    assert report["metrics"]["class_non_decreasing"] == 8


def test_accurate_scale_head_with_inactive_scale_router_fails_closed(
    tmp_path: Path,
) -> None:
    baseline = _write_arm(tmp_path, "baseline")
    method = _write_arm(
        tmp_path,
        "ra_glgm",
        delta=0.006,
        parameters=BASELINE_PARAMETERS
        + int(RA_EXPERIMENT_PROTOCOL["module"]["private_parameters"]),
    )
    evidence_path = method / "ra-epochs.jsonl"
    rows = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    for row in rows:
        row["scale_slope_rms"] = 0.0
        row["scale_slope_max_abs"] = 0.0
        row["scale_modulation_route_delta_mean"] = 0.0
        row["scale_modulation_route_delta_max"] = 0.0
    evidence_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = evaluate_gate(baseline, method)

    assert report["engineering"]["complete"] is True
    assert report["gate"]["checks"]["tail5_scale_pearson_at_least_0_40"] is True
    assert report["gate"]["checks"]["scale_slopes_nontrivial"] is False
    assert report["gate"]["checks"]["scale_modulation_changes_routes"] is False
    assert report["formal_eligible"] is False


def test_tail5_small_and_class_coverage_fail_closed(tmp_path: Path) -> None:
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
    for row in rows:
        row["ap_small"] -= 0.005
    _rewrite_evaluation_with_valid_chain(method / "locked-evaluation.jsonl", rows)

    report = evaluate_gate(baseline, method)

    assert report["gate"]["checks"]["tail5_map_delta_at_least_0_002"] is True
    assert report["gate"]["checks"]["tail5_ap_small_delta_at_least_0_0015"] is False
    assert report["gate"]["checks"]["at_least_7_of_10_classes_non_decreasing"] is False
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


def test_rehashed_metric_json_cannot_diverge_from_prediction_artifact(tmp_path: Path) -> None:
    baseline = _write_arm(tmp_path, "baseline")
    method = _write_arm(
        tmp_path,
        "ra_glgm",
        delta=0.006,
        parameters=BASELINE_PARAMETERS
        + int(RA_EXPERIMENT_PROTOCOL["module"]["private_parameters"]),
    )
    path = method / "locked-evaluation.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["map"] = 0.999
    _rewrite_evaluation_with_valid_chain(path, rows, sync_prediction_artifact=False)

    report = evaluate_gate(baseline, method)

    assert report["engineering"]["complete"] is False
    assert any(
        "differ from prediction rederivation" in error
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
