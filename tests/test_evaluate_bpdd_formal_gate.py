from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_bpdd_formal_gate.py"
CLASSES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
SCALES = ("tiny", "small", "medium", "large")


def _load_module():
    assert SCRIPT.is_file(), "BPDD Formal100 gate has not been implemented"
    spec = importlib.util.spec_from_file_location("evaluate_bpdd_formal_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest(variant: str, **identity_changes) -> dict:
    identity = {
        "source_sha256": "S" * 64,
        "protocol_sha256": "P" * 64,
        "fdr_protocol_sha256": "F" * 64,
        "initial_state_sha256": "I" * 64,
        "run_id": f"{variant}-formal-seed0-authority",
        "stage": "formal",
        "variant": variant,
        "seed": 0,
    }
    identity.update(identity_changes)
    return {
        "format_version": 1,
        "protocol_sha256": "P" * 64,
        "fdr_protocol_sha256": "F" * 64,
        "source": {"git_commit": "a" * 40, "tree_sha256": "T" * 64},
        "run_identity": identity,
        "initial_state": {
            "path": "/authority/formal-initial-state.pt",
            "sha256": "I" * 64,
        },
        "data": "/authority/formal-data.yaml",
        "screen_cutoff_epoch": None,
        "publication_queue": f"/runs/{variant}/publication-queue.jsonl",
    }


def _write_arm(
    root: Path,
    variant: str,
    *,
    tail_map_deltas: list[float] | None = None,
    tail_ap75_delta: float = 0.003,
    manifest: dict | None = None,
) -> Path:
    run = root / variant
    run.mkdir(parents=True)
    tail_map_deltas = tail_map_deltas or [0.003] * 10
    assert len(tail_map_deltas) == 10
    evidence = []
    results = []
    for epoch in range(1, 101):
        offset = 0.0
        if variant == "fdr_bpdd":
            offset = tail_map_deltas[epoch - 91] if epoch >= 91 else 0.002
        map_value = 0.10 + epoch * 0.0015 + offset
        map75 = 0.06 + epoch * 0.001 + (tail_ap75_delta if variant == "fdr_bpdd" else 0.0)
        map50 = map_value + 0.12
        precision = 0.30 + epoch * 0.001
        recall = 0.35 + epoch * 0.001
        row = {
            "completed_epoch": epoch,
            "variant": variant,
            "stage": "formal",
            "run_id": f"{variant}-formal-seed0-authority",
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map": map_value,
            "map75": map75,
            "loss_giou": 2.0 - epoch * 0.01,
            "loss_class": 1.5 - epoch * 0.007,
            "loss_bbox": 1.0 - epoch * 0.004,
            "loss_fgl": 0.5 - epoch * 0.002,
            "loss_fgl_aux": 0.3 - epoch * 0.001,
            "loss_bbox_pre": 1.1 - epoch * 0.003,
            "loss_giou_pre": 1.7 - epoch * 0.006,
            "loss_bpdd": 0.02 if variant == "fdr_bpdd" else None,
            "bpdd_active_edge_ratio": 0.25 if variant == "fdr_bpdd" else None,
            "bpdd_mean_reliability": 0.08 if variant == "fdr_bpdd" else None,
            "bpdd_mean_teacher_improvement": 0.03 if variant == "fdr_bpdd" else None,
            "bpdd_mixture_beats_final_ratio": 0.15 if variant == "fdr_bpdd" else None,
            "bpdd_mean_mixture_advantage_over_final": (
                0.01 if variant == "fdr_bpdd" else None
            ),
            "gradient_norm": 3.0,
            "fdr_gradient_norm": 2.0,
            "gradients_finite": True,
            "cuda_peak_mib": 1000.0,
        }
        evidence.append(row)
        results.append(
            {
                "epoch": epoch,
                "metrics/precision(B)": precision,
                "metrics/recall(B)": recall,
                "metrics/mAP50(B)": map50,
                "metrics/mAP50-95(B)": map_value,
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
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (run / "bpdd-run.json").write_text(
        json.dumps(manifest or _manifest(variant), allow_nan=False), encoding="utf-8"
    )
    return run


def _evaluation(
    variant: str,
    *,
    map_delta: float = 0.0,
    ap75_delta: float = 0.0,
    ap50_delta: float = 0.0,
    scale_deltas: dict[str, float] | None = None,
    class_deltas: dict[str, float] | None = None,
) -> dict:
    scale_deltas = scale_deltas or {}
    class_deltas = class_deltas or {}
    baseline = variant == "fdr"
    return {
        "format_version": 1,
        "evaluation_identity": {
            "source_sha256": "S" * 64,
            "protocol_sha256": "P" * 64,
            "fdr_protocol_sha256": "F" * 64,
            "initial_state_sha256": "I" * 64,
            "run_id": f"{variant}-formal-seed0-authority",
            "stage": "formal",
            "variant": variant,
            "seed": 0,
            "data": "/authority/formal-data.yaml",
        },
        "checkpoint": {
            "kind": "exact-final-ema",
            "completed_epoch": 100,
            "sha256": ("1" if baseline else "2") * 64,
            "sha256_verified": True,
            "remote_published": True,
            "remote_asset": f"release://bpdd-formal/{variant}-epoch-0100.pt",
        },
        "metrics": {
            "precision": 0.55 + map_delta / 2,
            "recall": 0.50 + map_delta / 3,
            "map50": 0.48 + ap50_delta,
            "map75": 0.29 + ap75_delta,
            "map": 0.30 + map_delta,
        },
        "scales": {
            name: 0.20 + index * 0.02 + scale_deltas.get(name, 0.0)
            for index, name in enumerate(SCALES)
        },
        "classes": {
            name: 0.18 + index * 0.01 + class_deltas.get(name, 0.0)
            for index, name in enumerate(CLASSES)
        },
    }


def _write_evaluation(root: Path, payload: dict) -> Path:
    variant = payload["evaluation_identity"]["variant"]
    path = root / f"{variant}-independent-final.json"
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    return path


def _valid_inputs(tmp_path: Path, **bpdd_eval_changes):
    fdr_run = _write_arm(tmp_path, "fdr")
    bpdd_run = _write_arm(tmp_path, "fdr_bpdd")
    fdr_eval = _write_evaluation(tmp_path, _evaluation("fdr"))
    bpdd_eval = _write_evaluation(
        tmp_path,
        _evaluation(
            "fdr_bpdd",
            map_delta=bpdd_eval_changes.pop("map_delta", 0.003),
            ap75_delta=bpdd_eval_changes.pop("ap75_delta", 0.001),
            ap50_delta=bpdd_eval_changes.pop("ap50_delta", -0.001),
            **bpdd_eval_changes,
        ),
    )
    return fdr_run, bpdd_run, fdr_eval, bpdd_eval


def test_cli_freezes_formal_pair_and_has_no_threshold_overrides() -> None:
    assert SCRIPT.is_file(), "BPDD Formal100 gate has not been implemented"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for option in ("--fdr-run", "--bpdd-run", "--fdr-eval", "--bpdd-eval", "--output"):
        assert option in result.stdout
    for forbidden in ("--threshold", "--epochs", "--seed", "--tail", "--stage"):
        assert forbidden not in result.stdout


def test_exact_frozen_thresholds_and_complete_formal_pair_pass(tmp_path: Path) -> None:
    module = _load_module()
    inputs = _valid_inputs(tmp_path)

    report = module.evaluate_formal_gate(*inputs)

    assert module.FROZEN_THRESHOLDS == {
        "final_map_delta_min": 0.0030,
        "final_ap75_delta_min": 0.0010,
        "final_ap50_delta_min": -0.0010,
        "tail10_map_delta_min": 0.0020,
        "tail10_ap75_delta_strict_min": 0.0,
        "last10_positive_map_epochs_min": 8,
        "scale_delta_floor": -0.005,
        "class_delta_floor": -0.010,
    }
    assert report["engineering"]["complete"] is True
    assert report["engineering"]["checks"]["fresh_formal_pair"] is True
    assert report["engineering"]["checks"]["continuous_100_epochs"] is True
    assert report["engineering"]["checks"]["bpdd_signal_live"] is True
    assert report["metrics"]["final"]["delta"]["map"] == pytest.approx(0.003)
    assert report["metrics"]["final"]["delta"]["map75"] == pytest.approx(0.001)
    assert report["metrics"]["final"]["delta"]["map50"] == pytest.approx(-0.001)
    assert report["metrics"]["tail10"]["delta"]["map"] == pytest.approx(0.003)
    assert report["metrics"]["last10_positive_map_epochs"]["count"] == 10
    assert report["gate"]["passed"] is True
    assert report["formal_success"] is True
    assert report["outcome"] == {"status": "passed", "exit_code": 0}


@pytest.mark.parametrize(
    ("change", "check"),
    [
        ({"map_delta": 0.002999}, "final_map_delta_at_least_0_0030"),
        ({"ap75_delta": 0.000999}, "final_ap75_delta_at_least_0_0010"),
        ({"ap50_delta": -0.001001}, "final_ap50_delta_at_least_minus_0_0010"),
    ],
)
def test_final_metric_below_any_frozen_threshold_is_scientific_failure(
    tmp_path: Path, change: dict, check: str
) -> None:
    module = _load_module()
    report = module.evaluate_formal_gate(*_valid_inputs(tmp_path, **change))

    assert report["engineering"]["complete"] is True
    assert report["gate"]["checks"][check] is False
    assert report["formal_success"] is False
    assert report["outcome"] == {"status": "scientific_failed", "exit_code": 3}


def test_tail10_and_positive_epoch_requirements_are_not_replaced_by_final_metrics(
    tmp_path: Path,
) -> None:
    module = _load_module()
    fdr_run = _write_arm(tmp_path, "fdr")
    bpdd_run = _write_arm(
        tmp_path,
        "fdr_bpdd",
        tail_map_deltas=[0.004] * 7 + [-0.001] * 3,
        tail_ap75_delta=0.0,
    )
    fdr_eval = _write_evaluation(tmp_path, _evaluation("fdr"))
    bpdd_eval = _write_evaluation(
        tmp_path,
        _evaluation("fdr_bpdd", map_delta=0.004, ap75_delta=0.002),
    )

    report = module.evaluate_formal_gate(fdr_run, bpdd_run, fdr_eval, bpdd_eval)

    assert report["metrics"]["final"]["delta"]["map"] == pytest.approx(0.004)
    assert report["gate"]["checks"]["tail10_ap75_strictly_positive"] is False
    assert report["gate"]["checks"]["last10_at_least_8_positive_map_deltas"] is False
    assert report["outcome"]["status"] == "scientific_failed"


@pytest.mark.parametrize(
    ("kind", "delta", "expected"),
    [
        ("scale", -0.005, True),
        ("scale", -0.005001, False),
        ("class", -0.010, True),
        ("class", -0.010001, False),
    ],
)
def test_scale_and_class_regression_floors_are_inclusive_and_fail_closed(
    tmp_path: Path, kind: str, delta: float, expected: bool
) -> None:
    module = _load_module()
    changes = {
        "scale_deltas" if kind == "scale" else "class_deltas": {
            "tiny" if kind == "scale" else "bus": delta
        }
    }
    report = module.evaluate_formal_gate(*_valid_inputs(tmp_path, **changes))

    check = "no_scale_below_minus_0_005" if kind == "scale" else "no_class_below_minus_0_010"
    assert report["gate"]["checks"][check] is expected
    assert report["formal_success"] is expected


@pytest.mark.parametrize("corruption", ("epoch_gap", "authority_drift", "inactive_bpdd"))
def test_incomplete_or_unpaired_formal_evidence_is_engineering_failure(
    tmp_path: Path, corruption: str
) -> None:
    module = _load_module()
    fdr_run, bpdd_run, fdr_eval, bpdd_eval = _valid_inputs(tmp_path)
    if corruption == "epoch_gap":
        path = bpdd_run / "bpdd-epochs.jsonl"
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        rows[50]["completed_epoch"] = 999
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")
    elif corruption == "authority_drift":
        path = bpdd_run / "bpdd-run.json"
        manifest = json.loads(path.read_text("utf-8"))
        manifest["run_identity"]["source_sha256"] = "X" * 64
        path.write_text(json.dumps(manifest), "utf-8")
    else:
        path = bpdd_run / "bpdd-epochs.jsonl"
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        for row in rows:
            row["bpdd_active_edge_ratio"] = 0.0
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")

    report = module.evaluate_formal_gate(fdr_run, bpdd_run, fdr_eval, bpdd_eval)

    assert report["engineering"]["complete"] is False
    assert report["formal_success"] is False
    assert report["outcome"] == {"status": "engineering_failed", "exit_code": 2}


def test_independent_exact_final_ema_must_link_to_the_formal_run(tmp_path: Path) -> None:
    module = _load_module()
    fdr_run, bpdd_run, fdr_eval, bpdd_eval = _valid_inputs(tmp_path)
    payload = json.loads(bpdd_eval.read_text("utf-8"))
    payload["checkpoint"]["remote_published"] = False
    payload["evaluation_identity"]["run_id"] = "wrong-run"
    bpdd_eval.write_text(json.dumps(payload), "utf-8")

    report = module.evaluate_formal_gate(fdr_run, bpdd_run, fdr_eval, bpdd_eval)

    assert report["engineering"]["checks"]["independent_final_evaluations"] is False
    assert report["outcome"]["status"] == "engineering_failed"


def test_additional_diagnostic_metrics_do_not_change_the_frozen_gate(tmp_path: Path) -> None:
    module = _load_module()
    fdr_run, bpdd_run, fdr_eval, bpdd_eval = _valid_inputs(tmp_path)
    for path in (fdr_eval, bpdd_eval):
        payload = json.loads(path.read_text("utf-8"))
        payload["metrics"]["f1"] = 0.52
        payload["scales"]["all"] = payload["metrics"]["map"]
        payload["classes"]["all"] = payload["metrics"]["map"]
        path.write_text(json.dumps(payload), "utf-8")

    report = module.evaluate_formal_gate(fdr_run, bpdd_run, fdr_eval, bpdd_eval)

    assert report["engineering"]["checks"]["independent_final_evaluations"] is True
    assert report["formal_success"] is True


def test_nonfinite_training_loss_is_engineering_failure(tmp_path: Path) -> None:
    module = _load_module()
    fdr_run, bpdd_run, fdr_eval, bpdd_eval = _valid_inputs(tmp_path)
    path = fdr_run / "bpdd-epochs.jsonl"
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    rows[49]["loss_giou"] = "nan"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")

    report = module.evaluate_formal_gate(fdr_run, bpdd_run, fdr_eval, bpdd_eval)

    assert report["engineering"]["checks"]["finite_training"] is False
    assert report["outcome"] == {"status": "engineering_failed", "exit_code": 2}


def test_cli_writes_scientific_failure_before_returning_nonzero(tmp_path: Path) -> None:
    inputs = _valid_inputs(tmp_path, map_delta=0.002)
    output = tmp_path / "formal-gate.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fdr-run",
            str(inputs[0]),
            "--bpdd-run",
            str(inputs[1]),
            "--fdr-eval",
            str(inputs[2]),
            "--bpdd-eval",
            str(inputs[3]),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 3
    report = json.loads(output.read_text("utf-8"))
    assert report["outcome"] == {"status": "scientific_failed", "exit_code": 3}
    assert report["formal_success"] is False


def test_report_is_create_only(tmp_path: Path) -> None:
    module = _load_module()
    output = tmp_path / "formal-gate.json"
    report = {"formal_success": False}

    assert module.write_create_only_report(output, report) == output.resolve()
    with pytest.raises(FileExistsError):
        module.write_create_only_report(output, report)
    assert json.loads(output.read_text("utf-8")) == report
