from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest
import torch

from src.itber_evaluation import (
    EVALUATION_CONSTANTS,
    assert_repeated_evaluations,
    compute_detection_metrics,
    compute_refinement_diagnostics,
    evaluate_formal_gate,
    evaluate_gate2,
    write_immutable_report,
)


def _predictions() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "boxes": torch.tensor(
                [
                    [0.20, 0.20, 8 / 640, 8 / 640],
                    [0.60, 0.60, 24 / 640, 24 / 640],
                ],
                dtype=torch.float32,
            ),
            "scores": torch.tensor([0.95, 0.90]),
            "classes": torch.tensor([0, 1]),
        }
    ]


def _targets() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "boxes": _predictions()[0]["boxes"].clone(),
            "classes": torch.tensor([0, 1]),
        }
    ]


def _metrics(**updates: float) -> dict[str, float]:
    values = {
        "map": 0.2400,
        "ap50": 0.4200,
        "ap75": 0.2400,
        "ap_tiny": 0.1000,
        "ap_small": 0.2000,
        "precision": 0.5000,
        "recall": 0.4300,
    }
    values.update(updates)
    return values


def _diagnostics(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "finite": True,
        "improvement_count": 101,
        "degradation_count": 100,
        "matched_correction_rms": 0.020,
        "unmatched_correction_rms": 0.005,
        "unmatched_to_matched_rms_ratio": 0.25,
        "gate_mean": 0.30,
        "gate_std": 0.10,
        "gate_p05": 0.05,
        "gate_p95": 0.80,
        "residual_rms": 0.20,
        "residual_abs_p95": 0.75,
        "detector_sha_before": "A" * 64,
        "detector_sha_after": "A" * 64,
    }
    values.update(updates)
    return values


def test_evaluation_constants_match_the_baseline_validation_contract() -> None:
    assert EVALUATION_CONSTANTS == {
        "seed": 0,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "device": "0",
        "max_det": 300,
        "nms": False,
        "cache": False,
        "conf": 0.001,
        "half": False,
        "repeats": 3,
    }


def test_evaluation_report_source_binds_runtime_amendment_and_both_environments() -> None:
    source = __import__("pathlib").Path("scripts/evaluate_itber.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        "RUNTIME_AMENDMENT_SHA256",
        "BASELINE_REFERENCE_ENVIRONMENT",
        "EXECUTION_ENVIRONMENT",
        '"runtime_amendment_sha256"',
        '"baseline_reference_environment"',
        '"execution_environment"',
    ):
        assert marker in source


def test_detection_metrics_include_full_tiny_and_small_ap() -> None:
    metrics = compute_detection_metrics(_predictions(), _targets(), image_size=640)

    assert set(metrics) == {
        "map",
        "ap50",
        "ap75",
        "ap_tiny",
        "ap_small",
        "precision",
        "recall",
    }
    assert metrics["map"] > 0.99
    assert metrics["ap50"] > 0.99
    assert metrics["ap75"] > 0.99
    assert metrics["ap_tiny"] > 0.99
    assert metrics["ap_small"] > 0.99


def test_refinement_diagnostics_count_iou_direction_and_correction_safety() -> None:
    stock = torch.tensor([[[0.50, 0.50, 0.40, 0.40], [0.20, 0.20, 0.10, 0.10]]])
    refined = torch.tensor([[[0.50, 0.50, 0.50, 0.50], [0.20, 0.20, 0.10, 0.10]]])
    targets = torch.tensor([[0.50, 0.50, 0.50, 0.50]])
    matches = [(torch.tensor([0]), torch.tensor([0]))]
    correction = torch.tensor([[[0.10, 0.10, 0.10, 0.10], [0.01, 0.01, 0.01, 0.01]]])
    gates = torch.tensor([[[0.2] * 4, [0.1] * 4]])
    residuals = torch.tensor([[[0.5] * 4, [0.1] * 4]])

    report = compute_refinement_diagnostics(
        stock,
        refined,
        targets,
        matches,
        correction,
        gates,
        residuals,
    )

    assert report["improvement_count"] == 1
    assert report["degradation_count"] == 0
    assert report["matched_iou_delta_mean"] > 0
    assert report["matched_correction_rms"] == pytest.approx(0.1)
    assert report["unmatched_correction_rms"] == pytest.approx(0.01)
    assert report["unmatched_to_matched_rms_ratio"] == pytest.approx(0.1)
    assert report["finite"] is True


def test_three_repeated_evaluations_must_be_bitwise_value_identical() -> None:
    report = {
        "stock": _metrics(),
        "refined": _metrics(map=0.242),
        "diagnostics": _diagnostics(),
    }
    accepted = assert_repeated_evaluations([copy.deepcopy(report) for _ in range(3)])
    assert accepted == report

    changed = [copy.deepcopy(report) for _ in range(3)]
    changed[2]["refined"]["map"] += 1e-12
    with pytest.raises(ValueError, match="repeat 3"):
        assert_repeated_evaluations(changed)


def test_gate2_uses_exact_unrounded_boundaries() -> None:
    stock = _metrics()
    refined = _metrics(
        map=stock["map"] + 0.002,
        ap75=stock["ap75"] + 0.003,
        ap50=stock["ap50"] - 0.0005,
        ap_tiny=stock["ap_tiny"] + 1e-12,
    )
    passed = evaluate_gate2(stock, refined, _diagnostics())

    assert passed["status"] == "passed"
    assert all(passed["conditions"].values())

    failed = evaluate_gate2(stock, dict(refined, map=stock["map"] + 0.002 - 1e-12), _diagnostics())
    assert failed["status"] == "scientific_failed"
    assert failed["conditions"]["map_gain"] is False


@pytest.mark.parametrize(
    ("diagnostic_update", "condition"),
    [
        ({"improvement_count": 100}, "matched_iou_majority"),
        ({"unmatched_to_matched_rms_ratio": 0.250000000001}, "unmatched_correction_safe"),
        ({"gate_std": 0.0}, "refinement_active_unsaturated"),
        ({"residual_abs_p95": 1.0}, "refinement_active_unsaturated"),
    ],
)
def test_gate2_rejects_unsafe_or_collapsed_refinement(
    diagnostic_update: dict[str, object], condition: str
) -> None:
    stock = _metrics()
    refined = _metrics(map=0.242, ap75=0.243, ap50=0.4195, ap_small=0.201)
    decision = evaluate_gate2(stock, refined, _diagnostics(**diagnostic_update))

    assert decision["status"] == "scientific_failed"
    assert decision["conditions"][condition] is False


def test_formal_gate_requires_larger_gain_tail5_and_frozen_detector() -> None:
    stock = _metrics()
    refined = _metrics(map=0.243, ap75=0.245, ap_small=0.201)
    passed = evaluate_formal_gate(stock, refined, _diagnostics(), tail5_map_delta=1e-12)
    assert passed["status"] == "passed"

    changed = evaluate_formal_gate(
        stock,
        refined,
        _diagnostics(detector_sha_after="B" * 64),
        tail5_map_delta=1e-12,
    )
    assert changed["status"] == "engineering_invalid"
    assert changed["conditions"]["detector_unchanged"] is False


def test_immutable_report_refuses_changed_content(tmp_path) -> None:
    path = tmp_path / "evaluation.json"
    report = {"stock": _metrics(), "refined": _metrics(map=0.242)}

    write_immutable_report(path, report)
    write_immutable_report(path, copy.deepcopy(report))
    assert json.loads(path.read_text(encoding="utf-8")) == report

    with pytest.raises(FileExistsError):
        write_immutable_report(path, {**report, "changed": True})


def test_evaluation_cli_exposes_no_scientific_parameter_overrides() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_itber.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for allowed in ("--stage", "--baseline-checkpoint", "--private-checkpoint", "--dataset-root", "--gate1-cache-manifest", "--output"):
        assert allowed in result.stdout
    for forbidden in ("--imgsz", "--batch", "--workers", "--max-det", "--nms", "--repeats", "--seed"):
        assert forbidden not in result.stdout
