from __future__ import annotations

import json

from scripts.evaluate_lpr_gate import evaluate_run, evaluate_screen, write_report


def test_screen_passes_at_exact_baseline() -> None:
    report = evaluate_screen(
        map=0.04098,
        map50=0.08404,
        val_giou=1.2702,
        val_l1=0.19467,
        finite=True,
        gate_active=True,
    )

    assert report.passed


def test_screen_rejects_lower_map_even_when_other_metrics_improve() -> None:
    report = evaluate_screen(
        map=0.04097,
        map50=0.10,
        val_giou=1.0,
        val_l1=0.1,
        finite=True,
        gate_active=True,
    )

    assert not report.passed


def test_screen_accepts_frozen_map50_tradeoff() -> None:
    report = evaluate_screen(
        map=0.0430,
        map50=0.0821,
        val_giou=1.20,
        val_l1=0.20,
        finite=True,
        gate_active=True,
    )

    assert report.passed


def test_screen_rejects_nonfinite_run() -> None:
    report = evaluate_screen(
        map=0.05,
        map50=0.10,
        val_giou=1.0,
        val_l1=0.1,
        finite=False,
        gate_active=True,
    )

    assert not report.passed


def test_run_report_contains_all_checks_values_and_recommendation(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.csv").write_text(
        "epoch,metrics/mAP50(B),metrics/mAP50-95(B),val/giou_loss,val/l1_loss\n"
        "10,0.08404,0.04098,1.2702,0.19467\n",
        encoding="utf-8",
    )
    (run / "lpr_diagnostics.jsonl").write_text(
        json.dumps(
            {
                "epoch": 10,
                "map75": 0.02,
                "gates": [0.001] * 6,
                "residual_mean": 0.1,
                "residual_max": 0.4,
                "lpr_grad_norm": 0.03,
                "cuda_peak_mib": 12000,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_run(run)
    output = tmp_path / "screen.json"
    write_report(report, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert report.passed
    assert set(payload["checks"]) == {"map", "map50_tradeoff", "localization", "finite", "gate_active"}
    assert payload["measured"]["map"] == 0.04098
    assert payload["baseline"]["map"] == 0.04098
    assert payload["recommendation"] == "resume_to_100"
