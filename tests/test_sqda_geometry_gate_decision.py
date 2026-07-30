from __future__ import annotations

import json

from src.sqda_geometry_gate_decision import decide_g1_admission
from src.sqda_geometry_gate_decision import decide_g1_result


def _branch(*, tp: int, fp: int, fn: int) -> dict:
    return {
        "fixed_baseline_threshold": {
            "confidence_threshold": 0.50,
            "error": {"all": {"tp": tp, "fp": fp, "fn": fn}},
        }
    }


def test_geometry_gate_admission_requires_evidence_of_geometry_fp_harm() -> None:
    decision = decide_g1_admission(
        {
            "training_signal": False,
            "branches": {
                "full": _branch(tp=90, fp=20, fn=10),
                "semantic_only": _branch(tp=90, fp=16, fn=10),
            },
        }
    )

    assert decision["passed"] is True
    assert decision["criteria"]["precision_non_decrease"] is True
    assert decision["criteria"]["recall_within_tolerance"] is True
    assert decision["criteria"]["geometry_fp_excess"] is True


def test_geometry_gate_admission_rejects_when_semantic_only_loses_precision() -> None:
    decision = decide_g1_admission(
        {
            "training_signal": False,
            "branches": {
                "full": _branch(tp=90, fp=20, fn=10),
                "semantic_only": _branch(tp=90, fp=21, fn=10),
            },
        }
    )

    assert decision["passed"] is False
    assert decision["criteria"]["precision_non_decrease"] is False


def test_admission_cli_writes_machine_readable_decision(tmp_path, monkeypatch) -> None:
    from scripts.decide_sqda_geometry_admission import main

    diagnosis = {
        "training_signal": False,
        "branches": {
            "full": _branch(tp=90, fp=20, fn=10),
            "semantic_only": _branch(tp=90, fp=16, fn=10),
        },
    }
    source = tmp_path / "diagnosis.json"
    output = tmp_path / "decision.json"
    source.write_text(json.dumps(diagnosis), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["decision.py", "--diagnosis", str(source), "--output", str(output)])

    main()

    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True


def test_g1_result_gate_requires_precision_recall_ap_and_no_lower_saturation() -> None:
    full = {
        "coco": {"ap": 0.2000, "ap_small": 0.1000},
        "fixed_baseline_threshold": {"error": {"all": {"tp": 90, "fp": 10, "fn": 10}}},
        "pr_f1_curve": {"best_f1": {"precision": 0.90}},
    }
    candidate = {
        "coco": {"ap": 0.1999, "ap_small": 0.0999},
        "fixed_baseline_threshold": {"error": {"all": {"tp": 90, "fp": 9, "fn": 10}}},
        "pr_f1_curve": {"best_f1": {"precision": 0.91}},
        "gate": {"lower_bound_fraction": 0.0},
    }

    decision = decide_g1_result(full, candidate)

    assert decision["passed"] is True
    assert decision["criteria"]["map_within_tolerance"] is True
    assert decision["criteria"]["gate_not_saturated_low"] is True
