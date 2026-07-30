from __future__ import annotations

import json

from src.sqda_geometry_gate_decision import decide_g1_admission


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
