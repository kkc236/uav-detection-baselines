from __future__ import annotations

from pathlib import Path

from scripts.compare_ebc_qp_d2_arms import (
    METRIC_KEYS,
    classify_a1,
    classify_four_arm_isolation,
    compare_four_arm_metrics,
    compare_metrics,
    evaluate_metric_trajectory,
    load_results,
)


def _metrics(offset: float) -> dict[str, float]:
    return {name: index + offset for index, name in enumerate(METRIC_KEYS, start=1)}


def test_compare_metrics_reports_all_three_isolation_deltas():
    comparison = compare_metrics({"a0": _metrics(0.0), "a1": _metrics(0.5), "a2": _metrics(1.0)})

    assert set(comparison) == {"a1_minus_a0", "a2_minus_a1", "a2_minus_a0"}
    for pair in comparison.values():
        assert set(pair) == set(METRIC_KEYS)
    assert comparison["a1_minus_a0"]["metrics/mAP50-95(B)"] == 0.5
    assert comparison["a2_minus_a1"]["metrics/Recall-tiny"] == 0.5
    assert comparison["a2_minus_a0"]["metrics/mAP50(B)"] == 1.0


def test_compare_four_arm_metrics_separates_p2_injection_and_ebc_effects():
    comparison = compare_four_arm_metrics(
        {
            "a0": _metrics(0.0),
            "a1_no_injection": _metrics(0.25),
            "a1": _metrics(0.5),
            "a2": _metrics(1.0),
        }
    )

    assert set(comparison) == {
        "p2_training_without_injection",
        "query_injection",
        "ebc",
        "full_method",
    }
    assert comparison["p2_training_without_injection"]["metrics/mAP50-95(B)"] == 0.25
    assert comparison["query_injection"]["metrics/mAP50-95(B)"] == 0.25
    assert comparison["ebc"]["metrics/mAP50-95(B)"] == 0.5
    assert comparison["full_method"]["metrics/mAP50-95(B)"] == 1.0


def test_a1_metric_trajectory_uses_active_epochs_final_three_and_tiny_recall():
    a0 = [
        {"epoch": epoch, "metrics/mAP50-95(B)": 0.01 * epoch}
        for epoch in range(1, 11)
    ]
    a1 = [
        {"epoch": epoch, "metrics/mAP50-95(B)": 0.01 * epoch + (0.001 if epoch >= 4 else 0.0)}
        for epoch in range(1, 11)
    ]

    result = evaluate_metric_trajectory(
        a0,
        a1,
        a0_tiny_recall=0.10,
        a1_tiny_recall=0.11,
    )

    assert result["passed"] is True
    assert result["active_epochs"] == 7
    assert result["wins"] == 7
    assert result["final_three_mean_delta"] > 0


def test_a1_branch_decision_separates_effective_unclear_and_ineffective():
    positive = {"n_gain": 12, "n_loss": 7, "v_replace": 5}
    negative = {"n_gain": 4, "n_loss": 9, "v_replace": -5}

    assert classify_a1(metric_gate_passed=True, mechanism=positive)["decision"] == "P2_EFFECTIVE"
    assert classify_a1(metric_gate_passed=True, mechanism=negative)["decision"] == "QUERY_INJECTION_UNCLEAR"
    assert classify_a1(metric_gate_passed=False, mechanism=positive)["decision"] == "QUERY_INJECTION_UNCLEAR"
    assert classify_a1(metric_gate_passed=False, mechanism=negative)["decision"] == "P2_INEFFECTIVE"


def test_four_arm_decision_develops_qg_only_when_p2_metric_signal_is_effective():
    negative = {"n_gain": 4, "n_loss": 9, "v_replace": -5}
    positive = {"n_gain": 12, "n_loss": 7, "v_replace": 5}

    result = classify_four_arm_isolation(
        p2_metric_gate_passed=True,
        query_metric_gate_passed=False,
        query_mechanism=negative,
    )
    assert result["decision"] == "P2_SIGNAL_EFFECTIVE_QUERY_INJECTION_FAILED"
    assert result["next_step"] == "DESIGN_QG_P2"
    assert result["query_injection_gate_passed"] is False

    result = classify_four_arm_isolation(
        p2_metric_gate_passed=False,
        query_metric_gate_passed=True,
        query_mechanism=positive,
    )
    assert result["decision"] == "QUERY_INJECTION_EFFECTIVE_P2_SIGNAL_UNCONFIRMED"
    assert result["next_step"] == "REPEAT_P2_ISOLATION"

    result = classify_four_arm_isolation(
        p2_metric_gate_passed=False,
        query_metric_gate_passed=False,
        query_mechanism=negative,
    )
    assert result["decision"] == "P2_FORMULATION_INEFFECTIVE"
    assert result["next_step"] == "STOP_CURRENT_P2_FORMULATION"


def test_result_trajectory_loader_ignores_blank_nontrajectory_metrics(tmp_path: Path):
    results = tmp_path / "results.csv"
    results.write_text(
        "epoch,metrics/mAP50-95(B),metrics/AP-tiny\n"
        "4,0.001,\n",
        encoding="utf-8",
    )

    assert load_results(results) == [{"epoch": 4.0, "metrics/mAP50-95(B)": 0.001}]
