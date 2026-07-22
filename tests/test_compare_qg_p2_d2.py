from __future__ import annotations

from scripts.compare_qg_p2_d2 import classify_qg_p2, compare_paired_metrics


def test_compare_paired_metrics_reports_method_minus_control():
    control = {"metrics/mAP50-95(B)": 0.01, "metrics/Recall-tiny": 0.02}
    method = {"metrics/mAP50-95(B)": 0.03, "metrics/Recall-tiny": 0.05}

    assert compare_paired_metrics(control, method) == {
        "metrics/mAP50-95(B)": 0.019999999999999997,
        "metrics/Recall-tiny": 0.030000000000000002,
    }


def test_qg_p2_requires_both_metric_and_mechanism_gates_for_100_epochs():
    positive = {"n_gain": 12, "n_loss": 7, "v_replace": 5}
    negative = {"n_gain": 4, "n_loss": 9, "v_replace": -5}

    passed = classify_qg_p2(metric_gate_passed=True, mechanism=positive)
    assert passed["joint_gate_passed"] is True
    assert passed["decision"] == "ENTER_100_EPOCH"

    metric_only = classify_qg_p2(metric_gate_passed=True, mechanism=negative)
    assert metric_only["joint_gate_passed"] is False
    assert metric_only["decision"] == "ITERATE_QG_P2"

    mechanism_only = classify_qg_p2(metric_gate_passed=False, mechanism=positive)
    assert mechanism_only["joint_gate_passed"] is False
    assert mechanism_only["decision"] == "ITERATE_QG_P2"
