from __future__ import annotations

from scripts.evaluate_lpr_g import expected_completed_epoch
from src.lpr_g_evaluation import evaluate_screen_gate


def _arm(final_map, tail_map, final_ap75, tail_ap75, final_map50, tail_map50):
    return {
        "final": {"map": final_map, "ap75": final_ap75, "map50": final_map50},
        "tail10": {"map": tail_map, "ap75": tail_ap75, "map50": tail_map50},
    }


def test_independent_evaluation_uses_screen_cutoff_and_formal_total() -> None:
    assert expected_completed_epoch("screen") == 30
    assert expected_completed_epoch("formal") == 100


def test_screen_gate_requires_final_tail_and_same_checkpoint_refinement_wins() -> None:
    control = _arm(0.10, 0.09, 0.05, 0.04, 0.20, 0.19)
    method = _arm(0.101, 0.091, 0.051, 0.040, 0.200, 0.189)
    ablation = {
        "refined": {"map": 0.101, "ap75": 0.051},
        "stock": {"map": 0.10, "ap75": 0.05},
    }
    activity = {"finite": True, "gate_p95": 0.01, "residual_rms": 0.001}

    result = evaluate_screen_gate(
        control, method, ablation, activity, engineering_valid=True
    )

    assert result["passed"] is True
    assert all(result["conditions"].values())
    assert result["status"] == "passed"


def test_equal_map_is_not_a_win_and_invalid_engineering_blocks_science() -> None:
    arm = _arm(0.10, 0.09, 0.05, 0.04, 0.20, 0.19)
    ablation = {
        "refined": {"map": 0.10, "ap75": 0.05},
        "stock": {"map": 0.10, "ap75": 0.05},
    }
    activity = {"finite": True, "gate_p95": 0.01, "residual_rms": 0.001}

    assert evaluate_screen_gate(arm, arm, ablation, activity, True)["passed"] is False
    assert (
        evaluate_screen_gate(arm, arm, ablation, activity, False)["status"]
        == "engineering_invalid"
    )


def test_map50_floor_is_inclusive_but_activity_threshold_is_strict() -> None:
    control = _arm(0.10, 0.09, 0.05, 0.04, 0.20, 0.19)
    method = _arm(0.101, 0.091, 0.051, 0.041, 0.199, 0.189)
    ablation = {
        "refined": {"map": 0.101, "ap75": 0.051},
        "stock": {"map": 0.10, "ap75": 0.05},
    }
    activity = {"finite": True, "gate_p95": 0.001, "residual_rms": 0.001}

    result = evaluate_screen_gate(control, method, ablation, activity, True)

    assert result["conditions"]["map50_floor"] is True
    assert result["conditions"]["refinement_active"] is False
    assert result["status"] == "scientific_failed"
