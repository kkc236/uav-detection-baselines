from __future__ import annotations

import copy

from src.ascv_loc_adjudicator import adjudicate_formal, adjudicate_screen


METRICS = ("mAP50-95", "AP-tiny-SBR", "tiny_recall", "AP75", "AP-large-SBR")


def _view(mAP: float) -> dict[str, float]:
    return {
        "mAP50-95": mAP,
        "AP-tiny-SBR": mAP,
        "tiny_recall": mAP,
        "AP75": mAP,
        "AP-large-SBR": mAP,
    }


def _passing_records() -> dict:
    records = {}
    for seed, gain in ((0, 0.02), (1, 0.01), (2, -0.001)):
        control_a = _view(0.10)
        control_c = _view(0.11)
        treatment_a = _view(0.10)
        treatment_c = _view(0.11 + gain)
        records[str(seed)] = {
            "control": {"A": control_a, "C": control_c},
            "ascv": {"A": treatment_a, "C": treatment_c},
        }
    return records


def test_screen_go_requires_three_seed_paired_dc_and_did() -> None:
    decision = adjudicate_screen(_passing_records())

    assert decision["decision"] == "SCREEN_GO"
    assert decision["failures"] == []
    assert decision["aggregate"]["mAP50-95"]["dC_wins"] == 2
    assert decision["aggregate"]["mAP50-95"]["DID_wins"] == 2
    assert decision["aggregate"]["mAP50-95"]["dC_mean"] > 0
    assert decision["aggregate"]["mAP50-95"]["DID_mean"] > 0


def test_screen_all_zero_cannot_pass_on_ties() -> None:
    records = _passing_records()
    for seed in records.values():
        seed["ascv"] = copy.deepcopy(seed["control"])

    decision = adjudicate_screen(records)

    assert decision["decision"] == "ASCV_LOC_STOP"
    assert "mAP_dC_wins<2" in decision["failures"]
    assert "mAP_DID_wins<2" in decision["failures"]


def test_screen_enforces_per_seed_no_collapse_guard() -> None:
    records = _passing_records()
    records["1"]["ascv"]["C"]["mAP50-95"] = 0.01

    decision = adjudicate_screen(records)

    assert decision["decision"] == "ASCV_LOC_STOP"
    assert "seed1_treatment_C_mAP<0.8_control_C" in decision["failures"]


def test_screen_enforces_nonnegative_aggregate_absolute_guards() -> None:
    records = _passing_records()
    for seed in records.values():
        seed["ascv"]["C"]["AP75"] = seed["control"]["C"]["AP75"] - 0.01

    decision = adjudicate_screen(records)

    assert decision["decision"] == "ASCV_LOC_STOP"
    assert "mean_dC_AP75<0" in decision["failures"]


def test_screen_rejects_missing_or_nonfinite_metrics_as_invalid() -> None:
    records = _passing_records()
    del records["2"]["ascv"]["C"]["tiny_recall"]
    decision = adjudicate_screen(records)
    assert decision["decision"] == "INVALID"


def _formal_passing_records() -> dict:
    records = {}
    for seed in ("0", "1", "2"):
        control_a = _view(0.20)
        control_c = _view(0.21)
        treatment_a = _view(0.20)
        treatment_c = {
            "mAP50-95": 0.225,
            "AP-tiny-SBR": 0.215,
            "tiny_recall": 0.225,
            "AP75": 0.205,
            "AP-large-SBR": 0.205,
        }
        records[seed] = {
            "control": {"A": control_a, "C": control_c},
            "ascv": {"A": treatment_a, "C": treatment_c},
        }
    return records


def test_formal_seed0_and_three_seed_paper_gates_apply_original_five_thresholds() -> None:
    seed0 = adjudicate_formal({"0": _formal_passing_records()["0"]}, require_three_seeds=False)
    assert seed0["decision"] == "FORMAL_SEED0_GO"

    paper = adjudicate_formal(_formal_passing_records(), require_three_seeds=True)
    assert paper["decision"] == "PAPER_READY"
    assert paper["five_gate_mean"]["AP-tiny-SBR"]["passed"] is True
    assert paper["five_gate_mean"]["mAP50-95"]["passed"] is True
    assert paper["five_gate_mean"]["tiny_recall"]["passed"] is True
    assert paper["five_gate_mean"]["AP75"]["passed"] is True
    assert paper["five_gate_mean"]["AP-large-SBR"]["passed"] is True


def test_formal_fails_when_any_original_gate_or_attribution_gate_fails() -> None:
    records = _formal_passing_records()
    records["0"]["ascv"]["C"]["AP-large-SBR"] = 0.19
    decision = adjudicate_formal({"0": records["0"]}, require_three_seeds=False)
    assert decision["decision"] == "ASCV_LOC_STOP"
    assert "five_gate_AP-large-SBR" in decision["failures"]

    records = _formal_passing_records()
    records["0"]["control"]["C"]["mAP50-95"] = records["0"]["ascv"]["C"]["mAP50-95"] + 0.01
    decision = adjudicate_formal({"0": records["0"]}, require_three_seeds=False)
    assert decision["decision"] == "ASCV_LOC_STOP"
    assert "seed0_dC_mAP<=0" in decision["failures"]

    records = _passing_records()
    records["0"]["control"]["A"]["mAP50-95"] = float("nan")
    decision = adjudicate_screen(records)
    assert decision["decision"] == "INVALID"
