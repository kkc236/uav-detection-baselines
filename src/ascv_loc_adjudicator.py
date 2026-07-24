from __future__ import annotations

import math
from statistics import fmean

from src.ascv_loc_protocol import FROZEN_FORMAL_THRESHOLDS, FROZEN_SCREEN_GATE

SCREEN_METRICS = (
    "mAP50-95",
    "AP-tiny-SBR",
    "tiny_recall",
    "AP75",
    "AP-large-SBR",
)
SCREEN_GUARD_METRICS = SCREEN_METRICS[1:]
FORMAL_THRESHOLDS = FROZEN_FORMAL_THRESHOLDS


def _finite_metric(record: dict, metric: str) -> float:
    value = record[metric]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"metric {metric} is not finite")
    return float(value)


def adjudicate_screen(records: dict) -> dict:
    try:
        if set(records) != {"0", "1", "2"}:
            raise ValueError("screen requires exactly seeds 0, 1, and 2")
        per_seed: dict[str, dict] = {}
        for seed in ("0", "1", "2"):
            seed_record = records[seed]
            if set(seed_record) != {"control", "ascv"}:
                raise ValueError(f"seed {seed} does not contain the exact paired arms")
            values: dict[str, dict[str, float]] = {}
            for arm in ("control", "ascv"):
                if set(seed_record[arm]) != {"A", "C"}:
                    raise ValueError(f"seed {seed} arm {arm} does not contain A/C")
                values[arm] = {}
                for view in ("A", "C"):
                    for metric in SCREEN_METRICS:
                        values[arm][f"{view}:{metric}"] = _finite_metric(seed_record[arm][view], metric)
            deltas = {}
            for metric in SCREEN_METRICS:
                d_c = values["ascv"][f"C:{metric}"] - values["control"][f"C:{metric}"]
                d_a = values["ascv"][f"A:{metric}"] - values["control"][f"A:{metric}"]
                deltas[metric] = {"dC": d_c, "dA": d_a, "DID": d_c - d_a}
            per_seed[seed] = {
                "absolute": seed_record,
                "deltas": deltas,
            }
    except (KeyError, TypeError, ValueError) as error:
        return {"decision": "INVALID", "failures": [str(error)]}

    aggregate = {}
    for metric in SCREEN_METRICS:
        d_cs = [per_seed[seed]["deltas"][metric]["dC"] for seed in ("0", "1", "2")]
        d_as = [per_seed[seed]["deltas"][metric]["dA"] for seed in ("0", "1", "2")]
        dids = [per_seed[seed]["deltas"][metric]["DID"] for seed in ("0", "1", "2")]
        aggregate[metric] = {
            "dC_mean": fmean(d_cs),
            "dC_wins": sum(value > 0 for value in d_cs),
            "dA_mean": fmean(d_as),
            "DID_mean": fmean(dids),
            "DID_wins": sum(value > 0 for value in dids),
        }

    failures = []
    primary = aggregate["mAP50-95"]
    if primary["dC_wins"] < FROZEN_SCREEN_GATE["mAP_dC_wins_minimum"]:
        failures.append("mAP_dC_wins<2")
    if primary["dC_mean"] <= 0:
        failures.append("mean_mAP_dC<=0")
    if primary["DID_wins"] < FROZEN_SCREEN_GATE["mAP_DID_wins_minimum"]:
        failures.append("mAP_DID_wins<2")
    if primary["DID_mean"] <= 0:
        failures.append("mean_mAP_DID<=0")
    for seed in ("0", "1", "2"):
        treatment_c = float(per_seed[seed]["absolute"]["ascv"]["C"]["mAP50-95"])
        control_c = float(per_seed[seed]["absolute"]["control"]["C"]["mAP50-95"])
        if treatment_c < FROZEN_SCREEN_GATE["per_seed_treatment_C_over_control_C_minimum"] * control_c:
            failures.append(f"seed{seed}_treatment_C_mAP<0.8_control_C")
    for metric in SCREEN_GUARD_METRICS:
        if aggregate[metric]["dC_mean"] < 0:
            failures.append(f"mean_dC_{metric}<0")

    return {
        "schema_version": "ascv-loc-screen-adjudication/v1",
        "decision": "SCREEN_GO" if not failures else "ASCV_LOC_STOP",
        "failures": failures,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def adjudicate_formal(records: dict, *, require_three_seeds: bool) -> dict:
    expected = {"0", "1", "2"} if require_three_seeds else {"0"}
    try:
        if set(records) != expected:
            raise ValueError(f"formal adjudication requires exactly seeds {sorted(expected)}")
        per_seed = {}
        for seed in sorted(expected):
            seed_record = records[seed]
            if set(seed_record) != {"control", "ascv"}:
                raise ValueError(f"seed {seed} does not contain the exact paired arms")
            for arm in ("control", "ascv"):
                if set(seed_record[arm]) != {"A", "C"}:
                    raise ValueError(f"seed {seed} arm {arm} does not contain A/C")
                for view in ("A", "C"):
                    for metric in SCREEN_METRICS:
                        _finite_metric(seed_record[arm][view], metric)
            deltas = {}
            for metric in SCREEN_METRICS:
                treatment_ca = (
                    float(seed_record["ascv"]["C"][metric])
                    - float(seed_record["ascv"]["A"][metric])
                )
                d_c = (
                    float(seed_record["ascv"]["C"][metric])
                    - float(seed_record["control"]["C"][metric])
                )
                d_a = (
                    float(seed_record["ascv"]["A"][metric])
                    - float(seed_record["control"]["A"][metric])
                )
                deltas[metric] = {
                    "treatment_C_minus_A": treatment_ca,
                    "dC": d_c,
                    "dA": d_a,
                    "DID": d_c - d_a,
                }
            per_seed[seed] = deltas
    except (KeyError, TypeError, ValueError) as error:
        return {"decision": "INVALID", "failures": [str(error)]}

    five_gate = {}
    failures = []
    for metric, threshold in FORMAL_THRESHOLDS.items():
        values = [per_seed[seed][metric]["treatment_C_minus_A"] for seed in sorted(expected)]
        mean_value = fmean(values)
        passed = mean_value >= threshold
        five_gate[metric] = {
            "mean_treatment_C_minus_A": mean_value,
            "threshold": threshold,
            "passed": passed,
        }
        if not passed:
            failures.append(f"five_gate_{metric}")

    d_cs = [per_seed[seed]["mAP50-95"]["dC"] for seed in sorted(expected)]
    dids = [per_seed[seed]["mAP50-95"]["DID"] for seed in sorted(expected)]
    attribution = {
        "dC_mAP_mean": fmean(d_cs),
        "dC_mAP_wins": sum(value > 0 for value in d_cs),
        "DID_mAP_mean": fmean(dids),
        "DID_mAP_wins": sum(value > 0 for value in dids),
    }
    if require_three_seeds:
        if attribution["dC_mAP_wins"] < 2:
            failures.append("dC_mAP_wins<2")
        if attribution["dC_mAP_mean"] <= 0:
            failures.append("mean_dC_mAP<=0")
        if attribution["DID_mAP_wins"] < 2:
            failures.append("DID_mAP_wins<2")
        if attribution["DID_mAP_mean"] <= 0:
            failures.append("mean_DID_mAP<=0")
        passing_decision = "PAPER_READY"
    else:
        if attribution["dC_mAP_mean"] <= 0:
            failures.append("seed0_dC_mAP<=0")
        if attribution["DID_mAP_mean"] <= 0:
            failures.append("seed0_DID_mAP<=0")
        passing_decision = "FORMAL_SEED0_GO"

    return {
        "schema_version": "ascv-loc-formal-adjudication/v1",
        "decision": passing_decision if not failures else "ASCV_LOC_STOP",
        "failures": failures,
        "per_seed": per_seed,
        "five_gate_mean": five_gate,
        "attribution": attribution,
    }
