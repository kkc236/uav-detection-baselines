from __future__ import annotations

import math
import json
import re
from pathlib import Path
from statistics import fmean

from src.ascv_loc_protocol import FROZEN_FORMAL_THRESHOLDS, FROZEN_SCREEN_GATE, sha256_file

SCREEN_METRICS = (
    "mAP50-95",
    "AP-tiny-SBR",
    "tiny_recall",
    "AP75",
    "AP-large-SBR",
)
SCREEN_GUARD_METRICS = SCREEN_METRICS[1:]
FORMAL_THRESHOLDS = FROZEN_FORMAL_THRESHOLDS


def adjudicate_preflight(summaries: dict) -> dict:
    """Adjudicate the exact paired one-batch runtime before longer training."""

    failures: list[str] = []
    common_fields = (
        "protocol_manifest_sha256",
        "protocol_source_commit",
        "source_repo_bundle_sha256",
        "source_upstream_bundle_sha256",
        "initial_state_sha256",
        "initial_state_common_fingerprint",
        "data_sha256",
        "subset_binding",
        "batch_canaries",
        "seed",
        "optimizer",
    )
    try:
        if set(summaries) != {"control", "ascv"}:
            raise ValueError("preflight requires exact control/ascv summaries")
        control = summaries["control"]
        ascv = summaries["ascv"]
        canaries = control["batch_canaries"]
        if (
            not isinstance(canaries, list)
            or len(canaries) != 1
            or not isinstance(canaries[0], dict)
            or set(canaries[0]) != {"epoch", "batch", "sha256"}
            or canaries[0]["epoch"] != 0
            or canaries[0]["batch"] != 1
            or not isinstance(canaries[0]["sha256"], str)
            or re.fullmatch(r"[0-9A-F]{64}", canaries[0]["sha256"]) is None
        ):
            raise ValueError("preflight batch canary schema drift")
        for arm, summary in (("control", control), ("ascv", ascv)):
            if summary["schema_version"] != "ascv-loc-training-summary/v2":
                raise ValueError(f"{arm} training summary schema drift")
            if summary["stage"] != "PREFLIGHT_1" or summary["arm"] != arm or summary["seed"] != 0:
                raise ValueError(f"{arm} preflight identity drift")
            for field in common_fields:
                if summary[field] != control[field]:
                    raise ValueError(f"paired preflight mismatch: {field}")
            if summary["batch"] != 8 or summary["workers"] != 8:
                raise ValueError(f"{arm} batch/workers drift")
            if summary["observed_tensor_batch_sizes"] != [8]:
                raise ValueError(f"{arm} observed tensor batch drift")
            if summary["loader"] != {
                "trainer_batch_size": 8,
                "per_rank_batch_size": 8,
                "loader_batch_size": 8,
                "loader_num_workers": 8,
            }:
                raise ValueError(f"{arm} loader contract drift")
            optimizer = summary["optimizer"]
            if (
                optimizer["class"] != "MuSGD"
                or optimizer["requested_lr0"] != 0.01
                or optimizer["requested_momentum"] != 0.937
                or not optimizer["groups"]
                or any(group["momentum"] != 0.937 for group in optimizer["groups"])
            ):
                raise ValueError(f"{arm} optimizer contract drift")
            if (
                summary["amp"] is not True
                or summary["amp_scale"] != 128.0
                or summary["amp_scale_min"] != 128.0
                or summary["amp_scale_max"] != 128.0
            ):
                raise ValueError(f"{arm} AMP contract drift")
            if summary["successful_batches"] != 1 or summary["optimizer_attempts"] != 1:
                raise ValueError(f"{arm} one-batch execution drift")
            if summary["internal_validation_bypass_count"] != 1:
                raise ValueError(f"{arm} internal validator bypass drift")
            if summary["test_loader_is_none"] is not True:
                raise ValueError(f"{arm} constructed a test loader")
            if summary["hardware"] != {
                "gpu": "NVIDIA GeForce RTX 4090",
                "device": "cuda:0",
            }:
                raise ValueError(f"{arm} hardware contract drift")
            peak = _finite_metric(summary, "cuda_peak_reserved_mib")
            if peak >= 24 * 1024:
                raise ValueError(f"{arm} peak CUDA reservation reached 24 GiB")
            checkpoint = summary["checkpoint"]
            if (
                set(checkpoint) != {"kind", "path", "sha256"}
                or checkpoint["kind"] != "last.pt"
                or not checkpoint["path"]
                or not checkpoint["sha256"]
            ):
                raise ValueError(f"{arm} checkpoint binding drift")
        if control["local_forward_calls"] != 0:
            raise ValueError("control executed a local forward")
        if control["local_forward_call_histogram"] != {"1": 0, "2": 0}:
            raise ValueError("control local-forward histogram drift")
        if control["local_bn_preserved_batches"] != 0:
            raise ValueError("control local BN accounting drift")
        if ascv["local_forward_calls"] != 2:
            raise ValueError("ASCV preflight did not prove checkpoint recompute")
        if ascv["local_forward_call_histogram"] != {"1": 0, "2": 1}:
            raise ValueError("ASCV local-forward histogram drift")
        if ascv["local_bn_preserved_batches"] != 1:
            raise ValueError("ASCV local BN preservation drift")
    except (KeyError, TypeError, ValueError) as error:
        failures.append(str(error))

    if failures:
        return {
            "schema_version": "ascv-loc-preflight-adjudication/v1",
            "decision": "INVALID",
            "failures": failures,
        }
    return {
        "schema_version": "ascv-loc-preflight-adjudication/v1",
        "decision": "PREFLIGHT_GO",
        "failures": [],
        "protocol_manifest_sha256": control["protocol_manifest_sha256"],
        "protocol_source_commit": control["protocol_source_commit"],
        "checks": {
            "paired_batch_canary": control["batch_canaries"],
            "control_checkpoint": control["checkpoint"],
            "ascv_checkpoint": ascv["checkpoint"],
            "matched_optimizer": control["optimizer"],
        },
    }


def build_preflight_gate(summaries: dict, inputs: list[dict]) -> dict:
    adjudication = adjudicate_preflight(summaries)
    if adjudication["decision"] != "PREFLIGHT_GO":
        return {
            "schema_version": "ascv-loc-preflight-gate/v1",
            "decision": adjudication["decision"],
            "protocol": {},
            "seed": 0,
            "inputs": inputs,
            "matched_contract": {},
            "checks": {},
            "failures": adjudication["failures"],
        }
    control = summaries["control"]
    return {
        "schema_version": "ascv-loc-preflight-gate/v1",
        "decision": "PREFLIGHT_GO",
        "protocol": {
            "manifest_sha256": control["protocol_manifest_sha256"],
            "source_commit": control["protocol_source_commit"],
            "repo_bundle_sha256": control["source_repo_bundle_sha256"],
            "upstream_bundle_sha256": control["source_upstream_bundle_sha256"],
        },
        "seed": 0,
        "inputs": inputs,
        "matched_contract": {
            "initial_state_sha256": control["initial_state_sha256"],
            "initial_state_common_fingerprint": control["initial_state_common_fingerprint"],
            "data_sha256": control["data_sha256"],
            "subset_binding": control["subset_binding"],
            "batch_canaries": control["batch_canaries"],
        },
        "checks": adjudication["checks"],
        "failures": [],
    }


def replay_preflight_gate(gate: dict) -> dict:
    expected_keys = {
        "schema_version",
        "decision",
        "protocol",
        "seed",
        "inputs",
        "matched_contract",
        "checks",
        "failures",
    }
    if set(gate) != expected_keys or gate.get("schema_version") != "ascv-loc-preflight-gate/v1":
        raise ValueError("preflight gate schema drift")
    inputs = gate.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ValueError("preflight gate requires two bound inputs")
    summaries = {}
    for record in inputs:
        if set(record) != {"arm", "summary", "checkpoint"}:
            raise ValueError("preflight input schema drift")
        arm = record["arm"]
        if arm not in {"control", "ascv"} or arm in summaries:
            raise ValueError("preflight input arm drift")
        summary_record = record["summary"]
        checkpoint_record = record["checkpoint"]
        if set(summary_record) != {"path", "sha256"}:
            raise ValueError("preflight summary binding drift")
        if set(checkpoint_record) != {"kind", "path", "sha256"}:
            raise ValueError("preflight checkpoint binding drift")
        summary_path = Path(summary_record["path"]).resolve()
        checkpoint_path = Path(checkpoint_record["path"]).resolve()
        if sha256_file(summary_path) != summary_record["sha256"]:
            raise ValueError("preflight summary checksum mismatch")
        if sha256_file(checkpoint_path) != checkpoint_record["sha256"]:
            raise ValueError("preflight checkpoint checksum mismatch")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("arm") != arm or summary.get("checkpoint") != checkpoint_record:
            raise ValueError("preflight summary/checkpoint binding mismatch")
        summaries[arm] = summary
    rebuilt = build_preflight_gate(summaries, inputs)
    if rebuilt != gate:
        raise ValueError("preflight gate does not replay exactly")
    return rebuilt


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
