from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.freeze_ebc_qp_d2_results import artifact_record


METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/AP-tiny",
    "metrics/Recall-tiny",
)


def compare_metrics(arms: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    for arm in ("a0", "a1", "a2"):
        if arm not in arms:
            raise KeyError(f"missing metric arm: {arm}")
        for key in METRIC_KEYS:
            _finite_metric(arms[arm], key, arm)
    return {
        "a1_minus_a0": _metric_delta(arms["a1"], arms["a0"]),
        "a2_minus_a1": _metric_delta(arms["a2"], arms["a1"]),
        "a2_minus_a0": _metric_delta(arms["a2"], arms["a0"]),
    }


def compare_four_arm_metrics(arms: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    for arm in ("a0", "a1_no_injection", "a1", "a2"):
        if arm not in arms:
            raise KeyError(f"missing metric arm: {arm}")
        for key in METRIC_KEYS:
            _finite_metric(arms[arm], key, arm)
    return {
        "p2_training_without_injection": _metric_delta(arms["a1_no_injection"], arms["a0"]),
        "query_injection": _metric_delta(arms["a1"], arms["a1_no_injection"]),
        "ebc": _metric_delta(arms["a2"], arms["a1"]),
        "full_method": _metric_delta(arms["a2"], arms["a0"]),
    }


def evaluate_metric_trajectory(
    control_rows: Iterable[dict[str, float]],
    method_rows: Iterable[dict[str, float]],
    *,
    a0_tiny_recall: float,
    a1_tiny_recall: float,
) -> dict[str, Any]:
    key = "metrics/mAP50-95(B)"
    control = {int(row["epoch"]): float(row[key]) for row in control_rows}
    method = {int(row["epoch"]): float(row[key]) for row in method_rows}
    expected_epochs = list(range(4, 11))
    missing = [epoch for epoch in expected_epochs if epoch not in control or epoch not in method]
    if missing:
        raise ValueError(f"missing active trajectory epochs: {missing}")

    deltas = [method[epoch] - control[epoch] for epoch in expected_epochs]
    tolerance = 1e-12
    wins = sum(delta > tolerance for delta in deltas)
    losses = sum(delta < -tolerance for delta in deltas)
    ties = len(deltas) - wins - losses
    final_three = deltas[-3:]
    final_three_mean_delta = sum(final_three) / len(final_three)
    tiny_recall_delta = float(a1_tiny_recall) - float(a0_tiny_recall)
    passed = wins + ties >= 4 and final_three_mean_delta >= -tolerance and tiny_recall_delta > tolerance
    return {
        "passed": passed,
        "active_epochs": len(deltas),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "final_three_mean_delta": final_three_mean_delta,
        "tiny_recall_delta": tiny_recall_delta,
        "epoch_deltas": {str(epoch): delta for epoch, delta in zip(expected_epochs, deltas)},
    }


def classify_a1(*, metric_gate_passed: bool, mechanism: dict[str, Any]) -> dict[str, Any]:
    n_gain = int(mechanism["n_gain"])
    n_loss = int(mechanism["n_loss"])
    v_replace = int(mechanism["v_replace"])
    mechanism_passed = n_gain > n_loss and v_replace > 0
    if metric_gate_passed and mechanism_passed:
        decision = "P2_EFFECTIVE"
        next_step = "DESIGN_QG_P2"
    elif metric_gate_passed or mechanism_passed:
        decision = "QUERY_INJECTION_UNCLEAR"
        next_step = "RUN_A1_NO_INJECTION"
    else:
        decision = "P2_INEFFECTIVE"
        next_step = "STOP_CURRENT_P2_FORMULATION"
    return {
        "decision": decision,
        "next_step": next_step,
        "metric_gate_passed": bool(metric_gate_passed),
        "mechanism_gate_passed": mechanism_passed,
        "n_gain_gt_n_loss": n_gain > n_loss,
        "v_replace_positive": v_replace > 0,
    }


def classify_four_arm_isolation(
    *,
    p2_metric_gate_passed: bool,
    query_metric_gate_passed: bool,
    query_mechanism: dict[str, Any],
) -> dict[str, Any]:
    n_gain = int(query_mechanism["n_gain"])
    n_loss = int(query_mechanism["n_loss"])
    v_replace = int(query_mechanism["v_replace"])
    query_mechanism_gate_passed = n_gain > n_loss and v_replace > 0
    query_injection_gate_passed = bool(query_metric_gate_passed and query_mechanism_gate_passed)

    if p2_metric_gate_passed:
        if query_injection_gate_passed:
            decision = "P2_AND_QUERY_INJECTION_EFFECTIVE"
        else:
            decision = "P2_SIGNAL_EFFECTIVE_QUERY_INJECTION_FAILED"
        next_step = "DESIGN_QG_P2"
    elif query_injection_gate_passed:
        decision = "QUERY_INJECTION_EFFECTIVE_P2_SIGNAL_UNCONFIRMED"
        next_step = "REPEAT_P2_ISOLATION"
    else:
        decision = "P2_FORMULATION_INEFFECTIVE"
        next_step = "STOP_CURRENT_P2_FORMULATION"

    return {
        "decision": decision,
        "next_step": next_step,
        "p2_metric_gate_passed": bool(p2_metric_gate_passed),
        "query_metric_gate_passed": bool(query_metric_gate_passed),
        "query_mechanism_gate_passed": query_mechanism_gate_passed,
        "query_injection_gate_passed": query_injection_gate_passed,
        "n_gain_gt_n_loss": n_gain > n_loss,
        "v_replace_positive": v_replace > 0,
    }


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact {source}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {source}")
    return value


def load_results(path: str | Path) -> list[dict[str, float]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    trajectory_key = "metrics/mAP50-95(B)"
    return [
        {"epoch": float(row["epoch"]), trajectory_key: float(row[trajectory_key])}
        for row in rows
    ]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    no_injection_paths = (
        args.a1_no_injection_diagnostics,
        args.a1_no_injection_results,
    )
    if any(path is not None for path in no_injection_paths) and not all(
        path is not None for path in no_injection_paths
    ):
        raise ValueError("four-arm comparison requires both A1-no-injection artifacts")
    four_arm = all(path is not None for path in no_injection_paths)
    exact_paths = {
        "a0": args.a0_exact,
        "a1": args.a1_diagnostics,
        "a2": args.a2_diagnostics,
    }
    if four_arm:
        exact_paths["a1_no_injection"] = args.a1_no_injection_diagnostics
    exact = {arm: load_json(path) for arm, path in exact_paths.items()}
    metrics = {arm: _extract_metrics(record, arm) for arm, record in exact.items()}
    results_paths = {"a0": args.a0_results, "a1": args.a1_results, "a2": args.a2_results}
    if four_arm:
        results_paths["a1_no_injection"] = args.a1_no_injection_results
    results = {arm: load_results(path) for arm, path in results_paths.items()}
    protocol = load_json(args.protocol_manifest)
    freeze = load_json(args.freeze_manifest)
    _validate_protocol(protocol, freeze)

    trajectories = {
        "a1_vs_a0": evaluate_metric_trajectory(
            results["a0"],
            results["a1"],
            a0_tiny_recall=metrics["a0"]["metrics/Recall-tiny"],
            a1_tiny_recall=metrics["a1"]["metrics/Recall-tiny"],
        ),
        "a2_vs_a0": evaluate_metric_trajectory(
            results["a0"],
            results["a2"],
            a0_tiny_recall=metrics["a0"]["metrics/Recall-tiny"],
            a1_tiny_recall=metrics["a2"]["metrics/Recall-tiny"],
        ),
        "a2_vs_a1": evaluate_metric_trajectory(
            results["a1"],
            results["a2"],
            a0_tiny_recall=metrics["a1"]["metrics/Recall-tiny"],
            a1_tiny_recall=metrics["a2"]["metrics/Recall-tiny"],
        ),
    }
    mechanism = {arm: exact[arm].get("mechanism") for arm in ("a1", "a2")}
    if not all(isinstance(value, dict) for value in mechanism.values()):
        raise ValueError("A1 and A2 diagnostics must contain mechanism objects")
    if four_arm:
        mechanism["a1_no_injection"] = exact["a1_no_injection"].get("mechanism")
        if not isinstance(mechanism["a1_no_injection"], dict):
            raise ValueError("A1-no-injection diagnostics must contain a mechanism object")

    source_paths = {
        "a0_exact": args.a0_exact,
        "a1_diagnostics": args.a1_diagnostics,
        "a2_diagnostics": args.a2_diagnostics,
        "a0_results": args.a0_results,
        "a1_results": args.a1_results,
        "a2_results": args.a2_results,
        "protocol_manifest": args.protocol_manifest,
        "freeze_manifest": args.freeze_manifest,
    }
    if four_arm:
        source_paths["a1_no_injection_diagnostics"] = args.a1_no_injection_diagnostics
        source_paths["a1_no_injection_results"] = args.a1_no_injection_results

    report = {
        "format_version": 2 if four_arm else 1,
        "protocol": {
            "signature": protocol["signature"],
            "initial_state_sha256": protocol["initial_state"]["sha256"],
            "subset_sha256": protocol["subset"]["sha256"],
            "dataset_sha256": protocol["dataset"]["sha256"],
            "seed": protocol["seed"],
        },
        "artifacts": {name: artifact_record(path) for name, path in sorted(source_paths.items())},
        "metrics": metrics,
        "metric_deltas": compare_four_arm_metrics(metrics) if four_arm else compare_metrics(metrics),
        "trajectories": trajectories,
        "mechanism": mechanism,
        "a1_decision": classify_a1(
            metric_gate_passed=trajectories["a1_vs_a0"]["passed"],
            mechanism=mechanism["a1"],
        ),
    }
    if four_arm:
        four_arm_trajectories = {
            "p2_training_without_injection": evaluate_metric_trajectory(
                results["a0"],
                results["a1_no_injection"],
                a0_tiny_recall=metrics["a0"]["metrics/Recall-tiny"],
                a1_tiny_recall=metrics["a1_no_injection"]["metrics/Recall-tiny"],
            ),
            "query_injection": evaluate_metric_trajectory(
                results["a1_no_injection"],
                results["a1"],
                a0_tiny_recall=metrics["a1_no_injection"]["metrics/Recall-tiny"],
                a1_tiny_recall=metrics["a1"]["metrics/Recall-tiny"],
            ),
            "ebc": trajectories["a2_vs_a1"],
        }
        report["four_arm_trajectories"] = four_arm_trajectories
        report["four_arm_decision"] = classify_four_arm_isolation(
            p2_metric_gate_passed=four_arm_trajectories["p2_training_without_injection"]["passed"],
            query_metric_gate_passed=four_arm_trajectories["query_injection"]["passed"],
            query_mechanism=mechanism["a1"],
        )
    return report


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    destination = Path(path)
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to replace changed tri-arm report: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{destination}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare exact EBC-QP D2 isolation evidence.")
    parser.add_argument("--a0-exact", type=Path, required=True)
    parser.add_argument("--a1-diagnostics", type=Path, required=True)
    parser.add_argument("--a2-diagnostics", type=Path, required=True)
    parser.add_argument("--a1-no-injection-diagnostics", type=Path)
    parser.add_argument("--a0-results", type=Path, required=True)
    parser.add_argument("--a1-results", type=Path, required=True)
    parser.add_argument("--a2-results", type=Path, required=True)
    parser.add_argument("--a1-no-injection-results", type=Path)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_report(args)
    write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _metric_delta(first: dict[str, float], second: dict[str, float]) -> dict[str, float]:
    return {key: float(first[key]) - float(second[key]) for key in METRIC_KEYS}


def _finite_metric(metrics: dict[str, float], key: str, arm: str) -> float:
    try:
        value = float(metrics[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid {key} for {arm}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} for {arm}")
    return value


def _extract_metrics(record: dict[str, Any], arm: str) -> dict[str, float]:
    raw = record.get("metrics")
    if not isinstance(raw, dict):
        raise ValueError(f"{arm} exact artifact has no metrics object")
    return {key: _finite_metric(raw, key, arm) for key in METRIC_KEYS}


def _validate_protocol(protocol: dict[str, Any], freeze: dict[str, Any]) -> None:
    required = ("signature", "initial_state", "subset", "dataset", "seed")
    missing = [key for key in required if key not in protocol]
    if missing:
        raise ValueError(f"protocol manifest is missing fields: {missing}")
    if freeze.get("protocol_signature") != protocol["signature"]:
        raise ValueError("freeze manifest protocol signature mismatch")


if __name__ == "__main__":
    main()
