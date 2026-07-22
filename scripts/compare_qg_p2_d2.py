from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_ebc_qp_d2_arms import METRIC_KEYS, evaluate_metric_trajectory, load_json, load_results
from scripts.freeze_ebc_qp_d2_results import artifact_record


def compare_paired_metrics(
    control: dict[str, float], method: dict[str, float]
) -> dict[str, float]:
    if set(control) != set(method):
        raise ValueError("control and method metric keys differ")
    result = {}
    for key in control:
        control_value = float(control[key])
        method_value = float(method[key])
        if not math.isfinite(control_value) or not math.isfinite(method_value):
            raise ValueError(f"non-finite paired metric: {key}")
        result[key] = method_value - control_value
    return result


def classify_qg_p2(*, metric_gate_passed: bool, mechanism: dict[str, Any]) -> dict[str, Any]:
    n_gain = int(mechanism["n_gain"])
    n_loss = int(mechanism["n_loss"])
    v_replace = int(mechanism["v_replace"])
    mechanism_gate_passed = n_gain > n_loss and v_replace > 0
    joint_gate_passed = bool(metric_gate_passed and mechanism_gate_passed)
    return {
        "decision": "ENTER_100_EPOCH" if joint_gate_passed else "ITERATE_QG_P2",
        "metric_gate_passed": bool(metric_gate_passed),
        "mechanism_gate_passed": mechanism_gate_passed,
        "joint_gate_passed": joint_gate_passed,
        "n_gain_gt_n_loss": n_gain > n_loss,
        "v_replace_positive": v_replace > 0,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    control_exact = load_json(args.control_exact)
    method_diagnostics = load_json(args.method_diagnostics)
    protocol = load_json(args.protocol_manifest)
    preflight = load_json(args.preflight)
    resources = load_json(args.resource_report) if args.resource_report else None

    control_metrics = _extract_metrics(control_exact, "control")
    method_metrics = _extract_metrics(method_diagnostics, "qg-p2")
    mechanism = method_diagnostics.get("mechanism")
    if not isinstance(mechanism, dict):
        raise ValueError("QG-P2 diagnostics must contain a mechanism object")
    _validate_protocol(protocol)
    if not bool(preflight.get("passed")):
        raise ValueError("QG-P2 CUDA preflight did not pass")

    trajectory = evaluate_metric_trajectory(
        load_results(args.control_results),
        load_results(args.method_results),
        a0_tiny_recall=control_metrics["metrics/Recall-tiny"],
        a1_tiny_recall=method_metrics["metrics/Recall-tiny"],
    )
    decision = classify_qg_p2(
        metric_gate_passed=trajectory["passed"],
        mechanism=mechanism,
    )
    sources = {
        "control_exact": args.control_exact,
        "control_results": args.control_results,
        "method_diagnostics": args.method_diagnostics,
        "method_results": args.method_results,
        "preflight": args.preflight,
        "protocol_manifest": args.protocol_manifest,
    }
    if args.resource_report:
        sources["resource_report"] = args.resource_report

    report = {
        "format_version": 1,
        "protocol": {
            "signature": protocol["signature"],
            "initial_state_sha256": protocol["initial_state"]["sha256"],
            "subset_sha256": protocol["subset"]["sha256"],
            "dataset_sha256": protocol["dataset"]["sha256"],
            "seed": protocol["seed"],
        },
        "artifacts": {
            name: artifact_record(path) for name, path in sorted(sources.items())
        },
        "metrics": {"control": control_metrics, "qg-p2": method_metrics},
        "metric_delta_qg_p2_minus_control": compare_paired_metrics(
            control_metrics, method_metrics
        ),
        "metric_trajectory": trajectory,
        "mechanism": mechanism,
        "preflight": preflight,
        "resources": resources,
        "gate": decision,
    }
    return report


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    destination = Path(path)
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to replace changed QG-P2 report: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{destination}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare strict paired QG-P2 D2 evidence.")
    parser.add_argument("--control-exact", type=Path, required=True)
    parser.add_argument("--method-diagnostics", type=Path, required=True)
    parser.add_argument("--control-results", type=Path, required=True)
    parser.add_argument("--method-results", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--resource-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_report(args)
    write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _extract_metrics(record: dict[str, Any], arm: str) -> dict[str, float]:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{arm} artifact has no metrics object")
    result = {}
    for key in METRIC_KEYS:
        try:
            value = float(metrics[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"missing or invalid {key} for {arm}") from error
        if not math.isfinite(value):
            raise ValueError(f"non-finite {key} for {arm}")
        result[key] = value
    return result


def _validate_protocol(protocol: dict[str, Any]) -> None:
    required = ("signature", "initial_state", "subset", "dataset", "seed")
    missing = [key for key in required if key not in protocol]
    if missing:
        raise ValueError(f"protocol manifest is missing fields: {missing}")


if __name__ == "__main__":
    main()
