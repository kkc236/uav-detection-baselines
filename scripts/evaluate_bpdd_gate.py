"""Evaluate the frozen seed0 paired 30-epoch FDR/BPDD screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_fdr_gate as common


EXPECTED_EPOCHS = 30
EXPECTED_STAGE = "screen"
EXPECTED_SEED = 0
VARIANTS = ("fdr", "fdr_bpdd")
BPDD_SIGNAL_FIELDS = (
    "loss_bpdd",
    "bpdd_active_edge_ratio",
    "bpdd_mean_reliability",
    "bpdd_mean_teacher_improvement",
    "bpdd_mixture_beats_final_ratio",
    "bpdd_mean_mixture_advantage_over_final",
)


def _manifest_valid(manifest: Mapping[str, Any], variant: str) -> bool:
    identity = manifest.get("run_identity")
    return bool(
        isinstance(identity, Mapping)
        and manifest.get("format_version") == 1
        and isinstance(manifest.get("source"), Mapping)
        and bool(manifest.get("protocol_sha256"))
        and isinstance(manifest.get("initial_state"), Mapping)
        and bool(manifest.get("data"))
        and manifest.get("screen_cutoff_epoch") == EXPECTED_EPOCHS
        and identity.get("stage") == EXPECTED_STAGE
        and identity.get("variant") == variant
        and identity.get("seed") == EXPECTED_SEED
        and identity.get("protocol_sha256") == manifest.get("protocol_sha256")
        and bool(identity.get("run_id"))
        and bool(identity.get("source_sha256"))
        and bool(identity.get("fdr_protocol_sha256"))
        and bool(identity.get("initial_state_sha256"))
    )


def _row_authority_valid(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], variant: str
) -> bool:
    identity = manifest.get("run_identity", {})
    return all(
        row.get("variant") == variant
        and row.get("stage") == EXPECTED_STAGE
        and row.get("run_id") == identity.get("run_id")
        for row in rows
    )


def _paired_authority(fdr: Mapping[str, Any], bpdd: Mapping[str, Any]) -> bool:
    left = fdr.get("run_identity")
    right = bpdd.get("run_identity")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    manifest_fields = (
        "format_version",
        "protocol_sha256",
        "source",
        "initial_state",
        "data",
        "screen_cutoff_epoch",
    )
    identity_fields = (
        "source_sha256",
        "protocol_sha256",
        "fdr_protocol_sha256",
        "initial_state_sha256",
        "stage",
        "seed",
    )
    return bool(
        all(fdr.get(field) == bpdd.get(field) for field in manifest_fields)
        and all(left.get(field) == right.get(field) for field in identity_fields)
        and left.get("variant") == "fdr"
        and right.get("variant") == "fdr_bpdd"
        and left.get("run_id") != right.get("run_id")
    )


def _bpdd_signal_live(rows: Sequence[Mapping[str, Any]]) -> bool:
    if len(rows) != EXPECTED_EPOCHS:
        return False
    try:
        for row in rows:
            for field in BPDD_SIGNAL_FIELDS:
                common._number(row[field])
            active = common._number(row["bpdd_active_edge_ratio"])
            reliability = common._number(row["bpdd_mean_reliability"])
            if not (0.0 <= active <= 1.0 and 0.0 <= reliability <= 1.0):
                return False
        return any(common._number(row["bpdd_active_edge_ratio"]) > 0 for row in rows)
    except (KeyError, TypeError, ValueError):
        return False


def _load_arm(run: str | Path, variant: str) -> dict[str, Any]:
    root = Path(run).resolve()
    errors: list[str] = []
    manifest: dict[str, Any] = {}
    evidence: list[dict[str, Any]] = []
    results: list[dict[str, str]] = []
    paths = {
        "manifest": root / "bpdd-run.json",
        "evidence": root / "bpdd-epochs.jsonl",
        "results": root / "results.csv",
    }
    for label, path in paths.items():
        if not path.is_file():
            errors.append(f"{variant} missing {label}: {path}")
    if not errors:
        try:
            manifest = common._read_json(paths["manifest"])
            evidence = common._read_jsonl(paths["evidence"])
            results = common._read_csv(paths["results"])
        except (OSError, UnicodeError, json.JSONDecodeError, csv.Error, ValueError) as error:
            errors.append(f"{variant} unreadable evidence: {error}")
    checks = {
        "manifest_valid": _manifest_valid(manifest, variant),
        "jsonl_30_continuous": common._continuous_jsonl(evidence),
        "results_30_continuous": common._continuous_results(results),
        "row_authority": _row_authority_valid(evidence, manifest, variant),
        "finite_metrics": common._finite_evidence(evidence),
        "jsonl_results_consistent": common._results_consistent(evidence, results),
    }
    for name, passed in checks.items():
        if not passed:
            errors.append(f"{variant} failed {name}")
    return {
        "run_dir": str(root),
        "manifest": manifest,
        "evidence": evidence,
        "results": results,
        "checks": checks,
        "errors": errors,
    }


def _metric_window(
    fdr_rows: Sequence[Mapping[str, Any]],
    bpdd_rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> dict[str, Any]:
    raw = common._metric_window(fdr_rows, bpdd_rows, indices)
    return {
        "epochs": raw["epochs"],
        "fdr": raw["control"],
        "fdr_bpdd": raw["fdr"],
        "delta": raw["delta"],
    }


def _best_summary(
    fdr_rows: Sequence[Mapping[str, Any]],
    bpdd_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    raw = common._best_summary(fdr_rows, bpdd_rows)
    return {
        "selection": raw["selection"],
        "fdr": raw["control"],
        "fdr_bpdd": raw["fdr"],
        "delta": raw["delta"],
    }


def evaluate_gate(fdr_run: str | Path, bpdd_run: str | Path) -> dict[str, Any]:
    """Fail closed unless BPDD beats a strict same-authority FDR screen."""

    fdr = _load_arm(fdr_run, "fdr")
    bpdd = _load_arm(bpdd_run, "fdr_bpdd")
    paired = _paired_authority(fdr["manifest"], bpdd["manifest"])
    bpdd_signal = _bpdd_signal_live(bpdd["evidence"])
    engineering_checks = {
        "fdr_complete": all(fdr["checks"].values()),
        "bpdd_complete": all(bpdd["checks"].values()),
        "continuous_30_epochs": (
            fdr["checks"]["jsonl_30_continuous"]
            and fdr["checks"]["results_30_continuous"]
            and bpdd["checks"]["jsonl_30_continuous"]
            and bpdd["checks"]["results_30_continuous"]
        ),
        "paired_authority": paired,
        "jsonl_results_consistent": (
            fdr["checks"]["jsonl_results_consistent"]
            and bpdd["checks"]["jsonl_results_consistent"]
        ),
        "finite_metrics": fdr["checks"]["finite_metrics"] and bpdd["checks"]["finite_metrics"],
        "bpdd_signal_live": bpdd_signal,
    }
    errors = [*fdr["errors"], *bpdd["errors"]]
    if not paired:
        errors.append("FDR/BPDD authority manifests are not a strict pair")
    if not bpdd_signal:
        errors.append("BPDD loss/activity evidence is missing, non-finite, or inactive")
    engineering_complete = all(engineering_checks.values())

    metrics: dict[str, Any] = {"final": None, "tail3": None, "best": None}
    if common._finite_evidence(fdr["evidence"]) and common._finite_evidence(bpdd["evidence"]):
        metrics = {
            "final": _metric_window(fdr["evidence"], bpdd["evidence"], [29]),
            "tail3": _metric_window(fdr["evidence"], bpdd["evidence"], [27, 28, 29]),
            "best": _best_summary(fdr["evidence"], bpdd["evidence"]),
        }
    gate_checks = {
        "final_map_strictly_positive": bool(metrics["final"] and metrics["final"]["delta"]["map"] > 0),
        "tail3_mean_map_strictly_positive": bool(metrics["tail3"] and metrics["tail3"]["delta"]["map"] > 0),
        "final_ap75_strictly_positive": bool(metrics["final"] and metrics["final"]["delta"]["map75"] > 0),
    }
    eligible = engineering_complete and all(gate_checks.values())
    return {
        "format_version": 1,
        "gate_name": "BPDD-paired-screen-seed0",
        "frozen_protocol": {
            "epochs": 30,
            "stage": "screen",
            "seed": 0,
            "scientific_thresholds": {
                "final_map_delta": ">0",
                "tail3_mean_map_delta": ">0",
                "final_ap75_delta": ">0",
            },
        },
        "runs": {"fdr": fdr["run_dir"], "fdr_bpdd": bpdd["run_dir"]},
        "engineering": {
            "complete": engineering_complete,
            "checks": engineering_checks,
            "arm_checks": {"fdr": fdr["checks"], "fdr_bpdd": bpdd["checks"]},
            "errors": errors,
        },
        "metrics": metrics,
        "loss_trends": {
            "fdr": common._loss_trends(fdr["evidence"], fdr["results"]),
            "fdr_bpdd": common._loss_trends(bpdd["evidence"], bpdd["results"]),
        },
        "bpdd_diagnostics": {
            field: {
                "final": common._number(bpdd["evidence"][-1][field]),
                "tail3_mean": common._mean(
                    [common._number(row[field]) for row in bpdd["evidence"][-3:]]
                ),
            }
            for field in BPDD_SIGNAL_FIELDS
        } if bpdd_signal else None,
        "gate": {"checks": gate_checks, "passed": eligible},
        "formal_eligible": eligible,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdr-run", type=Path, required=True)
    parser.add_argument("--bpdd-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_gate(args.fdr_run, args.bpdd_run)
    common.write_create_only_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not report["engineering"]["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
