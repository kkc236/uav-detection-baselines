"""Evaluate the frozen fresh seed0 Formal100 FDR versus FDR+BPDD pair."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_bpdd_gate as screen_gate
from scripts import evaluate_fdr_gate as common


EXPECTED_EPOCHS = 100
EXPECTED_STAGE = "formal"
EXPECTED_SEED = 0
VARIANTS = ("fdr", "fdr_bpdd")
METRIC_FIELDS = ("map", "map50", "map75", "precision", "recall")
SCALES = ("tiny", "small", "medium", "large")
CLASSES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
FROZEN_THRESHOLDS = {
    "final_map_delta_min": 0.0030,
    "final_ap75_delta_min": 0.0010,
    "final_ap50_delta_min": -0.0010,
    "tail10_map_delta_min": 0.0020,
    "tail10_ap75_delta_strict_min": 0.0,
    "last10_positive_map_epochs_min": 8,
    "scale_delta_floor": -0.005,
    "class_delta_floor": -0.010,
}
ENGINEERING_FAILURE_EXIT = 2
SCIENTIFIC_FAILURE_EXIT = 3
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _decimal(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric value: {value!r}") from error
    if not result.is_finite():
        raise ValueError(f"non-finite numeric value: {value!r}")
    return result


def _mean(values: Sequence[Any]) -> Decimal:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum((_decimal(value) for value in values), Decimal(0)) / len(values)


def _continuous_evidence(rows: Sequence[Mapping[str, Any]]) -> bool:
    try:
        return len(rows) == EXPECTED_EPOCHS and [
            int(row["completed_epoch"]) for row in rows
        ] == list(range(1, EXPECTED_EPOCHS + 1))
    except (KeyError, TypeError, ValueError):
        return False


def _continuous_results(rows: Sequence[Mapping[str, Any]]) -> bool:
    if len(rows) != EXPECTED_EPOCHS:
        return False
    try:
        epochs = [int(float(row["epoch"])) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False
    return epochs in (
        list(range(EXPECTED_EPOCHS)),
        list(range(1, EXPECTED_EPOCHS + 1)),
    )


def _manifest_valid(manifest: Mapping[str, Any], variant: str) -> bool:
    identity = manifest.get("run_identity")
    initial_state = manifest.get("initial_state")
    source = manifest.get("source")
    if not all(isinstance(value, Mapping) for value in (identity, initial_state, source)):
        return False
    assert isinstance(identity, Mapping)
    assert isinstance(initial_state, Mapping)
    return bool(
        manifest.get("format_version") == 1
        and manifest.get("protocol_sha256") == identity.get("protocol_sha256")
        and manifest.get("fdr_protocol_sha256") == identity.get("fdr_protocol_sha256")
        and initial_state.get("sha256") == identity.get("initial_state_sha256")
        and bool(manifest.get("data"))
        and manifest.get("screen_cutoff_epoch") is None
        and identity.get("stage") == EXPECTED_STAGE
        and identity.get("variant") == variant
        and identity.get("seed") == EXPECTED_SEED
        and isinstance(identity.get("run_id"), str)
        and identity.get("run_id", "").startswith(
            f"{variant}-{EXPECTED_STAGE}-seed{EXPECTED_SEED}-"
        )
        and bool(identity.get("source_sha256"))
        and bool(identity.get("protocol_sha256"))
        and bool(identity.get("fdr_protocol_sha256"))
        and bool(identity.get("initial_state_sha256"))
    )


def _row_authority_valid(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], variant: str
) -> bool:
    identity = manifest.get("run_identity", {})
    return bool(rows) and all(
        row.get("variant") == variant
        and row.get("stage") == EXPECTED_STAGE
        and row.get("run_id") == identity.get("run_id")
        for row in rows
    )


def _finite_training(rows: Sequence[Mapping[str, Any]]) -> bool:
    if len(rows) != EXPECTED_EPOCHS:
        return False
    try:
        for row in rows:
            for field in (*METRIC_FIELDS, "gradient_norm", "fdr_gradient_norm", "cuda_peak_mib"):
                common._number(row[field])
            for field, value in row.items():
                if field.startswith("loss_") and value is not None:
                    common._number(value)
            if row.get("gradients_finite") is not True:
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _results_consistent(
    evidence: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]
) -> bool:
    if len(evidence) != EXPECTED_EPOCHS or len(results) != EXPECTED_EPOCHS:
        return False
    try:
        for evidence_row, result_row in zip(evidence, results, strict=True):
            for evidence_field, result_field in common.CSV_METRIC_FIELDS.items():
                if not math.isclose(
                    common._number(evidence_row[evidence_field]),
                    common._number(result_row[result_field]),
                    rel_tol=common.RESULT_TOLERANCE,
                    abs_tol=common.RESULT_TOLERANCE,
                ):
                    return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _bpdd_signal_live(rows: Sequence[Mapping[str, Any]]) -> bool:
    if len(rows) != EXPECTED_EPOCHS:
        return False
    try:
        for row in rows:
            for field in screen_gate.BPDD_SIGNAL_FIELDS:
                common._number(row[field])
            active = common._number(row["bpdd_active_edge_ratio"])
            reliability = common._number(row["bpdd_mean_reliability"])
            if not (0.0 <= active <= 1.0 and 0.0 <= reliability <= 1.0):
                return False
        return any(common._number(row["bpdd_active_edge_ratio"]) > 0.0 for row in rows)
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
        "jsonl_100_continuous": _continuous_evidence(evidence),
        "results_100_continuous": _continuous_results(results),
        "row_authority": _row_authority_valid(evidence, manifest, variant),
        "finite_training": _finite_training(evidence),
        "jsonl_results_consistent": _results_consistent(evidence, results),
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


def _paired_authority(fdr: Mapping[str, Any], bpdd: Mapping[str, Any]) -> bool:
    left = fdr.get("run_identity")
    right = bpdd.get("run_identity")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    manifest_fields = (
        "format_version",
        "protocol_sha256",
        "fdr_protocol_sha256",
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
        and left.get("stage") == right.get("stage") == EXPECTED_STAGE
        and left.get("seed") == right.get("seed") == EXPECTED_SEED
        and left.get("variant") == "fdr"
        and right.get("variant") == "fdr_bpdd"
        and left.get("run_id") != right.get("run_id")
    )


def _valid_metric_mapping(value: Any, expected: Sequence[str]) -> bool:
    if not isinstance(value, Mapping) or not set(expected).issubset(value):
        return False
    try:
        for field in expected:
            common._number(value[field])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _evaluation_valid(
    evaluation: Mapping[str, Any], manifest: Mapping[str, Any], variant: str
) -> bool:
    identity = evaluation.get("evaluation_identity")
    run_identity = manifest.get("run_identity")
    checkpoint = evaluation.get("checkpoint")
    if not all(isinstance(value, Mapping) for value in (identity, run_identity, checkpoint)):
        return False
    assert isinstance(identity, Mapping)
    assert isinstance(run_identity, Mapping)
    assert isinstance(checkpoint, Mapping)
    identity_fields = (
        "source_sha256",
        "protocol_sha256",
        "fdr_protocol_sha256",
        "initial_state_sha256",
        "run_id",
        "stage",
        "variant",
        "seed",
    )
    return bool(
        evaluation.get("format_version") == 1
        and all(identity.get(field) == run_identity.get(field) for field in identity_fields)
        and identity.get("data") == manifest.get("data")
        and identity.get("stage") == EXPECTED_STAGE
        and identity.get("variant") == variant
        and identity.get("seed") == EXPECTED_SEED
        and checkpoint.get("kind") == "exact-final-ema"
        and checkpoint.get("completed_epoch") == EXPECTED_EPOCHS
        and isinstance(checkpoint.get("sha256"), str)
        and _SHA256.fullmatch(checkpoint["sha256"]) is not None
        and checkpoint.get("sha256_verified") is True
        and checkpoint.get("remote_published") is True
        and bool(checkpoint.get("remote_asset"))
        and _valid_metric_mapping(evaluation.get("metrics"), METRIC_FIELDS)
        and _valid_metric_mapping(evaluation.get("scales"), SCALES)
        and _valid_metric_mapping(evaluation.get("classes"), CLASSES)
    )


def _load_evaluation(
    path: str | Path, manifest: Mapping[str, Any], variant: str
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    errors: list[str] = []
    payload: dict[str, Any] = {}
    if not resolved.is_file():
        errors.append(f"{variant} missing independent evaluation: {resolved}")
    else:
        try:
            payload = common._read_json(resolved)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"{variant} unreadable independent evaluation: {error}")
    valid = _evaluation_valid(payload, manifest, variant)
    if not valid:
        errors.append(f"{variant} independent exact-final-EMA evaluation is invalid")
    return {"path": str(resolved), "payload": payload, "valid": valid, "errors": errors}


def _metric_summary(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_values = {name: _decimal(left[name]) for name in METRIC_FIELDS}
    right_values = {name: _decimal(right[name]) for name in METRIC_FIELDS}
    return {
        "fdr": {name: float(value) for name, value in left_values.items()},
        "fdr_bpdd": {name: float(value) for name, value in right_values.items()},
        "delta": {
            name: float(right_values[name] - left_values[name]) for name in METRIC_FIELDS
        },
    }


def _tail_summary(
    fdr_rows: Sequence[Mapping[str, Any]], bpdd_rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    left = {name: _mean([row[name] for row in fdr_rows[-10:]]) for name in METRIC_FIELDS}
    right = {name: _mean([row[name] for row in bpdd_rows[-10:]]) for name in METRIC_FIELDS}
    deltas = [
        _decimal(bpdd_row["map"]) - _decimal(fdr_row["map"])
        for fdr_row, bpdd_row in zip(fdr_rows[-10:], bpdd_rows[-10:], strict=True)
    ]
    summary = {
        "epochs": list(range(91, 101)),
        "fdr": {name: float(value) for name, value in left.items()},
        "fdr_bpdd": {name: float(value) for name, value in right.items()},
        "delta": {name: float(right[name] - left[name]) for name in METRIC_FIELDS},
    }
    positive = {
        "count": sum(delta > 0 for delta in deltas),
        "required": FROZEN_THRESHOLDS["last10_positive_map_epochs_min"],
        "epochs": [91 + index for index, delta in enumerate(deltas) if delta > 0],
        "deltas": [float(delta) for delta in deltas],
    }
    return summary, positive


def _group_summary(
    left: Mapping[str, Any], right: Mapping[str, Any], names: Sequence[str]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in names:
        left_value = _decimal(left[name])
        right_value = _decimal(right[name])
        output[name] = {
            "fdr": float(left_value),
            "fdr_bpdd": float(right_value),
            "delta": float(right_value - left_value),
        }
    return output


def evaluate_formal_gate(
    fdr_run: str | Path,
    bpdd_run: str | Path,
    fdr_eval: str | Path,
    bpdd_eval: str | Path,
) -> dict[str, Any]:
    """Fail closed unless a fresh exact-authority BPDD Formal100 pair passes."""

    fdr = _load_arm(fdr_run, "fdr")
    bpdd = _load_arm(bpdd_run, "fdr_bpdd")
    paired = _paired_authority(fdr["manifest"], bpdd["manifest"])
    bpdd_signal = _bpdd_signal_live(bpdd["evidence"])
    fdr_evaluation = _load_evaluation(fdr_eval, fdr["manifest"], "fdr")
    bpdd_evaluation = _load_evaluation(bpdd_eval, bpdd["manifest"], "fdr_bpdd")
    evaluation_pair = fdr_evaluation["valid"] and bpdd_evaluation["valid"]
    engineering_checks = {
        "fdr_complete": all(fdr["checks"].values()),
        "bpdd_complete": all(bpdd["checks"].values()),
        "continuous_100_epochs": bool(
            fdr["checks"]["jsonl_100_continuous"]
            and fdr["checks"]["results_100_continuous"]
            and bpdd["checks"]["jsonl_100_continuous"]
            and bpdd["checks"]["results_100_continuous"]
        ),
        "fresh_formal_pair": paired,
        "same_source_protocol_initial_data_authority": paired,
        "jsonl_results_consistent": bool(
            fdr["checks"]["jsonl_results_consistent"]
            and bpdd["checks"]["jsonl_results_consistent"]
        ),
        "finite_training": bool(
            fdr["checks"]["finite_training"] and bpdd["checks"]["finite_training"]
        ),
        "bpdd_signal_live": bpdd_signal,
        "independent_final_evaluations": evaluation_pair,
    }
    errors = [
        *fdr["errors"],
        *bpdd["errors"],
        *fdr_evaluation["errors"],
        *bpdd_evaluation["errors"],
    ]
    if not paired:
        errors.append("FDR/BPDD manifests are not a fresh same-authority Formal100 pair")
    if not bpdd_signal:
        errors.append("BPDD loss/activity evidence is missing, non-finite, or inactive")
    engineering_complete = all(engineering_checks.values())

    metrics: dict[str, Any] = {
        "final": None,
        "tail10": None,
        "last10_positive_map_epochs": None,
        "scales": None,
        "classes": None,
    }
    if evaluation_pair and _finite_training(fdr["evidence"]) and _finite_training(bpdd["evidence"]):
        tail10, positive = _tail_summary(fdr["evidence"], bpdd["evidence"])
        metrics = {
            "final": _metric_summary(
                fdr_evaluation["payload"]["metrics"],
                bpdd_evaluation["payload"]["metrics"],
            ),
            "tail10": tail10,
            "last10_positive_map_epochs": positive,
            "scales": _group_summary(
                fdr_evaluation["payload"]["scales"],
                bpdd_evaluation["payload"]["scales"],
                SCALES,
            ),
            "classes": _group_summary(
                fdr_evaluation["payload"]["classes"],
                bpdd_evaluation["payload"]["classes"],
                CLASSES,
            ),
        }

    thresholds = {name: _decimal(value) for name, value in FROZEN_THRESHOLDS.items()}
    final = metrics["final"]
    tail10 = metrics["tail10"]
    positive = metrics["last10_positive_map_epochs"]
    scales = metrics["scales"]
    classes = metrics["classes"]
    gate_checks = {
        "final_map_delta_at_least_0_0030": bool(
            final and _decimal(final["delta"]["map"]) >= thresholds["final_map_delta_min"]
        ),
        "final_ap75_delta_at_least_0_0010": bool(
            final and _decimal(final["delta"]["map75"]) >= thresholds["final_ap75_delta_min"]
        ),
        "final_ap50_delta_at_least_minus_0_0010": bool(
            final and _decimal(final["delta"]["map50"]) >= thresholds["final_ap50_delta_min"]
        ),
        "tail10_map_delta_at_least_0_0020": bool(
            tail10 and _decimal(tail10["delta"]["map"]) >= thresholds["tail10_map_delta_min"]
        ),
        "tail10_ap75_strictly_positive": bool(
            tail10
            and _decimal(tail10["delta"]["map75"])
            > thresholds["tail10_ap75_delta_strict_min"]
        ),
        "last10_at_least_8_positive_map_deltas": bool(
            positive
            and positive["count"] >= FROZEN_THRESHOLDS["last10_positive_map_epochs_min"]
        ),
        "no_scale_below_minus_0_005": bool(
            scales
            and min(_decimal(value["delta"]) for value in scales.values())
            >= thresholds["scale_delta_floor"]
        ),
        "no_class_below_minus_0_010": bool(
            classes
            and min(_decimal(value["delta"]) for value in classes.values())
            >= thresholds["class_delta_floor"]
        ),
    }
    formal_success = engineering_complete and all(gate_checks.values())
    if not engineering_complete:
        outcome = {"status": "engineering_failed", "exit_code": ENGINEERING_FAILURE_EXIT}
    elif not formal_success:
        outcome = {"status": "scientific_failed", "exit_code": SCIENTIFIC_FAILURE_EXIT}
    else:
        outcome = {"status": "passed", "exit_code": 0}
    return {
        "format_version": 1,
        "gate_name": "BPDD-paired-formal100-seed0",
        "frozen_protocol": {
            "epochs": EXPECTED_EPOCHS,
            "stage": EXPECTED_STAGE,
            "seed": EXPECTED_SEED,
            "variants": list(VARIANTS),
            "comparison": "fresh_fdr_vs_fresh_fdr_bpdd_only",
            "scientific_thresholds": dict(FROZEN_THRESHOLDS),
        },
        "runs": {"fdr": fdr["run_dir"], "fdr_bpdd": bpdd["run_dir"]},
        "evaluations": {
            "fdr": fdr_evaluation["path"],
            "fdr_bpdd": bpdd_evaluation["path"],
        },
        "engineering": {
            "complete": engineering_complete,
            "checks": engineering_checks,
            "arm_checks": {"fdr": fdr["checks"], "fdr_bpdd": bpdd["checks"]},
            "errors": errors,
        },
        "metrics": metrics,
        "bpdd_diagnostics": (
            {
                field: {
                    "final": common._number(bpdd["evidence"][-1][field]),
                    "tail10_mean": common._mean(
                        [common._number(row[field]) for row in bpdd["evidence"][-10:]]
                    ),
                }
                for field in screen_gate.BPDD_SIGNAL_FIELDS
            }
            if bpdd_signal
            else None
        ),
        "gate": {"checks": gate_checks, "passed": formal_success},
        "formal_success": formal_success,
        "outcome": outcome,
    }


write_create_only_report = common.write_create_only_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdr-run", type=Path, required=True)
    parser.add_argument("--bpdd-run", type=Path, required=True)
    parser.add_argument("--fdr-eval", type=Path, required=True)
    parser.add_argument("--bpdd-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_formal_gate(
        args.fdr_run,
        args.bpdd_run,
        args.fdr_eval,
        args.bpdd_eval,
    )
    write_create_only_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(report["outcome"]["exit_code"])


if __name__ == "__main__":
    main()
