"""Evaluate the frozen seed0 paired 30-epoch FDR Gate2 screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


EXPECTED_EPOCHS = 30
EXPECTED_STAGE = "screen"
EXPECTED_SEED = 0
RESULT_TOLERANCE = 1e-5
METRIC_FIELDS = ("map", "map75", "precision", "recall")
CSV_METRIC_FIELDS = {
    "map": "metrics/mAP50-95(B)",
    "map50": "metrics/mAP50(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
}


def _number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return result


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [
            {
                str(key).strip(): str(value).strip()
                for key, value in row.items()
                if key is not None and value is not None
            }
            for row in reader
        ]


def _continuous_jsonl(rows: Sequence[Mapping[str, Any]]) -> bool:
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
    if not isinstance(identity, Mapping):
        return False
    return (
        manifest.get("format_version") == 1
        and isinstance(manifest.get("source"), Mapping)
        and bool(manifest.get("protocol_sha256"))
        and isinstance(manifest.get("initial_state"), Mapping)
        and bool(manifest.get("data"))
        and manifest.get("screen_cutoff_epoch") == EXPECTED_EPOCHS
        and identity.get("stage") == EXPECTED_STAGE
        and identity.get("variant") == variant
        and identity.get("seed") == EXPECTED_SEED
        and isinstance(identity.get("run_id"), str)
        and bool(identity.get("run_id"))
        and identity.get("protocol_sha256") == manifest.get("protocol_sha256")
        and bool(identity.get("source_sha256"))
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


def _paired_authority(
    control: Mapping[str, Any], fdr: Mapping[str, Any]
) -> bool:
    control_identity = control.get("run_identity")
    fdr_identity = fdr.get("run_identity")
    if not isinstance(control_identity, Mapping) or not isinstance(
        fdr_identity, Mapping
    ):
        return False
    shared_manifest_fields = (
        "format_version",
        "protocol_sha256",
        "source",
        "initial_state",
        "data",
        "screen_cutoff_epoch",
    )
    shared_identity_fields = (
        "source_sha256",
        "protocol_sha256",
        "stage",
        "seed",
    )
    return (
        all(control.get(field) == fdr.get(field) for field in shared_manifest_fields)
        and all(
            control_identity.get(field) == fdr_identity.get(field)
            for field in shared_identity_fields
        )
        and control_identity.get("variant") == "control"
        and fdr_identity.get("variant") == "fdr"
        and control_identity.get("run_id") != fdr_identity.get("run_id")
    )


def _finite_evidence(rows: Sequence[Mapping[str, Any]]) -> bool:
    if len(rows) != EXPECTED_EPOCHS:
        return False
    try:
        for row in rows:
            for field in (*METRIC_FIELDS, "map50"):
                _number(row[field])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _results_consistent(
    evidence: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]
) -> bool:
    if len(evidence) != EXPECTED_EPOCHS or len(results) != EXPECTED_EPOCHS:
        return False
    try:
        for evidence_row, results_row in zip(evidence, results, strict=True):
            for evidence_field, results_field in CSV_METRIC_FIELDS.items():
                if not math.isclose(
                    _number(evidence_row[evidence_field]),
                    _number(results_row[results_field]),
                    rel_tol=RESULT_TOLERANCE,
                    abs_tol=RESULT_TOLERANCE,
                ):
                    return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _load_arm(run: Path, variant: str) -> dict[str, Any]:
    run = Path(run).resolve()
    errors: list[str] = []
    manifest: dict[str, Any] = {}
    evidence: list[dict[str, Any]] = []
    results: list[dict[str, str]] = []
    paths = {
        "manifest": run / "fdr-run.json",
        "evidence": run / "fdr-epochs.jsonl",
        "results": run / "results.csv",
    }
    for label, path in paths.items():
        if not path.is_file():
            errors.append(f"{variant} missing {label}: {path}")
    if not errors:
        try:
            manifest = _read_json(paths["manifest"])
            evidence = _read_jsonl(paths["evidence"])
            results = _read_csv(paths["results"])
        except (OSError, UnicodeError, json.JSONDecodeError, csv.Error, ValueError) as error:
            errors.append(f"{variant} unreadable evidence: {error}")

    checks = {
        "manifest_valid": _manifest_valid(manifest, variant),
        "jsonl_30_continuous": _continuous_jsonl(evidence),
        "results_30_continuous": _continuous_results(results),
        "row_authority": _row_authority_valid(evidence, manifest, variant),
        "finite_metrics": _finite_evidence(evidence),
        "jsonl_results_consistent": _results_consistent(evidence, results),
    }
    for name, passed in checks.items():
        if not passed:
            errors.append(f"{variant} failed {name}")
    return {
        "run_dir": str(run),
        "manifest": manifest,
        "evidence": evidence,
        "results": results,
        "checks": checks,
        "errors": errors,
    }


def _metric_values(row: Mapping[str, Any]) -> dict[str, float]:
    return {field: _number(row[field]) for field in METRIC_FIELDS}


def _metric_delta(
    control: Mapping[str, float], fdr: Mapping[str, float]
) -> dict[str, float]:
    return {field: fdr[field] - control[field] for field in METRIC_FIELDS}


def _metric_window(
    control_rows: Sequence[Mapping[str, Any]],
    fdr_rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> dict[str, Any]:
    control = {
        field: _mean([_number(control_rows[index][field]) for index in indices])
        for field in METRIC_FIELDS
    }
    fdr = {
        field: _mean([_number(fdr_rows[index][field]) for index in indices])
        for field in METRIC_FIELDS
    }
    return {
        "epochs": [int(control_rows[index]["completed_epoch"]) for index in indices],
        "control": control,
        "fdr": fdr,
        "delta": _metric_delta(control, fdr),
    }


def _best_summary(
    control_rows: Sequence[Mapping[str, Any]],
    fdr_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    control_row = max(control_rows, key=lambda row: _number(row["map"]))
    fdr_row = max(fdr_rows, key=lambda row: _number(row["map"]))
    control = {
        "epoch": int(control_row["completed_epoch"]),
        **_metric_values(control_row),
    }
    fdr = {"epoch": int(fdr_row["completed_epoch"]), **_metric_values(fdr_row)}
    return {
        "selection": "independent_highest_map_epoch",
        "control": control,
        "fdr": fdr,
        "delta": _metric_delta(control, fdr),
    }


def _linear_slope(values: Sequence[float]) -> float:
    center = (len(values) - 1) / 2
    denominator = sum((index - center) ** 2 for index in range(len(values)))
    if denominator == 0:
        return 0.0
    mean_value = _mean(values)
    return sum(
        (index - center) * (value - mean_value)
        for index, value in enumerate(values)
    ) / denominator


def _loss_series(
    evidence: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]
) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    evidence_names = sorted(
        {name for row in evidence for name in row if name.startswith("loss_")}
    )
    results_names = sorted(
        {name for row in results for name in row if "loss" in name.lower()}
    )
    for name in evidence_names:
        try:
            values = [_number(row[name]) for row in evidence]
        except (KeyError, TypeError, ValueError):
            continue
        if len(values) == EXPECTED_EPOCHS:
            series[name] = values
    for name in results_names:
        try:
            values = [_number(row[name]) for row in results]
        except (KeyError, TypeError, ValueError):
            continue
        if len(values) == EXPECTED_EPOCHS:
            series[name] = values
    return series


def _loss_trends(
    evidence: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    trends: dict[str, dict[str, Any]] = {}
    for name, values in _loss_series(evidence, results).items():
        first3 = _mean(values[:3])
        tail3 = _mean(values[-3:])
        delta = tail3 - first3
        trends[name] = {
            "first3_mean": first3,
            "tail3_mean": tail3,
            "tail3_minus_first3": delta,
            "final": values[-1],
            "minimum": min(values),
            "linear_slope": _linear_slope(values),
            "direction": "decreasing" if delta < 0 else "increasing" if delta > 0 else "flat",
        }
    return trends


def _empty_metrics() -> dict[str, Any]:
    return {"final": None, "tail3": None, "best": None}


def evaluate_gate(control_run: str | Path, fdr_run: str | Path) -> dict[str, Any]:
    """Evaluate one immutable control/FDR seed0 screen pair without mutating it."""
    control = _load_arm(Path(control_run), "control")
    fdr = _load_arm(Path(fdr_run), "fdr")
    paired_authority = _paired_authority(control["manifest"], fdr["manifest"])

    arm_checks = [*control["checks"].values(), *fdr["checks"].values()]
    engineering_checks = {
        "control_complete": all(control["checks"].values()),
        "fdr_complete": all(fdr["checks"].values()),
        "continuous_30_epochs": (
            control["checks"]["jsonl_30_continuous"]
            and control["checks"]["results_30_continuous"]
            and fdr["checks"]["jsonl_30_continuous"]
            and fdr["checks"]["results_30_continuous"]
        ),
        "same_stage_seed0": (
            control["checks"]["manifest_valid"]
            and fdr["checks"]["manifest_valid"]
            and control["checks"]["row_authority"]
            and fdr["checks"]["row_authority"]
        ),
        "paired_authority": paired_authority,
        "jsonl_results_consistent": (
            control["checks"]["jsonl_results_consistent"]
            and fdr["checks"]["jsonl_results_consistent"]
        ),
        "finite_metrics": (
            control["checks"]["finite_metrics"]
            and fdr["checks"]["finite_metrics"]
        ),
    }
    engineering_complete = all(arm_checks) and all(engineering_checks.values())
    errors = [*control["errors"], *fdr["errors"]]
    if not paired_authority:
        errors.append("control/fdr authority manifests are not a strict pair")

    metrics = _empty_metrics()
    control_rows = control["evidence"]
    fdr_rows = fdr["evidence"]
    if _finite_evidence(control_rows) and _finite_evidence(fdr_rows):
        metrics = {
            "final": _metric_window(control_rows, fdr_rows, [EXPECTED_EPOCHS - 1]),
            "tail3": _metric_window(
                control_rows, fdr_rows, list(range(EXPECTED_EPOCHS - 3, EXPECTED_EPOCHS))
            ),
            "best": _best_summary(control_rows, fdr_rows),
        }

    gate_checks = {
        "final_map_strictly_positive": bool(
            metrics["final"] is not None and metrics["final"]["delta"]["map"] > 0
        ),
        "tail3_mean_map_strictly_positive": bool(
            metrics["tail3"] is not None and metrics["tail3"]["delta"]["map"] > 0
        ),
        "final_ap75_strictly_positive": bool(
            metrics["final"] is not None and metrics["final"]["delta"]["map75"] > 0
        ),
    }
    formal_eligible = engineering_complete and all(gate_checks.values())
    return {
        "format_version": 1,
        "gate_name": "FDR-Gate2-paired-screen-seed0",
        "frozen_protocol": {
            "epochs": EXPECTED_EPOCHS,
            "stage": EXPECTED_STAGE,
            "seed": EXPECTED_SEED,
            "scientific_thresholds": {
                "final_map_delta": ">0",
                "tail3_mean_map_delta": ">0",
                "final_ap75_delta": ">0",
            },
        },
        "runs": {
            "control": control["run_dir"],
            "fdr": fdr["run_dir"],
        },
        "engineering": {
            "complete": engineering_complete,
            "checks": engineering_checks,
            "arm_checks": {
                "control": control["checks"],
                "fdr": fdr["checks"],
            },
            "errors": errors,
        },
        "metrics": metrics,
        "loss_trends": {
            "control": _loss_trends(control["evidence"], control["results"]),
            "fdr": _loss_trends(fdr["evidence"], fdr["results"]),
        },
        "gate": {
            "checks": gate_checks,
            "passed": formal_eligible,
        },
        "formal_eligible": formal_eligible,
    }


def write_create_only_report(output: str | Path, report: Mapping[str, Any]) -> Path:
    """Atomically create one immutable Gate2 report; never replace evidence."""
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--fdr-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_gate(args.control_run, args.fdr_run)
    write_create_only_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not report["engineering"]["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
