"""Evaluate the immutable Screen30 gate for FDR versus FDR+RA-GLGM."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ra_experiment_protocol import (  # noqa: E402
    BASELINE_PARAMETERS,
    MAX_PARAMETER_INCREASE_RATIO,
    MAX_PEAK_VRAM_MIB,
    RA_EXPERIMENT_PROTOCOL,
    RA_EXPERIMENT_PROTOCOL_SHA256,
    continuous_epochs,
    file_sha256,
    finite_number,
    paired_manifests,
    read_json,
    read_jsonl,
    validate_runtime_identity,
)
from scripts.validate_ra_resume import validate_optimizer_evidence  # noqa: E402


EXPECTED_EPOCHS = 30
TAIL_EPOCHS = (28, 29, 30)
FORMAL_EXPECTED_EPOCHS = 100
FORMAL_TAIL_EPOCHS = (98, 99, 100)
STANDARD_METRICS = ("map", "map50", "map75", "precision", "recall")
DETAILED_METRICS = (*STANDARD_METRICS, "ap_tiny", "ap_small")


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values)


def _checkpoint_path(run: Path, completed_epoch: int) -> Path:
    return run / "weights" / f"epoch{completed_epoch - 1}.pt"


def _queue_integrity(
    run: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    strict_hashes: bool,
    expected_epochs: int = EXPECTED_EPOCHS,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if len(rows) != expected_epochs:
        errors.append(f"queue has {len(rows)} records, expected {expected_epochs}")
        return False, errors
    expected_snapshots = {
        _checkpoint_path(run, epoch).resolve()
        for epoch in range(1, expected_epochs + 1)
    }
    actual_snapshots = {path.resolve() for path in (run / "weights").glob("epoch*.pt")}
    if actual_snapshots != expected_snapshots:
        errors.append("epoch checkpoint set is not exactly one snapshot per completed epoch")
    for epoch, row in enumerate(rows, 1):
        checkpoint = _checkpoint_path(run, epoch).resolve()
        if row.get("run_id") != run_id or int(row.get("completed_epoch", -1)) != epoch:
            errors.append(f"queue authority mismatch at epoch {epoch}")
            continue
        if row.get("status") != "pending":
            errors.append(f"queue status is not local pending at epoch {epoch}")
        try:
            recorded = Path(str(row["checkpoint"])).resolve()
        except (KeyError, TypeError):
            errors.append(f"queue checkpoint path missing at epoch {epoch}")
            continue
        if recorded != checkpoint or not checkpoint.is_file():
            errors.append(f"checkpoint path/file mismatch at epoch {epoch}")
            continue
        if strict_hashes and str(row.get("checkpoint_sha256", "")).upper() != file_sha256(checkpoint):
            errors.append(f"checkpoint SHA256 mismatch at epoch {epoch}")
    return not errors, errors


def _detailed_integrity(
    rows: Sequence[Mapping[str, Any]],
    *,
    run: Path,
    identity: Mapping[str, Any],
    queue: Sequence[Mapping[str, Any]],
    evaluator_sha256: str,
    parameter_count: Any,
    tail_epochs: Sequence[int] = TAIL_EPOCHS,
    stage: str = "screen",
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        epochs = [int(row["completed_epoch"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False, ["locked evaluation epochs are invalid"]
    if epochs != list(tail_epochs):
        errors.append(f"locked evaluation epochs must be exactly {list(tail_epochs)}")
    queue_by_epoch = {
        int(row.get("completed_epoch", -1)): row
        for row in queue
        if isinstance(row.get("completed_epoch"), int)
    }
    for row in rows:
        epoch = row.get("completed_epoch", "?")
        expected_checkpoint = (
            _checkpoint_path(run, int(epoch)).resolve()
            if isinstance(epoch, int) and epoch > 0
            else None
        )
        queued = queue_by_epoch.get(int(epoch)) if isinstance(epoch, int) else None
        if row.get("evaluator_sha256") != evaluator_sha256:
            errors.append(f"locked evaluator authority mismatch at epoch {epoch}")
        if (
            row.get("variant") != identity.get("variant")
            or row.get("stage") != stage
            or row.get("run_id") != identity.get("run_id")
        ):
            errors.append(f"locked evaluation run authority mismatch at epoch {epoch}")
        if (
            expected_checkpoint is None
            or queued is None
            or Path(str(row.get("checkpoint", ""))).resolve() != expected_checkpoint
            or row.get("checkpoint_sha256") != queued.get("checkpoint_sha256")
        ):
            errors.append(f"locked evaluation checkpoint authority mismatch at epoch {epoch}")
        elif not expected_checkpoint.is_file() or row.get("checkpoint_sha256") != file_sha256(
            expected_checkpoint
        ):
            errors.append(f"locked evaluation checkpoint SHA256 mismatch at epoch {epoch}")
        if row.get("model_parameters") != parameter_count:
            errors.append(f"locked evaluation parameter count mismatch at epoch {epoch}")
        if not all(
            finite_number(row.get(metric)) and 0.0 <= float(row[metric]) <= 1.0
            for metric in DETAILED_METRICS
        ):
            errors.append(f"non-finite or missing detailed metric at epoch {epoch}")
        class_ap = row.get("class_ap")
        if (
            not isinstance(class_ap, list)
            or len(class_ap) != int(RA_EXPERIMENT_PROTOCOL["screen_gate"]["classes"])
            or not all(finite_number(value) and 0.0 <= float(value) <= 1.0 for value in class_ap)
        ):
            errors.append(f"class_ap must contain ten finite values at epoch {epoch}")
    return not errors, errors


def _load_arm(
    run_dir: str | Path,
    variant: str,
    *,
    strict_hashes: bool,
    stage: str = "screen",
    expected_epochs: int = EXPECTED_EPOCHS,
    tail_epochs: Sequence[int] = TAIL_EPOCHS,
) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    errors: list[str] = []
    try:
        manifest = read_json(run / "ra-run.json")
        identity = validate_runtime_identity(manifest, variant=variant, stage=stage)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        manifest, identity = {}, {}
        errors.append(f"invalid {variant} runtime manifest: {error}")
    try:
        evidence = read_jsonl(run / "ra-epochs.jsonl")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        evidence = []
        errors.append(f"invalid {variant} epoch evidence: {error}")
    try:
        queue = read_jsonl(run / "publication-queue.jsonl")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        queue = []
        errors.append(f"invalid {variant} publication queue: {error}")
    try:
        detailed = read_jsonl(run / "locked-evaluation.jsonl")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        detailed = []
        errors.append(f"invalid {variant} locked evaluation: {error}")

    evidence_continuous = continuous_epochs(evidence, expected_epochs)
    if not evidence_continuous:
        errors.append(
            f"{variant} epoch evidence is not continuous 1..{expected_epochs}"
        )
    finite_evidence = all(
        all(finite_number(row.get(metric)) for metric in STANDARD_METRICS)
        and finite_number(row.get("cuda_peak_mib"))
        and row.get("run_id") == identity.get("run_id")
        and row.get("variant") == variant
        and row.get("stage") == stage
        for row in evidence
    )
    if not finite_evidence:
        errors.append(f"{variant} epoch evidence contains non-finite or foreign rows")
    evaluator_sha = str(manifest.get("locked_evaluator_sha256", ""))
    detailed_valid, detailed_errors = _detailed_integrity(
        detailed,
        run=run,
        identity=identity,
        queue=queue,
        evaluator_sha256=evaluator_sha,
        parameter_count=manifest.get("model_parameters"),
        tail_epochs=tail_epochs,
        stage=stage,
    )
    errors.extend(f"{variant} {message}" for message in detailed_errors)
    queue_valid, queue_errors = _queue_integrity(
        run,
        queue,
        run_id=str(identity.get("run_id", "")),
        strict_hashes=strict_hashes,
        expected_epochs=expected_epochs,
    )
    errors.extend(f"{variant} {message}" for message in queue_errors)
    parameter_count = manifest.get("model_parameters")
    if not isinstance(parameter_count, int) or parameter_count <= 0:
        errors.append(f"{variant} model parameter count is invalid")
    try:
        optimizer = validate_optimizer_evidence(
            run / "optimizer-evidence.jsonl",
            run_id=str(identity.get("run_id", "")),
            variant=variant,
            stage=stage,
            completed_epochs=expected_epochs,
        )
        optimizer_valid = True
    except ValueError as error:
        optimizer = None
        optimizer_valid = False
        errors.append(f"{variant} optimizer evidence is invalid: {error}")
    checks = {
        "manifest": bool(manifest and identity),
        "continuous_epochs": evidence_continuous,
        "finite_epoch_evidence": finite_evidence,
        "checkpoint_queue_integrity": queue_valid,
        "locked_evaluation": detailed_valid,
        "optimizer_evidence": optimizer_valid,
        "parameter_count": isinstance(parameter_count, int) and parameter_count > 0,
    }
    return {
        "run_dir": str(run),
        "manifest": manifest,
        "identity": identity,
        "evidence": evidence,
        "detailed": detailed,
        "parameter_count": parameter_count,
        "optimizer": optimizer,
        "checks": checks,
        "errors": errors,
    }


def _metric_window(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    final = {name: float(rows[-1][name]) for name in DETAILED_METRICS}
    tail3 = {
        name: _mean([float(row[name]) for row in rows])
        for name in DETAILED_METRICS
    }
    return {"final": final, "tail3": tail3}


def evaluate_gate(
    baseline_run: str | Path,
    method_run: str | Path,
    *,
    strict_checkpoint_hashes: bool = True,
) -> dict[str, Any]:
    """Audit one Screen30 pair and apply all preregistered scientific gates."""
    baseline = _load_arm(baseline_run, "baseline", strict_hashes=strict_checkpoint_hashes)
    method = _load_arm(method_run, "ra_glgm", strict_hashes=strict_checkpoint_hashes)
    pair_valid = paired_manifests(baseline["manifest"], method["manifest"], stage="screen")
    if not pair_valid:
        baseline["errors"].append("baseline/RA manifests are not a strict same-GPU pair")
    engineering = {
        "baseline_complete": all(baseline["checks"].values()),
        "method_complete": all(method["checks"].values()),
        "strict_pair": pair_valid,
    }
    engineering_complete = all(engineering.values())

    metrics: dict[str, Any] | None = None
    checks: dict[str, bool] = {
        "epoch30_map_delta_at_least_0_005": False,
        "tail3_map_delta_positive": False,
        "epoch30_recall_delta_positive": False,
        "epoch30_ap50_delta_positive": False,
        "epoch30_ap75_non_degrading": False,
        "epoch30_ap_tiny_delta_positive": False,
        "epoch30_ap_small_delta_positive": False,
        "at_least_7_of_10_classes_improve": False,
        "parameter_increase_at_most_10_percent": False,
        "peak_vram_below_22_gib": False,
    }
    if engineering_complete:
        baseline_metrics = _metric_window(baseline["detailed"])
        method_metrics = _metric_window(method["detailed"])
        final_delta = {
            name: method_metrics["final"][name] - baseline_metrics["final"][name]
            for name in DETAILED_METRICS
        }
        tail3_delta = {
            name: method_metrics["tail3"][name] - baseline_metrics["tail3"][name]
            for name in DETAILED_METRICS
        }
        class_delta = [
            float(method["detailed"][-1]["class_ap"][index])
            - float(baseline["detailed"][-1]["class_ap"][index])
            for index in range(10)
        ]
        class_wins = sum(delta > 0 for delta in class_delta)
        baseline_parameters = int(baseline["parameter_count"])
        method_parameters = int(method["parameter_count"])
        parameter_ratio = (method_parameters - baseline_parameters) / baseline_parameters
        peak_vram = max(float(row["cuda_peak_mib"]) for row in method["evidence"])
        metrics = {
            "baseline": baseline_metrics,
            "ra_glgm": method_metrics,
            "final_delta": final_delta,
            "tail3_delta": tail3_delta,
            "class_ap_delta": class_delta,
            "class_wins": class_wins,
            "parameters": {
                "frozen_expected_baseline": BASELINE_PARAMETERS,
                "baseline": baseline_parameters,
                "ra_glgm": method_parameters,
                "increase_ratio": parameter_ratio,
            },
            "ra_glgm_peak_vram_mib": peak_vram,
        }
        checks = {
            "epoch30_map_delta_at_least_0_005": final_delta["map"] >= 0.005,
            "tail3_map_delta_positive": tail3_delta["map"] > 0,
            "epoch30_recall_delta_positive": final_delta["recall"] > 0,
            "epoch30_ap50_delta_positive": final_delta["map50"] > 0,
            "epoch30_ap75_non_degrading": final_delta["map75"] >= 0,
            "epoch30_ap_tiny_delta_positive": final_delta["ap_tiny"] > 0,
            "epoch30_ap_small_delta_positive": final_delta["ap_small"] > 0,
            "at_least_7_of_10_classes_improve": class_wins >= 7,
            "parameter_increase_at_most_10_percent": (
                baseline_parameters == BASELINE_PARAMETERS
                and method_parameters
                == BASELINE_PARAMETERS
                + int(RA_EXPERIMENT_PROTOCOL["module"]["private_parameters"])
                and 0 <= parameter_ratio <= MAX_PARAMETER_INCREASE_RATIO
            ),
            "peak_vram_below_22_gib": peak_vram < MAX_PEAK_VRAM_MIB,
        }
    passed = engineering_complete and all(checks.values())
    return {
        "format_version": 1,
        "gate_name": "RA-GLGM-Screen30-v1",
        "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
        "engineering": {
            "complete": engineering_complete,
            "checks": engineering,
            "arm_checks": {
                "baseline": baseline["checks"],
                "ra_glgm": method["checks"],
            },
            "errors": [*baseline["errors"], *method["errors"]],
        },
        "metrics": metrics,
        "gate": {"checks": checks, "passed": passed},
        "formal_eligible": passed,
        "formal_instruction": (
            "start_fresh_from_paired_scratch_initial_state"
            if passed
            else "do_not_start_formal100"
        ),
    }


def evaluate_formal_report(
    baseline_run: str | Path,
    method_run: str | Path,
    *,
    strict_checkpoint_hashes: bool = True,
) -> dict[str, Any]:
    """Audit and compare the frozen epoch100 and tail-three Formal100 evidence."""

    baseline = _load_arm(
        baseline_run,
        "baseline",
        strict_hashes=strict_checkpoint_hashes,
        stage="formal",
        expected_epochs=FORMAL_EXPECTED_EPOCHS,
        tail_epochs=FORMAL_TAIL_EPOCHS,
    )
    method = _load_arm(
        method_run,
        "ra_glgm",
        strict_hashes=strict_checkpoint_hashes,
        stage="formal",
        expected_epochs=FORMAL_EXPECTED_EPOCHS,
        tail_epochs=FORMAL_TAIL_EPOCHS,
    )
    pair_valid = paired_manifests(
        baseline["manifest"], method["manifest"], stage="formal"
    )
    if not pair_valid:
        baseline["errors"].append("baseline/RA Formal100 manifests are not a strict same-GPU pair")
    engineering = {
        "baseline_complete": all(baseline["checks"].values()),
        "method_complete": all(method["checks"].values()),
        "strict_pair": pair_valid,
    }
    engineering_complete = all(engineering.values())
    metrics: dict[str, Any] | None = None
    checks = {
        "epoch100_map_delta_at_least_0_005": False,
        "tail3_map_delta_positive": False,
        "epoch100_recall_delta_positive": False,
        "epoch100_ap50_delta_positive": False,
        "epoch100_ap75_non_degrading": False,
        "epoch100_ap_tiny_delta_positive": False,
        "epoch100_ap_small_delta_positive": False,
        "at_least_7_of_10_classes_improve": False,
        "exact_frozen_parameter_delta": False,
        "peak_vram_below_22_gib": False,
    }
    if engineering_complete:
        baseline_metrics = _metric_window(baseline["detailed"])
        method_metrics = _metric_window(method["detailed"])
        final_delta = {
            name: method_metrics["final"][name] - baseline_metrics["final"][name]
            for name in DETAILED_METRICS
        }
        tail3_delta = {
            name: method_metrics["tail3"][name] - baseline_metrics["tail3"][name]
            for name in DETAILED_METRICS
        }
        class_delta = [
            float(method["detailed"][-1]["class_ap"][index])
            - float(baseline["detailed"][-1]["class_ap"][index])
            for index in range(10)
        ]
        class_wins = sum(delta > 0 for delta in class_delta)
        baseline_parameters = int(baseline["parameter_count"])
        method_parameters = int(method["parameter_count"])
        peak_vram = max(float(row["cuda_peak_mib"]) for row in method["evidence"])
        metrics = {
            "baseline": baseline_metrics,
            "ra_glgm": method_metrics,
            "epoch100_delta": final_delta,
            "tail3_delta": tail3_delta,
            "class_ap_delta": class_delta,
            "class_wins": class_wins,
            "parameters": {
                "baseline": baseline_parameters,
                "ra_glgm": method_parameters,
                "increase": method_parameters - baseline_parameters,
            },
            "ra_glgm_peak_vram_mib": peak_vram,
        }
        checks = {
            "epoch100_map_delta_at_least_0_005": final_delta["map"] >= 0.005,
            "tail3_map_delta_positive": tail3_delta["map"] > 0,
            "epoch100_recall_delta_positive": final_delta["recall"] > 0,
            "epoch100_ap50_delta_positive": final_delta["map50"] > 0,
            "epoch100_ap75_non_degrading": final_delta["map75"] >= 0,
            "epoch100_ap_tiny_delta_positive": final_delta["ap_tiny"] > 0,
            "epoch100_ap_small_delta_positive": final_delta["ap_small"] > 0,
            "at_least_7_of_10_classes_improve": class_wins >= 7,
            "exact_frozen_parameter_delta": (
                baseline_parameters == BASELINE_PARAMETERS
                and method_parameters - baseline_parameters
                == int(RA_EXPERIMENT_PROTOCOL["module"]["private_parameters"])
            ),
            "peak_vram_below_22_gib": peak_vram < MAX_PEAK_VRAM_MIB,
        }
    success = engineering_complete and all(checks.values())
    return {
        "format_version": 1,
        "report_name": "RA-GLGM-Formal100-v1",
        "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
        "primary_evidence": ["epoch100", "tail3_mean"],
        "engineering": {
            "complete": engineering_complete,
            "checks": engineering,
            "arm_checks": {
                "baseline": baseline["checks"],
                "ra_glgm": method["checks"],
            },
            "errors": [*baseline["errors"], *method["errors"]],
        },
        "metrics": metrics,
        "scientific": {"checks": checks, "passed": success},
        "formal_success": success,
        "best_checkpoint_policy": "supplemental only; not used for primary conclusion",
    }


def validate_evaluated_arm(
    run_dir: str | Path, *, variant: str, stage: str
) -> dict[str, Any]:
    """Validate an existing create-only locked evaluation before reuse."""

    if stage == "screen":
        expected_epochs, tail_epochs = EXPECTED_EPOCHS, TAIL_EPOCHS
    elif stage == "formal":
        expected_epochs, tail_epochs = FORMAL_EXPECTED_EPOCHS, FORMAL_TAIL_EPOCHS
    else:
        raise ValueError(f"locked evaluation is unsupported for stage: {stage}")
    arm = _load_arm(
        run_dir,
        variant,
        strict_hashes=True,
        stage=stage,
        expected_epochs=expected_epochs,
        tail_epochs=tail_epochs,
    )
    if not all(arm["checks"].values()) or arm["errors"]:
        raise ValueError(
            f"existing locked evaluation failed audit for {stage}/{variant}: "
            f"{arm['errors'][:3]}"
        )
    return arm


def validate_screen_gate_report(
    path: str | Path,
    *,
    baseline_run: str | Path,
    method_run: str | Path,
) -> dict[str, Any]:
    """Recompute Screen30 from its bound artifacts before any Formal100 action."""

    recorded = read_json(path)
    expected = evaluate_gate(baseline_run, method_run)
    if recorded != expected:
        raise ValueError("Screen30 gate report differs from frozen recomputation")
    if (
        recorded.get("formal_eligible") is not True
        or recorded.get("formal_instruction")
        != "start_fresh_from_paired_scratch_initial_state"
    ):
        raise ValueError("Screen30 gate did not authorize Formal100")
    return recorded


def validate_formal_report(
    path: str | Path,
    *,
    baseline_run: str | Path,
    method_run: str | Path,
) -> dict[str, Any]:
    """Recompute the completed Formal100 comparison before accepting it."""

    recorded = read_json(path)
    expected = evaluate_formal_report(baseline_run, method_run)
    if recorded != expected:
        raise ValueError("Formal100 report differs from frozen recomputation")
    if recorded.get("engineering", {}).get("complete") is not True:
        raise ValueError("Formal100 report failed engineering audit")
    return recorded


def write_create_only_report(output: str | Path, report: Mapping[str, Any]) -> Path:
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
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
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--ra-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen", "formal"), default="screen")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = (
        evaluate_gate(args.baseline_run, args.ra_run)
        if args.stage == "screen"
        else evaluate_formal_report(args.baseline_run, args.ra_run)
    )
    write_create_only_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not report["engineering"]["complete"]:
        raise SystemExit(2)
    if args.stage == "screen" and not report["formal_eligible"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
