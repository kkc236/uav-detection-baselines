"""Compare exact paired LPR-G v2 evidence with the pre-registered gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lpr_g_evaluation import evaluate_screen_gate


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing paired evidence: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing paired results: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return [
            {str(key).strip(): str(value).strip() for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _require_epochs(records: list[dict], *, expected_epochs: int, label: str) -> None:
    expected = list(range(1, expected_epochs + 1))
    actual = [int(record["epoch"]) for record in records]
    if actual != expected:
        raise ValueError(f"{label} must contain exactly epochs 1-{expected_epochs}: {actual}")


def _optimizer_valid(records: list[dict]) -> bool:
    if not records:
        return False
    attempts = [int(record.get("optimizer_attempt", -1)) for record in records]
    return attempts == list(range(1, len(records) + 1)) and all(
        record.get("amp_scale_before") == 128.0
        and record.get("amp_scale_after") == 128.0
        and record.get("amp_step_skipped") is False
        and record.get("gradient_norm_finite") is True
        for record in records
    )


def _finite(values) -> bool:
    try:
        return all(value is not None and math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def load_arm_evidence(
    run: str | Path,
    *,
    method: bool,
    expected_epochs: int = 50,
) -> dict[str, Any]:
    """Load one arm while rejecting duplicates, gaps, and post-fit rows."""
    run = Path(run).resolve()
    results = _read_csv(run / "results.csv")
    diagnostics = _read_jsonl(run / "lpr_g_diagnostics.jsonl")
    audits = _read_jsonl(run / "common_state_audit.jsonl")
    ledger = _read_jsonl(run / "publication-ledger.jsonl")
    optimizer = _read_jsonl(run / "optimizer-evidence.jsonl")

    _require_epochs(results, expected_epochs=expected_epochs, label="results.csv")
    _require_epochs(diagnostics, expected_epochs=expected_epochs, label="diagnostics")
    _require_epochs(audits, expected_epochs=expected_epochs, label="common-state audit")
    completed_epochs = [int(record.get("completed_epoch", -1)) for record in ledger]
    expected = list(range(1, expected_epochs + 1))
    if completed_epochs != expected or any(record.get("verified") is not True for record in ledger):
        raise ValueError(
            f"publication ledger must contain verified completed epochs 1-{expected_epochs}: "
            f"{completed_epochs}"
        )

    maps = [float(record["metrics/mAP50-95(B)"]) for record in results]
    maps50 = [float(record["metrics/mAP50(B)"]) for record in results]
    ap75 = [float(record["map75"]) for record in diagnostics]
    if not _finite([*maps, *maps50, *ap75]):
        raise FloatingPointError(f"non-finite paired metrics: {run}")
    tail = slice(expected_epochs - 10, expected_epochs)
    metrics = {
        "final": {"map": maps[-1], "map50": maps50[-1], "ap75": ap75[-1]},
        "tail10": {
            "map": statistics.fmean(maps[tail]),
            "map50": statistics.fmean(maps50[tail]),
            "ap75": statistics.fmean(ap75[tail]),
        },
    }
    private_fields = (
        "loss_bbox_refine",
        "loss_giou_refine",
        "gate_p95",
        "residual_rms",
        "lpr_g_gradient_norm",
    )
    private_finite = True
    if method:
        private_finite = _finite(
            record.get(field)
            for record in diagnostics
            for field in private_fields
        )
    return {
        "run": str(run),
        "method": method,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "audits": audits,
        "publication_records": len(ledger),
        "optimizer_valid": _optimizer_valid(optimizer),
        "private_finite": private_finite,
    }


def compare_runs(
    control_run: str | Path,
    method_run: str | Path,
    *,
    ablation: dict[str, Any],
    benchmark: dict[str, Any] | None,
    expected_epochs: int = 50,
) -> dict[str, Any]:
    """Compare paired artifacts and return the frozen decision plus raw evidence."""
    control = load_arm_evidence(control_run, method=False, expected_epochs=expected_epochs)
    method = load_arm_evidence(method_run, method=True, expected_epochs=expected_epochs)
    common_state_equal = all(
        control_record.get("common_model_sha256")
        == method_record.get("common_model_sha256")
        and control_record.get("common_optimizer_sha256")
        == method_record.get("common_optimizer_sha256")
        for control_record, method_record in zip(control["audits"], method["audits"])
    )
    publication_records = control["publication_records"] + method["publication_records"]
    engineering = {
        "expected_epochs_per_arm": expected_epochs,
        "common_state_equal": common_state_equal,
        "control_optimizer_valid": control["optimizer_valid"],
        "method_optimizer_valid": method["optimizer_valid"],
        "method_private_finite": method["private_finite"],
        "publication_records": publication_records,
        "publication_complete": publication_records == 2 * expected_epochs,
    }
    engineering_valid = all(
        (
            engineering["common_state_equal"],
            engineering["control_optimizer_valid"],
            engineering["method_optimizer_valid"],
            engineering["method_private_finite"],
            engineering["publication_complete"],
        )
    )
    final_diagnostic = method["diagnostics"][-1]
    activity = {
        "finite": method["private_finite"],
        "gate_p95": final_diagnostic.get("gate_p95"),
        "residual_rms": final_diagnostic.get("residual_rms"),
        "loss_bbox_refine": final_diagnostic.get("loss_bbox_refine"),
        "loss_giou_refine": final_diagnostic.get("loss_giou_refine"),
        "lpr_g_gradient_norm": final_diagnostic.get("lpr_g_gradient_norm"),
        "efficiency_measured": benchmark is not None,
    }
    decision = evaluate_screen_gate(
        control["metrics"],
        method["metrics"],
        ablation,
        activity,
        engineering_valid,
    )
    return {
        **decision,
        "engineering": engineering,
        "control": control["metrics"],
        "method": method["metrics"],
        "same_checkpoint_ablation": ablation,
        "benchmark": benchmark,
        "runs": {"control": control["run"], "lprg": method["run"]},
    }


def write_immutable_report(path: Path, report: dict[str, Any]) -> None:
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to replace changed comparison report: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare strict paired LPR-G v2 evidence.")
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--method-run", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen", "formal"), default="screen")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ablation = json.loads(args.ablation.read_text(encoding="utf-8"))
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    report = compare_runs(
        args.control_run,
        args.method_run,
        ablation=ablation,
        benchmark=benchmark,
        expected_epochs=50 if args.stage == "screen" else 100,
    )
    write_immutable_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
