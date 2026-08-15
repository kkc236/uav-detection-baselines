"""Evaluate the frozen three-seed paired LPR screening gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


EXPECTED_SCREEN_EPOCHS = 10
EXPECTED_OPTIMIZER_ATTEMPTS = 145


@dataclass(frozen=True)
class ArmScreenMetrics:
    final_map: float
    tail3_map: float
    tail3_l1: float
    tail3_giou: float
    finite: bool
    optimizer_valid: bool
    gate_active: bool
    run_path: str


def _mean(values) -> float:
    return statistics.fmean(values)


def evaluate_paired_screen(
    controls: dict[int, ArmScreenMetrics],
    methods: dict[int, ArmScreenMetrics],
) -> dict:
    required = {0, 1, 2}
    if set(controls) != required or set(methods) != required:
        raise ValueError("paired LPR screen requires control and method arms for seeds 0, 1, and 2")

    pairs = {}
    final_deltas = []
    tail_deltas = []
    l1_deltas = []
    giou_deltas = []
    floor_checks = []
    for seed in sorted(required):
        control = controls[seed]
        method = methods[seed]
        delta = {
            "final_map": method.final_map - control.final_map,
            "tail3_map": method.tail3_map - control.tail3_map,
            "tail3_l1": method.tail3_l1 - control.tail3_l1,
            "tail3_giou": method.tail3_giou - control.tail3_giou,
        }
        final_deltas.append(delta["final_map"])
        tail_deltas.append(delta["tail3_map"])
        l1_deltas.append(delta["tail3_l1"])
        giou_deltas.append(delta["tail3_giou"])
        floor_checks.append(
            method.tail3_map >= 0.8 * control.tail3_map
            if control.tail3_map > 0
            else method.tail3_map >= control.tail3_map
        )
        pairs[str(seed)] = {
            "control": asdict(control),
            "lpr": asdict(method),
            "delta": delta,
        }

    checks = {
        "runtime": all(arm.finite and arm.optimizer_valid for arm in [*controls.values(), *methods.values()]),
        "final_map": sum(delta > 0 for delta in final_deltas) >= 2 and _mean(final_deltas) > 0,
        "tail3_map": sum(delta > 0 for delta in tail_deltas) >= 2 and _mean(tail_deltas) > 0,
        "tail_floor": all(floor_checks),
        "localization": _mean(l1_deltas) < 0 or _mean(giou_deltas) < 0,
        "lpr_evidence": all(arm.gate_active for arm in methods.values()),
    }
    if all(checks.values()):
        recommendation = "fresh_full_data_seed0_pair"
    elif not checks["lpr_evidence"]:
        recommendation = "alpha_lr_multiplier_10"
    elif not checks["runtime"]:
        recommendation = "rerun_invalid_pair_without_parameter_change"
    elif not checks["localization"]:
        recommendation = "last_two_layers_only"
    else:
        recommendation = "max_gate_0.25"
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "pairs": pairs,
        "aggregate_delta": {
            "final_map_mean": _mean(final_deltas),
            "tail3_map_mean": _mean(tail_deltas),
            "tail3_l1_mean": _mean(l1_deltas),
            "tail3_giou_mean": _mean(giou_deltas),
            "final_wins": sum(delta > 0 for delta in final_deltas),
            "tail3_wins": sum(delta > 0 for delta in tail_deltas),
        },
        "recommendation": recommendation,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(stream)]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _optimizer_valid(records: list[dict]) -> bool:
    return (
        len(records) == EXPECTED_OPTIMIZER_ATTEMPTS
        and [record.get("optimizer_attempt") for record in records]
        == list(range(1, EXPECTED_OPTIMIZER_ATTEMPTS + 1))
        and all(
            record.get("amp_scale_before") == 128.0
            and record.get("amp_scale_after") == 128.0
            and record.get("amp_step_skipped") is False
            and record.get("gradient_norm_finite") is True
            for record in records
        )
    )


def _training_diagnostics(records: list[dict], run: Path) -> list[dict]:
    expected = list(range(1, EXPECTED_SCREEN_EPOCHS + 1))
    epochs = [record.get("epoch") for record in records]
    if epochs == expected:
        return records
    if epochs == [*expected, EXPECTED_SCREEN_EPOCHS + 1]:
        return records[:EXPECTED_SCREEN_EPOCHS]
    raise ValueError(f"LPR diagnostics must contain training epochs 1-10 only: {run}; epochs={epochs}")


def load_arm(run_path: str | Path, *, lpr: bool) -> ArmScreenMetrics:
    run = Path(run_path).resolve()
    rows = _read_csv(run / "results.csv")
    if len(rows) != EXPECTED_SCREEN_EPOCHS:
        raise ValueError(f"screen arm must contain exactly 10 result rows: {run}")
    maps = [float(row["metrics/mAP50-95(B)"]) for row in rows]
    l1s = [float(row["val/l1_loss"]) for row in rows]
    gious = [float(row["val/giou_loss"]) for row in rows]
    optimizer_records = _read_jsonl(run / "optimizer-evidence.jsonl")
    diagnostic_values = []
    gate_active = True
    if lpr:
        diagnostics = _training_diagnostics(_read_jsonl(run / "lpr_diagnostics.jsonl"), run)
        diagnostic_values = [
            float(value)
            for record in diagnostics
            for value in (
                *record["gates"],
                record["residual_mean"],
                record["residual_max"],
                record["lpr_grad_norm"],
            )
        ]
        gate_active = all(
            float(record["lpr_grad_norm"]) > 0 and any(abs(float(gate)) > 1e-8 for gate in record["gates"])
            for record in diagnostics
        )
    numeric = [*maps, *l1s, *gious, *diagnostic_values]
    return ArmScreenMetrics(
        final_map=maps[-1],
        tail3_map=_mean(maps[-3:]),
        tail3_l1=_mean(l1s[-3:]),
        tail3_giou=_mean(gious[-3:]),
        finite=all(math.isfinite(value) for value in numeric),
        optimizer_valid=_optimizer_valid(optimizer_records),
        gate_active=gate_active,
        run_path=str(run),
    )


def evaluate_pairs_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = manifest.get("pairs", {})
    controls = {int(seed): load_arm(record["control"], lpr=False) for seed, record in pairs.items()}
    methods = {int(seed): load_arm(record["lpr"], lpr=True) for seed, record in pairs.items()}
    return evaluate_paired_screen(controls, methods)


def write_report(report: dict, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate three strict paired LPR screening seeds.")
    parser.add_argument("--pairs-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_pairs_manifest(args.pairs_manifest)
    write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
