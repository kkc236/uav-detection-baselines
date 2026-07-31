"""Evaluate the frozen 10-epoch LPR screening gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


BASELINE = {
    "map": 0.04098,
    "map50": 0.08404,
    "val_giou": 1.27020,
    "val_l1": 0.19467,
}


@dataclass(frozen=True)
class ScreenReport:
    passed: bool
    checks: dict[str, bool]
    measured: dict[str, Any]
    baseline: dict[str, float]
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fallback(checks: dict[str, bool]) -> str:
    if all(checks.values()):
        return "resume_to_100"
    if not checks["finite"]:
        return "max_gate_0.25"
    if not checks["gate_active"]:
        return "alpha_lr_multiplier_10"
    if not checks["localization"]:
        return "last_two_layers_only"
    return "max_gate_0.25"


def evaluate_screen(
    *,
    map: float,
    map50: float,
    val_giou: float,
    val_l1: float,
    finite: bool,
    gate_active: bool,
    diagnostics: dict[str, Any] | None = None,
) -> ScreenReport:
    numeric_finite = all(math.isfinite(value) for value in (map, map50, val_giou, val_l1))
    checks = {
        "map": map >= BASELINE["map"],
        "map50_tradeoff": map50 >= BASELINE["map50"]
        or (map - BASELINE["map"] >= 0.002 and BASELINE["map50"] - map50 <= 0.002),
        "localization": val_l1 <= BASELINE["val_l1"] or val_giou <= BASELINE["val_giou"] * 0.98,
        "finite": bool(finite and numeric_finite),
        "gate_active": bool(gate_active),
    }
    measured = {
        "map": float(map),
        "map50": float(map50),
        "val_giou": float(val_giou),
        "val_l1": float(val_l1),
        **(diagnostics or {}),
    }
    return ScreenReport(
        passed=all(checks.values()),
        checks=checks,
        measured=measured,
        baseline=BASELINE.copy(),
        recommendation=_fallback(checks),
    )


def _last_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No result rows found in {path}")
    return {key.strip(): value.strip() for key, value in rows[-1].items()}


def _last_jsonl_row(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No diagnostic rows found in {path}")
    return json.loads(lines[-1])


def evaluate_run(run: str | Path) -> ScreenReport:
    run_path = Path(run)
    row = _last_csv_row(run_path / "results.csv")
    diagnostics = _last_jsonl_row(run_path / "lpr_diagnostics.jsonl")
    gates = [float(value) for value in diagnostics["gates"]]
    grad_norm = float(diagnostics["lpr_grad_norm"])
    diagnostic_numbers = [
        *gates,
        grad_norm,
        float(diagnostics["residual_mean"]),
        float(diagnostics["residual_max"]),
        float(diagnostics["map75"]),
    ]
    finite = all(math.isfinite(value) for value in diagnostic_numbers)
    gate_active = grad_norm > 0 and any(abs(value) > 1e-8 for value in gates)
    return evaluate_screen(
        map=float(row["metrics/mAP50-95(B)"]),
        map50=float(row["metrics/mAP50(B)"]),
        val_giou=float(row["val/giou_loss"]),
        val_l1=float(row["val/l1_loss"]),
        finite=finite,
        gate_active=gate_active,
        diagnostics={
            "epoch": int(float(row["epoch"])),
            "map75": float(diagnostics["map75"]),
            "gates": gates,
            "residual_mean": float(diagnostics["residual_mean"]),
            "residual_max": float(diagnostics["residual_max"]),
            "lpr_grad_norm": grad_norm,
            "cuda_peak_mib": float(diagnostics["cuda_peak_mib"]),
        },
    )


def write_report(report: ScreenReport, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a completed LPR 10-epoch run.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_run(args.run)
    write_report(report, args.output)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
