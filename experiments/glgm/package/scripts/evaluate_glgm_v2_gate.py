#!/usr/bin/env python3
"""Evaluate preregistered GLGM-v2 Screen10 or Screen30 promotion gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = PACKAGE_ROOT / "GLGM_V2_PREREGISTRATION.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name}: {value}")
    return result


def load_results(receipt: dict[str, Any]) -> list[dict[str, float]]:
    path = Path(receipt["training_completion"]["results_csv"])
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    parsed = []
    for row in rows:
        parsed.append(
            {key: finite(value, f"{path}:{key}") for key, value in row.items()}
        )
    return parsed


def delta_percent(candidate: float, control: float) -> float:
    return 100.0 * (candidate - control) / control


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    actual: float,
    operator: str,
    threshold: float,
) -> None:
    passed = actual >= threshold if operator == ">=" else actual <= threshold
    checks.append(
        {
            "name": name,
            "actual": actual,
            "operator": operator,
            "threshold": threshold,
            "pass": passed,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen10", "screen30"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.work_root.resolve()
    artifacts = root / "artifacts"
    prereg = load_json(PREREGISTRATION)
    manifest = load_json(artifacts / "paired_preflight_manifest.json")
    if manifest.get("experiment_variant") not in {
        row["id"] for row in prereg["variants"]
    }:
        raise RuntimeError(
            f"unregistered variant: {manifest.get('experiment_variant')}"
        )
    expected_epochs = 10 if args.stage == "screen10" else 30
    control_receipt = load_json(artifacts / "control-train-receipt.json")
    method_receipt = load_json(artifacts / "glgm-train-receipt.json")
    control_rows = load_results(control_receipt)
    method_rows = load_results(method_receipt)
    if len(control_rows) != expected_epochs or len(method_rows) != expected_epochs:
        raise RuntimeError(
            f"{args.stage} requires {expected_epochs} completed epochs, got "
            f"control={len(control_rows)}, method={len(method_rows)}"
        )

    checks: list[dict[str, Any]] = []
    efficiency = prereg["efficiency"]
    private_delta = finite(manifest["parameter_delta"], "parameter_delta")
    reduction = 100.0 * (1.0 - private_delta / prereg["baseline_v1_private_parameters"])
    add_check(
        checks,
        "private_parameter_reduction_from_v1_percent",
        reduction,
        ">=",
        efficiency["private_parameter_reduction_from_v1_min_percent"],
    )
    add_check(
        checks,
        "total_parameter_delta_percent",
        finite(manifest["parameter_delta_percent"], "parameter_delta_percent"),
        "<=",
        efficiency["total_parameter_delta_percent_max"],
    )

    control_bench = load_json(artifacts / "control-best-benchmark.json")
    method_bench = load_json(artifacts / "glgm-best-benchmark.json")
    for key, threshold_key in (
        ("mean_ms", "mean_latency_delta_percent_max"),
        ("p95_ms", "p95_latency_delta_percent_max"),
        ("peak_allocated_vram_bytes", "peak_vram_delta_percent_max"),
    ):
        actual = delta_percent(
            finite(method_bench[key], key), finite(control_bench[key], key)
        )
        add_check(
            checks, f"{key}_delta_percent", actual, "<=", efficiency[threshold_key]
        )

    map_key = "metrics/mAP50-95(B)"
    recall_key = "metrics/recall(B)"
    early_map_deltas = [
        100.0 * (method_rows[index][map_key] - control_rows[index][map_key])
        for index in range(2, 10)
    ]
    mean_early_map = sum(early_map_deltas) / len(early_map_deltas)
    screen10 = prereg["screen10"]
    add_check(
        checks,
        "mean_map_delta_pp_epochs_3_10",
        mean_early_map,
        ">=",
        screen10["mean_map_delta_pp_epochs_3_10_min"],
    )
    add_check(
        checks,
        "epoch10_map_delta_pp",
        100.0 * (method_rows[9][map_key] - control_rows[9][map_key]),
        ">=",
        screen10["epoch10_map_delta_pp_min"],
    )
    add_check(
        checks,
        "epoch10_recall_delta_pp",
        100.0 * (method_rows[9][recall_key] - control_rows[9][recall_key]),
        ">=",
        screen10["epoch10_recall_delta_pp_min"],
    )

    last_comparison = load_json(artifacts / "comparison-last.json")
    best_comparison = load_json(artifacts / "comparison-best.json")
    if args.stage == "screen30":
        screen30 = prereg["screen30"]
        add_check(
            checks,
            "last_map_delta_pp",
            finite(
                last_comparison["metrics"]["map50_95"]["percentage_point_delta"],
                "last_map",
            ),
            ">=",
            screen30["last_map_delta_pp_min"],
        )
        add_check(
            checks,
            "best_map_delta_pp",
            finite(
                best_comparison["metrics"]["map50_95"]["percentage_point_delta"],
                "best_map",
            ),
            ">=",
            screen30["best_map_delta_pp_min"],
        )
        add_check(
            checks,
            "last_recall_delta_pp",
            finite(
                last_comparison["metrics"]["recall"]["percentage_point_delta"],
                "last_recall",
            ),
            ">=",
            screen30["last_recall_delta_pp_min"],
        )
        for metric in screen30["last_non_degrading_metrics"]:
            add_check(
                checks,
                f"last_{metric}_delta_pp",
                finite(
                    last_comparison["metrics"][metric]["percentage_point_delta"], metric
                ),
                ">=",
                0.0,
            )
        class_rows = last_comparison["per_class_delta"]
        non_degrading = sum(
            finite(row["delta"]["map50_95"], row["name"]) >= 0.0 for row in class_rows
        )
        key_classes = set(screen30["key_classes"])
        key_non_degrading = sum(
            row["name"] in key_classes
            and finite(row["delta"]["map50_95"], row["name"]) >= 0.0
            for row in class_rows
        )
        add_check(
            checks,
            "non_degrading_class_count",
            float(non_degrading),
            ">=",
            float(screen30["non_degrading_class_count_min"]),
        )
        add_check(
            checks,
            "key_non_degrading_class_count",
            float(key_non_degrading),
            ">=",
            float(screen30["key_non_degrading_class_count_min"]),
        )

    payload = {
        "schema": "glgm-v2-promotion-gate-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "variant": manifest["experiment_variant"],
        "work_root": str(root),
        "pass": all(row["pass"] for row in checks),
        "checks": checks,
        "ranking": {
            "last_map_delta_pp": finite(
                last_comparison["metrics"]["map50_95"]["percentage_point_delta"],
                "ranking_map",
            ),
            "last_recall_delta_pp": finite(
                last_comparison["metrics"]["recall"]["percentage_point_delta"],
                "ranking_recall",
            ),
            "mean_early_map_delta_pp": mean_early_map,
        },
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
