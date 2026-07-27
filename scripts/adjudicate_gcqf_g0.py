"""Aggregate the two preregistered GCQF G0 seed evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from src.sbr_artifacts import atomic_write_json, sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate seed0/seed1 GCQF accuracy gates."
    )
    parser.add_argument("--seed0", type=Path, required=True)
    parser.add_argument("--seed1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _delta(seed: Mapping[str, Any], group: str, metric: str) -> float:
    return float(seed["deltas"][group][metric])


def two_seed_gate(
    seeds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(seeds) != 2:
        raise ValueError("GCQF G0 requires exactly two seeds")
    map_values = [
        _delta(seed, "full_minus_anchor", "mAP50-95")
        for seed in seeds
    ]
    tiny_values = [
        _delta(seed, "full_minus_anchor", "AP-tiny-SBR")
        for seed in seeds
    ]
    ap75_values = [
        _delta(seed, "full_minus_anchor", "AP75")
        for seed in seeds
    ]
    medium_values = [
        _delta(seed, "full_minus_anchor", "AP-medium-SBR")
        for seed in seeds
    ]
    large_values = [
        _delta(seed, "full_minus_anchor", "AP-large-SBR")
        for seed in seeds
    ]
    global_large_values = [
        _delta(seed, "full_minus_global", "AP-large-SBR")
        for seed in seeds
    ]
    averages = {
        "mAP50-95": mean(map_values),
        "AP-tiny-SBR": mean(tiny_values),
        "AP75": mean(ap75_values),
        "AP-medium-SBR": mean(medium_values),
        "AP-large-SBR": mean(large_values),
        "AP-large-SBR-vs-global": mean(global_large_values),
    }
    checks = {
        "both_anchor_exact": all(
            seed["anchor_reference"]["exact"] is True for seed in seeds
        ),
        "both_protected_exact": all(
            seed["protected_global_exact"] is True for seed in seeds
        ),
        "both_residual_active": all(
            seed["per_seed_gate"]["residual_is_active"] is True
            for seed in seeds
        ),
        "both_residual_not_saturated": all(
            seed["per_seed_gate"]["residual_not_saturated"] is True
            for seed in seeds
        ),
        "both_seed_map_positive": all(value > 0.0 for value in map_values),
        "mean_map_at_least_0_003": averages["mAP50-95"] >= 0.003,
        "mean_tiny_or_ap75_material": (
            averages["AP-tiny-SBR"] >= 0.005
            or averages["AP75"] >= 0.003
        ),
        "mean_medium_within_budget": (
            averages["AP-medium-SBR"] >= -0.002
        ),
        "mean_large_within_fixed_budget": (
            averages["AP-large-SBR"] >= -0.002
        ),
        "mean_large_within_global_budget": (
            averages["AP-large-SBR-vs-global"] >= -0.005
        ),
    }
    checks["advance_accuracy"] = all(checks.values())
    return {
        **checks,
        "averages": averages,
        "per_seed_map": map_values,
    }


def adjudicate(args: argparse.Namespace) -> tuple[Path, bool]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    paths = [args.seed0.resolve(), args.seed1.resolve()]
    seeds = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in paths
    ]
    if any(
        seed.get("schema_version") != "gcte-gcqf-five-state/v1"
        for seed in seeds
    ):
        raise ValueError("GCQF seed evaluation schema drift")
    gate = two_seed_gate(seeds)
    result = {
        "schema_version": "gcte-gcqf-two-seed-adjudication/v1",
        "seeds": [
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "module": seed["module"],
            }
            for path, seed in zip(paths, seeds, strict=True)
        ],
        "gate": gate,
    }
    atomic_write_json(output, result)
    print(
        f"GCQF_TWO_SEED_COMPLETE "
        f"advance={gate['advance_accuracy']} output={output}",
        flush=True,
    )
    return output, bool(gate["advance_accuracy"])


def main() -> None:
    path, passed = adjudicate(build_parser().parse_args())
    print(path)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
