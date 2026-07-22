from __future__ import annotations

import argparse
import json
import math
import statistics
from hashlib import sha256
from pathlib import Path
from typing import Any


SEEDS = (0, 1, 2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply the frozen three-seed TSGR-P2 E1 gate.")
    parser.add_argument("--control-run", action="append", required=True, type=_seed_path)
    parser.add_argument("--tsgr-run", action="append", required=True, type=_seed_path)
    parser.add_argument("--control-diagnostics", action="append", required=True, type=_seed_paths)
    parser.add_argument("--tsgr-diagnostics", action="append", required=True, type=_seed_paths)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_report(
    control_runs: dict[int, dict],
    tsgr_runs: dict[int, dict],
    control_diagnostics: dict[int, list[dict]],
    tsgr_diagnostics: dict[int, list[dict]],
) -> dict[str, Any]:
    for mapping in (control_runs, tsgr_runs, control_diagnostics, tsgr_diagnostics):
        if set(mapping) != set(SEEDS):
            raise ValueError(f"E1 evidence must contain seeds {SEEDS}")

    seed_reports = {}
    experiment_signatures = set()
    subset_signatures = set()
    for seed in SEEDS:
        control = control_runs[seed]
        method = tsgr_runs[seed]
        _validate_run_manifest(control, seed=seed, arm="control")
        _validate_run_manifest(method, seed=seed, arm="tsgr-p2")
        for run in (control, method):
            experiment_signatures.add(run["protocol"]["experiment_signature"])
            subset_signatures.add(run["protocol"]["subset"]["sha256"])
        if control["protocol"]["signature"] != method["protocol"]["signature"]:
            raise ValueError(f"seed {seed} arms used different protocol manifests")
        if control["protocol"]["initial_state_sha256"] != method["protocol"]["initial_state_sha256"]:
            raise ValueError(f"seed {seed} arms used different initial states")

        control_stock = _diagnostic_tail(control_diagnostics[seed], control)
        method_stock = _diagnostic_tail(tsgr_diagnostics[seed], method)
        final_delta = method["results"]["final_map50_95"] - control["results"]["final_map50_95"]
        tail_delta = method["results"]["tail3_map50_95"] - control["results"]["tail3_map50_95"]
        coverage_delta = method_stock["coverage"] - control_stock["coverage"]
        rank_delta = method_stock["normalized_best_rank"] - control_stock["normalized_best_rank"]
        seed_reports[str(seed)] = {
            "control": {
                "final_map50_95": control["results"]["final_map50_95"],
                "tail3_map50_95": control["results"]["tail3_map50_95"],
                "final_ap_tiny": control["results"]["final_ap_tiny"],
                "final_recall_tiny": control["results"]["final_recall_tiny"],
                **control_stock,
            },
            "tsgr_p2": {
                "final_map50_95": method["results"]["final_map50_95"],
                "tail3_map50_95": method["results"]["tail3_map50_95"],
                "final_ap_tiny": method["results"]["final_ap_tiny"],
                "final_recall_tiny": method["results"]["final_recall_tiny"],
                **method_stock,
            },
            "delta_tsgr_minus_control": {
                "final_map50_95": final_delta,
                "tail3_map50_95": tail_delta,
                "stock_top300_coverage": coverage_delta,
                "normalized_best_rank": rank_delta,
            },
            "no_collapse": method["results"]["tail3_map50_95"] >= 0.8 * control["results"]["tail3_map50_95"],
        }

    if len(experiment_signatures) != 1:
        raise ValueError("E1 runs do not share one frozen experiment signature")
    if len(subset_signatures) != 1:
        raise ValueError("E1 runs do not share one frozen subset")

    deltas = [seed_reports[str(seed)]["delta_tsgr_minus_control"] for seed in SEEDS]
    final_wins = sum(delta["final_map50_95"] > 0 for delta in deltas)
    tail_wins = sum(delta["tail3_map50_95"] > 0 for delta in deltas)
    coverage_wins = sum(delta["stock_top300_coverage"] > 0 for delta in deltas)
    rank_wins = sum(delta["normalized_best_rank"] < 0 for delta in deltas)
    mean_deltas = {
        key: statistics.mean(delta[key] for delta in deltas)
        for key in ("final_map50_95", "tail3_map50_95", "stock_top300_coverage", "normalized_best_rank")
    }
    conditions = {
        "final_wins_at_least_2_of_3": final_wins >= 2,
        "mean_final_delta_positive": mean_deltas["final_map50_95"] > 0,
        "tail_wins_at_least_2_of_3": tail_wins >= 2,
        "mean_tail_delta_positive": mean_deltas["tail3_map50_95"] > 0,
        "no_seed_collapsed_below_80_percent": all(seed_reports[str(seed)]["no_collapse"] for seed in SEEDS),
        "coverage_wins_at_least_2_of_3": coverage_wins >= 2,
        "mean_coverage_delta_positive": mean_deltas["stock_top300_coverage"] > 0,
        "rank_wins_at_least_2_of_3": rank_wins >= 2,
        "mean_normalized_rank_delta_negative": mean_deltas["normalized_best_rank"] < 0,
    }
    passed = all(conditions.values())
    report = {
        "format_version": 1,
        "classification": "TSGR_E1_PASS" if passed else "TSGR_E1_FAIL",
        "passed": passed,
        "experiment_signature": next(iter(experiment_signatures)),
        "subset_sha256": next(iter(subset_signatures)),
        "seeds": seed_reports,
        "aggregate": {
            "final_wins": final_wins,
            "tail_wins": tail_wins,
            "coverage_wins": coverage_wins,
            "rank_wins": rank_wins,
            "mean_deltas": mean_deltas,
        },
        "conditions": conditions,
    }
    report["signature"] = _json_sha256(report)
    return report


def _validate_run_manifest(run: dict, *, seed: int, arm: str) -> None:
    signed = dict(run)
    signature = signed.pop("signature", None)
    if signature != _json_sha256(signed):
        raise ValueError(f"invalid {arm} seed {seed} run-manifest signature")
    if run.get("stage") != "e1" or run.get("arm") != arm or run.get("seed") != seed:
        raise ValueError(f"wrong E1 run identity for {arm} seed {seed}")
    if run.get("controlled_amp", {}).get("skipped_attempts") != 0:
        raise ValueError(f"AMP skip in {arm} seed {seed}")
    if run.get("results", {}).get("epochs") != 10:
        raise ValueError(f"incomplete {arm} seed {seed} run")
    if arm == "tsgr-p2":
        expected = {
            "lambda_p2": 0.1,
            "lambda_ebc": 0.0,
            "lambda_quality": 0.0,
            "query_injection_enabled": False,
            "quality_gated_p2": False,
            "quality_weighted_ebc": False,
            "learnable_fusion_gamma": False,
            "p2_c2_grad_scale": 0.1,
            "contribution_separated_aux_gradients": True,
        }
        config = run.get("ebc_config", {})
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError(f"changed TSGR config in seed {seed}")
        evidence = run.get("optimizer_evidence", {})
        if evidence.get("p2_entry_count_max") != 0 or evidence.get("ordinary_query_count_values") != [300]:
            raise ValueError(f"query isolation failed in TSGR seed {seed}")


def _diagnostic_tail(records: list[dict], run: dict) -> dict[str, float]:
    if len(records) != 3:
        raise ValueError("stock-query tail requires exactly three checkpoint diagnostics")
    checkpoint_hashes = {item["sha256"] for item in run.get("epoch_checkpoints", [])}
    for record in records:
        signed = dict(record)
        signature = signed.pop("signature", None)
        if signature != _json_sha256(signed):
            raise ValueError("invalid stock-query diagnostic signature")
        if record.get("checkpoint_sha256") not in checkpoint_hashes:
            raise ValueError("stock-query diagnostic checkpoint is not in its run manifest")
        stock = record.get("stock_query", {})
        for key in ("stock_top300_coverage", "normalized_best_rank_mean"):
            value = float(stock[key])
            if not math.isfinite(value):
                raise ValueError(f"non-finite stock-query diagnostic {key}")
    return {
        "coverage": statistics.mean(float(record["stock_query"]["stock_top300_coverage"]) for record in records),
        "normalized_best_rank": statistics.mean(
            float(record["stock_query"]["normalized_best_rank_mean"]) for record in records
        ),
    }


def _seed_path(value: str) -> tuple[int, Path]:
    seed, separator, path = value.partition("=")
    if not separator or not seed.isdigit():
        raise argparse.ArgumentTypeError("expected SEED=PATH")
    return int(seed), Path(path)


def _seed_paths(value: str) -> tuple[int, list[Path]]:
    seed, path = _seed_path(value)
    paths = [Path(item) for item in str(path).split(",") if item]
    if len(paths) != 3:
        raise argparse.ArgumentTypeError("expected SEED=PATH,PATH,PATH")
    return seed, paths


def _mapping(items: list[tuple[int, Any]], label: str) -> dict[int, Any]:
    result = dict(items)
    if len(result) != len(items):
        raise ValueError(f"duplicate seed in {label}")
    return result


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _json_sha256(payload: object) -> str:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256(content).hexdigest().upper()


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to replace E1 report: {args.output}")
    control_paths = _mapping(args.control_run, "control runs")
    tsgr_paths = _mapping(args.tsgr_run, "TSGR runs")
    control_diagnostic_paths = _mapping(args.control_diagnostics, "control diagnostics")
    tsgr_diagnostic_paths = _mapping(args.tsgr_diagnostics, "TSGR diagnostics")
    report = build_report(
        {seed: _load_json(path) for seed, path in control_paths.items()},
        {seed: _load_json(path) for seed, path in tsgr_paths.items()},
        {seed: [_load_json(path) for path in paths] for seed, paths in control_diagnostic_paths.items()},
        {seed: [_load_json(path) for path in paths] for seed, paths in tsgr_diagnostic_paths.items()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
