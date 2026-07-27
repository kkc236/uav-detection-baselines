from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics.data.utils import check_det_dataset

from scripts.evaluate_gcmv_plec import (
    DEFAULT_MODEL,
    _build_dataset,
    _run_arm,
    jsonable,
    metric_deltas,
)
from src.sbr_artifacts import (
    atomic_write_json,
    environment_info,
    load_dataset,
)
from src.sbr_metrics import evaluate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Control, Method-On, and Method-Off."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--method-checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    return parser


def advance_gate(
    metrics: dict[str, dict],
    runtime: dict[str, dict],
) -> dict[str, bool]:
    total = metric_deltas(metrics["control"], metrics["method_on"])
    direct = metric_deltas(metrics["method_off"], metrics["method_on"])
    gate = (
        runtime["method_on"]
        .get("gcmv_diagnostics", {})
        .get("peg_gate", {})
    )
    gamma = (
        runtime["method_on"]
        .get("checkpoint", {})
        .get("gcmv_gamma", 0.0)
    )
    checks = {
        "tiny_ap_improves_control": total.get("AP-tiny-SBR", 0.0) > 0.0,
        "tiny_recall_improves_control": (
            total.get("tiny_recall", 0.0) > 0.0
        ),
        "map_nonnegative": total.get("mAP50-95", -1.0) >= 0.0,
        "medium_within_budget": (
            total.get("AP-medium-SBR", -1.0) >= -0.002
        ),
        "large_within_budget": (
            total.get("AP-large-SBR", -1.0) >= -0.005
        ),
        "direct_tiny_ap_positive": (
            direct.get("AP-tiny-SBR", 0.0) > 0.0
        ),
        "gamma_materially_open": (
            isinstance(gamma, (int, float))
            and abs(float(gamma)) >= 1e-4
        ),
        "spatial_gate_nondegenerate": (
            gate.get("count", 0.0) > 0.0
            and gate.get("std", 0.0) >= 1e-4
            and gate.get("max", 0.0) > gate.get("min", 0.0)
        ),
    }
    checks["advance"] = all(checks.values())
    return checks


def evaluate(args: argparse.Namespace) -> Path:
    if args.batch <= 0 or args.workers < 0:
        raise ValueError("batch must be positive and workers non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("three-state GCMV evaluation requires CUDA")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(f"cuda:{args.device}")
    data = check_det_dataset(args.data, autodownload=False)
    authority = load_dataset(args.data, split="val")
    dataset = _build_dataset(data, batch=args.batch)
    if len(dataset) != int(authority["image_count"]):
        raise RuntimeError("validation dataset count does not match authority")

    rows: dict[str, list[dict]] = {}
    runtime: dict[str, dict] = {}
    arms = (
        ("control", "control", args.control_checkpoint),
        ("method_on", "method", args.method_checkpoint),
        ("method_off", "method_off", args.method_checkpoint),
    )
    for result_name, execution_arm, checkpoint in arms:
        rows[result_name], runtime[result_name] = _run_arm(
            arm=execution_arm,
            checkpoint=checkpoint,
            model_path=args.model,
            data=data,
            dataset=dataset,
            authority=authority,
            batch=args.batch,
            workers=args.workers,
            device=device,
        )
    metrics = {
        name: evaluate_dataset(arm_rows)
        for name, arm_rows in rows.items()
    }
    deltas = {
        "total_method_on_minus_control": metric_deltas(
            metrics["control"], metrics["method_on"]
        ),
        "direct_method_on_minus_method_off": metric_deltas(
            metrics["method_off"], metrics["method_on"]
        ),
        "training_drift_method_off_minus_control": metric_deltas(
            metrics["control"], metrics["method_off"]
        ),
    }
    result = {
        "schema_version": "gcmv-ei-warmstart-three-state/v1",
        "data": {
            "yaml": str(Path(args.data).resolve()),
            "yaml_sha256": authority["yaml_hash"],
            "dataset_signature": authority["dataset_signature"],
            "image_count": authority["image_count"],
        },
        "metrics": metrics,
        "deltas": deltas,
        "runtime": runtime,
        "advance_gate": advance_gate(metrics, runtime),
        "environment": environment_info(),
    }
    atomic_write_json(output, jsonable(result))
    return output


def main() -> None:
    print(evaluate(build_parser().parse_args()))


if __name__ == "__main__":
    main()
