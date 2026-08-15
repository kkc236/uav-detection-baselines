"""Benchmark identical ordinary-FDR inference graphs from paired Formal100 arms."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bpdd_formal_evaluation import (  # noqa: E402
    load_exact_final_checkpoint,
    write_create_only_json,
)


BENCHMARK_PROTOCOL = {
    "imgsz": 640,
    "batch": 1,
    "half": True,
    "warmup": 50,
    "runs": 200,
    "measurement": "alternating",
}


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("latency samples cannot be empty")
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    samples = [float(value) for value in values]
    if not samples or not all(math.isfinite(value) and value > 0 for value in samples):
        raise ValueError("latency samples must be finite and positive")
    mean = sum(samples) / len(samples)
    return {
        "mean_ms": mean,
        "p50_ms": _percentile(samples, 0.50),
        "p95_ms": _percentile(samples, 0.95),
        "fps": 1000.0 / mean,
    }


def measurement_order(iteration: int) -> tuple[str, str]:
    return ("fdr", "fdr_bpdd") if iteration % 2 == 0 else ("fdr_bpdd", "fdr")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_pair_latency(
    fdr: torch.nn.Module,
    fdr_bpdd: torch.nn.Module,
    *,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    if device.type != "cuda":
        raise RuntimeError("the frozen BPDD latency protocol requires CUDA FP16")
    image = torch.randn(
        BENCHMARK_PROTOCOL["batch"],
        3,
        BENCHMARK_PROTOCOL["imgsz"],
        BENCHMARK_PROTOCOL["imgsz"],
        device=device,
        dtype=torch.float16,
    )
    models = {
        "fdr": fdr.to(device).eval().half(),
        "fdr_bpdd": fdr_bpdd.to(device).eval().half(),
    }
    samples: dict[str, list[float]] = {"fdr": [], "fdr_bpdd": []}
    with torch.inference_mode():
        for iteration in range(BENCHMARK_PROTOCOL["warmup"]):
            for name in measurement_order(iteration):
                models[name].predict(image)
        _synchronize(device)
        for iteration in range(BENCHMARK_PROTOCOL["runs"]):
            for name in measurement_order(iteration):
                _synchronize(device)
                start = time.perf_counter()
                models[name].predict(image)
                _synchronize(device)
                samples[name].append((time.perf_counter() - start) * 1000.0)
    return latency_summary(samples["fdr"]), latency_summary(samples["fdr_bpdd"])


def build_efficiency_report(
    fdr: Mapping[str, Any],
    fdr_bpdd: Mapping[str, Any],
    *,
    fdr_latency: Mapping[str, float],
    bpdd_latency: Mapping[str, float],
) -> dict[str, Any]:
    if int(fdr["parameters"]) != int(fdr_bpdd["parameters"]):
        raise ValueError("FDR and FDR+BPDD inference parameters must be strictly equal")
    if float(fdr["gflops"]) != float(fdr_bpdd["gflops"]):
        raise ValueError("FDR and FDR+BPDD inference GFLOPs must be strictly equal")
    return {
        "deployment_graph": "ordinary-fdr-for-both-arms",
        "parameters": {
            "fdr": int(fdr["parameters"]),
            "fdr_bpdd": int(fdr_bpdd["parameters"]),
            "delta": 0,
            "strictly_equal": True,
        },
        "gflops": {
            "fdr": float(fdr["gflops"]),
            "fdr_bpdd": float(fdr_bpdd["gflops"]),
            "delta": 0.0,
            "strictly_equal": True,
        },
        "latency": {
            "fdr": dict(fdr_latency),
            "fdr_bpdd": dict(bpdd_latency),
        },
    }


def _model_cost(model: torch.nn.Module) -> dict[str, int | float]:
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "gflops": float(get_flops(model, imgsz=BENCHMARK_PROTOCOL["imgsz"])),
    }


def _read_evaluation(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"independent evaluation is not an object: {path}")
    return payload


def validate_evaluation_pair(
    fdr: Mapping[str, Any], fdr_bpdd: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind efficiency evidence to the already-independent Formal100 pair."""

    identities: dict[str, Mapping[str, Any]] = {}
    checkpoints: dict[str, Mapping[str, Any]] = {}
    for variant, evaluation in (("fdr", fdr), ("fdr_bpdd", fdr_bpdd)):
        identity = evaluation.get("evaluation_identity")
        checkpoint = evaluation.get("checkpoint")
        if not isinstance(identity, Mapping) or not isinstance(checkpoint, Mapping):
            raise ValueError("benchmark paired authority is missing evaluation identity")
        if (
            evaluation.get("format_version") != 1
            or identity.get("variant") != variant
            or identity.get("stage") != "formal"
            or identity.get("seed") != 0
            or checkpoint.get("kind") != "exact-final-ema"
            or checkpoint.get("completed_epoch") != 100
            or checkpoint.get("sha256_verified") is not True
            or checkpoint.get("remote_published") is not True
            or not checkpoint.get("remote_asset")
            or len(str(checkpoint.get("sha256", ""))) != 64
        ):
            raise ValueError(f"benchmark paired authority is invalid for {variant}")
        identities[variant] = identity
        checkpoints[variant] = checkpoint
    common_fields = (
        "source_sha256",
        "protocol_sha256",
        "fdr_protocol_sha256",
        "initial_state_sha256",
        "stage",
        "seed",
        "data",
        "dataset_sha256",
    )
    if any(
        identities["fdr"].get(field) != identities["fdr_bpdd"].get(field)
        for field in common_fields
    ):
        raise ValueError("benchmark paired authority differs between Formal100 arms")
    if identities["fdr"].get("run_id") == identities["fdr_bpdd"].get("run_id"):
        raise ValueError("benchmark paired authority must contain distinct arm run ids")
    return {
        "comparison": "fresh-fdr-vs-fresh-fdr-bpdd",
        "runs": {
            variant: str(identities[variant]["run_id"])
            for variant in ("fdr", "fdr_bpdd")
        },
        "data": {
            "yaml": str(identities["fdr"]["data"]),
            "dataset_sha256": str(identities["fdr"]["dataset_sha256"]),
        },
        "remote_assets": {
            variant: str(checkpoints[variant]["remote_asset"])
            for variant in ("fdr", "fdr_bpdd")
        },
        "checkpoint_sha256": {
            variant: str(checkpoints[variant]["sha256"]).upper()
            for variant in ("fdr", "fdr_bpdd")
        },
    }


def benchmark_formal_pair(
    *,
    fdr_checkpoint: str | Path,
    bpdd_checkpoint: str | Path,
    fdr_evaluation: str | Path,
    bpdd_evaluation: str | Path,
    output: str | Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen BPDD benchmark requires an available CUDA GPU")
    pair_authority = validate_evaluation_pair(
        _read_evaluation(fdr_evaluation),
        _read_evaluation(bpdd_evaluation),
    )
    fdr = load_exact_final_checkpoint(
        fdr_checkpoint,
        expected_sha256=pair_authority["checkpoint_sha256"]["fdr"],
    )
    bpdd = load_exact_final_checkpoint(
        bpdd_checkpoint,
        expected_sha256=pair_authority["checkpoint_sha256"]["fdr_bpdd"],
    )
    if fdr.metadata["kind"] != "exact-final-ema" or bpdd.metadata["kind"] != "exact-final-ema":
        raise ValueError("paired benchmark requires exact-final EMA checkpoints")
    fdr_cost = _model_cost(fdr.model)
    bpdd_cost = _model_cost(bpdd.model)
    fdr_latency, bpdd_latency = benchmark_pair_latency(
        fdr.model,
        bpdd.model,
        device=torch_device,
    )
    report = {
        "format_version": 1,
        "protocol": {**BENCHMARK_PROTOCOL, "device": str(torch_device)},
        "pair_authority": pair_authority,
        "checkpoints": {"fdr": fdr.metadata, "fdr_bpdd": bpdd.metadata},
        "efficiency": build_efficiency_report(
            fdr_cost,
            bpdd_cost,
            fdr_latency=fdr_latency,
            bpdd_latency=bpdd_latency,
        ),
    }
    write_create_only_json(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdr-checkpoint", type=Path, required=True)
    parser.add_argument("--bpdd-checkpoint", type=Path, required=True)
    parser.add_argument("--fdr-evaluation", type=Path, required=True)
    parser.add_argument("--bpdd-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = benchmark_formal_pair(
        fdr_checkpoint=args.fdr_checkpoint,
        bpdd_checkpoint=args.bpdd_checkpoint,
        fdr_evaluation=args.fdr_evaluation,
        bpdd_evaluation=args.bpdd_evaluation,
        output=args.output,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
