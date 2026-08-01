"""Benchmark truthful parameter, GFLOPs, and latency overhead for I-TBER."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from ultralytics.utils.torch_utils import get_flops

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.train_itber import (  # noqa: E402
    TRAINING_CONSTANTS,
    validate_gate1_cache_manifest,
    validate_resume_checkpoint,
)
from src.itber_protocol import (  # noqa: E402
    EXPECTED_BASELINE_SHA256,
    module_state_sha256,
)
from src.lpr_protocol import file_sha256  # noqa: E402
from src.rtdetr_itber import FrozenITBERAdapter  # noqa: E402


BENCHMARK_PROTOCOL = {
    "device": "cuda:0",
    "input": [1, 3, 640, 640],
    "dtype": "float16",
    "warmup": 50,
    "iterations": 200,
    "synchronized_cuda": True,
    "targets_nonblocking": {
        "parameters_percent": 1.0,
        "gflops_percent": 1.0,
        "latency_percent": 3.0,
    },
}


def percentage_increase(value: float, baseline: float) -> float:
    if baseline <= 0:
        raise ValueError("benchmark baseline must be positive")
    return (float(value) / float(baseline) - 1.0) * 100.0


def parameter_report(
    baseline: torch.nn.Module,
    method: torch.nn.Module,
) -> dict[str, float | int]:
    baseline_total = sum(parameter.numel() for parameter in baseline.parameters())
    method_total = sum(parameter.numel() for parameter in method.parameters())
    private_total = sum(parameter.numel() for parameter in method.refiner.parameters())
    detector_total = sum(parameter.numel() for parameter in method.detector.parameters())
    if detector_total != baseline_total:
        raise RuntimeError("I-TBER detector parameter count differs from baseline")
    if method_total - baseline_total != private_total:
        raise RuntimeError("I-TBER overhead is not attributable only to private parameters")
    return {
        "baseline_total": baseline_total,
        "method_total": method_total,
        "private_total": private_total,
        "increase_percent": percentage_increase(method_total, baseline_total),
    }


def measurement_order(iteration: int) -> tuple[str, str, str]:
    names = ("control", "stock", "refined")
    offset = iteration % len(names)
    return names[offset:] + names[:offset]


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("I-TBER latency samples must not be empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": sum(values) / len(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
    }


def benchmark_latency(
    baseline: torch.nn.Module,
    method: FrozenITBERAdapter,
    *,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Rotate control/stock/refined order and synchronize every measurement."""
    if device.type != "cuda":
        raise RuntimeError("I-TBER latency benchmark requires CUDA")
    baseline = baseline.to(device).half().eval()
    method = method.to(device).half().eval()
    image = torch.randn(
        *BENCHMARK_PROTOCOL["input"],
        device=device,
        dtype=torch.float16,
    )
    samples: dict[str, list[float]] = {name: [] for name in ("control", "stock", "refined")}

    def infer(name: str) -> None:
        if name == "control":
            baseline.predict(image)
        else:
            method.set_output_mode(name)
            method(image)

    with torch.inference_mode():
        for iteration in range(BENCHMARK_PROTOCOL["warmup"]):
            for name in measurement_order(iteration):
                infer(name)
        torch.cuda.synchronize(device)
        for iteration in range(BENCHMARK_PROTOCOL["iterations"]):
            for name in measurement_order(iteration):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                infer(name)
                torch.cuda.synchronize(device)
                samples[name].append((time.perf_counter() - start) * 1000.0)
    return {name: _latency_summary(values) for name, values in samples.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "formal"), required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--private-checkpoint", type=Path, required=True)
    parser.add_argument("--gate1-cache-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _write_immutable(path: Path, report: dict) -> None:
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to replace changed I-TBER benchmark: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the I-TBER benchmark")
    if file_sha256(args.baseline_checkpoint) != EXPECTED_BASELINE_SHA256:
        raise ValueError("I-TBER benchmark baseline authority mismatch")
    cache_sha = validate_gate1_cache_manifest(args.gate1_cache_manifest)
    artifact = torch.load(args.private_checkpoint, map_location="cpu", weights_only=False)
    validate_resume_checkpoint(
        artifact,
        stage=args.stage,
        cache_manifest_sha256=cache_sha,
    )

    from ultralytics import RTDETR

    device = torch.device(BENCHMARK_PROTOCOL["device"])
    baseline = RTDETR(str(args.baseline_checkpoint)).model.eval()
    detector = RTDETR(str(args.baseline_checkpoint)).model.to(device).eval()
    detector.requires_grad_(False)
    method = FrozenITBERAdapter.from_detector(
        detector,
        private_seed=TRAINING_CONSTANTS["private_seed"],
        probe="p3",
        image_size=BENCHMARK_PROTOCOL["input"][-1],
        rho=0.05,
    ).to(device).eval()
    method.refiner.load_state_dict(artifact["refiner"], strict=True)
    if module_state_sha256(method.detector) != artifact.get("detector_sha_after"):
        raise ValueError("I-TBER benchmark detector authority mismatch")

    parameters = parameter_report(baseline, method)
    baseline_flops = float(get_flops(baseline, imgsz=640))
    method.set_output_mode("stock")
    stock_flops = float(get_flops(method, imgsz=640))
    method.set_output_mode("refined")
    refined_flops = float(get_flops(method, imgsz=640))
    if min(baseline_flops, stock_flops, refined_flops) <= 0:
        raise RuntimeError("I-TBER GFLOPs profiler did not return positive values")
    latency = benchmark_latency(baseline, method, device=device)
    report = {
        "format_version": 1,
        "design_version": "itber-v1.1",
        "stage": args.stage,
        "seed": 0,
        "epoch": int(artifact["epoch"]),
        "protocol": BENCHMARK_PROTOCOL,
        "parameters": parameters,
        "gflops": {
            "baseline": baseline_flops,
            "stock": stock_flops,
            "refined": refined_flops,
            "stock_increase_percent": percentage_increase(stock_flops, baseline_flops),
            "refined_increase_percent": percentage_increase(refined_flops, baseline_flops),
        },
        "latency": {
            **latency,
            "stock_increase_percent": percentage_increase(
                latency["stock"]["mean_ms"], latency["control"]["mean_ms"]
            ),
            "refined_increase_percent": percentage_increase(
                latency["refined"]["mean_ms"], latency["control"]["mean_ms"]
            ),
        },
    }
    _write_immutable(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
