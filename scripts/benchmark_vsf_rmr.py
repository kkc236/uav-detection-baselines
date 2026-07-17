from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rtdetr_vsf_rmr import VSFRMRDetectionModel


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("latency values must not be empty")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("latency values must not be empty")
    return {
        "mean_ms": sum(values) / len(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
    }


def percentage_increase(value: float, baseline: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    return (float(value) / float(baseline) - 1.0) * 100.0


def measurement_order(iteration: int) -> tuple[str, str]:
    return ("baseline", "vsf_rmr") if iteration % 2 == 0 else ("vsf_rmr", "baseline")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_latency(
    model: torch.nn.Module,
    *,
    batch: int,
    imgsz: int,
    warmup: int,
    iterations: int,
    device: torch.device,
    half: bool,
) -> dict[str, float]:
    dtype = torch.float16 if half and device.type == "cuda" else torch.float32
    image = torch.randn(batch, 3, imgsz, imgsz, device=device, dtype=dtype)
    model = model.to(device).eval()
    if dtype == torch.float16:
        model = model.half()
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            model.predict(image)
        _synchronize(device)
        for _ in range(iterations):
            start = time.perf_counter()
            model.predict(image)
            _synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.0)
    return latency_summary(samples)


def benchmark_pair_latency(
    baseline: torch.nn.Module,
    vsf_rmr: torch.nn.Module,
    *,
    batch: int,
    imgsz: int,
    warmup: int,
    iterations: int,
    device: torch.device,
    half: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    dtype = torch.float16 if half and device.type == "cuda" else torch.float32
    image = torch.randn(batch, 3, imgsz, imgsz, device=device, dtype=dtype)
    models = {"baseline": baseline.to(device).eval(), "vsf_rmr": vsf_rmr.to(device).eval()}
    if dtype == torch.float16:
        models = {name: model.half() for name, model in models.items()}
    samples: dict[str, list[float]] = {"baseline": [], "vsf_rmr": []}
    with torch.inference_mode():
        for iteration in range(warmup):
            for name in measurement_order(iteration):
                models[name].predict(image)
        _synchronize(device)
        for iteration in range(iterations):
            for name in measurement_order(iteration):
                _synchronize(device)
                start = time.perf_counter()
                models[name].predict(image)
                _synchronize(device)
                samples[name].append((time.perf_counter() - start) * 1000.0)
    return latency_summary(samples["baseline"]), latency_summary(samples["vsf_rmr"])


def _load(model, checkpoint: Path | None):
    if checkpoint is not None:
        model.load(str(checkpoint))
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare matched RT-DETR-L and VSF-RMR inference cost.")
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--vsf-checkpoint", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    baseline = _load(RTDETRDetectionModel("rtdetr-l.yaml", verbose=False), args.baseline_checkpoint)
    vsf = _load(VSFRMRDetectionModel("rtdetr-l.yaml", verbose=False), args.vsf_checkpoint)
    baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
    vsf_params = sum(parameter.numel() for parameter in vsf.parameters())
    baseline_flops = float(get_flops(baseline, imgsz=args.imgsz))
    vsf_flops = float(get_flops(vsf, imgsz=args.imgsz))

    report = {
        "protocol": {
            "device": str(device),
            "imgsz": args.imgsz,
            "half": bool(args.half),
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "parameters": {
            "baseline": baseline_params,
            "vsf_rmr": vsf_params,
            "increase_percent": percentage_increase(vsf_params, baseline_params),
        },
        "gflops": {
            "baseline": baseline_flops,
            "vsf_rmr": vsf_flops,
            "increase_percent": percentage_increase(vsf_flops, baseline_flops),
        },
        "latency": {},
    }
    for batch in args.batches:
        baseline_latency, vsf_latency = benchmark_pair_latency(
            baseline,
            vsf,
            batch=batch,
            imgsz=args.imgsz,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
            half=args.half,
        )
        report["latency"][str(batch)] = {
            "baseline": baseline_latency,
            "vsf_rmr": vsf_latency,
            "mean_increase_percent": percentage_increase(
                vsf_latency["mean_ms"], baseline_latency["mean_ms"]
            ),
        }

    report["acceptance"] = {
        "parameters_under_3_percent": report["parameters"]["increase_percent"] <= 3.0,
        "gflops_under_5_percent": report["gflops"]["increase_percent"] <= 5.0,
        "latency_under_5_percent": all(
            value["mean_increase_percent"] <= 5.0 for value in report["latency"].values()
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
