"""Measure LPR parameter, compute, and end-to-end decoder overhead."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rtdetr_lpr import LPRRTDETRDetectionModel


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def _percent_delta(candidate: float, baseline: float) -> float | None:
    return 100.0 * (candidate - baseline) / baseline if baseline else None


def _latency_ms(
    model: torch.nn.Module,
    image: torch.Tensor,
    warmup: int,
    runs: int,
) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(image)

        samples = []
        if image.is_cuda:
            torch.cuda.synchronize(image.device)
            for _ in range(runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(image)
                end.record()
                torch.cuda.synchronize(image.device)
                samples.append(float(start.elapsed_time(end)))
        else:
            for _ in range(runs):
                start = time.perf_counter()
                model(image)
                samples.append((time.perf_counter() - start) * 1000.0)

    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, max(0, round(0.95 * len(ordered)) - 1))
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[p95_index],
    }


def benchmark(
    *,
    device: str = "0",
    imgsz: int = 640,
    warmup: int = 20,
    runs: int = 100,
) -> dict[str, Any]:
    if runs < 1 or warmup < 0:
        raise ValueError("runs must be positive and warmup must be non-negative")
    torch_device = torch.device(f"cuda:{device}" if device.isdigit() and torch.cuda.is_available() else device)

    torch.manual_seed(0)
    stock = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    lpr = LPRRTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    incompatible = lpr.load_state_dict(stock.state_dict(), strict=False)
    if incompatible.unexpected_keys or any("lpr_refiners" not in key for key in incompatible.missing_keys):
        raise RuntimeError(f"Stock-to-LPR state transfer failed: {incompatible}")

    stock_counts = parameter_counts(stock)
    lpr_counts = parameter_counts(lpr)
    stock_flops = float(get_flops(stock, imgsz=imgsz))
    lpr_flops = float(get_flops(lpr, imgsz=imgsz))

    stock = stock.to(torch_device)
    lpr = lpr.to(torch_device)
    image = torch.rand(1, 3, imgsz, imgsz, device=torch_device)
    stock_latency = _latency_ms(stock, image, warmup=warmup, runs=runs)
    lpr_latency = _latency_ms(lpr, image, warmup=warmup, runs=runs)

    report = {
        "device": str(torch_device),
        "imgsz": imgsz,
        "warmup": warmup,
        "runs": runs,
        "stock": {"parameters": stock_counts, "gflops": stock_flops, "latency": stock_latency},
        "lpr": {"parameters": lpr_counts, "gflops": lpr_flops, "latency": lpr_latency},
        "delta_percent": {
            "parameters": _percent_delta(lpr_counts["total"], stock_counts["total"]),
            "gflops": _percent_delta(lpr_flops, stock_flops),
            "latency_mean": _percent_delta(lpr_latency["mean_ms"], stock_latency["mean_ms"]),
        },
        "preferred_limits_percent": {"parameters": 1.0, "gflops": 1.0, "latency_mean": 3.0},
    }
    return report


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark stock and LPR RT-DETR-L overhead.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = benchmark(device=args.device, imgsz=args.imgsz, warmup=args.warmup, runs=args.runs)
    write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
