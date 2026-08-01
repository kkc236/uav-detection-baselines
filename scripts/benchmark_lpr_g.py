"""Measure truthful parameter, GFLOPs, and latency overhead for LPR-G v2."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lpr_g_protocol import load_lpr_g_initial_state, validate_lpr_g_initial_state
from src.rtdetr_lpr_g import LPRGRTDETRDetectionModel


IMGSZ = 640
WARMUP = 50
ITERATIONS = 200


def percentage_increase(value: float, baseline: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    return (float(value) / float(baseline) - 1.0) * 100.0


def parameter_report(
    control: torch.nn.Module,
    method: torch.nn.Module,
) -> dict[str, float | int]:
    control_total = sum(parameter.numel() for parameter in control.parameters())
    method_total = sum(parameter.numel() for parameter in method.parameters())
    private_total = sum(
        parameter.numel()
        for name, parameter in method.named_parameters()
        if "lpr_g_refiner." in name
    )
    if method_total - control_total != private_total:
        raise RuntimeError("non-private parameters differ between control and LPR-G")
    return {
        "control_total": control_total,
        "method_total": method_total,
        "private_total": private_total,
        "increase_percent": percentage_increase(method_total, control_total),
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("latency samples must not be empty")
    return {
        "mean_ms": sum(values) / len(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
    }


def benchmark_latency_modes(
    control: RTDETRDetectionModel,
    method: LPRGRTDETRDetectionModel,
    *,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Alternate control/stock/refined timing to reduce thermal-order bias."""
    if device.type != "cuda":
        raise RuntimeError("frozen LPR-G latency benchmark requires CUDA")
    control = control.to(device).half().eval()
    method = method.to(device).half().eval()
    image = torch.randn(1, 3, IMGSZ, IMGSZ, device=device, dtype=torch.float16)
    names = ("control", "method_stock", "method_refined")

    def infer(name: str) -> None:
        if name == "control":
            control.predict(image)
            return
        method.set_refinement_output("stock" if name == "method_stock" else "refined")
        method.predict(image)

    samples = {name: [] for name in names}
    with torch.inference_mode():
        for iteration in range(WARMUP):
            order = names[iteration % len(names) :] + names[: iteration % len(names)]
            for name in order:
                infer(name)
        torch.cuda.synchronize(device)
        for iteration in range(ITERATIONS):
            order = names[iteration % len(names) :] + names[: iteration % len(names)]
            for name in order:
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                infer(name)
                torch.cuda.synchronize(device)
                samples[name].append((time.perf_counter() - start) * 1000.0)
    return {name: _latency_summary(values) for name, values in samples.items()}


def build_report(
    control: RTDETRDetectionModel,
    method: LPRGRTDETRDetectionModel,
    *,
    device: torch.device,
) -> dict:
    parameters = parameter_report(control, method)
    control_flops = float(get_flops(control, imgsz=IMGSZ))
    method.set_refinement_output("stock")
    method_stock_flops = float(get_flops(method, imgsz=IMGSZ))
    method.set_refinement_output("refined")
    method_refined_flops = float(get_flops(method, imgsz=IMGSZ))
    latency = benchmark_latency_modes(control, method, device=device)
    return {
        "protocol": {
            "device": str(device),
            "input": [1, 3, IMGSZ, IMGSZ],
            "dtype": "float16",
            "warmup": WARMUP,
            "iterations": ITERATIONS,
            "synchronized_cuda": True,
        },
        "parameters": parameters,
        "gflops": {
            "control": control_flops,
            "method_stock": method_stock_flops,
            "method_refined": method_refined_flops,
            "stock_increase_percent": percentage_increase(method_stock_flops, control_flops),
            "refined_increase_percent": percentage_increase(
                method_refined_flops, control_flops
            ),
        },
        "latency": {
            **latency,
            "stock_increase_percent": percentage_increase(
                latency["method_stock"]["mean_ms"], latency["control"]["mean_ms"]
            ),
            "refined_increase_percent": percentage_increase(
                latency["method_refined"]["mean_ms"], latency["control"]["mean_ms"]
            ),
        },
        "targets_nonblocking": {
            "parameters_percent": 1.0,
            "gflops_percent": 1.0,
            "latency_percent": 3.0,
        },
    }


def _write_immutable(path: Path, report: dict) -> None:
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to replace changed benchmark report: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark frozen seed0 RT-DETR-L control and LPR-G v2."
    )
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the frozen LPR-G benchmark")
    artifact = torch.load(args.initial_state, map_location="cpu", weights_only=False)
    validate_lpr_g_initial_state(artifact, seed=0)
    control = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    method = LPRGRTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    load_lpr_g_initial_state(control, artifact, variant="control")
    load_lpr_g_initial_state(method, artifact, variant="lprg")
    report = build_report(control, method, device=torch.device("cuda:0"))
    _write_immutable(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
