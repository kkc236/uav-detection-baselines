from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def percentage_increase(value: float, baseline: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    return (float(value) / float(baseline) - 1.0) * 100.0


def measurement_order(iteration: int) -> tuple[str, str]:
    return ("control", "qg-p2") if iteration % 2 == 0 else ("qg-p2", "control")


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("latency values must not be empty")
    ordered = sorted(values)
    return {
        "mean_ms": sum(values) / len(values),
        "p50_ms": _percentile(ordered, 0.50),
        "p95_ms": _percentile(ordered, 0.95),
    }


def benchmark_pair_latency(
    control: torch.nn.Module,
    method: torch.nn.Module,
    *,
    batch: int,
    imgsz: int,
    warmup: int,
    iterations: int,
    device: torch.device,
    half: bool,
) -> dict[str, dict[str, float]]:
    dtype = torch.float16 if half else torch.float32
    image = torch.randn(batch, 3, imgsz, imgsz, device=device, dtype=dtype)
    models = {
        "control": _prepare_model(control, device, half),
        "qg-p2": _prepare_model(method, device, half),
    }
    samples: dict[str, list[float]] = {"control": [], "qg-p2": []}
    with torch.inference_mode():
        for iteration in range(warmup):
            for name in measurement_order(iteration):
                models[name].predict(image)
        torch.cuda.synchronize(device)
        for iteration in range(iterations):
            for name in measurement_order(iteration):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                models[name].predict(image)
                torch.cuda.synchronize(device)
                samples[name].append((time.perf_counter() - start) * 1000.0)
    return {name: latency_summary(values) for name, values in samples.items()}


def benchmark_peak_memory(
    model: torch.nn.Module,
    *,
    batch: int,
    imgsz: int,
    warmup: int,
    device: torch.device,
    half: bool,
) -> float:
    torch.cuda.empty_cache()
    prepared = _prepare_model(model, device, half)
    dtype = torch.float16 if half else torch.float32
    image = torch.randn(batch, 3, imgsz, imgsz, device=device, dtype=dtype)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(warmup):
            prepared.predict(image)
    torch.cuda.synchronize(device)
    peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2
    prepared.to("cpu").float()
    del image
    torch.cuda.empty_cache()
    return peak_mib


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark strict paired RT-DETR-L and QG-P2 inference cost.")
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--method-checkpoint", type=Path, required=True)
    parser.add_argument("--control-results", type=Path)
    parser.add_argument("--method-results", type=Path)
    parser.add_argument("--control-training-peak-gib", type=float)
    parser.add_argument("--method-training-peak-gib", type=float)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("QG-P2 benchmark requires CUDA")

    control, control_checkpoint = _load_checkpoint(args.control_checkpoint, method=False)
    method, method_checkpoint = _load_checkpoint(args.method_checkpoint, method=True)
    control = _fuse(control)
    method = _fuse(method)
    params = {
        "control": sum(parameter.numel() for parameter in control.parameters()),
        "qg-p2": sum(parameter.numel() for parameter in method.parameters()),
    }
    flops = {
        "control": float(get_flops(control, imgsz=args.imgsz)),
        "qg-p2": float(get_flops(method, imgsz=args.imgsz)),
    }
    report: dict[str, Any] = {
        "protocol": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "imgsz": args.imgsz,
            "half": bool(args.half),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "measurement_order": "alternating",
        },
        "checkpoints": {
            "control": _checkpoint_record(args.control_checkpoint, control_checkpoint),
            "qg-p2": _checkpoint_record(args.method_checkpoint, method_checkpoint),
        },
        "parameters": _paired_measurement(params),
        "gflops": _paired_measurement(flops),
        "inference_peak_memory_mib": {},
        "latency": {},
    }
    for batch in args.batches:
        memory = {
            "control": benchmark_peak_memory(
                control, batch=batch, imgsz=args.imgsz, warmup=args.warmup,
                device=device, half=args.half,
            ),
            "qg-p2": benchmark_peak_memory(
                method, batch=batch, imgsz=args.imgsz, warmup=args.warmup,
                device=device, half=args.half,
            ),
        }
        report["inference_peak_memory_mib"][str(batch)] = _paired_measurement(memory)
        latency = benchmark_pair_latency(
            control,
            method,
            batch=batch,
            imgsz=args.imgsz,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
            half=args.half,
        )
        report["latency"][str(batch)] = {
            **latency,
            "mean_increase_percent": percentage_increase(
                latency["qg-p2"]["mean_ms"], latency["control"]["mean_ms"]
            ),
        }
        control.to("cpu").float()
        method.to("cpu").float()
        torch.cuda.empty_cache()

    if args.control_results and args.method_results:
        epoch_time = {
            "control": _epoch_time(args.control_results),
            "qg-p2": _epoch_time(args.method_results),
        }
        report["epoch_time_seconds"] = {
            **epoch_time,
            "mean_increase_percent": percentage_increase(
                epoch_time["qg-p2"]["mean"], epoch_time["control"]["mean"]
            ),
        }
    if args.control_training_peak_gib is not None and args.method_training_peak_gib is not None:
        report["training_peak_memory_gib"] = _paired_measurement(
            {"control": args.control_training_peak_gib, "qg-p2": args.method_training_peak_gib}
        )

    _write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _load_checkpoint(path: Path, *, method: bool) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema")
    if model is None:
        model = checkpoint.get("model")
    if model is None:
        raise ValueError(f"checkpoint has no model: {path}")
    metadata = checkpoint.get("ebc_qp")
    if method:
        config = metadata.get("config") if isinstance(metadata, dict) else None
        if not isinstance(config, dict) or not bool(config.get("quality_gated_p2")):
            raise ValueError("method checkpoint is not QG-P2")
        if float(config.get("lambda_ebc", -1.0)) != 0.0:
            raise ValueError("QG-P2 method checkpoint has nonzero EBC weight")
    return model.float(), checkpoint


def _prepare_model(model: torch.nn.Module, device: torch.device, half: bool) -> torch.nn.Module:
    prepared = model.to(device).eval()
    return prepared.half() if half else prepared.float()


def _fuse(model: torch.nn.Module) -> torch.nn.Module:
    try:
        return model.fuse(verbose=False)
    except TypeError:
        return model.fuse()


def _paired_measurement(values: dict[str, float | int]) -> dict[str, float | int]:
    return {
        **values,
        "increase_percent": percentage_increase(values["qg-p2"], values["control"]),
    }


def _epoch_time(path: Path) -> dict[str, float | int]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "time" not in rows[0]:
        raise ValueError(f"results CSV has no cumulative time: {path}")
    cumulative = [float(row["time"]) for row in rows]
    return {
        "epochs": len(cumulative),
        "total": cumulative[-1],
        "mean": cumulative[-1] / len(cumulative),
    }


def _checkpoint_record(path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _percentile(ordered: list[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"refusing to replace changed benchmark report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
