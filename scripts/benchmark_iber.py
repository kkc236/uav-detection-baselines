"""Benchmark truthful IBER-BE parameter, operation, and latency overhead."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.iber_protocol import (  # noqa: E402
    DESIGN_VERSION,
    EXPECTED_BASELINE_SHA256,
    PRIVATE_SEED,
    PROTOCOL_SHA256,
    RUNTIME_AMENDMENT,
    RUNTIME_AMENDMENT_SHA256,
    execution_environment,
    file_sha256,
    module_state_sha256,
    write_immutable_report,
)
from src.rtdetr_iber import FrozenIBERAdapter  # noqa: E402


BENCHMARK_PROTOCOL = {
    "device": "cuda:0",
    "input": [1, 3, 640, 640],
    "dtype": "float16",
    "warmup": 50,
    "iterations": 200,
    "synchronized_cuda": True,
    "order": "alternating_stock_refined",
    "targets_nonblocking": {
        "parameters_percent": 1.0,
        "gflops_percent": 1.0,
        "latency_percent": 3.0,
    },
}
PINNED_THOP_DISTRIBUTION = "ultralytics-thop"
PINNED_THOP_VERSION = "2.0.18"


def percentage_increase(value: float, stock: float) -> float:
    """Return percentage increase while refusing a nonpositive denominator."""
    if not math.isfinite(float(stock)) or float(stock) <= 0.0:
        raise ValueError("benchmark requires a positive stock denominator")
    if not math.isfinite(float(value)):
        raise ValueError("benchmark value must be finite")
    return (float(value) / float(stock) - 1.0) * 100.0


def _parameter_signature(module: nn.Module) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((name, tuple(parameter.shape)) for name, parameter in module.named_parameters())


def parameter_report(stock: nn.Module, method: nn.Module) -> dict[str, float | int]:
    """Attribute every additional parameter, and only those, to the refiner."""
    if not hasattr(method, "detector") or not hasattr(method, "refiner"):
        raise TypeError("IBER-BE method must expose detector and refiner modules")
    if _parameter_signature(stock) != _parameter_signature(method.detector):
        raise RuntimeError("IBER-BE stock and detector parameter structures differ")
    stock_total = sum(parameter.numel() for parameter in stock.parameters())
    detector_total = sum(parameter.numel() for parameter in method.detector.parameters())
    refined_total = sum(parameter.numel() for parameter in method.parameters())
    private_total = sum(parameter.numel() for parameter in method.refiner.parameters())
    if stock_total <= 0:
        raise ValueError("benchmark requires a positive stock parameter denominator")
    if detector_total != stock_total:
        raise RuntimeError("IBER-BE detector parameter count differs from stock")
    if refined_total - detector_total != private_total:
        raise RuntimeError("IBER-BE overhead is not attributable only to private parameters")
    return {
        "stock_total": stock_total,
        "refined_total": refined_total,
        "private_total": private_total,
        "increase_percent": percentage_increase(refined_total, stock_total),
    }


def private_operation_report(
    refiner: nn.Module,
    *,
    batch: int = 1,
    queries: int = 300,
    f3_spatial: tuple[int, int] = (80, 80),
) -> dict[str, Any]:
    """Count functional grid_sample work omitted by module MAC profilers."""
    if batch <= 0 or queries <= 0 or min(f3_spatial) <= 0:
        raise ValueError("private operation dimensions must be positive")
    per_query = batch * queries
    f3_height, f3_width = (int(value) for value in f3_spatial)

    f3_projection = refiner.f3_projection
    if not isinstance(f3_projection, nn.Conv2d) or f3_projection.kernel_size != (1, 1):
        raise TypeError("IBER-BE operation accounting requires the frozen 1x1 F3 projection")
    projected_channels = int(f3_projection.out_channels)
    # Bilinear grid_sample uses four multiply-add contributions per sampled scalar.
    # The following reduction averages the three along-edge positions.
    f3_sample_scalars = per_query * 4 * 3 * 3 * projected_channels
    f3_reduced_scalars = per_query * 4 * 3 * projected_channels
    f3_grid_flops = 7 * f3_sample_scalars + 3 * f3_reduced_scalars
    rgb_channels = 3
    rgb_sample_scalars = per_query * 4 * 3 * 5 * rgb_channels
    rgb_reduced_scalars = per_query * 4 * 5 * rgb_channels
    rgb_grid_flops = 7 * rgb_sample_scalars + 3 * rgb_reduced_scalars

    components = {
        "f3_grid_sample": f3_grid_flops,
        "rgb_grid_sample": rgb_grid_flops,
    }
    if any(type(value) is not int or value <= 0 for value in components.values()):
        raise RuntimeError("IBER-BE private operation accounting produced an invalid component")
    private_flops = sum(components.values())
    return {
        "convention": "explicit_bilinear_seven_flops_per_scalar",
        "batch": batch,
        "queries": queries,
        "f3_spatial": [f3_height, f3_width],
        "components_flops": components,
        "private_flops": private_flops,
        "private_gflops": private_flops / 1e9,
    }


def validate_thop_authority(
    distributions: Sequence[str],
    distribution_version: str,
    module_version: str,
) -> dict[str, str]:
    """Reject legacy, conflicting, or drifted THOP installations."""
    normalized = [value.lower() for value in distributions]
    if normalized != [PINNED_THOP_DISTRIBUTION]:
        raise RuntimeError("IBER-BE THOP distribution authority is conflicting or invalid")
    if (
        distribution_version != PINNED_THOP_VERSION
        or module_version != PINNED_THOP_VERSION
    ):
        raise RuntimeError("IBER-BE THOP version authority drifted")
    return {
        "distribution": PINNED_THOP_DISTRIBUTION,
        "version": PINNED_THOP_VERSION,
    }


def _pinned_thop_profiler() -> tuple[Any, dict[str, str]]:
    from importlib.metadata import packages_distributions, version

    try:
        import thop
    except ImportError as error:
        raise RuntimeError(
            "IBER-BE GFLOPs benchmark requires ultralytics-thop==2.0.18"
        ) from error
    authority = validate_thop_authority(
        packages_distributions().get("thop", []),
        version(PINNED_THOP_DISTRIBUTION),
        str(getattr(thop, "__version__", "")),
    )
    return thop.profile, authority


def profile_private_gflops(
    refiner: nn.Module,
    *,
    batch: int = 1,
    queries: int = 300,
    f3_spatial: tuple[int, int] = (80, 80),
    image_size: int = 640,
    classes: int = 10,
    profiler: Any | None = None,
) -> dict[str, Any]:
    """Profile private modules with the same THOP 2-FLOP/MAC stock convention."""
    if min(batch, queries, *f3_spatial, image_size, classes) <= 0:
        raise ValueError("private profiler dimensions must be positive")
    from copy import deepcopy

    if profiler is None:
        profiler, profiler_authority = _pinned_thop_profiler()
    else:
        profiler_authority = {
            "distribution": "injected_test_profiler",
            "version": "test",
        }

    module = deepcopy(refiner).cpu().float().eval()
    hidden_dim = int(module.query_path[1].in_features)
    f3_channels = int(module.f3_projection.in_channels)
    hidden = torch.zeros(batch, queries, hidden_dim, dtype=torch.float32)
    stock_boxes = torch.full((batch, queries, 4), 0.25, dtype=torch.float32)
    stock_scores = torch.zeros(batch, queries, classes, dtype=torch.float32)
    f3 = torch.zeros(batch, f3_channels, *f3_spatial, dtype=torch.float32)
    image = torch.zeros(batch, 3, image_size, image_size, dtype=torch.float32)
    with torch.inference_mode():
        macs, _parameters = profiler(
            module,
            inputs=(hidden, stock_boxes, stock_scores, f3, image),
            verbose=False,
        )
    profiled_gflops = float(macs) * 2.0 / 1e9
    if not math.isfinite(profiled_gflops) or profiled_gflops <= 0.0:
        raise RuntimeError("IBER-BE private THOP profile is not positive")
    return {
        "profile": "thop_two_flops_per_mac",
        "profiled_private_gflops": profiled_gflops,
        "profiler_authority": profiler_authority,
        "includes": [
            "f3_projection",
            "private_encoders",
            "gate_heads",
            "residual_heads",
        ],
    }


def build_gflops_report(
    *,
    stock_gflops: float,
    profiled_private_gflops: float,
    sampling_operations: Mapping[str, Any],
    profiler_authority: Mapping[str, str],
) -> dict[str, Any]:
    """Combine same-convention module MACs with explicit functional sampling."""
    if not math.isfinite(float(stock_gflops)) or float(stock_gflops) <= 0.0:
        raise ValueError("benchmark requires a positive stock GFLOPs denominator")
    if not math.isfinite(float(profiled_private_gflops)) or profiled_private_gflops <= 0:
        raise ValueError("IBER-BE profiled private GFLOPs must be positive")
    if dict(profiler_authority) != {
        "distribution": PINNED_THOP_DISTRIBUTION,
        "version": PINNED_THOP_VERSION,
    }:
        raise RuntimeError("IBER-BE THOP profiler authority is invalid")
    private_flops = sampling_operations.get("private_flops")
    grid_components = sampling_operations.get("components_flops")
    if (
        type(private_flops) is not int
        or private_flops <= 0
        or not isinstance(grid_components, Mapping)
        or set(grid_components) != {"f3_grid_sample", "rgb_grid_sample"}
        or sum(grid_components.values()) != private_flops
    ):
        raise ValueError("IBER-BE explicit grid_sample accounting is invalid")
    grid_sample_gflops = private_flops / 1e9
    private_gflops = float(profiled_private_gflops) + grid_sample_gflops
    refined_gflops = float(stock_gflops) + private_gflops
    return {
        "stock": float(stock_gflops),
        "profiled_private": float(profiled_private_gflops),
        "explicit_grid_sample": grid_sample_gflops,
        "private": private_gflops,
        "refined": refined_gflops,
        "increase_percent": percentage_increase(refined_gflops, float(stock_gflops)),
        "grid_sample_components_flops": dict(grid_components),
        "profiler_authority": dict(profiler_authority),
        "accounting_convention": {
            "stock_and_private_modules": "thop_two_flops_per_mac",
            "functional_grid_sample": "explicit_bilinear_seven_flops_per_scalar",
            "elementwise_operations": "excluded_from_both_stock_and_private_thop_totals",
        },
    }


def measurement_order(iteration: int) -> tuple[str, str]:
    """Alternate the timed first mode to reduce order and thermal bias."""
    return ("stock", "refined") if iteration % 2 == 0 else ("refined", "stock")


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def latency_summary(values: Sequence[float]) -> dict[str, Any]:
    """Keep all raw samples and report robust center and tail percentiles."""
    raw = [float(value) for value in values]
    if not raw:
        raise ValueError("IBER-BE latency samples must not be empty")
    if not all(math.isfinite(value) and value > 0 for value in raw):
        raise ValueError("IBER-BE latency samples must be finite and positive")
    return {
        "raw_samples_ms": raw,
        "sample_count": len(raw),
        "mean_ms": math.fsum(raw) / len(raw),
        "median_ms": _percentile(raw, 0.50),
        "p50_ms": _percentile(raw, 0.50),
        "p90_ms": _percentile(raw, 0.90),
        "p95_ms": _percentile(raw, 0.95),
        "p99_ms": _percentile(raw, 0.99),
        "min_ms": min(raw),
        "max_ms": max(raw),
    }


def benchmark_latency(
    stock: nn.Module,
    method: FrozenIBERAdapter,
    *,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """Measure FP16 batch-1 stock/refined paths in one alternating CUDA process."""
    if device.type != "cuda":
        raise RuntimeError("IBER-BE latency benchmark requires CUDA")
    stock = stock.to(device).half().eval()
    method = method.to(device).half().eval()
    image = torch.randn(
        *BENCHMARK_PROTOCOL["input"],
        device=device,
        dtype=torch.float16,
    )
    samples: dict[str, list[float]] = {"stock": [], "refined": []}
    method.set_output_mode("refined")

    def infer(mode: str) -> None:
        if mode == "stock":
            stock.predict(image)
        else:
            method(image)

    with torch.inference_mode():
        for iteration in range(BENCHMARK_PROTOCOL["warmup"]):
            for mode in measurement_order(iteration):
                infer(mode)
        torch.cuda.synchronize(device)
        for iteration in range(BENCHMARK_PROTOCOL["iterations"]):
            for mode in measurement_order(iteration):
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                infer(mode)
                torch.cuda.synchronize(device)
                samples[mode].append((time.perf_counter() - started) * 1000.0)
    return {mode: latency_summary(values) for mode, values in samples.items()}


def current_gpu_environment() -> dict[str, Any]:
    """Record the actual GPU, driver, memory, and software runtime."""
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row.strip() for row in query.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("IBER-BE benchmark requires exactly one visible GPU")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 3:
        raise RuntimeError("nvidia-smi returned an invalid IBER-BE environment row")
    import torchvision
    import ultralytics

    return {
        "gpu": fields[0],
        "driver": fields[1],
        "reported_memory_mib": int(fields[2]),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
    }


def validate_checkpoint_source_commit(
    checkpoint_commit: str, deployed_commit: str
) -> str:
    """Bind the private checkpoint to the exact source being benchmarked."""
    values = (checkpoint_commit, deployed_commit)
    if any(
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in value)
        for value in values
    ):
        raise ValueError("IBER-BE benchmark source_commit is invalid")
    checkpoint_normalized, deployed_normalized = (
        value.lower() for value in values
    )
    if checkpoint_normalized != deployed_normalized:
        raise ValueError("IBER-BE checkpoint source_commit differs from deployed source_commit")
    return checkpoint_normalized


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower()


def build_report(
    *,
    parameters: Mapping[str, Any],
    gflops: Mapping[str, Any],
    latency: Mapping[str, Mapping[str, Any]],
    gpu_environment: Mapping[str, Any],
    checkpoint_epoch: int,
    baseline_sha256: str,
    private_checkpoint_sha256: str,
    source_commit: str,
    detector_sha256: str,
) -> dict[str, Any]:
    """Build a factual report; engineering targets never alter scientific state."""
    stock_median = float(latency["stock"]["median_ms"])
    refined_median = float(latency["refined"]["median_ms"])
    latency_increase = percentage_increase(refined_median, stock_median)
    targets = BENCHMARK_PROTOCOL["targets_nonblocking"]
    return {
        "format_version": 1,
        "design_version": DESIGN_VERSION,
        "stage": "screen",
        "seed": 0,
        "checkpoint_epoch": int(checkpoint_epoch),
        "protocol": dict(BENCHMARK_PROTOCOL),
        "baseline_reference_environment": execution_environment(),
        "gpu_environment": dict(gpu_environment),
        "runtime_amendment": dict(RUNTIME_AMENDMENT),
        "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
        "artifact_authority": {
            "baseline_sha256": baseline_sha256,
            "private_checkpoint_sha256": private_checkpoint_sha256,
            "source_commit": source_commit,
            "protocol_sha256": PROTOCOL_SHA256,
            "detector_sha256": detector_sha256,
        },
        "parameters": dict(parameters),
        "gflops": dict(gflops),
        "latency": {
            "stock": dict(latency["stock"]),
            "refined": dict(latency["refined"]),
            "increase_percent": latency_increase,
        },
        "target_observations": {
            "parameters_under_1_percent": float(parameters["increase_percent"])
            < float(targets["parameters_percent"]),
            "gflops_under_1_percent": float(gflops["increase_percent"])
            < float(targets["gflops_percent"]),
            "latency_under_3_percent": latency_increase
            < float(targets["latency_percent"]),
            "scientific_gate": False,
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--private-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the IBER-BE benchmark")
    baseline_path = args.baseline_checkpoint.resolve()
    baseline_sha256 = file_sha256(baseline_path)
    if baseline_sha256 != EXPECTED_BASELINE_SHA256:
        raise ValueError("IBER-BE benchmark baseline SHA256 mismatch")
    gpu_environment = current_gpu_environment()
    if gpu_environment != execution_environment():
        raise ValueError("IBER-BE benchmark execution environment mismatch")

    private_checkpoint = args.private_checkpoint.resolve()
    private_checkpoint_sha256 = file_sha256(private_checkpoint)
    artifact = torch.load(
        private_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    from scripts.train_iber import validate_resume_checkpoint

    checkpoint_epoch = artifact.get("epoch")
    checkpoint_source_commit = artifact.get("source_commit")
    if type(checkpoint_epoch) is not int or not isinstance(checkpoint_source_commit, str):
        raise ValueError("IBER-BE benchmark private checkpoint identity is invalid")
    source_commit = validate_checkpoint_source_commit(
        checkpoint_source_commit,
        _source_commit(),
    )
    validate_resume_checkpoint(
        artifact,
        source_commit=source_commit,
        highest_verified_epoch=checkpoint_epoch,
    )

    from ultralytics import RTDETR
    from ultralytics.utils.torch_utils import get_flops

    device = torch.device(BENCHMARK_PROTOCOL["device"])
    stock = RTDETR(str(baseline_path)).model.eval()
    detector = RTDETR(str(baseline_path)).model.to(device).eval()
    detector.requires_grad_(False)
    with FrozenIBERAdapter.from_detector(
        detector,
        private_seed=PRIVATE_SEED,
        probe="b3",
        image_size=BENCHMARK_PROTOCOL["input"][-1],
        rho=0.05,
    ).to(device).eval() as method:
        method.refiner.load_state_dict(artifact["refiner"], strict=True)
        detector_sha256 = module_state_sha256(method.detector)
        if detector_sha256 != artifact.get("detector_sha_after"):
            raise ValueError("IBER-BE benchmark detector authority mismatch")
        parameters = parameter_report(stock, method)
        stock_gflops = float(get_flops(stock, imgsz=BENCHMARK_PROTOCOL["input"][-1]))
        profiled_private = profile_private_gflops(method.refiner)
        sampling_operations = private_operation_report(method.refiner)
        gflops = build_gflops_report(
            stock_gflops=stock_gflops,
            profiled_private_gflops=profiled_private["profiled_private_gflops"],
            sampling_operations=sampling_operations,
            profiler_authority=profiled_private["profiler_authority"],
        )
        latency = benchmark_latency(stock, method, device=device)
        report = build_report(
            parameters=parameters,
            gflops=gflops,
            latency=latency,
            gpu_environment=gpu_environment,
            checkpoint_epoch=checkpoint_epoch,
            baseline_sha256=baseline_sha256,
            private_checkpoint_sha256=private_checkpoint_sha256,
            source_commit=source_commit,
            detector_sha256=detector_sha256,
        )

    write_immutable_report(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
