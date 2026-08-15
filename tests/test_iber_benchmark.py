from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.benchmark_iber import (
    BENCHMARK_PROTOCOL,
    build_gflops_report,
    build_report,
    latency_summary,
    measurement_order,
    parameter_report,
    percentage_increase,
    private_operation_report,
    profile_private_gflops,
    validate_checkpoint_source_commit,
    validate_thop_authority,
)
from src.iber_protocol import (
    DESIGN_VERSION,
    PROTOCOL_SHA256,
    RUNTIME_AMENDMENT,
    RUNTIME_AMENDMENT_SHA256,
    execution_environment,
)
from src.iber_head import IBERRefiner


class _Method(nn.Module):
    def __init__(self, detector: nn.Module, *, foreign_parameter: bool = False) -> None:
        super().__init__()
        self.detector = detector
        self.refiner = nn.Sequential(nn.Linear(4, 3), nn.SiLU(), nn.Linear(3, 2))
        if foreign_parameter:
            self.foreign = nn.Parameter(torch.ones(1))


class _PrivateShape(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_path = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 4), nn.SiLU())
        self.geometry_path = nn.Sequential(nn.Linear(8, 2), nn.SiLU())
        self.base_trunk = nn.Sequential(
            nn.Linear(14, 4), nn.SiLU(), nn.Linear(4, 4), nn.SiLU()
        )
        self.f3_projection = nn.Conv2d(6, 3, kernel_size=1)
        self.f3_encoder = nn.Sequential(nn.Linear(9, 3), nn.SiLU())
        self.rgb_encoder = nn.Sequential(
            nn.Linear(15, 2), nn.LayerNorm(2), nn.SiLU()
        )
        self.boundary_encoder = nn.Sequential(nn.Linear(5, 3), nn.SiLU())
        self.boundary_trunk = nn.Sequential(
            nn.Linear(8, 4), nn.SiLU(), nn.Linear(4, 4), nn.SiLU()
        )
        self.base_gate_head = nn.Linear(4, 1)
        self.boundary_gate_head = nn.Linear(4, 1)
        self.base_residual_head = nn.Linear(4, 1)
        self.boundary_residual_head = nn.Linear(4, 1)


def test_benchmark_protocol_is_frozen_fp16_batch1_cuda() -> None:
    assert BENCHMARK_PROTOCOL == {
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


def test_parameter_report_attributes_every_and_only_private_parameter() -> None:
    stock = nn.Linear(4, 4)
    method = _Method(nn.Linear(4, 4))
    method.detector.load_state_dict(stock.state_dict())

    report = parameter_report(stock, method)

    expected_private = sum(parameter.numel() for parameter in method.refiner.parameters())
    assert report == {
        "stock_total": sum(parameter.numel() for parameter in stock.parameters()),
        "refined_total": sum(parameter.numel() for parameter in method.parameters()),
        "private_total": expected_private,
        "increase_percent": pytest.approx(
            100.0 * expected_private / sum(parameter.numel() for parameter in stock.parameters())
        ),
    }
    with pytest.raises(RuntimeError, match="private"):
        parameter_report(stock, _Method(nn.Linear(4, 4), foreign_parameter=True))


def test_positive_stock_denominator_is_mandatory_everywhere() -> None:
    assert percentage_increase(101.0, 100.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="positive stock"):
        percentage_increase(1.0, 0.0)
    with pytest.raises(ValueError, match="positive stock"):
        build_gflops_report(
            stock_gflops=0.0,
            profiled_private_gflops=1.0,
            sampling_operations={"private_flops": 1},
            profiler_authority={
                "distribution": "ultralytics-thop",
                "version": "2.0.18",
            },
        )


def test_private_operation_report_counts_both_functional_grid_samples() -> None:
    report = private_operation_report(
        _PrivateShape(),
        batch=1,
        queries=2,
        f3_spatial=(4, 5),
    )

    assert set(report["components_flops"]) == {
        "f3_grid_sample",
        "rgb_grid_sample",
    }
    assert all(value > 0 for value in report["components_flops"].values())
    assert report["private_flops"] == sum(report["components_flops"].values())
    assert report["private_gflops"] == pytest.approx(report["private_flops"] / 1e9)


def test_private_profiler_uses_same_two_flops_per_mac_convention() -> None:
    refiner = IBERRefiner(
        hidden_dim=8,
        f3_channels=6,
        private_seed=10_000,
        image_size=16,
    )
    def profiler(module: nn.Module, inputs: tuple[torch.Tensor, ...], verbose: bool):
        assert verbose is False
        with torch.inference_mode():
            module(*inputs)
        return 500_000.0, float(sum(p.numel() for p in module.parameters()))

    report = profile_private_gflops(
        refiner,
        batch=1,
        queries=2,
        f3_spatial=(4, 5),
        image_size=16,
        profiler=profiler,
    )
    assert report["profile"] == "thop_two_flops_per_mac"
    assert report["profiled_private_gflops"] > 0
    assert report["profiler_authority"] == {
        "distribution": "injected_test_profiler",
        "version": "test",
    }
    assert report["includes"] == [
        "f3_projection",
        "private_encoders",
        "gate_heads",
        "residual_heads",
    ]


def test_real_pinned_thop_profiles_iber_refiner_and_combines_grid_samples() -> None:
    refiner = IBERRefiner(
        hidden_dim=8,
        f3_channels=6,
        private_seed=10_000,
        image_size=16,
    )
    profiled = profile_private_gflops(
        refiner,
        batch=1,
        queries=2,
        f3_spatial=(4, 5),
        image_size=16,
    )
    sampling = private_operation_report(
        refiner,
        batch=1,
        queries=2,
        f3_spatial=(4, 5),
    )
    report = build_gflops_report(
        stock_gflops=100.0,
        profiled_private_gflops=profiled["profiled_private_gflops"],
        sampling_operations=sampling,
        profiler_authority=profiled["profiler_authority"],
    )

    assert math.isfinite(profiled["profiled_private_gflops"])
    assert profiled["profiled_private_gflops"] > 0
    assert profiled["profiler_authority"] == {
        "distribution": "ultralytics-thop",
        "version": "2.0.18",
    }
    assert report["private"] == pytest.approx(
        profiled["profiled_private_gflops"] + sampling["private_gflops"]
    )
    assert report["explicit_grid_sample"] == pytest.approx(
        sampling["private_gflops"]
    )


def test_gflops_combines_same_profiler_private_delta_and_explicit_sampling() -> None:
    report = build_gflops_report(
        stock_gflops=100.0,
        profiled_private_gflops=0.4,
        sampling_operations={
            "private_flops": 100_000_000,
            "components_flops": {
                "f3_grid_sample": 90_000_000,
                "rgb_grid_sample": 10_000_000,
            },
        },
        profiler_authority={
            "distribution": "ultralytics-thop",
            "version": "2.0.18",
        },
    )
    assert report["stock"] == 100.0
    assert report["profiled_private"] == 0.4
    assert report["explicit_grid_sample"] == 0.1
    assert report["private"] == pytest.approx(0.5)
    assert report["refined"] == pytest.approx(100.5)
    assert report["increase_percent"] == pytest.approx(0.5)
    assert report["accounting_convention"] == {
        "stock_and_private_modules": "thop_two_flops_per_mac",
        "functional_grid_sample": "explicit_bilinear_seven_flops_per_scalar",
        "elementwise_operations": "excluded_from_both_stock_and_private_thop_totals",
    }
    assert report["profiler_authority"] == {
        "distribution": "ultralytics-thop",
        "version": "2.0.18",
    }


def test_thop_authority_rejects_conflicting_or_drifted_distribution() -> None:
    assert validate_thop_authority(
        ["ultralytics-thop"], "2.0.18", "2.0.18"
    ) == {"distribution": "ultralytics-thop", "version": "2.0.18"}
    for distributions, distribution_version, module_version in (
        (["thop"], "0.1.1", "0.1.1"),
        (["thop", "ultralytics-thop"], "2.0.18", "2.0.18"),
        (["ultralytics-thop"], "2.1.6", "2.1.6"),
        (["ultralytics-thop"], "2.0.18", "2.1.6"),
    ):
        with pytest.raises(RuntimeError, match="THOP"):
            validate_thop_authority(
                distributions, distribution_version, module_version
            )


def test_measurement_order_alternates_stock_and_refined() -> None:
    assert measurement_order(0) == ("stock", "refined")
    assert measurement_order(1) == ("refined", "stock")
    assert measurement_order(2) == measurement_order(0)


def test_latency_summary_preserves_raw_samples_median_and_percentiles() -> None:
    summary = latency_summary([4.0, 1.0, 3.0, 2.0])
    assert summary["raw_samples_ms"] == [4.0, 1.0, 3.0, 2.0]
    assert summary["sample_count"] == 4
    assert summary["median_ms"] == pytest.approx(2.5)
    assert summary["p50_ms"] == pytest.approx(2.5)
    assert summary["p90_ms"] == pytest.approx(3.7)
    assert summary["p95_ms"] == pytest.approx(3.85)
    assert summary["p99_ms"] == pytest.approx(3.97)
    with pytest.raises(ValueError, match="samples"):
        latency_summary([])


def test_report_contains_honest_nonblocking_targets_and_runtime_authority() -> None:
    parameters = {
        "stock_total": 1000,
        "refined_total": 1005,
        "private_total": 5,
        "increase_percent": 0.5,
    }
    gflops = build_gflops_report(
        stock_gflops=100.0,
        profiled_private_gflops=0.4,
        sampling_operations={
            "private_flops": 100_000_000,
            "components_flops": {
                "f3_grid_sample": 90_000_000,
                "rgb_grid_sample": 10_000_000,
            },
        },
        profiler_authority={
            "distribution": "ultralytics-thop",
            "version": "2.0.18",
        },
    )
    latency = {
        "stock": latency_summary([10.0, 10.2]),
        "refined": latency_summary([10.4, 10.6]),
    }
    environment = execution_environment()

    report = build_report(
        parameters=parameters,
        gflops=gflops,
        latency=latency,
        gpu_environment=environment,
        checkpoint_epoch=30,
        baseline_sha256="A" * 64,
        private_checkpoint_sha256="B" * 64,
        source_commit="c" * 40,
        detector_sha256="D" * 64,
    )

    assert report["design_version"] == DESIGN_VERSION
    assert report["gpu_environment"] == environment
    assert report["runtime_amendment"] == dict(RUNTIME_AMENDMENT)
    assert report["runtime_amendment_sha256"] == RUNTIME_AMENDMENT_SHA256
    assert report["artifact_authority"] == {
        "baseline_sha256": "A" * 64,
        "private_checkpoint_sha256": "B" * 64,
        "source_commit": "c" * 40,
        "protocol_sha256": PROTOCOL_SHA256,
        "detector_sha256": "D" * 64,
    }
    assert report["latency"]["increase_percent"] == pytest.approx(
        percentage_increase(10.5, 10.1)
    )
    assert report["target_observations"] == {
        "parameters_under_1_percent": True,
        "gflops_under_1_percent": True,
        "latency_under_3_percent": False,
        "scientific_gate": False,
    }


def test_checkpoint_source_commit_must_equal_deployed_source() -> None:
    commit = "a" * 40
    assert validate_checkpoint_source_commit(commit, commit.upper()) == commit
    with pytest.raises(ValueError, match="source_commit"):
        validate_checkpoint_source_commit(commit, "b" * 40)
    with pytest.raises(ValueError, match="source_commit"):
        validate_checkpoint_source_commit("not-a-commit", "not-a-commit")


def test_cli_has_only_artifact_paths_and_frozen_protocol() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_iber.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for allowed in ("--baseline-checkpoint", "--private-checkpoint", "--output"):
        assert allowed in result.stdout
    for forbidden in ("--imgsz", "--warmup", "--iterations", "--device", "--half"):
        assert forbidden not in result.stdout


def test_source_is_independent_iber_and_records_cuda_measurement_authority() -> None:
    source = Path("scripts/benchmark_iber.py").read_text(encoding="utf-8")
    for marker in (
        "FrozenIBERAdapter",
        "DESIGN_VERSION",
        "RUNTIME_AMENDMENT_SHA256",
        "torch.float16",
        "torch.cuda.synchronize",
        '"f3_grid_sample"',
        '"rgb_grid_sample"',
        '"raw_samples_ms"',
    ):
        assert marker in source
    forbidden = ("I-TBER", "itber-v1.1", "FrozenITBERAdapter", "rtdetr_itber")
    assert all(marker not in source for marker in forbidden)
    latency_source = source[source.index("def benchmark_latency") : source.index("def current_gpu_environment")]
    assert latency_source.index('method.set_output_mode("refined")') < latency_source.index(
        "def infer"
    )
