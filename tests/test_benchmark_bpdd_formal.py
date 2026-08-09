from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn

from scripts import benchmark_bpdd_formal as benchmark


def test_benchmark_protocol_is_frozen_to_fp16_batch1() -> None:
    assert benchmark.BENCHMARK_PROTOCOL == {
        "imgsz": 640,
        "batch": 1,
        "half": True,
        "warmup": 50,
        "runs": 200,
        "measurement": "alternating",
    }
    parser = benchmark.build_parser()
    assert not any(
        option in parser._option_string_actions
        for option in ("--imgsz", "--batch", "--half", "--warmup", "--runs")
    )


def test_latency_summary_reports_mean_p50_p95_and_fps() -> None:
    summary = benchmark.latency_summary([1.0, 2.0, 3.0, 4.0, 10.0])
    assert summary == {
        "mean_ms": 4.0,
        "p50_ms": 3.0,
        "p95_ms": pytest.approx(8.8),
        "fps": 250.0,
    }


def test_measurement_order_alternates_fdr_and_bpdd() -> None:
    assert benchmark.measurement_order(0) == ("fdr", "fdr_bpdd")
    assert benchmark.measurement_order(1) == ("fdr_bpdd", "fdr")


def test_efficiency_report_requires_identical_parameters_and_gflops() -> None:
    fdr = {"parameters": 33_000_000, "gflops": 100.0}
    bpdd = {"parameters": 33_000_000, "gflops": 100.0}
    report = benchmark.build_efficiency_report(
        fdr,
        bpdd,
        fdr_latency={"mean_ms": 20.0, "p50_ms": 19.0, "p95_ms": 22.0, "fps": 50.0},
        bpdd_latency={"mean_ms": 20.2, "p50_ms": 19.2, "p95_ms": 22.2, "fps": 1000 / 20.2},
    )
    assert report["parameters"]["strictly_equal"] is True
    assert report["gflops"]["strictly_equal"] is True
    assert report["deployment_graph"] == "ordinary-fdr-for-both-arms"

    with pytest.raises(ValueError, match="parameters"):
        benchmark.build_efficiency_report(
            fdr,
            {**bpdd, "parameters": 33_000_001},
            fdr_latency=report["latency"]["fdr"],
            bpdd_latency=report["latency"]["fdr_bpdd"],
        )
    with pytest.raises(ValueError, match="GFLOPs"):
        benchmark.build_efficiency_report(
            fdr,
            {**bpdd, "gflops": 100.1},
            fdr_latency=report["latency"]["fdr"],
            bpdd_latency=report["latency"]["fdr_bpdd"],
        )


def test_create_only_benchmark_output(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"
    payload = {"status": "complete"}
    assert benchmark.write_create_only_json(output, payload) == output.resolve()
    with pytest.raises(FileExistsError):
        benchmark.write_create_only_json(output, payload)
    assert json.loads(output.read_text("utf-8")) == payload


def _evaluation(variant: str, checkpoint_sha: str) -> dict:
    return {
        "format_version": 1,
        "evaluation_identity": {
            "source_sha256": "S" * 64,
            "protocol_sha256": "P" * 64,
            "fdr_protocol_sha256": "F" * 64,
            "initial_state_sha256": "I" * 64,
            "run_id": f"{variant}-formal-seed0-authority",
            "stage": "formal",
            "variant": variant,
            "seed": 0,
            "data": "/authority/formal.yaml",
            "dataset_sha256": "D" * 64,
        },
        "checkpoint": {
            "kind": "exact-final-ema",
            "completed_epoch": 100,
            "sha256": checkpoint_sha,
            "sha256_verified": True,
            "ema_state_sha256": ("A" if variant == "fdr" else "B") * 64,
            "remote_published": True,
            "remote_asset": f"https://example.invalid/release#{variant}-epoch-0100.pt",
        },
    }


def test_pair_benchmark_binds_independent_run_data_and_publication_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fdr_eval = tmp_path / "fdr-eval.json"
    bpdd_eval = tmp_path / "bpdd-eval.json"
    fdr_eval.write_text(json.dumps(_evaluation("fdr", "1" * 64)), "utf-8")
    bpdd_eval.write_text(json.dumps(_evaluation("fdr_bpdd", "2" * 64)), "utf-8")
    output = tmp_path / "benchmark.json"

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(__import__("torch").ones(1))

    def fake_load(_path, *, expected_sha256):
        return SimpleNamespace(
            model=TinyModel(),
            metadata={
                "kind": "exact-final-ema",
                "completed_epoch": 100,
                "sha256": expected_sha256,
                "sha256_verified": True,
                "ema_state_sha256": "E" * 64,
            },
        )

    monkeypatch.setattr(benchmark, "load_exact_final_checkpoint", fake_load)
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        benchmark,
        "_model_cost",
        lambda _model: {"parameters": 33_156_614, "gflops": 100.0},
    )
    monkeypatch.setattr(
        benchmark,
        "benchmark_pair_latency",
        lambda *_args, **_kwargs: (
            {"mean_ms": 20.0, "p50_ms": 19.0, "p95_ms": 22.0, "fps": 50.0},
            {"mean_ms": 20.1, "p50_ms": 19.1, "p95_ms": 22.1, "fps": 1000 / 20.1},
        ),
    )

    report = benchmark.benchmark_formal_pair(
        fdr_checkpoint=tmp_path / "fdr" / "epoch99.pt",
        bpdd_checkpoint=tmp_path / "bpdd" / "epoch99.pt",
        fdr_evaluation=fdr_eval,
        bpdd_evaluation=bpdd_eval,
        output=output,
    )

    assert report["pair_authority"]["comparison"] == "fresh-fdr-vs-fresh-fdr-bpdd"
    assert report["pair_authority"]["data"] == {
        "yaml": "/authority/formal.yaml",
        "dataset_sha256": "D" * 64,
    }
    assert report["pair_authority"]["runs"] == {
        "fdr": "fdr-formal-seed0-authority",
        "fdr_bpdd": "fdr_bpdd-formal-seed0-authority",
    }
    assert report["pair_authority"]["remote_assets"]["fdr"].endswith(
        "fdr-epoch-0100.pt"
    )
    assert json.loads(output.read_text("utf-8")) == report


def test_benchmark_rejects_mismatched_pair_authority(tmp_path: Path) -> None:
    fdr = _evaluation("fdr", "1" * 64)
    bpdd = _evaluation("fdr_bpdd", "2" * 64)
    bpdd["evaluation_identity"]["dataset_sha256"] = "X" * 64

    with pytest.raises(ValueError, match="paired authority"):
        benchmark.validate_evaluation_pair(fdr, bpdd)
