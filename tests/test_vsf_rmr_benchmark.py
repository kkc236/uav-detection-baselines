from __future__ import annotations

import pytest

from scripts.benchmark_vsf_rmr import build_parser, latency_summary, measurement_order, percentage_increase


def test_latency_summary_reports_mean_p50_and_p95():
    summary = latency_summary([1.0, 2.0, 3.0, 4.0, 10.0])

    assert summary["mean_ms"] == 4.0
    assert summary["p50_ms"] == 3.0
    assert summary["p95_ms"] == pytest.approx(8.8)


def test_percentage_increase_uses_baseline_as_denominator():
    assert percentage_increase(105.0, 100.0) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="positive"):
        percentage_increase(1.0, 0.0)


def test_benchmark_defaults_follow_frozen_protocol():
    args = build_parser().parse_args([])

    assert args.imgsz == 640
    assert args.warmup == 50
    assert args.iterations == 200
    assert args.batches == [1, 8]
    assert args.half is True


def test_pair_measurement_alternates_model_order_to_cancel_clock_drift():
    assert measurement_order(0) == ("baseline", "vsf_rmr")
    assert measurement_order(1) == ("vsf_rmr", "baseline")
