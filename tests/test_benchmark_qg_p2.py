from __future__ import annotations

import pytest

from scripts.benchmark_qg_p2 import build_parser, measurement_order, percentage_increase


def test_qg_benchmark_defaults_follow_frozen_protocol():
    args = build_parser().parse_args(
        ["--control-checkpoint", "control.pt", "--method-checkpoint", "method.pt"]
    )

    assert args.imgsz == 640
    assert args.batches == [1, 8]
    assert args.warmup == 50
    assert args.iterations == 200
    assert args.half is True


def test_qg_benchmark_alternates_order_and_reports_relative_increase():
    assert measurement_order(0) == ("control", "qg-p2")
    assert measurement_order(1) == ("qg-p2", "control")
    assert percentage_increase(105.0, 100.0) == pytest.approx(5.0)

    with pytest.raises(ValueError, match="positive"):
        percentage_increase(1.0, 0.0)
