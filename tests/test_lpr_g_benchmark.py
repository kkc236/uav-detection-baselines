from __future__ import annotations

from scripts.benchmark_lpr_g import parameter_report
from src.rtdetr_lpr_g import LPRGRTDETRDetectionModel
from ultralytics.nn.tasks import RTDETRDetectionModel


def test_parameter_report_attributes_only_private_overhead() -> None:
    stock = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    method = LPRGRTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)

    report = parameter_report(stock, method)

    assert report["method_total"] - report["control_total"] == report["private_total"]
    assert report["private_total"] > 0
