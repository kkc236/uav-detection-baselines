import pytest
import torch

from src.rtdetr_ebc_qp import NormalizedUpdateMonitor


def test_monitor_uses_post_optimizer_delta_not_raw_gradient():
    stock = torch.nn.Parameter(torch.tensor([10.0]))
    p2 = torch.nn.Parameter(torch.tensor([1.0]))
    monitor = NormalizedUpdateMonitor([p2], [stock], limit=10.0, patience=2, max_steps=200)

    monitor.snapshot()
    with torch.no_grad():
        p2.add_(0.2)
        stock.add_(0.1)
    record = monitor.observe()

    assert record.u_p2 == pytest.approx(0.2, abs=1e-6)
    assert record.u_stock == pytest.approx(0.01, abs=1e-6)
    assert record.ratio == pytest.approx(20.0, abs=1e-4)


def test_one_spike_does_not_abort_but_twenty_consecutive_steps_do():
    monitor = _make_monitor(limit=10.0, patience=20)

    for _ in range(19):
        assert _observe_ratio(monitor, 11.0).abort is False
    assert _observe_ratio(monitor, 9.0).abort is False
    for _ in range(19):
        assert _observe_ratio(monitor, 11.0).abort is False
    assert _observe_ratio(monitor, 11.0).abort is True


def test_monitor_stops_collecting_after_step_200():
    monitor = _make_monitor(max_steps=200)

    for _ in range(201):
        record = _observe_ratio(monitor, 1.0)

    assert record.monitored is False
    assert len(monitor.trace) == 200


def test_scaler_skipped_step_records_zero_deltas_without_abort():
    monitor = _make_monitor(limit=10.0, patience=1)

    monitor.snapshot()
    record = monitor.observe()

    assert record.u_p2 == 0.0
    assert record.u_stock == 0.0
    assert record.ratio == 0.0
    assert record.abort is False


def _make_monitor(
    limit: float = 10.0,
    patience: int = 20,
    max_steps: int = 200,
) -> NormalizedUpdateMonitor:
    p2 = torch.nn.Parameter(torch.tensor([1.0]))
    stock = torch.nn.Parameter(torch.tensor([1.0]))
    return NormalizedUpdateMonitor([p2], [stock], limit=limit, patience=patience, max_steps=max_steps)


def _observe_ratio(monitor: NormalizedUpdateMonitor, ratio: float):
    monitor.snapshot()
    with torch.no_grad():
        monitor.p2[0].mul_(1.0 + ratio * 0.001)
        monitor.stock[0].mul_(1.001)
    return monitor.observe()
