import pytest
import torch

from src.rtdetr_ebc_qp import (
    NormalizedUpdateMonitor,
    add_clipped_gradient_contribution,
    is_tsgr_shallow_parameter_name,
    partition_optimizer_parameters,
    prepare_contribution_separated_gradients,
)


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


def test_snapshot_keeps_parameter_copies_on_the_training_device():
    parameter = _DeviceKeepingParameter()
    monitor = NormalizedUpdateMonitor([parameter], [], limit=10.0, patience=1, max_steps=1)

    monitor.snapshot()

    assert monitor._before_p2 == [parameter]


def test_separate_gradient_clipping_prevents_aux_norm_from_rescaling_stock():
    stock = torch.nn.Parameter(torch.zeros(2))
    auxiliary = torch.nn.Parameter(torch.zeros(2))
    stock.grad = torch.tensor([3.0, 4.0])

    result = add_clipped_gradient_contribution(
        [(auxiliary, torch.tensor([0.0, 20.0]))],
        max_norm=10.0,
    )

    torch.testing.assert_close(stock.grad, torch.tensor([3.0, 4.0]))
    assert auxiliary.grad.norm().item() == pytest.approx(10.0, abs=1e-5)
    assert result["preclip_norm"] == pytest.approx(20.0)
    assert result["clip_coefficient"] == pytest.approx(0.5, abs=1e-6)


def test_tsgr_shallow_boundary_is_exactly_model_zero_and_one():
    assert is_tsgr_shallow_parameter_name("model.0.conv.weight")
    assert is_tsgr_shallow_parameter_name("model.1.block.weight")
    assert not is_tsgr_shallow_parameter_name("model.2.conv.weight")
    assert not is_tsgr_shallow_parameter_name("model.10.block.weight")
    assert not is_tsgr_shallow_parameter_name("model.28.decoder.weight")


def test_optimizer_parameter_partition_is_complete_and_disjoint():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stock_head = torch.nn.Linear(2, 2)
            self.p2_adapter = torch.nn.Linear(2, 2)
            self.p2_fusion_gamma = torch.nn.Parameter(torch.ones(()))

    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    stock, auxiliary = partition_optimizer_parameters(model, optimizer)

    stock_ids = {id(parameter) for parameter in stock}
    auxiliary_ids = {id(parameter) for parameter in auxiliary}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    }
    assert stock_ids.isdisjoint(auxiliary_ids)
    assert stock_ids | auxiliary_ids == optimizer_ids
    assert id(model.p2_adapter.weight) in auxiliary_ids
    assert id(model.p2_fusion_gamma) in auxiliary_ids
    assert id(model.stock_head.weight) in stock_ids


def test_contribution_separated_prepare_keeps_deep_stock_clip_independent():
    class TinyTSGRModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.ModuleList(
                [torch.nn.Linear(2, 1, bias=False) for _index in range(3)]
            )
            self.p2_adapter = torch.nn.Linear(2, 1, bias=False)
            self._buffer = {
                "model.0.weight": torch.tensor([[20.0, 0.0]]),
                "p2_adapter.weight": torch.tensor([[0.0, 20.0]]),
            }

        def pop_isolated_auxiliary_gradients(self):
            buffer, self._buffer = self._buffer, {}
            return buffer, 1.0

    class IdentityScaler:
        @staticmethod
        def get_scale():
            return 1.0

        @staticmethod
        def unscale_(_optimizer):
            return None

    model = TinyTSGRModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    stock, auxiliary = partition_optimizer_parameters(model, optimizer)
    model.model[0].weight.grad = torch.tensor([[3.0, 0.0]])
    model.model[1].weight.grad = torch.zeros_like(model.model[1].weight)
    model.model[2].weight.grad = torch.tensor([[0.0, 4.0]])

    diagnostics = prepare_contribution_separated_gradients(
        model,
        optimizer,
        IdentityScaler(),
        stock,
        auxiliary,
        max_norm=10.0,
    )

    torch.testing.assert_close(model.model[0].weight.grad, torch.tensor([[13.0, 0.0]]), atol=1e-5, rtol=0)
    torch.testing.assert_close(model.model[2].weight.grad, torch.tensor([[0.0, 4.0]]), atol=0, rtol=0)
    torch.testing.assert_close(model.p2_adapter.weight.grad, torch.tensor([[0.0, 10.0]]), atol=1e-5, rtol=0)
    assert diagnostics["pure_stock_preclip_norm"] == pytest.approx(5.0)
    assert diagnostics["pure_stock_shallow_preclip_norm"] == pytest.approx(3.0)
    assert diagnostics["pure_stock_clip_coefficient"] == pytest.approx(1.0)
    assert diagnostics["routed_shallow_preclip_norm"] == pytest.approx(20.0)
    assert diagnostics["aux_private_preclip_norm"] == pytest.approx(20.0)


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


class _DeviceKeepingParameter:
    requires_grad = True

    def detach(self):
        return self

    def float(self):
        return self

    def clone(self):
        return self

    def to(self, *args, **kwargs):
        raise AssertionError("snapshot must not transfer parameters to the host")
