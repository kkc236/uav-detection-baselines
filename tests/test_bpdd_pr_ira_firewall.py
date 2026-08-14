from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch
from torch import nn

from src.rtdetr_fdr_bpdd import FDRBPDDTrainer
from src.rtdetr_fdr_bpdd_pr_ira import (
    FDRBPDDPRIRADetectionModel,
    FDRBPDDPRIRATrainer,
)


@pytest.fixture(scope="module")
def firewall_model() -> FDRBPDDPRIRADetectionModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71_031)
        model = FDRBPDDPRIRADetectionModel(
            nc=10,
            verbose=False,
            experiment_seed=0,
        )
    model.train()
    model.pr_ira.set_training_progress(30, 30)
    with torch.no_grad():
        model.pr_ira.amplitude.fill_(0.2)

    torch.manual_seed(81_031)
    batch = _batch()
    model.loss(batch)
    evidence = model.last_fdr_evidence
    assert evidence is not None
    assignments = model.criterion.normal_assignment_snapshot()
    _logits, target_indices, _right, _left = (
        model.criterion._matched_bpdd_inputs(
            evidence.corner_logits,
            evidence.pre_boxes,
            batch["bboxes"],
            assignments[-1],
        )
    )
    assert target_indices.shape == (1, 4)
    final_layer = model.model[-1].dec_bbox_head[-1].layers[-1]
    bins = final_layer.bias.numel() // 4
    with torch.no_grad():
        final_layer.weight.zero_()
        final_layer.bias.fill_(-10.0)
        for edge, target in enumerate(target_indices[0]):
            index = int(target)
            final_layer.bias[edge * bins + index] = 10.0
            final_layer.bias[edge * bins + index + 1] = 10.0
    model.clear_pr_ira_firewall_buffer()
    model.zero_grad(set_to_none=True)
    return model


def _batch() -> dict[str, torch.Tensor]:
    return {
        "img": torch.zeros(1, 3, 128, 128),
        "cls": torch.tensor([[1]], dtype=torch.float32),
        "bboxes": torch.tensor(
            [[0.50, 0.50, 0.20, 0.20]],
            dtype=torch.float32,
        ),
        "batch_idx": torch.tensor([0], dtype=torch.float32),
    }


def _clone_grad(parameter: nn.Parameter) -> torch.Tensor:
    if parameter.grad is None:
        return torch.zeros_like(parameter)
    return parameter.grad.detach().clone()


def _backward_snapshot(
    model: FDRBPDDPRIRADetectionModel,
    *,
    objective: str,
    subtract_firewall: bool,
) -> dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    model.clear_pr_ira_firewall_buffer()
    torch.manual_seed(81_031)
    total, _displayed = model.loss(_batch())
    if objective == "main":
        assert model.last_main_loss is not None
        selected = model.last_main_loss
    elif objective == "total":
        selected = total
    else:
        raise AssertionError(f"unknown objective: {objective}")
    selected.backward()
    if subtract_firewall:
        model.subtract_pr_ira_firewall_buffer()
    snapshot = {
        name: _clone_grad(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    model.clear_pr_ira_firewall_buffer()
    return snapshot


def _synthetic_losses(
    parameters: tuple[nn.Parameter, ...],
    microbatch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    main_scale = float(microbatch + 1) / 97.0
    bpdd_scale = float(microbatch + 3) / 131.0
    main = sum(
        parameter.square().mean() * main_scale
        for parameter in parameters
    )
    bpdd = sum(
        parameter.sum() / max(parameter.numel(), 1) * bpdd_scale
        for parameter in parameters
    )
    return main, bpdd


def test_real_graph_firewall_matches_main_only_and_preserves_bpdd_elsewhere(
    firewall_model: FDRBPDDPRIRADetectionModel,
) -> None:
    model = firewall_model
    private_ids = {
        id(parameter) for parameter in model.pr_ira_private_parameters()
    }

    main_gradients = _backward_snapshot(
        model,
        objective="main",
        subtract_firewall=False,
    )
    total_gradients = _backward_snapshot(
        model,
        objective="total",
        subtract_firewall=False,
    )
    firewall_gradients = _backward_snapshot(
        model,
        objective="total",
        subtract_firewall=True,
    )

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        expected = (
            main_gradients[name]
            if id(parameter) in private_ids
            else total_gradients[name]
        )
        torch.testing.assert_close(
            firewall_gradients[name],
            expected,
            rtol=1e-5,
            atol=1e-7,
        )

    fdr_bpdd_contributions = [
        total_gradients[name] - main_gradients[name]
        for name in total_gradients
        if ".dec_bbox_head." in name
    ]
    assert fdr_bpdd_contributions
    assert any(torch.count_nonzero(value) > 0 for value in fdr_bpdd_contributions)


def test_eight_microbatch_amp128_buffer_is_unscaled_fp32_and_subtracts_exactly(
    firewall_model: FDRBPDDPRIRADetectionModel,
) -> None:
    model = firewall_model
    model.zero_grad(set_to_none=True)
    model.clear_pr_ira_firewall_buffer()
    parameters = model.pr_ira_private_parameters()
    optimizer = torch.optim.SGD(parameters, lr=0.0)
    scaler = torch.amp.GradScaler(
        "cpu",
        enabled=True,
        init_scale=128.0,
        growth_interval=2**31 - 1,
    )
    expected_main = [torch.zeros_like(parameter) for parameter in parameters]
    expected_bpdd = [torch.zeros_like(parameter) for parameter in parameters]

    for microbatch in range(8):
        main_loss, bpdd_loss = _synthetic_losses(parameters, microbatch)
        main_gradients = torch.autograd.grad(
            main_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        bpdd_gradients = torch.autograd.grad(
            bpdd_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        for index, gradient in enumerate(main_gradients):
            if gradient is not None:
                expected_main[index].add_(gradient.detach())
        for index, gradient in enumerate(bpdd_gradients):
            if gradient is not None:
                expected_bpdd[index].add_(gradient.detach())

        model.capture_pr_ira_firewall_gradient(bpdd_loss)
        scaler.scale(main_loss + bpdd_loss).backward()

    assert float(scaler.get_scale()) == 128.0
    assert model.pr_ira_firewall_buffer_size == len(parameters)
    assert set(model._pr_ira_firewall_buffer) == {
        id(parameter) for parameter in parameters
    }
    for parameter, expected in zip(parameters, expected_bpdd, strict=True):
        entry = model._pr_ira_firewall_buffer[id(parameter)]
        assert entry.parameter is parameter
        assert entry.gradient.dtype == torch.float32
        assert entry.gradient.device == parameter.device
        assert not entry.gradient.requires_grad
        torch.testing.assert_close(entry.gradient, expected.float(), rtol=0, atol=0)

    scaler.unscale_(optimizer)
    model.subtract_pr_ira_firewall_buffer()
    for parameter, expected in zip(parameters, expected_main, strict=True):
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, expected, rtol=1e-5, atol=1e-7)

    _main, pending_bpdd = _synthetic_losses(parameters, 0)
    with pytest.raises(RuntimeError, match="half-complete"):
        model.capture_pr_ira_firewall_gradient(pending_bpdd)
    model.clear_pr_ira_firewall_buffer()
    assert model.pr_ira_firewall_buffer_empty


def test_buffer_contract_rejects_count_shape_dtype_device_and_finiteness(
    firewall_model: FDRBPDDPRIRADetectionModel,
) -> None:
    model = firewall_model
    model.clear_pr_ira_firewall_buffer()
    parameters = model.pr_ira_private_parameters()
    _main, bpdd = _synthetic_losses(parameters, 1)
    model.capture_pr_ira_firewall_gradient(bpdd)
    buffer = model._pr_ira_firewall_buffer
    identifier, entry = next(iter(buffer.items()))
    original = entry.gradient

    removed = buffer.pop(identifier)
    with pytest.raises(RuntimeError, match="count"):
        model.validate_pr_ira_firewall_buffer()
    buffer[identifier] = removed

    entry.gradient = torch.zeros((1,), device=original.device)
    with pytest.raises(RuntimeError, match="shape"):
        model.validate_pr_ira_firewall_buffer()
    entry.gradient = original

    entry.gradient = original.double()
    with pytest.raises(RuntimeError, match="dtype"):
        model.validate_pr_ira_firewall_buffer()
    entry.gradient = original

    entry.gradient = torch.empty(original.shape, device="meta")
    with pytest.raises(RuntimeError, match="device"):
        model.validate_pr_ira_firewall_buffer()
    entry.gradient = original

    entry.gradient = original.clone()
    entry.gradient.view(-1)[0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        model.validate_pr_ira_firewall_buffer()
    model.clear_pr_ira_firewall_buffer()


def test_capture_and_failed_microbatch_errors_clear_pending_state(
    firewall_model: FDRBPDDPRIRADetectionModel,
) -> None:
    model = firewall_model
    parameters = model.pr_ira_private_parameters()
    model.clear_pr_ira_firewall_buffer()
    _main, bpdd = _synthetic_losses(parameters, 2)
    model.capture_pr_ira_firewall_gradient(bpdd)
    assert not model.pr_ira_firewall_buffer_empty

    invalid = parameters[0].sum() * float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        model.capture_pr_ira_firewall_gradient(invalid)
    assert model.pr_ira_firewall_buffer_empty

    _main, bpdd = _synthetic_losses(parameters, 3)
    model.capture_pr_ira_firewall_gradient(bpdd)
    with pytest.raises(KeyError):
        model.loss({"img": torch.zeros(1, 3, 128, 128)})
    assert model.pr_ira_firewall_buffer_empty


class _RecordingScaler:
    def __init__(self, events: list[str], *, fail_step: bool = False) -> None:
        self.events = events
        self.fail_step = fail_step

    def get_scale(self) -> float:
        return 128.0

    def unscale_(self, _optimizer: object) -> None:
        self.events.append("unscale")

    def step(self, _optimizer: object) -> None:
        self.events.append("step")
        if self.fail_step:
            raise RuntimeError("synthetic optimizer failure")

    def update(self) -> None:
        self.events.append("update")


class _RecordingOptimizer:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def zero_grad(self) -> None:
        self.events.append("zero_grad")


class _RecordingEMA:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def update(self, _model: nn.Module) -> None:
        self.events.append("ema")


def test_optimizer_step_uses_one_global_clip_and_preserves_amp128_evidence(
    firewall_model: FDRBPDDPRIRADetectionModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    records: list[dict[str, object]] = []
    model = firewall_model
    trainer = FDRBPDDPRIRATrainer.__new__(FDRBPDDPRIRATrainer)
    trainer.model = model
    trainer.scaler = _RecordingScaler(events)
    trainer.optimizer = _RecordingOptimizer(events)
    trainer.ema = _RecordingEMA(events)
    trainer._record_optimizer_evidence = records.append
    monkeypatch.setattr(
        model,
        "subtract_pr_ira_firewall_buffer",
        lambda: events.append("subtract"),
        raising=False,
    )

    def record_suppression() -> bool:
        events.append("suppress")
        return False

    monkeypatch.setattr(
        trainer,
        "suppress_pr_ira_inactive_gradients",
        record_suppression,
    )
    monkeypatch.setattr(
        model,
        "clear_pr_ira_firewall_buffer",
        lambda: events.append("clear"),
        raising=False,
    )
    clipped: list[tuple[set[int], float]] = []

    def record_clip(
        parameters: Iterable[nn.Parameter],
        max_norm: float,
    ) -> torch.Tensor:
        events.append("clip")
        clipped.append(({id(parameter) for parameter in parameters}, max_norm))
        return torch.tensor(3.5)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)

    trainer.optimizer_step()

    assert events == [
        "unscale",
        "subtract",
        "suppress",
        "clip",
        "step",
        "update",
        "zero_grad",
        "clear",
        "ema",
    ]
    assert clipped == [
        ({id(parameter) for parameter in model.parameters()}, 10.0)
    ]
    assert trainer.last_gradient_norms == {"gradient_norm": 3.5}
    assert records == [
        {
            "amp_scale_before": 128.0,
            "amp_scale_after": 128.0,
            "amp_step_skipped": False,
            "gradient_norm": 3.5,
            "gradient_norm_finite": True,
        }
    ]


def test_optimizer_error_resets_gradients_and_firewall(
    firewall_model: FDRBPDDPRIRADetectionModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    model = firewall_model
    trainer = FDRBPDDPRIRATrainer.__new__(FDRBPDDPRIRATrainer)
    trainer.model = model
    trainer.scaler = _RecordingScaler(events, fail_step=True)
    trainer.optimizer = _RecordingOptimizer(events)
    trainer.ema = None
    trainer._record_optimizer_evidence = lambda _record: None
    monkeypatch.setattr(
        model,
        "subtract_pr_ira_firewall_buffer",
        lambda: events.append("subtract"),
        raising=False,
    )
    monkeypatch.setattr(
        model,
        "clear_pr_ira_firewall_buffer",
        lambda: events.append("clear"),
        raising=False,
    )
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda _parameters, max_norm: torch.tensor(max_norm),
    )

    with pytest.raises(RuntimeError, match="synthetic optimizer failure"):
        trainer.optimizer_step()

    assert events == [
        "unscale",
        "subtract",
        "step",
        "zero_grad",
        "clear",
    ]


def test_explicit_reset_optimizer_step_and_checkpoint_keep_buffer_empty(
    firewall_model: FDRBPDDPRIRADetectionModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = firewall_model
    parameters = model.pr_ira_private_parameters()
    optimizer = torch.optim.SGD(parameters, lr=0.0)
    scaler = torch.amp.GradScaler(
        "cpu",
        enabled=True,
        init_scale=128.0,
        growth_interval=2**31 - 1,
    )
    trainer = FDRBPDDPRIRATrainer.__new__(FDRBPDDPRIRATrainer)
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.scaler = scaler
    trainer.ema = None
    trainer._record_optimizer_evidence = lambda _record: None

    model.clear_pr_ira_firewall_buffer()
    _main, bpdd = _synthetic_losses(parameters, 4)
    model.capture_pr_ira_firewall_gradient(bpdd)
    scaler.scale(bpdd).backward()
    trainer.reset_pr_ira_firewall_state()
    assert model.pr_ira_firewall_buffer_empty
    assert all(parameter.grad is None for parameter in parameters)

    _main, bpdd = _synthetic_losses(parameters, 5)
    model.capture_pr_ira_firewall_gradient(bpdd)
    scaler.scale(bpdd).backward()
    trainer.optimizer_step()
    assert model.pr_ira_firewall_buffer_empty
    assert all(parameter.grad is None for parameter in parameters)
    assert float(scaler.get_scale()) == 128.0

    monkeypatch.setattr(
        FDRBPDDTrainer,
        "save_model",
        lambda _self: "saved",
    )
    assert trainer.save_model() == "saved"
    assert model.pr_ira_firewall_buffer_empty

    _main, pending = _synthetic_losses(parameters, 6)
    model.capture_pr_ira_firewall_gradient(pending)
    with pytest.raises(RuntimeError, match="firewall buffer must be empty"):
        trainer.save_model()
    model.clear_pr_ira_firewall_buffer()
