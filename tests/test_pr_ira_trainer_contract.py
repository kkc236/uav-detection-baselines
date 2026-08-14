from __future__ import annotations

import pytest
import torch

from src.rtdetr_fdr_bpdd import FDRBPDDTrainer
from src.rtdetr_fdr_bpdd_pr_ira import (
    FDRBPDDPRIRADetectionModel,
    FDRBPDDPRIRATrainer,
)


@pytest.fixture(scope="module")
def combined_model() -> FDRBPDDPRIRADetectionModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91_031)
        return FDRBPDDPRIRADetectionModel(
            nc=10,
            verbose=False,
            experiment_seed=0,
        )


def test_build_optimizer_splits_private_groups_at_one_tenth_lr(
    combined_model: FDRBPDDPRIRADetectionModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_build(
        _self: object,
        model: torch.nn.Module,
        name: str = "MuSGD",
        lr: float = 0.01,
        momentum: float = 0.937,
        decay: float = 0.0005,
        iterations: float = 1e5,
    ) -> torch.optim.Optimizer:
        del name, iterations
        decay_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.ndim > 1
        ]
        no_decay_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.ndim <= 1
        ]
        return torch.optim.SGD(
            [
                {
                    "params": decay_parameters,
                    "lr": lr,
                    "weight_decay": decay,
                    "param_group": "weight",
                },
                {
                    "params": no_decay_parameters,
                    "lr": lr,
                    "weight_decay": 0.0,
                    "param_group": "bias",
                },
            ],
            lr=lr,
            momentum=momentum,
        )

    monkeypatch.setattr(FDRBPDDTrainer, "build_optimizer", fake_build)
    trainer = FDRBPDDPRIRATrainer.__new__(FDRBPDDPRIRATrainer)

    optimizer = trainer.build_optimizer(
        combined_model,
        name="MuSGD",
        lr=0.01,
        momentum=0.937,
        decay=0.0005,
        iterations=1000,
    )

    private_ids = {id(parameter) for parameter in combined_model.pr_ira.parameters()}
    all_group_ids: list[int] = []
    private_group_ids: set[int] = set()
    public_group_ids: set[int] = set()
    private_decays: set[float] = set()
    for group in optimizer.param_groups:
        identifiers = {id(parameter) for parameter in group["params"]}
        all_group_ids.extend(identifiers)
        if group.get("pr_ira_private"):
            assert group["lr"] == pytest.approx(0.001)
            assert identifiers <= private_ids
            private_group_ids |= identifiers
            private_decays.add(float(group["weight_decay"]))
        else:
            assert group["lr"] == pytest.approx(0.01)
            assert identifiers.isdisjoint(private_ids)
            public_group_ids |= identifiers

    expected_ids = {
        id(parameter)
        for parameter in combined_model.parameters()
        if parameter.requires_grad
    }
    assert len(all_group_ids) == len(set(all_group_ids))
    assert set(all_group_ids) == expected_ids
    assert private_group_ids == private_ids
    assert public_group_ids == expected_ids - private_ids
    assert private_decays == {0.0, 0.0005}


@pytest.mark.parametrize(
    ("epoch_zero_based", "epochs", "expected"),
    [
        (0, 30, 0.0),
        (2, 30, 0.0),
        (3, 30, 1.0 / 7.0),
        (9, 30, 1.0),
        (0, 100, 0.0),
        (9, 100, 0.0),
        (10, 100, 1.0 / 21.0),
        (30, 100, 1.0),
    ],
)
def test_model_train_applies_the_frozen_relative_schedule(
    combined_model: FDRBPDDPRIRADetectionModel,
    monkeypatch: pytest.MonkeyPatch,
    epoch_zero_based: int,
    epochs: int,
    expected: float,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        FDRBPDDTrainer,
        "_model_train",
        lambda _self: calls.append("stock_train"),
    )
    trainer = FDRBPDDPRIRATrainer.__new__(FDRBPDDPRIRATrainer)
    trainer.model = combined_model
    trainer.epoch = epoch_zero_based
    trainer.epochs = epochs

    trainer._model_train()

    assert calls == ["stock_train"]
    assert combined_model.pr_ira.open_ratio == pytest.approx(expected)


@pytest.mark.parametrize(
    ("epoch_zero_based", "epochs", "expected_suppressed"),
    [
        (2, 30, True),
        (3, 30, False),
        (29, 30, False),
        (9, 100, True),
        (10, 100, False),
        (59, 100, False),
        (60, 100, True),
        (99, 100, True),
    ],
)
def test_inactive_phases_suppress_all_private_gradients_only(
    combined_model: FDRBPDDPRIRADetectionModel,
    epoch_zero_based: int,
    epochs: int,
    expected_suppressed: bool,
) -> None:
    trainer = FDRBPDDPRIRATrainer.__new__(FDRBPDDPRIRATrainer)
    trainer.model = combined_model
    trainer.epoch = epoch_zero_based
    trainer.epochs = epochs
    private = combined_model.pr_ira_private_parameters()
    private_ids = {id(parameter) for parameter in private}
    public = next(
        parameter
        for parameter in combined_model.parameters()
        if parameter.requires_grad and id(parameter) not in private_ids
    )

    try:
        private_gradients = [torch.ones_like(parameter) for parameter in private]
        for parameter, gradient in zip(private, private_gradients, strict=True):
            parameter.grad = gradient
        public_gradient = torch.ones_like(public)
        public.grad = public_gradient

        assert trainer.suppress_pr_ira_inactive_gradients() is expected_suppressed
        if expected_suppressed:
            assert all(parameter.grad is None for parameter in private)
        else:
            assert all(
                parameter.grad is not None
                and torch.equal(parameter.grad, gradient)
                for parameter, gradient in zip(
                    private,
                    private_gradients,
                    strict=True,
                )
            )
        assert public.grad is not None
        assert torch.equal(public.grad, public_gradient)
    finally:
        for parameter in combined_model.parameters():
            parameter.grad = None


def test_missing_schedule_context_suppresses_private_gradients_only(
    combined_model: FDRBPDDPRIRADetectionModel,
) -> None:
    trainer = FDRBPDDPRIRATrainer.__new__(FDRBPDDPRIRATrainer)
    trainer.model = combined_model
    private = combined_model.pr_ira_private_parameters()
    private_ids = {id(parameter) for parameter in private}
    public = next(
        parameter
        for parameter in combined_model.parameters()
        if parameter.requires_grad and id(parameter) not in private_ids
    )

    try:
        for parameter in private:
            parameter.grad = torch.ones_like(parameter)
        public_gradient = torch.ones_like(public)
        public.grad = public_gradient

        assert trainer.suppress_pr_ira_inactive_gradients() is True
        assert all(parameter.grad is None for parameter in private)
        assert public.grad is not None
        assert torch.equal(public.grad, public_gradient)
    finally:
        for parameter in combined_model.parameters():
            parameter.grad = None


def test_late_freeze_preserves_private_sgd_parameter_and_momentum(
    combined_model: FDRBPDDPRIRADetectionModel,
) -> None:
    trainer = FDRBPDDPRIRATrainer.__new__(FDRBPDDPRIRATrainer)
    trainer.model = combined_model
    trainer.epoch = 60
    trainer.epochs = 100
    private_parameters = combined_model.pr_ira_private_parameters()
    private_ids = {id(parameter) for parameter in private_parameters}
    private = private_parameters[0]
    public = next(
        parameter
        for parameter in combined_model.parameters()
        if parameter.requires_grad and id(parameter) not in private_ids
    )
    optimizer = torch.optim.SGD(
        [private, public],
        lr=0.01,
        momentum=0.9,
        weight_decay=0.1,
    )
    private_original = private.detach().clone()
    public_original = public.detach().clone()

    try:
        private.grad = torch.ones_like(private)
        public.grad = torch.ones_like(public)
        optimizer.step()
        assert torch.count_nonzero(
            optimizer.state[private]["momentum_buffer"]
        ).item() > 0
        assert torch.count_nonzero(
            optimizer.state[public]["momentum_buffer"]
        ).item() > 0

        private_before_freeze = private.detach().clone()
        private_momentum_before_freeze = optimizer.state[private][
            "momentum_buffer"
        ].clone()
        public_before_freeze = public.detach().clone()
        private.grad = torch.full_like(private, 2.0)
        public.grad = torch.full_like(public, 2.0)

        assert trainer.suppress_pr_ira_inactive_gradients() is True
        assert private.grad is None
        assert public.grad is not None
        optimizer.step()

        assert torch.equal(private, private_before_freeze)
        assert torch.equal(
            optimizer.state[private]["momentum_buffer"],
            private_momentum_before_freeze,
        )
        assert not torch.equal(public, public_before_freeze)
    finally:
        optimizer.zero_grad(set_to_none=True)
        optimizer.state.clear()
        with torch.no_grad():
            private.copy_(private_original)
            public.copy_(public_original)
        for parameter in combined_model.parameters():
            parameter.grad = None


def test_memory_cleanup_preserves_normal_accumulation_and_resets_retry(
    combined_model: FDRBPDDPRIRADetectionModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    combined_model.clear_pr_ira_firewall_buffer()
    private = combined_model.pr_ira_private_parameters()
    loss = sum(parameter.square().mean() for parameter in private)
    combined_model.capture_pr_ira_firewall_gradient(loss)
    assert not combined_model.pr_ira_firewall_buffer_empty

    calls: list[object] = []
    monkeypatch.setattr(
        FDRBPDDTrainer,
        "_clear_memory",
        lambda _self, threshold=None: calls.append(threshold),
    )
    trainer = FDRBPDDPRIRATrainer.__new__(FDRBPDDPRIRATrainer)
    trainer.model = combined_model

    for parameter in private:
        parameter.grad = torch.ones_like(parameter)
    trainer._clear_memory(0.5)

    assert calls == [0.5]
    assert not combined_model.pr_ira_firewall_buffer_empty
    assert all(parameter.grad is not None for parameter in private)

    trainer._clear_memory()

    assert calls == [0.5, None]
    assert combined_model.pr_ira_firewall_buffer_empty
    assert all(parameter.grad is None for parameter in private)
