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


def test_identity_phase_suppresses_all_private_gradients_only(
    combined_model: FDRBPDDPRIRADetectionModel,
) -> None:
    trainer = FDRBPDDPRIRATrainer.__new__(FDRBPDDPRIRATrainer)
    trainer.model = combined_model
    private = combined_model.pr_ira_private_parameters()
    public = next(
        parameter
        for parameter in combined_model.parameters()
        if parameter.requires_grad and id(parameter) not in {id(item) for item in private}
    )

    combined_model.pr_ira.set_training_progress(3, 30)
    for parameter in private:
        parameter.grad = torch.ones_like(parameter)
    public.grad = torch.ones_like(public)

    assert trainer.suppress_pr_ira_identity_gradients() is True
    assert all(parameter.grad is None for parameter in private)
    assert public.grad is not None

    combined_model.pr_ira.set_training_progress(4, 30)
    for parameter in private:
        parameter.grad = torch.ones_like(parameter)

    assert trainer.suppress_pr_ira_identity_gradients() is False
    assert all(parameter.grad is not None for parameter in private)


def test_memory_cleanup_clears_pending_firewall_before_retry(
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

    trainer._clear_memory(0.5)

    assert calls == [0.5]
    assert combined_model.pr_ira_firewall_buffer_empty
