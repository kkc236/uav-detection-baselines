from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.pr_fia_protocol import build_run_identity
from src.rtdetr_fdr_bpdd import FDRBPDDTrainer
from src.rtdetr_fdr_bpdd_pr_fia import (
    FDRBPDDPRFIADetectionModel,
    FDRBPDDPRFIATrainer,
)


@pytest.fixture(scope="module")
def combined_model() -> FDRBPDDPRFIADetectionModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91_031)
        return FDRBPDDPRFIADetectionModel(
            nc=10,
            verbose=False,
            experiment_seed=0,
        )


def _run_identity(stage: str) -> dict[str, object]:
    return build_run_identity(
        {"git_commit": "a" * 40, "tree_sha256": "B" * 64},
        stage=stage,
        variant="fdr_bpdd_pr_fia",
        seed=0,
    )


@pytest.mark.parametrize(
    ("stage", "epochs"),
    [
        ("screen", 30),
        ("formal", 100),
    ],
)
def test_schedule_authority_accepts_only_the_stage_total(
    stage: str,
    epochs: int,
) -> None:
    trainer = FDRBPDDPRFIATrainer.__new__(FDRBPDDPRFIATrainer)
    trainer.pr_fia_run_identity = _run_identity(stage)

    assert trainer._validate_pr_fia_schedule_authority(epochs) is True


@pytest.mark.parametrize(
    ("stage", "epochs"),
    [
        ("screen", 100),
        ("formal", 30),
    ],
)
def test_schedule_authority_rejects_the_other_stage_total(
    stage: str,
    epochs: int,
) -> None:
    trainer = FDRBPDDPRFIATrainer.__new__(FDRBPDDPRFIATrainer)
    trainer.pr_fia_run_identity = _run_identity(stage)

    with pytest.raises(ValueError, match="schedule epochs"):
        trainer._validate_pr_fia_schedule_authority(epochs)


@pytest.mark.parametrize("missing_field", ["source_sha256", "stage"])
def test_schedule_authority_rejects_incomplete_identity(
    missing_field: str,
) -> None:
    identity = _run_identity("formal")
    del identity[missing_field]
    trainer = FDRBPDDPRFIATrainer.__new__(FDRBPDDPRFIATrainer)
    trainer.pr_fia_run_identity = identity

    with pytest.raises(ValueError, match=missing_field):
        trainer._validate_pr_fia_schedule_authority(100)


def test_schedule_authority_rejects_non_mapping_identity() -> None:
    trainer = FDRBPDDPRFIATrainer.__new__(FDRBPDDPRFIATrainer)
    trainer.pr_fia_run_identity = "formal"  # type: ignore[assignment]

    with pytest.raises(TypeError, match="Mapping"):
        trainer._validate_pr_fia_schedule_authority(100)


@pytest.mark.parametrize("stage", [None, "probe"])
def test_schedule_authority_rejects_malformed_stage(stage: object) -> None:
    identity = _run_identity("formal")
    identity["stage"] = stage
    trainer = FDRBPDDPRFIATrainer.__new__(FDRBPDDPRFIATrainer)
    trainer.pr_fia_run_identity = identity

    with pytest.raises(ValueError, match="stage"):
        trainer._validate_pr_fia_schedule_authority(100)


def test_schedule_authority_absence_fails_closed() -> None:
    trainer = FDRBPDDPRFIATrainer.__new__(FDRBPDDPRFIATrainer)

    assert trainer._validate_pr_fia_schedule_authority(100) is False


@pytest.mark.parametrize("identity", [None, "formal"])
def test_trainer_init_rejects_non_mapping_before_parent_initialization(
    monkeypatch: pytest.MonkeyPatch,
    identity: object,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        FDRBPDDTrainer,
        "__init__",
        lambda _self, *args, **kwargs: calls.append("parent"),
    )

    with pytest.raises(TypeError, match="Mapping"):
        FDRBPDDPRFIATrainer(pr_fia_run_identity=identity)  # type: ignore[arg-type]

    assert calls == []


def test_trainer_init_rejects_stage_only_identity_before_parent_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        FDRBPDDTrainer,
        "__init__",
        lambda _self, *args, **kwargs: calls.append("parent"),
    )
    complete_identity = _run_identity("formal")
    stage_only_identity = {"stage": complete_identity["stage"]}

    with pytest.raises(ValueError, match="source_sha256"):
        FDRBPDDPRFIATrainer(pr_fia_run_identity=stage_only_identity)

    assert calls == []


@pytest.mark.parametrize(
    ("stage", "epochs"),
    [
        ("screen", 30),
        ("formal", 100),
    ],
)
def test_trainer_init_copies_and_validates_identity_after_parent_initialization(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    epochs: int,
) -> None:
    source_identity = _run_identity(stage)
    parent_observations: list[dict[str, object]] = []

    def fake_parent_init(trainer: object, *args: object, **kwargs: object) -> None:
        del args, kwargs
        stored_identity = getattr(trainer, "pr_fia_run_identity")
        assert stored_identity == source_identity
        assert stored_identity is not source_identity
        parent_observations.append(stored_identity)
        trainer.epochs = epochs  # type: ignore[attr-defined]
        trainer.args = SimpleNamespace()  # type: ignore[attr-defined]

    monkeypatch.setattr(FDRBPDDTrainer, "__init__", fake_parent_init)

    trainer = FDRBPDDPRFIATrainer(pr_fia_run_identity=source_identity)

    assert parent_observations == [source_identity]
    assert trainer.pr_fia_run_identity == source_identity
    assert trainer.pr_fia_run_identity is not source_identity
    assert trainer.args.pr_fia_run_identity == source_identity
    assert trainer.args.pr_fia_run_identity is not trainer.pr_fia_run_identity


def test_trainer_init_rejects_identity_total_mismatch_after_parent_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_parent_init(trainer: object, *args: object, **kwargs: object) -> None:
        del args, kwargs
        calls.append("parent")
        trainer.epochs = 30  # type: ignore[attr-defined]
        trainer.args = SimpleNamespace()  # type: ignore[attr-defined]

    monkeypatch.setattr(FDRBPDDTrainer, "__init__", fake_parent_init)

    with pytest.raises(ValueError, match="schedule epochs"):
        FDRBPDDPRFIATrainer(pr_fia_run_identity=_run_identity("formal"))

    assert calls == ["parent"]


def test_build_optimizer_splits_private_groups_at_one_tenth_lr(
    combined_model: FDRBPDDPRFIADetectionModel,
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
    trainer = FDRBPDDPRFIATrainer.__new__(FDRBPDDPRFIATrainer)

    optimizer = trainer.build_optimizer(
        combined_model,
        name="MuSGD",
        lr=0.01,
        momentum=0.937,
        decay=0.0005,
        iterations=1000,
    )

    private_ids = {id(parameter) for parameter in combined_model.pr_fia.parameters()}
    all_group_ids: list[int] = []
    private_group_ids: set[int] = set()
    public_group_ids: set[int] = set()
    private_decays: set[float] = set()
    for group in optimizer.param_groups:
        identifiers = {id(parameter) for parameter in group["params"]}
        all_group_ids.extend(identifiers)
        if group.get("pr_fia_private"):
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
    combined_model: FDRBPDDPRFIADetectionModel,
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
    trainer = FDRBPDDPRFIATrainer.__new__(FDRBPDDPRFIATrainer)
    trainer.model = combined_model
    trainer.epoch = epoch_zero_based
    trainer.epochs = epochs
    trainer.pr_fia_run_identity = _run_identity(
        "screen" if epochs == 30 else "formal"
    )

    trainer._model_train()

    assert calls == ["stock_train"]
    assert combined_model.pr_fia.open_ratio == pytest.approx(expected)


def test_model_train_without_identity_fails_before_parent_or_schedule_mutation(
    combined_model: FDRBPDDPRFIADetectionModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        FDRBPDDTrainer,
        "_model_train",
        lambda _self: calls.append("stock_train"),
    )
    combined_model.pr_fia.set_training_progress(10, 30)
    open_ratio_before = combined_model.pr_fia.open_ratio
    trainer = FDRBPDDPRFIATrainer.__new__(FDRBPDDPRFIATrainer)
    trainer.model = combined_model
    trainer.epoch = 0
    trainer.epochs = 30

    with pytest.raises(RuntimeError, match="run identity"):
        trainer._model_train()

    assert calls == []
    assert combined_model.pr_fia.open_ratio == pytest.approx(open_ratio_before)


@pytest.mark.parametrize(
    ("stage", "epochs"),
    [
        ("formal", 30),
        ("screen", 100),
    ],
)
def test_model_train_rejects_stage_total_mismatch_before_parent_activity(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    epochs: int,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        FDRBPDDTrainer,
        "_model_train",
        lambda _self: calls.append("stock_train"),
    )
    trainer = FDRBPDDPRFIATrainer.__new__(FDRBPDDPRFIATrainer)
    trainer.pr_fia_run_identity = _run_identity(stage)
    trainer.epoch = 0
    trainer.epochs = epochs

    with pytest.raises(ValueError, match="schedule epochs"):
        trainer._model_train()

    assert calls == []


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
    combined_model: FDRBPDDPRFIADetectionModel,
    epoch_zero_based: int,
    epochs: int,
    expected_suppressed: bool,
) -> None:
    trainer = FDRBPDDPRFIATrainer.__new__(FDRBPDDPRFIATrainer)
    trainer.model = combined_model
    trainer.epoch = epoch_zero_based
    trainer.epochs = epochs
    trainer.pr_fia_run_identity = _run_identity(
        "screen" if epochs == 30 else "formal"
    )
    private = combined_model.pr_fia_private_parameters()
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

        assert trainer.suppress_pr_fia_inactive_gradients() is expected_suppressed
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


@pytest.mark.parametrize(
    "schedule_context",
    [
        {},
        {"pr_fia_run_identity": _run_identity("formal")},
        {"epoch": 10, "epochs": 100},
    ],
)
def test_missing_schedule_context_or_authority_suppresses_private_gradients_only(
    combined_model: FDRBPDDPRFIADetectionModel,
    schedule_context: dict[str, object],
) -> None:
    trainer = FDRBPDDPRFIATrainer.__new__(FDRBPDDPRFIATrainer)
    trainer.model = combined_model
    for name, value in schedule_context.items():
        setattr(trainer, name, value)
    private = combined_model.pr_fia_private_parameters()
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

        assert trainer.suppress_pr_fia_inactive_gradients() is True
        assert all(parameter.grad is None for parameter in private)
        assert public.grad is not None
        assert torch.equal(public.grad, public_gradient)
    finally:
        for parameter in combined_model.parameters():
            parameter.grad = None


@pytest.mark.parametrize(
    ("stage", "epochs"),
    [
        ("formal", 30),
        ("screen", 100),
    ],
)
def test_suppression_rejects_stage_total_mismatch_before_gradient_activity(
    combined_model: FDRBPDDPRFIADetectionModel,
    stage: str,
    epochs: int,
) -> None:
    trainer = FDRBPDDPRFIATrainer.__new__(FDRBPDDPRFIATrainer)
    trainer.model = combined_model
    trainer.epoch = 10
    trainer.epochs = epochs
    trainer.pr_fia_run_identity = _run_identity(stage)
    private = combined_model.pr_fia_private_parameters()
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

        with pytest.raises(ValueError, match="schedule epochs"):
            trainer.suppress_pr_fia_inactive_gradients()

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


def test_late_freeze_preserves_private_sgd_parameter_and_momentum(
    combined_model: FDRBPDDPRFIADetectionModel,
) -> None:
    trainer = FDRBPDDPRFIATrainer.__new__(FDRBPDDPRFIATrainer)
    trainer.model = combined_model
    trainer.epoch = 60
    trainer.epochs = 100
    trainer.pr_fia_run_identity = _run_identity("formal")
    private_parameters = combined_model.pr_fia_private_parameters()
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

        assert trainer.suppress_pr_fia_inactive_gradients() is True
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


def test_resume_revalidates_stage_total_before_parent_checkpoint_restoration(
    monkeypatch: pytest.MonkeyPatch,
    combined_model: FDRBPDDPRFIADetectionModel,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        FDRBPDDTrainer,
        "resume_training",
        lambda _self, checkpoint: calls.append(checkpoint),
    )
    trainer = FDRBPDDPRFIATrainer.__new__(FDRBPDDPRFIATrainer)
    trainer.resume = True
    trainer.epochs = 30
    trainer.pr_fia_run_identity = _run_identity("formal")
    trainer._pr_fia_full_resume_authority_validated = True
    trainer.model = combined_model
    combined_model.clear_pr_fia_firewall_buffer()
    for parameter in combined_model.parameters():
        parameter.grad = None

    with pytest.raises(ValueError, match="schedule epochs"):
        trainer.resume_training({"epoch": 1})

    assert calls == []


def test_resume_delegates_after_valid_stage_total_revalidation(
    monkeypatch: pytest.MonkeyPatch,
    combined_model: FDRBPDDPRFIADetectionModel,
) -> None:
    checkpoint = {"epoch": 1}
    calls: list[object] = []
    marker = object()
    monkeypatch.setattr(
        FDRBPDDTrainer,
        "resume_training",
        lambda _self, value: calls.append(value) or marker,
    )
    trainer = FDRBPDDPRFIATrainer.__new__(FDRBPDDPRFIATrainer)
    trainer.resume = True
    trainer.epochs = 100
    trainer.pr_fia_run_identity = _run_identity("formal")
    trainer._pr_fia_full_resume_authority_validated = True
    trainer.model = combined_model
    combined_model.clear_pr_fia_firewall_buffer()
    for parameter in combined_model.parameters():
        parameter.grad = None

    assert trainer.resume_training(checkpoint) is marker
    assert calls == [checkpoint]


def test_memory_cleanup_preserves_normal_accumulation_and_resets_retry(
    combined_model: FDRBPDDPRFIADetectionModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    combined_model.clear_pr_fia_firewall_buffer()
    private = combined_model.pr_fia_private_parameters()
    loss = sum(parameter.square().mean() for parameter in private)
    combined_model.capture_pr_fia_firewall_gradient(loss)
    assert not combined_model.pr_fia_firewall_buffer_empty

    calls: list[object] = []
    monkeypatch.setattr(
        FDRBPDDTrainer,
        "_clear_memory",
        lambda _self, threshold=None: calls.append(threshold),
    )
    trainer = FDRBPDDPRFIATrainer.__new__(FDRBPDDPRFIATrainer)
    trainer.model = combined_model

    for parameter in private:
        parameter.grad = torch.ones_like(parameter)
    trainer._clear_memory(0.5)

    assert calls == [0.5]
    assert not combined_model.pr_fia_firewall_buffer_empty
    assert all(parameter.grad is not None for parameter in private)

    trainer._clear_memory()

    assert calls == [0.5, None]
    assert combined_model.pr_fia_firewall_buffer_empty
    assert all(parameter.grad is None for parameter in private)
