from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import pytest
import torch

from src.tascv import TASCVLossResult
from src.tascv_diagnostics import (
    FROZEN_TASCV_MECHANISM_GATE,
    TASCVMechanismAccumulator,
    validate_tascv_checkpoint_runtime,
)
from src.tascv_stage import TASCVStage


def _result(
    *,
    tiny_pairs: int,
    advantage: float,
    wins: int,
    excluded: int = 0,
) -> TASCVLossResult:
    return TASCVLossResult(
        loss=torch.tensor(0.0, requires_grad=True),
        matched_pair_count=tiny_pairs + excluded,
        auxiliary_tiny_pair_count=tiny_pairs,
        excluded_non_tiny_pair_count=excluded,
        tiny_teacher_advantage_sum=torch.tensor(advantage),
        tiny_teacher_win_count=wins,
    )


def test_checkpoint_runtime_contract() -> None:
    validate_tascv_checkpoint_runtime(
        stage=TASCVStage.PREFLIGHT_1,
        calls=2,
        batchnorm_preserved=True,
    )
    for calls in (1, 2):
        validate_tascv_checkpoint_runtime(
            stage=TASCVStage.TINY_MECHANISM_500,
            calls=calls,
            batchnorm_preserved=True,
        )
    with pytest.raises(RuntimeError, match="RECOMPUTE"):
        validate_tascv_checkpoint_runtime(
            stage=TASCVStage.PREFLIGHT_1,
            calls=1,
            batchnorm_preserved=True,
        )
    with pytest.raises(RuntimeError, match="BATCHNORM"):
        validate_tascv_checkpoint_runtime(
            stage=TASCVStage.SCREEN_10,
            calls=1,
            batchnorm_preserved=False,
        )


@pytest.mark.parametrize("calls", (True, 1.9))
def test_checkpoint_runtime_rejects_non_integral_counts(calls) -> None:
    with pytest.raises(TypeError, match="calls"):
        validate_tascv_checkpoint_runtime(
            stage=TASCVStage.SCREEN_10,
            calls=calls,
            batchnorm_preserved=True,
        )


def test_mechanism_state_cannot_be_forged_by_public_counters() -> None:
    accumulator = TASCVMechanismAccumulator()
    for _ in range(100):
        accumulator.record(
            _result(tiny_pairs=1, advantage=1.0, wins=1)
        )

    with pytest.raises(AttributeError):
        accumulator.batches = 500
    assert accumulator.batches == 100
    assert accumulator.mechanism_gate()[0] is False
    with pytest.raises(TypeError):
        TASCVMechanismAccumulator(batches=500)


def test_zero_pair_batch_cannot_inject_advantage() -> None:
    accumulator = TASCVMechanismAccumulator()

    with pytest.raises(RuntimeError, match="RECORD_INVALID"):
        accumulator.record(
            _result(tiny_pairs=0, advantage=100.0, wins=0)
        )


def test_mechanism_gate_configuration_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        FROZEN_TASCV_MECHANISM_GATE.successful_batches = 400


@pytest.mark.parametrize("value", (True, 1.5))
def test_mechanism_record_rejects_non_integral_counts(value) -> None:
    result = _result(tiny_pairs=1, advantage=1.0, wins=1)
    object.__setattr__(result, "auxiliary_tiny_pair_count", value)
    with pytest.raises(TypeError, match="auxiliary_tiny_pair_count"):
        TASCVMechanismAccumulator().record(result)


def test_mechanism_uses_only_frozen_tail_401_to_500() -> None:
    accumulator = TASCVMechanismAccumulator()
    for _ in range(400):
        accumulator.record(
            _result(tiny_pairs=1, advantage=-100.0, wins=0)
        )
    for _ in range(100):
        accumulator.record(
            _result(tiny_pairs=1, advantage=1.0, wins=1)
        )

    summary = accumulator.summary()
    passed, failures = accumulator.mechanism_gate()

    assert summary["tail_window"] == [401, 500]
    assert summary["tail"]["tiny_pairs"] == 100
    assert summary["tail"]["tiny_batches_with_pairs"] == 100
    assert summary["tail"]["tiny_teacher_advantage_mean"] == 1.0
    assert summary["tail"]["tiny_teacher_win_rate"] == 1.0
    assert passed is True
    assert failures == []


def test_mechanism_batch_endpoint_is_not_runtime_configurable() -> None:
    assert "expected_batches" not in inspect.signature(
        TASCVMechanismAccumulator.mechanism_gate
    ).parameters
    accumulator = TASCVMechanismAccumulator()
    for _ in range(400):
        accumulator.record(
            _result(tiny_pairs=1, advantage=1.0, wins=1)
        )

    passed, failures = accumulator.mechanism_gate()

    assert passed is False
    assert "successful_batches=400, expected=500" in failures
    assert accumulator.summary()["tail_window"] == [301, 400]


def test_mechanism_gate_boundaries_are_strict_and_complete() -> None:
    accumulator = TASCVMechanismAccumulator()
    for _ in range(400):
        accumulator.record(_result(tiny_pairs=0, advantage=0.0, wins=0))
    for index in range(100):
        accumulator.record(
            _result(
                tiny_pairs=1,
                advantage=0.0,
                wins=1 if index < 50 else 0,
            )
        )

    passed, failures = accumulator.mechanism_gate()

    assert passed is False
    assert "tiny_teacher_advantage_mean<=0" in failures
    assert "tiny_teacher_win_rate<=0.5" in failures


@pytest.mark.parametrize(
    ("tiny_batches", "tiny_pairs", "expected_failure"),
    (
        (79, 100, "tail_tiny_batches_with_pairs<80"),
        (80, 99, "tail_tiny_pairs<100"),
        (80, 100, None),
    ),
)
def test_mechanism_tail_coverage_boundaries(
    tiny_batches,
    tiny_pairs,
    expected_failure,
) -> None:
    accumulator = TASCVMechanismAccumulator()
    for _ in range(400):
        accumulator.record(_result(tiny_pairs=0, advantage=0.0, wins=0))
    remaining_pairs = tiny_pairs
    for index in range(100):
        batches_left = tiny_batches - index
        pairs = (
            max(1, remaining_pairs - (batches_left - 1))
            if index < tiny_batches
            else 0
        )
        accumulator.record(
            _result(
                tiny_pairs=pairs,
                advantage=float(pairs),
                wins=pairs,
            )
        )
        remaining_pairs -= pairs

    passed, failures = accumulator.mechanism_gate()

    assert passed is (expected_failure is None)
    if expected_failure is not None:
        assert expected_failure in failures


@pytest.mark.parametrize("batches", (499, 501))
def test_mechanism_rejects_adjacent_batch_endpoints(batches) -> None:
    accumulator = TASCVMechanismAccumulator()
    for _ in range(batches):
        accumulator.record(
            _result(tiny_pairs=1, advantage=1.0, wins=1)
        )

    passed, failures = accumulator.mechanism_gate()

    assert passed is False
    assert any(item.startswith("successful_batches=") for item in failures)
    assert any(item.startswith("tail_window=") for item in failures)


def test_non_tiny_auxiliary_contribution_is_immediately_invalid() -> None:
    accumulator = TASCVMechanismAccumulator()

    with pytest.raises(RuntimeError, match="NON_TINY_AUXILIARY"):
        accumulator.record(
            _result(tiny_pairs=1, advantage=1.0, wins=1),
            auxiliary_non_tiny_pair_count=1,
        )
