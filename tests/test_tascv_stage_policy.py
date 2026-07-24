from __future__ import annotations

import pytest

from src.tascv_stage import (
    TASCVStage,
    allowed_observed_tensor_batch_sizes,
    allowed_seeds,
    stage_policy,
)


@pytest.mark.parametrize(
    (
        "stage",
        "epochs",
        "subset",
        "max_batches",
        "batches",
        "attempts",
        "seeds",
    ),
    (
        (TASCVStage.PREFLIGHT_1, 100, True, 1, 1, 1, {0}),
        (
            TASCVStage.TINY_MECHANISM_500,
            100,
            True,
            500,
            500,
            106,
            {1},
        ),
        (TASCVStage.SCREEN_10, 10, True, None, 810, 145, {0, 1, 2}),
        (
            TASCVStage.FORMAL_100,
            100,
            False,
            None,
            80_900,
            10_556,
            {0, 1, 2},
        ),
    ),
)
def test_tascv_stage_contract(
    stage,
    epochs,
    subset,
    max_batches,
    batches,
    attempts,
    seeds,
) -> None:
    policy = stage_policy(stage)

    assert policy.epochs == epochs
    assert policy.uses_hashed_subset is subset
    assert policy.builds_val_loader is False
    assert policy.validates_each_epoch is False
    assert policy.runs_final_eval is False
    assert policy.max_train_batches == max_batches
    assert policy.expected_successful_batches == batches
    assert policy.expected_optimizer_attempts == attempts
    assert allowed_seeds(stage) == frozenset(seeds)


def test_tascv_tensor_batch_contract() -> None:
    assert allowed_observed_tensor_batch_sizes(
        TASCVStage.PREFLIGHT_1
    ) == frozenset({8})
    for stage in (
        TASCVStage.TINY_MECHANISM_500,
        TASCVStage.SCREEN_10,
        TASCVStage.FORMAL_100,
    ):
        assert allowed_observed_tensor_batch_sizes(stage) == frozenset({7, 8})


def test_unknown_tascv_stage_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown T-ASCV stage"):
        stage_policy("tune")

