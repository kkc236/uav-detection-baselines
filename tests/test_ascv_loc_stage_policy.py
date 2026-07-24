from __future__ import annotations

import pytest

from src.ascv_loc_stage import (
    ASCVStage,
    allowed_observed_tensor_batch_sizes,
    stage_policy,
)


@pytest.mark.parametrize(
    ("stage", "epochs", "uses_subset", "builds_val", "validates_each_epoch", "runs_final_eval", "max_batches"),
    [
        (ASCVStage.PREFLIGHT_1, 100, True, False, False, False, 1),
        (ASCVStage.MECHANISM_500, 100, True, False, False, False, 500),
        (ASCVStage.SCREEN_10, 10, True, False, False, False, None),
        (ASCVStage.SEED0_100, 100, False, False, False, False, None),
        (ASCVStage.SEED1_100, 100, False, False, False, False, None),
        (ASCVStage.SEED2_100, 100, False, False, False, False, None),
    ],
)
def test_frozen_stage_policy(
    stage: ASCVStage,
    epochs: int,
    uses_subset: bool,
    builds_val: bool,
    validates_each_epoch: bool,
    runs_final_eval: bool,
    max_batches: int | None,
) -> None:
    policy = stage_policy(stage)

    assert policy.epochs == epochs
    assert policy.uses_hashed_subset is uses_subset
    assert policy.builds_val_loader is builds_val
    assert policy.validates_each_epoch is validates_each_epoch
    assert policy.runs_final_eval is runs_final_eval
    assert policy.max_train_batches == max_batches


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown ASCV-Loc stage"):
        stage_policy("tune-lambda")


@pytest.mark.parametrize(
    ("stage", "successful_batches", "optimizer_attempts"),
    [
        (ASCVStage.PREFLIGHT_1, 1, 1),
        (ASCVStage.MECHANISM_500, 500, 106),
        (ASCVStage.SCREEN_10, 810, 145),
        (ASCVStage.SEED0_100, 80_900, 10_556),
        (ASCVStage.SEED1_100, 80_900, 10_556),
        (ASCVStage.SEED2_100, 80_900, 10_556),
    ],
)
def test_stage_runtime_counts_are_frozen(
    stage: ASCVStage, successful_batches: int, optimizer_attempts: int
) -> None:
    policy = stage_policy(stage)
    assert policy.expected_successful_batches == successful_batches
    assert policy.expected_optimizer_attempts == optimizer_attempts


def test_observed_tensor_batch_contract_allows_only_the_frozen_tail_batch() -> None:
    assert allowed_observed_tensor_batch_sizes(ASCVStage.PREFLIGHT_1) == frozenset({8})
    for stage in (
        ASCVStage.MECHANISM_500,
        ASCVStage.SCREEN_10,
        ASCVStage.SEED0_100,
        ASCVStage.SEED1_100,
        ASCVStage.SEED2_100,
    ):
        assert allowed_observed_tensor_batch_sizes(stage) == frozenset({7, 8})
