from __future__ import annotations

import pytest

from src.ascv_loc_stage import ASCVStage, stage_policy


@pytest.mark.parametrize(
    ("stage", "epochs", "uses_subset", "builds_val", "validates_each_epoch", "runs_final_eval", "max_batches"),
    [
        (ASCVStage.MECHANISM_500, 100, True, False, False, False, 500),
        (ASCVStage.SCREEN_6, 6, True, False, False, False, None),
        (ASCVStage.FULL_20, 20, False, False, False, False, None),
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
