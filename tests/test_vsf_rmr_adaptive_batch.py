from __future__ import annotations

import pytest

from src.gpu_adaptive_batch import AdaptiveTrainingState, batch_policy_for_vram, scale_batch_policy


def test_4090_policy_starts_at_six_and_can_reach_eight():
    policy = batch_policy_for_vram(total_gib=24.0, free_gib=23.0)

    assert policy.levels == (2, 4, 6, 8)
    assert policy.initial_batch == 6


def test_eight_4090_policy_scales_global_batch_ladder():
    per_gpu = batch_policy_for_vram(total_gib=24.0, free_gib=23.0)

    policy = scale_batch_policy(per_gpu, world_size=8)

    assert policy.levels == (16, 32, 48, 64)
    assert policy.initial_batch == 48


def test_three_stable_epochs_promote_and_high_peak_demotes():
    state = AdaptiveTrainingState(levels=(2, 4, 6, 8), current_batch=6)

    for epoch in range(1, 4):
        state.record_epoch(completed_epoch=epoch, peak_gib=12.0, total_gib=24.0)

    assert state.current_batch == 8
    assert state.last_event == "promote"

    state.record_epoch(completed_epoch=4, peak_gib=23.0, total_gib=24.0)
    assert state.current_batch == 6
    assert state.last_event == "peak_demote"


def test_oom_and_numeric_failures_demote_without_falling_below_floor():
    state = AdaptiveTrainingState(levels=(2, 4, 6, 8), current_batch=6)

    assert state.record_oom() == 4
    assert state.cooldown_remaining == 5
    assert state.record_numeric_failure() == 2
    assert state.amp_enabled is False
    assert state.record_oom() == 2


def test_minimum_vram_error_is_experiment_neutral():
    with pytest.raises(ValueError) as error:
        batch_policy_for_vram(total_gib=16.0, free_gib=16.0)

    assert "IOQC" not in str(error.value)
    assert "RT-DETR-L" in str(error.value)

