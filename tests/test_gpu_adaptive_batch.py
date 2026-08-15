from __future__ import annotations

from pathlib import Path

from src.gpu_adaptive_batch import (
    AdaptiveTrainingState,
    batch_policy_for_vram,
    load_adaptive_state,
    save_adaptive_state,
    scale_batch_policy,
)


def test_vram_bands_select_documented_batch_ladders():
    assert batch_policy_for_vram(total_gib=24, free_gib=24).levels == (2, 4, 6, 8)
    assert batch_policy_for_vram(total_gib=24, free_gib=24).initial_batch == 6
    assert batch_policy_for_vram(total_gib=32, free_gib=32).levels == (4, 6, 8, 10, 12)
    assert batch_policy_for_vram(total_gib=32, free_gib=32).initial_batch == 8
    assert batch_policy_for_vram(total_gib=48, free_gib=48).levels == (6, 8, 12, 16, 20)
    assert batch_policy_for_vram(total_gib=48, free_gib=48).initial_batch == 12
    assert batch_policy_for_vram(total_gib=80, free_gib=80).levels == (8, 12, 16, 20, 24, 28)
    assert batch_policy_for_vram(total_gib=80, free_gib=80).initial_batch == 16


def test_low_startup_free_memory_reduces_initial_batch():
    policy = batch_policy_for_vram(total_gib=24, free_gib=18)

    assert policy.initial_batch == 4
    assert policy.reason == "startup_free_memory"


def test_batch_policy_scales_from_per_gpu_to_global_ddp_batch():
    policy = batch_policy_for_vram(total_gib=24, free_gib=24)

    scaled = scale_batch_policy(policy, world_size=8)

    assert scaled.levels == (16, 32, 48, 64)
    assert scaled.initial_batch == 48
    assert scaled.reason == "vram_default_x8"


def test_three_low_peak_epochs_promote_one_level():
    state = AdaptiveTrainingState(levels=(2, 4, 6, 8), current_batch=4)

    state.record_epoch(completed_epoch=1, peak_gib=18.0, total_gib=24.0)
    state.record_epoch(completed_epoch=2, peak_gib=18.0, total_gib=24.0)
    transition = state.record_epoch(completed_epoch=3, peak_gib=18.0, total_gib=24.0)

    assert transition == 6
    assert state.last_event == "promote"


def test_high_peak_proactively_demotes_and_starts_cooldown():
    state = AdaptiveTrainingState(levels=(2, 4, 6, 8), current_batch=8)

    transition = state.record_epoch(completed_epoch=5, peak_gib=22.8, total_gib=24.0)

    assert transition == 6
    assert state.cooldown_remaining == 5
    assert state.last_event == "peak_demote"


def test_oom_demotes_one_level_without_disabling_amp():
    state = AdaptiveTrainingState(levels=(2, 4, 6, 8), current_batch=6, amp_enabled=True)

    assert state.record_oom() == 4
    assert state.amp_enabled is True
    assert state.cooldown_remaining == 5
    assert state.oom_count == 1


def test_numeric_failure_disables_amp_and_demotes_one_level():
    state = AdaptiveTrainingState(levels=(2, 4, 6, 8), current_batch=6, amp_enabled=True)

    assert state.record_numeric_failure() == 4
    assert state.amp_enabled is False
    assert state.numeric_failure_count == 1
    assert state.last_event == "numeric_fp32_demote"


def test_adaptive_state_round_trip_is_atomic(tmp_path: Path):
    path = tmp_path / "adaptive_state.json"
    state = AdaptiveTrainingState(
        levels=(2, 4, 6, 8),
        current_batch=4,
        amp_enabled=False,
        completed_epoch=17,
        cooldown_remaining=2,
        checkpoint="run/weights/last.pt",
    )

    save_adaptive_state(path, state)

    assert load_adaptive_state(path) == state
    assert not path.with_suffix(".json.tmp").exists()
