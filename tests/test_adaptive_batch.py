from pathlib import Path

from src.adaptive_batch import BATCH_LEVELS, AdaptiveBatchState, load_state, save_state


def complete_stable_epochs(state: AdaptiveBatchState, count: int, peak_gib: float) -> None:
    for _ in range(count):
        state.record_epoch(peak_gib=peak_gib)


def test_default_batch_starts_at_16():
    assert AdaptiveBatchState().current_batch == 16
    assert BATCH_LEVELS == (10, 12, 14, 16, 18, 20)


def test_oom_from_batch_18_falls_back_to_16():
    state = AdaptiveBatchState(current_batch=18, completed_epoch=3)

    assert state.record_oom() == 16
    assert state.cooldown_remaining == 5
    assert state.oom_count == 1


def test_oom_reduces_one_level_until_emergency_batch_10():
    state = AdaptiveBatchState(current_batch=14, completed_epoch=10)

    assert state.record_oom() == 12
    assert state.record_oom() == 10


def test_repeated_oom_uses_capped_exponential_cooldown():
    state = AdaptiveBatchState(current_batch=18)

    state.record_oom()
    assert state.cooldown_remaining == 5
    state.current_batch = 18
    state.record_oom()
    assert state.cooldown_remaining == 10
    state.current_batch = 18
    state.record_oom()
    assert state.cooldown_remaining == 20
    state.current_batch = 18
    state.record_oom()
    assert state.cooldown_remaining == 20


def test_batch_14_promotes_to_16_after_cooldown_and_three_stable_epochs():
    state = AdaptiveBatchState(current_batch=14, cooldown_remaining=2)

    complete_stable_epochs(state, 2, peak_gib=20.0)
    assert state.current_batch == 14
    complete_stable_epochs(state, 3, peak_gib=25.0)

    assert state.current_batch == 16
    assert state.stable_epochs == 0


def test_batch_16_promotes_to_18_after_three_stable_epochs():
    state = AdaptiveBatchState(current_batch=16)

    complete_stable_epochs(state, 3, peak_gib=27.0)

    assert state.current_batch == 18


def test_batch_18_promotes_to_20_after_three_low_peak_epochs():
    state = AdaptiveBatchState(current_batch=18)

    complete_stable_epochs(state, 3, peak_gib=27.0)

    assert state.current_batch == 20


def test_oom_from_batch_20_falls_back_to_18():
    state = AdaptiveBatchState(current_batch=20)

    assert state.record_oom() == 18


def test_batch_10_recovers_to_12_after_three_stable_epochs():
    state = AdaptiveBatchState(current_batch=10)

    complete_stable_epochs(state, 3, peak_gib=21.0)

    assert state.current_batch == 12


def test_high_peak_at_batch_18_proactively_demotes():
    state = AdaptiveBatchState(current_batch=18)

    transition = state.record_epoch(peak_gib=29.0)

    assert transition == 16
    assert state.cooldown_remaining == 5
    assert state.last_event == "peak_demote"


def test_high_peak_at_batch_20_proactively_demotes_to_18():
    state = AdaptiveBatchState(current_batch=20)

    transition = state.record_epoch(peak_gib=29.0)

    assert transition == 18
    assert state.cooldown_remaining == 5
    assert state.last_event == "peak_demote"


def test_state_round_trip_is_atomic_and_complete(tmp_path: Path):
    path = tmp_path / "state.json"
    state = AdaptiveBatchState(
        current_batch=14,
        completed_epoch=22,
        cooldown_remaining=4,
        oom_count=2,
        stable_epochs=1,
        last_peak_gib=25.5,
        checkpoint="runs/example/weights/last.pt",
    )

    save_state(path, state)

    assert load_state(path) == state
    assert not path.with_suffix(".json.tmp").exists()
