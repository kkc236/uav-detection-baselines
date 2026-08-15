from pathlib import Path

from scripts.train_rtdetr_adaptive import (
    EXIT_PLANNED_RESTART,
    apply_child_result,
    build_child_command,
    build_status_payload,
    classify_child_exit,
    disable_internal_oom_retry,
    read_log_segment,
)
from src.adaptive_batch import AdaptiveBatchState


def test_child_command_uses_true_resume_checkpoint_and_current_batch():
    command = build_child_command(
        python_executable="python",
        script=Path("scripts/train_rtdetr_adaptive.py"),
        checkpoint=Path("runs/scratch/weights/last.pt"),
        state_path=Path("logs/adaptive_state.json"),
        batch=16,
    )

    assert command[0] == "python"
    assert Path(command[1]) == Path("scripts/train_rtdetr_adaptive.py")
    assert command[2] == "--child"
    assert Path(command[command.index("--checkpoint") + 1]) == Path("runs/scratch/weights/last.pt")
    assert command[command.index("--batch") + 1] == "16"
    assert "rtdetr-l.yaml" not in command


def test_exit_classifier_distinguishes_oom_planned_restart_and_success():
    assert classify_child_exit(1, "torch.OutOfMemoryError: CUDA out of memory") == "oom"
    assert classify_child_exit(EXIT_PLANNED_RESTART, "batch transition") == "planned_restart"
    assert classify_child_exit(0, "training complete") == "success"
    assert classify_child_exit(2, "dataset missing") == "failure"


def test_only_current_child_log_segment_is_classified(tmp_path: Path):
    log_path = tmp_path / "adaptive.log"
    log_path.write_text("old CUDA out of memory\n", encoding="utf-8")
    offset = log_path.stat().st_size
    with log_path.open("a", encoding="utf-8") as file:
        file.write("current training complete\n")

    current_output = read_log_segment(log_path, offset)

    assert "out of memory" not in current_output.lower()
    assert classify_child_exit(0, current_output) == "success"


def test_oom_result_demotes_one_level_without_advancing_epoch():
    state = AdaptiveBatchState(current_batch=16, completed_epoch=12)

    apply_child_result(state, "oom")

    assert state.current_batch == 14
    assert state.completed_epoch == 12
    assert state.last_event == "oom_demote"


def test_planned_restart_preserves_transition_recorded_by_callback():
    state = AdaptiveBatchState(current_batch=18, completed_epoch=20, last_event="promote")

    apply_child_result(state, "planned_restart")

    assert state.current_batch == 18
    assert state.completed_epoch == 20
    assert state.last_event == "promote"


def test_unexpected_failure_counter_resets_after_a_non_failure():
    state = AdaptiveBatchState()

    apply_child_result(state, "failure")
    apply_child_result(state, "failure")
    assert state.unexpected_failures == 2

    apply_child_result(state, "planned_restart")
    assert state.unexpected_failures == 0


def test_status_payload_accepts_process_state_without_name_collision():
    state = AdaptiveBatchState(current_batch=16, completed_epoch=5)

    payload = build_status_payload(state, process_state="running", pid=123)

    assert payload["process_state"] == "running"
    assert payload["batch"] == 16
    assert payload["pid"] == 123


def test_ultralytics_internal_batch_halving_is_disabled():
    class Trainer:
        _oom_retries = 0

    trainer = Trainer()

    disable_internal_oom_retry(trainer)

    assert trainer._oom_retries == 3
