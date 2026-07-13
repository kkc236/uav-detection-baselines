from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adaptive_batch import AdaptiveBatchState, load_state, save_state


EXIT_PLANNED_RESTART = 75
OOM_MARKERS = ("cuda out of memory", "outofmemoryerror", "cudnn_status_alloc_failed")


def build_child_command(
    *,
    python_executable: str,
    script: Path,
    checkpoint: Path,
    state_path: Path,
    batch: int,
) -> list[str]:
    return [
        python_executable,
        str(script),
        "--child",
        "--checkpoint",
        str(checkpoint),
        "--state",
        str(state_path),
        "--batch",
        str(batch),
    ]


def classify_child_exit(returncode: int, output: str) -> str:
    normalized = output.lower()
    if any(marker in normalized for marker in OOM_MARKERS):
        return "oom"
    if returncode == EXIT_PLANNED_RESTART:
        return "planned_restart"
    if returncode == 0:
        return "success"
    return "failure"


def apply_child_result(state: AdaptiveBatchState, result: str) -> AdaptiveBatchState:
    if result == "oom":
        state.record_oom()
        state.unexpected_failures = 0
    elif result in {"planned_restart", "success"}:
        state.unexpected_failures = 0
    elif result == "failure":
        state.unexpected_failures += 1
        state.last_event = "unexpected_failure"
    else:
        raise ValueError(f"Unknown child result: {result}")
    return state


def disable_internal_oom_retry(trainer: object) -> None:
    trainer._oom_retries = 3


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def read_log_segment(path: Path, offset: int, limit: int = 1024 * 1024) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as file:
        size = path.stat().st_size
        file.seek(max(offset, size - limit))
        return file.read().decode("utf-8", "replace")


def build_status_payload(batch_state: AdaptiveBatchState, **extra: object) -> dict[str, object]:
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "batch": batch_state.current_batch,
        "completed_epoch": batch_state.completed_epoch,
        "cooldown_remaining": batch_state.cooldown_remaining,
        "oom_count": batch_state.oom_count,
        "last_peak_gib": batch_state.last_peak_gib,
        "last_event": batch_state.last_event,
        **extra,
    }


def run_child(args: argparse.Namespace) -> int:
    import torch
    from ultralytics import RTDETR

    checkpoint = args.checkpoint.resolve()
    state_path = args.state.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = RTDETR(str(checkpoint))

    def reset_peak(_: object) -> None:
        torch.cuda.reset_peak_memory_stats()

    def checkpoint_saved(trainer: object) -> None:
        state = load_state(state_path)
        old_batch = state.current_batch
        peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
        state.checkpoint = str(checkpoint)
        state.record_epoch(peak_gib=peak_gib, completed_epoch=int(trainer.epoch) + 1)
        save_state(state_path, state)
        print(
            f"ADAPTIVE epoch={state.completed_epoch} peak_gib={peak_gib:.2f} "
            f"batch={old_batch}->{state.current_batch} event={state.last_event}",
            flush=True,
        )
        if state.current_batch != old_batch:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(EXIT_PLANNED_RESTART)

    model.add_callback("on_train_epoch_start", reset_peak)
    model.add_callback("on_train_start", disable_internal_oom_retry)
    model.add_callback("on_model_save", checkpoint_saved)
    model.train(
        resume=str(checkpoint),
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        save_period=args.save_period,
        plots=True,
    )
    return 0


def _acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Adaptive trainer already owns lock: {path}") from exc
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    return descriptor


def run_supervisor(args: argparse.Namespace) -> int:
    state_path = args.state.resolve()
    checkpoint = args.checkpoint.resolve()
    log_path = args.log.resolve()
    status_path = args.status.resolve()
    lock_path = args.lock.resolve()

    if state_path.exists():
        state = load_state(state_path)
    else:
        state = AdaptiveBatchState(
            current_batch=args.batch,
            completed_epoch=args.start_epoch,
            checkpoint=str(checkpoint),
        )
        save_state(state_path, state)

    lock_descriptor = _acquire_lock(lock_path)
    try:
        while state.completed_epoch < args.target_epoch:
            command = build_child_command(
                python_executable=sys.executable,
                script=Path(__file__).resolve(),
                checkpoint=checkpoint,
                state_path=state_path,
                batch=state.current_batch,
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_offset = log_path.stat().st_size if log_path.exists() else 0
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"\nSUPERVISOR launch batch={state.current_batch} epoch={state.completed_epoch}\n")
                log_file.flush()
                environment = os.environ.copy()
                environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                child = subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=args.workdir,
                    env=environment,
                )
                while child.poll() is None:
                    current = load_state(state_path)
                    _atomic_json(
                        status_path,
                        build_status_payload(current, process_state="running", pid=child.pid),
                    )
                    time.sleep(args.poll_interval)

            state = load_state(state_path)
            result = classify_child_exit(child.returncode, read_log_segment(log_path, log_offset))
            apply_child_result(state, result)
            save_state(state_path, state)
            _atomic_json(
                status_path,
                build_status_payload(state, process_state=result, returncode=child.returncode),
            )

            if result == "success" and state.completed_epoch >= args.target_epoch:
                return 0
            if state.unexpected_failures >= args.max_unexpected_failures:
                return 2

        return 0
    finally:
        os.close(lock_descriptor)
        lock_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive RT-DETR true-resume supervisor.")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=Path("logs/adaptive_rtdetr_state.json"))
    parser.add_argument("--log", type=Path, default=Path("logs/adaptive_rtdetr.log"))
    parser.add_argument("--status", type=Path, default=Path("logs/adaptive_rtdetr_status.json"))
    parser.add_argument("--lock", type=Path, default=Path("logs/adaptive_rtdetr.lock"))
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--start-epoch", type=int, default=3)
    parser.add_argument("--target-epoch", type=int, default=100)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--cache", default="ram")
    parser.add_argument("--save-period", type=int, default=5)
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument("--max-unexpected-failures", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_child(args) if args.child else run_supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())
