from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint_recovery import find_resume_checkpoint
from src.gpu_adaptive_batch import (
    AdaptiveTrainingState,
    batch_policy_for_vram,
    detect_gpu_profile,
    load_adaptive_state,
    save_adaptive_state,
    scale_batch_policy,
)


EXIT_PLANNED_RESTART = 75
OOM_MARKERS = ("cuda out of memory", "outofmemoryerror", "cudnn_status_alloc_failed")


def parse_device_indices(device: str) -> tuple[int, ...]:
    parts = tuple(part.strip() for part in device.split(",") if part.strip())
    if not parts:
        raise ValueError("At least one CUDA device is required")
    try:
        return tuple(int(part.removeprefix("cuda:")) for part in parts)
    except ValueError as error:
        raise ValueError(f"Invalid CUDA device list: {device!r}") from error


def parse_batch_levels(value: str) -> tuple[int, ...]:
    try:
        levels = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ValueError(f"Invalid batch level list: {value!r}") from error
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("Batch levels must contain positive integers")
    if any(left >= right for left, right in zip(levels, levels[1:])):
        raise ValueError("Batch levels must be strictly increasing")
    return levels


def build_child_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return environment


def classify_child_exit(returncode: int, output: str) -> str:
    normalized = output.lower()
    if "nonfinite_loss" in normalized:
        return "numeric_failure"
    if any(marker in normalized for marker in OOM_MARKERS):
        return "oom"
    if returncode == EXIT_PLANNED_RESTART:
        return "planned_restart"
    if returncode == 0:
        return "success"
    return "failure"


def build_child_command(
    *,
    python_executable: str,
    train_script: Path,
    project: Path,
    run_name: str,
    state_path: Path,
    batch: int,
    amp_enabled: bool,
    epochs: int,
    workers: int,
    device: str,
    save_period: int,
    optimizer: str,
    lr0: float,
    momentum: float,
    resume: Path | None,
) -> list[str]:
    command = [
        python_executable,
        str(train_script),
        "--epochs",
        str(epochs),
        "--batch",
        str(batch),
        "--workers",
        str(workers),
        "--device",
        device,
        "--save-period",
        str(save_period),
        "--optimizer",
        optimizer,
        "--lr0",
        str(lr0),
        "--momentum",
        str(momentum),
        "--project",
        str(project),
        "--name",
        run_name,
        "--state",
        str(state_path),
        "--amp",
        str(amp_enabled).lower(),
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def _pid_is_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def acquire_pid_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing_pid = int(path.read_text(encoding="ascii").strip())
        except ValueError:
            existing_pid = -1
        if existing_pid > 0 and _pid_is_alive(existing_pid):
            raise RuntimeError(f"IOQC-SA supervisor is already running with PID {existing_pid}")
        path.unlink(missing_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
    finally:
        os.close(descriptor)


def release_pid_lock(path: Path) -> None:
    if not path.exists():
        return
    try:
        owner = int(path.read_text(encoding="ascii").strip())
    except ValueError:
        owner = -1
    if owner in {-1, os.getpid()}:
        path.unlink(missing_ok=True)


def select_resume_checkpoint(run_dir: Path) -> Path | None:
    return find_resume_checkpoint(run_dir)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _read_log_segment(path: Path, offset: int, limit: int = 1024 * 1024) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as file:
        size = path.stat().st_size
        file.seek(max(offset, size - limit))
        return file.read().decode("utf-8", "replace")


def _status(state: AdaptiveTrainingState, **extra: object) -> dict[str, object]:
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "batch": state.current_batch,
        "amp": state.amp_enabled,
        "completed_epoch": state.completed_epoch,
        "last_event": state.last_event,
        "checkpoint": state.checkpoint,
        **extra,
    }


def run_supervisor(args: argparse.Namespace) -> int:
    device_indices = parse_device_indices(args.device)
    profile = detect_gpu_profile(device_indices[0])
    state_path = args.state.resolve()
    run_dir = (args.project / args.name).resolve()
    if state_path.exists():
        state = load_adaptive_state(state_path)
        if args.batch_levels is not None:
            if state.current_batch not in args.batch_levels:
                raise ValueError(
                    f"Current batch {state.current_batch} is not in configured ladder {args.batch_levels}"
                )
            state.levels = args.batch_levels
            save_adaptive_state(state_path, state)
    else:
        policy = scale_batch_policy(
            batch_policy_for_vram(total_gib=profile.total_gib, free_gib=profile.free_gib),
            world_size=len(device_indices),
        )
        levels = args.batch_levels or policy.levels
        initial = args.initial_batch or (policy.initial_batch if policy.initial_batch in levels else levels[0])
        if initial not in levels:
            raise ValueError(f"Initial batch {initial} is not in configured ladder {levels}")
        state = AdaptiveTrainingState(levels=levels, current_batch=initial)
        save_adaptive_state(state_path, state)

    acquire_pid_lock(args.lock.resolve())
    try:
        while state.completed_epoch < args.epochs:
            free_disk_gib = shutil.disk_usage(args.project.resolve()).free / 1024**3
            if free_disk_gib < args.min_free_gib:
                state.last_event = "low_disk_stop"
                save_adaptive_state(state_path, state)
                return 3

            resume = select_resume_checkpoint(run_dir)
            if resume is not None:
                state.checkpoint = str(resume)
                save_adaptive_state(state_path, state)
            command = build_child_command(
                python_executable=sys.executable,
                train_script=ROOT / "scripts" / "train_rtdetr_ioqc_sa.py",
                project=args.project.resolve(),
                run_name=args.name,
                state_path=state_path,
                batch=state.current_batch,
                amp_enabled=state.amp_enabled,
                epochs=args.epochs,
                workers=args.workers,
                device=args.device,
                save_period=args.save_period,
                optimizer=args.optimizer,
                lr0=args.lr0,
                momentum=args.momentum,
                resume=resume,
            )
            args.log.parent.mkdir(parents=True, exist_ok=True)
            offset = args.log.stat().st_size if args.log.exists() else 0
            with args.log.open("a", encoding="utf-8") as log_file:
                log_file.write(
                    f"\nSUPERVISOR batch={state.current_batch} amp={state.amp_enabled} "
                    f"epoch={state.completed_epoch} resume={resume}\n"
                )
                log_file.flush()
                environment = build_child_environment()
                child = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                received_signal: int | None = None

                def forward_signal(signum, _frame) -> None:
                    nonlocal received_signal
                    received_signal = signum
                    if child.poll() is None:
                        if os.name == "posix":
                            os.killpg(child.pid, signal.SIGTERM)
                        else:
                            child.terminate()

                previous_term = signal.signal(signal.SIGTERM, forward_signal)
                previous_int = signal.signal(signal.SIGINT, forward_signal)
                try:
                    child.wait()
                finally:
                    signal.signal(signal.SIGTERM, previous_term)
                    signal.signal(signal.SIGINT, previous_int)

            if received_signal is not None:
                state = load_adaptive_state(state_path)
                state.last_event = "signal_stop"
                save_adaptive_state(state_path, state)
                _atomic_json(args.status, _status(state, process_state="signal_stop", signal=received_signal))
                return 128 + received_signal

            state = load_adaptive_state(state_path)
            result = classify_child_exit(child.returncode, _read_log_segment(args.log, offset))
            if result == "oom":
                state.record_oom()
            elif result == "numeric_failure":
                state.record_numeric_failure()
            elif result in {"planned_restart", "success"}:
                state.unexpected_failures = 0
            else:
                state.unexpected_failures += 1
                state.last_event = "unexpected_failure"
            save_adaptive_state(state_path, state)
            _atomic_json(args.status, _status(state, process_state=result, returncode=child.returncode))

            if result == "success" and state.completed_epoch >= args.epochs:
                return 0
            if state.unexpected_failures >= args.max_unexpected_failures:
                return 2
            if result not in {"planned_restart", "success"}:
                time.sleep(min(args.restart_delay * max(1, state.unexpected_failures), 300))
        return 0
    finally:
        release_pid_lock(args.lock.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU-adaptive IOQC-SA RT-DETR-L supervisor.")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", default="scratch-rtdetr-l-ioqc-sa-100ep")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.000714)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--initial-batch", type=int)
    parser.add_argument("--batch-levels", type=parse_batch_levels)
    parser.add_argument("--min-free-gib", type=float, default=20.0)
    parser.add_argument("--restart-delay", type=int, default=30)
    parser.add_argument("--max-unexpected-failures", type=int, default=3)
    return parser


def main() -> int:
    return run_supervisor(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
