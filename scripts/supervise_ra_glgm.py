"""Audited sequential single-GPU supervisor for the gated RA-GLGM experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_ra_resume import (  # noqa: E402
    record_optimizer_recovery_generation,
    validate_resume,
)
from scripts.train_rtdetr_ra_glgm import load_authority  # noqa: E402
from src.fdr_protocol import canonical_json_bytes  # noqa: E402
from src.ra_experiment_protocol import (  # noqa: E402
    RA_EXPERIMENT_PROTOCOL_SHA256,
    file_sha256,
)
from src.ra_learnability_probe import validate_learnability_report  # noqa: E402
from scripts.evaluate_ra_glgm_gate import (  # noqa: E402
    validate_evaluated_arm,
    validate_formal_report,
    validate_screen10_gate_report,
    validate_screen_gate_report,
)


TRAIN_STEPS = (
    ("smoke", "baseline"),
    ("smoke", "ra_glgm"),
    ("screen10", "baseline"),
    ("screen10", "ra_glgm"),
    ("screen", "baseline"),
    ("screen", "ra_glgm"),
    ("formal", "baseline"),
    ("formal", "ra_glgm"),
)
STAGE_TARGETS = {"smoke": 2, "screen10": 10, "screen": 30, "formal": 100}


def run_name(stage: str, variant: str) -> str:
    return f"{stage}-seed0-{variant}-ra-glgm-v1.1"


def build_train_command(
    *,
    python: Path,
    protocol_manifest: Path,
    initial_state: Path,
    dataset_root: Path,
    output_root: Path,
    stage: str,
    variant: str,
    learnability_report: Path,
    resume: Path | None = None,
    screen_gate: Path | None = None,
    screen10_gate: Path | None = None,
) -> list[str]:
    command = [
        str(python),
        "scripts/train_rtdetr_ra_glgm.py",
        "--variant",
        variant,
        "--stage",
        stage,
        "--protocol-manifest",
        str(protocol_manifest.resolve()),
        "--initial-state",
        str(initial_state.resolve()),
        "--dataset-root",
        str(dataset_root.resolve()),
        "--output-root",
        str(output_root.resolve()),
        "--name",
        run_name(stage, variant),
    ]
    command.extend(("--learnability-report", str(learnability_report.resolve())))
    if resume is not None:
        command.extend(("--resume", str(resume.resolve())))
    if stage == "formal":
        if screen_gate is None:
            raise ValueError("Formal100 launch requires the immutable passing Screen30 Gate")
        command.extend(("--screen-gate", str(screen_gate.resolve())))
    if stage == "screen":
        if screen10_gate is None:
            raise ValueError("Screen30 launch requires the immutable passing Screen10 Gate")
        command.extend(("--screen10-gate", str(screen10_gate.resolve())))
    return command


def build_evaluator_command(
    *, python: Path, evaluator_script: Path, run: Path, protocol_manifest: Path, stage: str
) -> list[str]:
    epochs = (
        "8,9,10"
        if stage == "screen10"
        else "28,29,30"
        if stage == "screen"
        else "98,99,100"
    )
    return [
        str(python),
        str(evaluator_script.resolve()),
        "--run-dir",
        str(run.resolve()),
        "--protocol-manifest",
        str(protocol_manifest.resolve()),
        "--epochs",
        epochs,
        "--output",
        str((run / "locked-evaluation.jsonl").resolve()),
    ]


def build_gate_command(
    *, python: Path, output_root: Path, gate_output: Path, stage: str = "screen"
) -> list[str]:
    if stage not in {"screen10", "screen", "formal"}:
        raise ValueError(f"unknown RA comparison stage: {stage}")
    return [
        str(python),
        "scripts/evaluate_ra_glgm_gate.py",
        "--baseline-run",
        str((output_root / run_name(stage, "baseline")).resolve()),
        "--ra-run",
        str((output_root / run_name(stage, "ra_glgm")).resolve()),
        "--output",
        str(gate_output.resolve()),
        "--stage",
        stage,
    ]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_audit(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_audit_event(path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    """Append one canonical event linked to the preceding event digest."""
    rows = _read_audit(path)
    previous = rows[-1]["event_sha256"] if rows else "0" * 64
    payload = {
        "sequence": len(rows) + 1,
        "time": _utc_now(),
        "previous_sha256": previous,
        **dict(event),
    }
    payload["event_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def validate_audit_chain(path: Path) -> None:
    previous = "0" * 64
    for sequence, row in enumerate(_read_audit(path), 1):
        actual = dict(row)
        recorded = actual.pop("event_sha256", None)
        if actual.get("sequence") != sequence or actual.get("previous_sha256") != previous:
            raise ValueError("RA supervisor audit event sequence/hash link is broken")
        expected = hashlib.sha256(canonical_json_bytes(actual)).hexdigest().upper()
        if recorded != expected:
            raise ValueError("RA supervisor audit event digest is broken")
        previous = recorded


def acquire_lock(path: Path) -> None:
    """Acquire one PID/start-identity lock, safely replacing only a proven stale owner."""

    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {
        "pid": os.getpid(),
        "process_start_identity": _process_start_identity(os.getpid()),
    }
    if owner["process_start_identity"] is None:
        raise RuntimeError("cannot determine RA supervisor process start identity")
    payload = json.dumps(owner, sort_keys=True, separators=(",", ":")).encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _attempt in range(3):
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError as error:
            if path.is_symlink():
                raise RuntimeError(f"RA supervisor lock may not be a symlink: {path}") from error
            observed = path.read_bytes()
            try:
                existing = json.loads(observed.decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                try:
                    legacy_pid = int(observed.decode("ascii"))
                except (UnicodeDecodeError, ValueError):
                    legacy_pid = -1
                if legacy_pid > 0 and _pid_exists(legacy_pid):
                    raise RuntimeError(f"RA supervisor lock already exists: {path}") from error
            else:
                if isinstance(existing, Mapping):
                    pid = existing.get("pid")
                    identity = existing.get("process_start_identity")
                    if (
                        isinstance(pid, int)
                        and isinstance(identity, str)
                        and _process_start_identity(pid) == identity
                    ):
                        raise RuntimeError(f"RA supervisor lock already exists: {path}") from error
                elif isinstance(existing, int) and existing > 0 and _pid_exists(existing):
                    raise RuntimeError(f"RA supervisor lock already exists: {path}") from error
            if not path.exists() or path.read_bytes() != observed:
                continue
            path.unlink()
            continue
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return
    raise RuntimeError(f"RA supervisor lock changed during stale-owner takeover: {path}")


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_start_identity(pid: int) -> str | None:
    if pid <= 0:
        return None
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            fields = proc_stat.read_text(encoding="ascii").rsplit(") ", 1)[1].split()
            start_ticks = fields[19]
            boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
            return f"linux:{boot}:{start_ticks}"
        except (OSError, IndexError):
            return None
    try:
        import psutil

        return f"process:{pid}:{psutil.Process(pid).create_time():.6f}"
    except Exception:
        return None


def release_lock(path: Path) -> None:
    if not path.exists():
        return
    raw = path.read_text(encoding="ascii")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        try:
            legacy_pid = int(raw)
        except ValueError:
            legacy_pid = -1
        owned = legacy_pid == os.getpid()
    else:
        owned = isinstance(payload, Mapping) and (
            payload.get("pid") == os.getpid()
            and payload.get("process_start_identity") == _process_start_identity(os.getpid())
        )
    if owned:
        path.unlink(missing_ok=True)


def ensure_fresh_run_slot(run: Path) -> Path | None:
    """Atomically quarantine one manifest-free orphan before an exact-name fresh launch."""

    if run.is_symlink():
        raise RuntimeError(f"fresh RA run path may not be a symlink: {run}")
    if not run.exists():
        return None
    if (run / "ra-run.json").exists():
        raise RuntimeError(
            f"fresh RA run path contains runtime authority and must use recovery: {run}"
        )
    for attempt in range(100):
        destination = run.parent / (
            f".{run.name}.orphan-{time.time_ns()}-{os.getpid()}-{attempt:02d}"
        )
        if destination.exists() or destination.is_symlink():
            continue
        try:
            run.rename(destination)
        except FileExistsError:
            continue
        return destination.resolve()
    raise RuntimeError(f"cannot allocate a unique RA orphan quarantine path for {run}")


def _gpu_snapshot() -> list[dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    result = subprocess.run(
        [
            executable,
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    snapshots = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 6:
            snapshots.append(
                {
                    "index": int(fields[0]),
                    "uuid": fields[1],
                    "name": fields[2],
                    "utilization_percent": int(fields[3]),
                    "memory_used_mib": int(fields[4]),
                    "memory_total_mib": int(fields[5]),
                }
            )
    return snapshots


def _completed_epoch(run: Path) -> int:
    path = run / "ra-epochs.jsonl"
    if not path.exists():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def _status_payload(
    *, output_root: Path, stage: str, variant: str, process_state: str, pid: int | None
) -> dict[str, Any]:
    run = output_root / run_name(stage, variant)
    usage = shutil.disk_usage(output_root)
    return {
        "time": _utc_now(),
        "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
        "stage": stage,
        "variant": variant,
        "process_state": process_state,
        "pid": pid,
        "completed_epoch": _completed_epoch(run),
        "target_epoch": STAGE_TARGETS[stage],
        "gpu": _gpu_snapshot(),
        "disk_free_gib": round(usage.free / 1024**3, 2),
        "run_dir": str(run.resolve()),
    }


def _run_child(
    command: list[str],
    *,
    log: Path,
    status: Path,
    output_root: Path,
    stage: str,
    variant: str,
    poll_seconds: int,
    environment: Mapping[str, str],
) -> tuple[int, int | None]:
    log.parent.mkdir(parents=True, exist_ok=True)
    received_signal: int | None = None
    with log.open("ab") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=dict(environment))

        def forward(signum: int, _frame: object) -> None:
            nonlocal received_signal
            received_signal = signum
            if child.poll() is None:
                child.send_signal(signum)

        previous_term = signal.signal(signal.SIGTERM, forward)
        previous_int = signal.signal(signal.SIGINT, forward)
        try:
            while child.poll() is None:
                _atomic_json(
                    status,
                    _status_payload(
                        output_root=output_root,
                        stage=stage,
                        variant=variant,
                        process_state="running",
                        pid=child.pid,
                    ),
                )
                time.sleep(poll_seconds)
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)
    return int(child.returncode), received_signal


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    current = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(ROOT) + (os.pathsep + current if current else "")
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return environment


def revalidate_supervisor_authority(
    protocol: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-read immutable authority and source bytes at every process boundary."""

    current = load_authority(protocol)
    if current != dict(expected):
        raise ValueError("RA protocol manifest changed after supervisor start")
    return current


def validate_supervisor_evaluator(
    evaluator_script: Path, authority: Mapping[str, Any]
) -> Path:
    """Require the supervisor CLI to execute the exact manifest-bound evaluator."""

    evaluator = authority.get("locked_evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError("locked evaluator authority is missing")
    selected = evaluator_script.resolve()
    expected = Path(str(evaluator.get("path", ""))).resolve()
    if selected != expected:
        raise ValueError("supervisor evaluator path differs from protocol authority")
    if selected.is_symlink() or not selected.is_file():
        raise FileNotFoundError("locked evaluator script is missing")
    if file_sha256(selected) != str(evaluator.get("sha256", "")).upper():
        raise ValueError("supervisor evaluator bytes differ from protocol authority")
    return selected


def ensure_locked_evaluation(
    *,
    python: Path,
    evaluator_script: Path,
    protocol: Path,
    authority: Mapping[str, Any],
    run: Path,
    stage: str,
    variant: str,
    audit: Path,
    environment: Mapping[str, str],
) -> str:
    """Create or reuse one evaluation, then apply the identical post-audit."""

    revalidate_supervisor_authority(protocol, authority)
    evaluation_output = run / "locked-evaluation.jsonl"
    origin = "reused" if evaluation_output.exists() else "fresh"
    if origin == "reused":
        append_audit_event(
            audit,
            {
                "event": "locked_evaluation_reuse_detected",
                "stage": stage,
                "variant": variant,
            },
        )
    else:
        evaluator = build_evaluator_command(
            python=python,
            evaluator_script=evaluator_script,
            run=run,
            protocol_manifest=protocol,
            stage=stage,
        )
        append_audit_event(
            audit,
            {"event": "locked_evaluation_start", "stage": stage, "variant": variant},
        )
        result = subprocess.run(evaluator, cwd=ROOT, env=dict(environment), check=False)
        append_audit_event(
            audit,
            {
                "event": "locked_evaluation_exit",
                "stage": stage,
                "variant": variant,
                "returncode": result.returncode,
            },
        )
        if result.returncode != 0:
            raise RuntimeError(f"locked evaluator failed for {stage}/{variant}")
    revalidate_supervisor_authority(protocol, authority)
    validate_evaluated_arm(run, variant=variant, stage=stage)
    append_audit_event(
        audit,
        {
            "event": "locked_evaluation_verified",
            "stage": stage,
            "variant": variant,
            "origin": origin,
        },
    )
    return origin


def _validate_gate_for_formal(path: Path, output_root: Path) -> None:
    try:
        validate_screen_gate_report(
            path,
            baseline_run=output_root / run_name("screen", "baseline"),
            method_run=output_root / run_name("screen", "ra_glgm"),
        )
    except (OSError, ValueError) as error:
        raise RuntimeError("Screen30 gate did not authorize Formal100") from error


def _validate_gate_for_screen(path: Path, output_root: Path) -> None:
    try:
        validate_screen10_gate_report(
            path,
            baseline_run=output_root / run_name("screen10", "baseline"),
            method_run=output_root / run_name("screen10", "ra_glgm"),
        )
    except (OSError, ValueError) as error:
        raise RuntimeError("Screen10 gate did not authorize Screen30") from error


def _validate_formal_report(path: Path) -> dict[str, Any]:
    root = path.resolve().parent
    report = validate_formal_report(
        path,
        baseline_run=root / run_name("formal", "baseline"),
        method_run=root / run_name("formal", "ra_glgm"),
    )
    if (
        report.get("protocol_sha256") != RA_EXPERIMENT_PROTOCOL_SHA256
        or report.get("report_name") != "RA-GLGM-Formal100-v1.1"
        or report.get("primary_evidence") != ["epoch100", "tail3_mean"]
        or report.get("engineering", {}).get("complete") is not True
        or not isinstance(report.get("formal_success"), bool)
    ):
        raise RuntimeError("Formal100 report is absent, foreign, or incomplete")
    return report


def validate_smoke_advancement(decision: Mapping[str, Any], *, variant: str) -> None:
    """Fail closed before advancing from Smoke2 to any Screen10 arm."""
    valid = (
        decision.get("decision") == "complete"
        and decision.get("stage") == "smoke"
        and decision.get("completed_epoch") == STAGE_TARGETS["smoke"]
        and decision.get("amp_scale") == 128.0
        and decision.get("amp_skipped_steps") == 0
        and decision.get("public_gradient_finite") is True
        and decision.get("fdr_gradient_finite") is True
        and decision.get("public_gradient_nonzero") is True
        and decision.get("fdr_gradient_nonzero") is True
    )
    if variant == "ra_glgm":
        valid = valid and decision.get("ra_private_gradient_nonzero") is True
    if not valid:
        raise RuntimeError(f"Smoke2 optimizer/gradient gate failed for {variant}")


def run_supervisor(args: argparse.Namespace) -> int:
    protocol = args.protocol_manifest.resolve()
    initial_state = args.initial_state.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    screen10_gate_output = output_root / "RA_GLGM_SCREEN10_GATE.json"
    gate_output = output_root / "RA_GLGM_SCREEN30_GATE.json"
    formal_output = output_root / "RA_GLGM_FORMAL100_REPORT.json"
    audit = args.audit.resolve()
    status = args.status.resolve()
    lock = args.lock.resolve()
    authority = load_authority(protocol)
    if file_sha256(initial_state) != str(authority.get("initial_state", {}).get("sha256", "")).upper():
        raise ValueError("paired initial-state bytes differ from protocol authority")
    validate_learnability_report(
        args.learnability_report.resolve(), protocol_manifest=authority
    )
    validate_supervisor_evaluator(args.evaluator_script, authority)
    validate_audit_chain(audit)
    acquire_lock(lock)
    environment = _environment()
    append_audit_event(audit, {"event": "supervisor_start", "pid": os.getpid()})
    active_stage = "startup"
    active_variant = "none"
    try:
        for stage, variant in TRAIN_STEPS:
            active_stage, active_variant = stage, variant
            if stage == "screen":
                _validate_gate_for_screen(screen10_gate_output, output_root)
            if stage == "formal":
                _validate_gate_for_formal(gate_output, output_root)
            run = output_root / run_name(stage, variant)
            resume: Path | None = None
            attempts = 0
            while True:
                revalidate_supervisor_authority(protocol, authority)
                decision: dict[str, Any] | None = None
                if run.exists() and (run / "ra-run.json").is_file():
                    decision = validate_resume(
                        run,
                        variant=variant,
                        stage=stage,
                        protocol_manifest=protocol,
                        learnability_report=args.learnability_report,
                        screen_gate=gate_output if stage == "formal" else None,
                        screen10_gate=screen10_gate_output if stage == "screen" else None,
                    )
                    append_audit_event(audit, {"event": "recovery_audit", **decision})
                    if decision["decision"] == "complete":
                        if stage == "smoke":
                            validate_smoke_advancement(decision, variant=variant)
                            append_audit_event(
                                audit,
                                {
                                    "event": "smoke_optimizer_gate_passed",
                                    "variant": variant,
                                    "optimizer_attempts": decision["optimizer_attempts"],
                                    "optimizer_evidence_sha256": decision[
                                        "optimizer_evidence_sha256"
                                    ],
                                },
                            )
                        break
                    resume = Path(decision["checkpoint"])
                attempts += 1
                if attempts > args.max_attempts:
                    raise RuntimeError(f"recovery attempts exhausted for {stage}/{variant}")
                if resume is not None:
                    if decision is None:
                        raise RuntimeError("resume checkpoint has no audited recovery decision")
                    generation = record_optimizer_recovery_generation(run, decision)
                    append_audit_event(
                        audit,
                        {
                            "event": "optimizer_recovery_generation_recorded",
                            "stage": stage,
                            "variant": variant,
                            "generation": generation["generation"],
                            "discarded_attempt_count": generation[
                                "discarded_attempt_count"
                            ],
                            "checkpoint_sha256": generation["checkpoint_sha256"],
                        },
                    )
                else:
                    orphan = ensure_fresh_run_slot(run)
                    if orphan is not None:
                        append_audit_event(
                            audit,
                            {
                                "event": "manifest_free_run_quarantined",
                                "stage": stage,
                                "variant": variant,
                                "original": str(run),
                                "quarantine": str(orphan),
                            },
                        )
                revalidate_supervisor_authority(protocol, authority)
                command = build_train_command(
                    python=args.python,
                    protocol_manifest=protocol,
                    initial_state=initial_state,
                    dataset_root=args.dataset_root,
                    output_root=output_root,
                    stage=stage,
                    variant=variant,
                    learnability_report=args.learnability_report,
                    resume=resume,
                    screen_gate=gate_output if stage == "formal" else None,
                    screen10_gate=screen10_gate_output if stage == "screen" else None,
                )
                append_audit_event(
                    audit,
                    {
                        "event": "arm_launch",
                        "stage": stage,
                        "variant": variant,
                        "attempt": attempts,
                        "resume": str(resume) if resume else None,
                    },
                )
                returncode, received_signal = _run_child(
                    command,
                    log=args.log.resolve(),
                    status=status,
                    output_root=output_root,
                    stage=stage,
                    variant=variant,
                    poll_seconds=args.poll_seconds,
                    environment=environment,
                )
                append_audit_event(
                    audit,
                    {
                        "event": "arm_exit",
                        "stage": stage,
                        "variant": variant,
                        "returncode": returncode,
                        "signal": received_signal,
                    },
                )
                if received_signal is not None:
                    interrupted = _status_payload(
                        output_root=output_root,
                        stage=stage,
                        variant=variant,
                        process_state="interrupted",
                        pid=None,
                    )
                    interrupted["signal"] = received_signal
                    _atomic_json(
                        status,
                        interrupted,
                    )
                    return 128 + received_signal
                # Success and failure both pass through the same strict recovery audit.

            if stage in {"screen10", "screen", "formal"}:
                ensure_locked_evaluation(
                    python=args.python,
                    evaluator_script=args.evaluator_script,
                    protocol=protocol,
                    authority=authority,
                    run=run,
                    stage=stage,
                    variant=variant,
                    audit=audit,
                    environment=environment,
                )

            if stage == "screen10" and variant == "ra_glgm":
                revalidate_supervisor_authority(protocol, authority)
                if screen10_gate_output.exists():
                    _validate_gate_for_screen(screen10_gate_output, output_root)
                    append_audit_event(
                        audit,
                        {"event": "screen10_gate_reused_after_recomputation"},
                    )
                else:
                    result = subprocess.run(
                        build_gate_command(
                            python=args.python,
                            output_root=output_root,
                            gate_output=screen10_gate_output,
                            stage="screen10",
                        ),
                        cwd=ROOT,
                        env=environment,
                        check=False,
                    )
                    append_audit_event(
                        audit,
                        {"event": "screen10_gate_exit", "returncode": result.returncode},
                    )
                    if result.returncode != 0:
                        _validate_gate_for_screen(
                            screen10_gate_output, output_root
                        )  # Raises a precise failed-gate error.
                revalidate_supervisor_authority(protocol, authority)
                _validate_gate_for_screen(screen10_gate_output, output_root)

            if stage == "screen" and variant == "ra_glgm":
                revalidate_supervisor_authority(protocol, authority)
                if gate_output.exists():
                    _validate_gate_for_formal(gate_output, output_root)
                    append_audit_event(audit, {"event": "screen_gate_reused_after_recomputation"})
                else:
                    result = subprocess.run(
                        build_gate_command(
                            python=args.python,
                            output_root=output_root,
                            gate_output=gate_output,
                            stage="screen",
                        ),
                        cwd=ROOT,
                        env=environment,
                        check=False,
                    )
                    append_audit_event(audit, {"event": "screen_gate_exit", "returncode": result.returncode})
                    if result.returncode != 0:
                        _validate_gate_for_formal(
                            gate_output, output_root
                        )  # Raises a precise failed-gate error.
                revalidate_supervisor_authority(protocol, authority)
                _validate_gate_for_formal(gate_output, output_root)

            if stage == "formal" and variant == "ra_glgm":
                revalidate_supervisor_authority(protocol, authority)
                if formal_output.exists():
                    _validate_formal_report(formal_output)
                    append_audit_event(audit, {"event": "formal_report_reused_after_recomputation"})
                else:
                    result = subprocess.run(
                        build_gate_command(
                            python=args.python,
                            output_root=output_root,
                            gate_output=formal_output,
                            stage="formal",
                        ),
                        cwd=ROOT,
                        env=environment,
                        check=False,
                    )
                    append_audit_event(
                        audit,
                        {"event": "formal_report_exit", "returncode": result.returncode},
                    )
                    if result.returncode != 0:
                        raise RuntimeError("Formal100 paired report failed engineering audit")
                    _validate_formal_report(formal_output)
                revalidate_supervisor_authority(protocol, authority)
                _validate_formal_report(formal_output)

        revalidate_supervisor_authority(protocol, authority)
        formal_report = _validate_formal_report(formal_output)
        append_audit_event(
            audit,
            {
                "event": "experiment_complete",
                "formal_success": formal_report["formal_success"],
            },
        )
        _atomic_json(
            status,
            {
                "time": _utc_now(),
                "process_state": "complete",
                "screen10_gate": str(screen10_gate_output),
                "screen_gate": str(gate_output),
                "formal_report": str(formal_output),
                "formal_success": formal_report["formal_success"],
            },
        )
        return 0
    except BaseException as error:
        failed: dict[str, Any] = {
            "time": _utc_now(),
            "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
            "process_state": "failed",
            "stage": active_stage,
            "variant": active_variant,
            "pid": None,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if active_stage in STAGE_TARGETS and active_variant in {"baseline", "ra_glgm"}:
            failed = {
                **_status_payload(
                    output_root=output_root,
                    stage=active_stage,
                    variant=active_variant,
                    process_state="failed",
                    pid=None,
                ),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        try:
            _atomic_json(status, failed)
        except OSError:
            pass
        raise
    finally:
        release_lock(lock)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--learnability-report", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluator-script", type=Path, default=ROOT / "scripts" / "evaluate_ra_glgm_checkpoints.py")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser


def main() -> int:
    return run_supervisor(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
