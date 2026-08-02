"""Supervise the immutable IBER-BE Gate-0, Gate-1, and screen workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.iber_protocol import (  # noqa: E402
    DESIGN_VERSION,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    RUNTIME_AMENDMENT_SHA256,
    SCREEN_EPOCHS,
    file_sha256,
    write_immutable_report,
)
from src.lpr_protocol import (  # noqa: E402
    CATEGORY_NAMES,
    category_mapping_sha256,
    dataset_signature,
    select_hashed_subset,
    subset_signature,
)


PHASE_ORDER = (
    "authority",
    "gate0",
    "stock_authority",
    "cache",
    "probe",
    "screen30",
    "screen_decision",
)
TERMINAL_PHASES = frozenset(
    ("engineering_invalid", "scientific_failed", "screen_complete")
)
ACCEPTED_ENGINEERING_STATUSES = frozenset(
    ("passed", "passed_with_runtime_amendment")
)
EXPECTED_CATEGORY_SHA256 = (
    "1455A1F5D9FA9988799815B36CF2CE6D5044B5BD7CAD7CC614D6B5E5059EF2A6"
)
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_CHECKPOINT_PATTERN = re.compile(r"epoch-(\d{4})\.pt")
_CREDENTIAL_OPTIONS = frozenset(
    ("--token", "--password", "--secret", "--api-key", "--github-token")
)
_CREDENTIAL_VALUE_PATTERNS = (
    re.compile(r"github_pat_[A-Za-z0-9_]+", re.IGNORECASE),
    re.compile(r"gh[pousr]_[A-Za-z0-9]+", re.IGNORECASE),
    re.compile(r"https?://[^/\s:@]+:[^/\s@]+@", re.IGNORECASE),
)


@dataclass(frozen=True)
class PipelineEvidence:
    authority: str | None
    gate0: str | None
    stock_authority: str | None
    cache_complete: bool
    gate1: str | None
    screen_completed_epoch: int
    screen_verified_epoch: int
    screen_decision: str | None
    source_consistent: bool = True


@dataclass(frozen=True)
class PipelinePaths:
    baseline_checkpoint: Path
    dataset_root: Path
    run_root: Path
    cache_root: Path
    publication_config: Path
    device: str = "0"

    @property
    def screen_root(self) -> Path:
        return self.run_root / "screen"

    @property
    def gate1_decision(self) -> Path:
        return self.run_root / "probe" / "gate1-decision.json"


def _valid_epoch(value: object) -> bool:
    return type(value) is int and 0 <= value <= SCREEN_EPOCHS


def next_pipeline_phase(evidence: PipelineEvidence) -> str:
    """Return the sole phase allowed by the currently verified evidence."""
    if not evidence.source_consistent:
        return "engineering_invalid"
    if not _valid_epoch(evidence.screen_completed_epoch) or not _valid_epoch(
        evidence.screen_verified_epoch
    ):
        return "engineering_invalid"
    if evidence.screen_verified_epoch > evidence.screen_completed_epoch:
        return "engineering_invalid"

    for status in (
        evidence.authority,
        evidence.gate0,
        evidence.stock_authority,
    ):
        if status == "engineering_invalid":
            return "engineering_invalid"

    if evidence.authority is None:
        return "authority"
    if evidence.authority not in ACCEPTED_ENGINEERING_STATUSES:
        return "engineering_invalid"
    if evidence.gate0 is None:
        return "gate0"
    if evidence.gate0 not in ACCEPTED_ENGINEERING_STATUSES:
        return "engineering_invalid"
    if evidence.stock_authority is None:
        return "stock_authority"
    if evidence.stock_authority not in ACCEPTED_ENGINEERING_STATUSES:
        return "engineering_invalid"
    if not evidence.cache_complete:
        return "cache"
    if evidence.gate1 is None:
        return "probe"
    if evidence.gate1 == "engineering_invalid":
        return "engineering_invalid"
    if evidence.gate1 != "passed":
        return "scientific_failed"
    if evidence.screen_verified_epoch < SCREEN_EPOCHS:
        return "screen30"
    if evidence.screen_completed_epoch != SCREEN_EPOCHS:
        return "engineering_invalid"
    if evidence.screen_decision is None:
        return "screen_decision"
    if evidence.screen_decision == "engineering_invalid":
        return "engineering_invalid"
    if evidence.screen_decision != "passed":
        return "scientific_failed"
    return "screen_complete"


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_state(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Durably replace state while preserving its append-only history."""
    destination = Path(path)
    candidate = dict(payload)
    history = candidate.get("history")
    if not isinstance(history, list):
        raise ValueError("IBER-BE pipeline state history must be a list")
    if destination.is_file():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        previous_history = previous.get("history")
        if not isinstance(previous_history, list) or history[: len(previous_history)] != previous_history:
            raise ValueError("IBER-BE pipeline history is not append-only")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_json(candidate))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _assert_credential_free(command: Sequence[str]) -> None:
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError("subprocess command must be a non-empty argument sequence")
    arguments = [str(value) for value in command]
    lowered = [value.lower() for value in arguments]
    if any(
        value in _CREDENTIAL_OPTIONS
        or any(value.startswith(option + "=") for option in _CREDENTIAL_OPTIONS)
        for value in lowered
    ):
        raise ValueError("credential-bearing command options are forbidden")
    joined = " ".join(arguments)
    if any(pattern.search(joined) for pattern in _CREDENTIAL_VALUE_PATTERNS):
        raise ValueError("credential material is forbidden in subprocess commands")


def _redact_credentials(value: str) -> str:
    redacted = value
    for pattern in _CREDENTIAL_VALUE_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def run_tracked_subprocess(
    command: Sequence[str],
    *,
    phase: str,
    run_root: str | Path,
    state_path: str | Path,
    state: dict[str, Any],
) -> int:
    """Run one credential-free child and atomically track its process group."""
    arguments = [str(value) for value in command]
    _assert_credential_free(arguments)
    root = Path(run_root)
    log_path = root / "logs" / f"{phase}-{time.time_ns()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    popen_options: dict[str, Any] = {
        "cwd": REPOSITORY_ROOT,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
    }
    if os.name == "nt":
        popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_options["start_new_session"] = True

    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        log.write(
            "command="
            + json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)
            + "\n"
        )
        log.flush()
        process = subprocess.Popen(arguments, **popen_options)
        active = {
            "phase": phase,
            "pid": process.pid,
            "process_group_id": process.pid,
            "command": arguments,
            "log": str(log_path.resolve()),
            "started_time_ns": time.time_ns(),
        }
        state["active_process"] = active
        atomic_write_state(state_path, state)
        if process.stdout is None:
            raise RuntimeError("IBER-BE child output pipe is unavailable")
        for line in process.stdout:
            log.write(_redact_credentials(line))
            log.flush()
        return_code = process.wait()

    finished = {
        **state.pop("active_process"),
        "return_code": return_code,
        "finished_time_ns": time.time_ns(),
    }
    state["last_process"] = finished
    atomic_write_state(state_path, state)
    return return_code


def _source_commit() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip().lower()
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise RuntimeError("IBER-BE source commit is invalid")
    return commit


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"IBER-BE artifact is not an object: {path}")
    if payload.get("design_version") != DESIGN_VERSION:
        raise ValueError(f"IBER-BE artifact identity mismatch: {path}")
    return payload


def _status(payload: Mapping[str, Any]) -> str | None:
    decision = payload.get("decision")
    source = decision if isinstance(decision, Mapping) else payload
    value = source.get("status")
    return str(value) if value is not None else None


def _nested_source(payload: Mapping[str, Any], *keys: str) -> str:
    value: object = payload
    for key in keys:
        if not isinstance(value, Mapping):
            raise ValueError("IBER-BE prerequisite source commit is missing")
        value = value.get(key)
    commit = str(value).lower()
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError("IBER-BE prerequisite source commit is invalid")
    return commit


def validate_prerequisite_sources(
    reports: Mapping[str, Mapping[str, Any]], expected_source_commit: str
) -> str:
    """Require every pre-screen authority to bind the same source commit."""
    expected = expected_source_commit.lower()
    if _COMMIT_PATTERN.fullmatch(expected) is None:
        raise ValueError("IBER-BE expected source commit is invalid")
    required = {"authority", "gate0", "stock_authority", "cache", "gate1"}
    if set(reports) != required:
        raise ValueError("IBER-BE prerequisite reports are incomplete")
    commits = [
        _nested_source(reports["authority"], "source_commit"),
        _nested_source(reports["gate0"], "authority", "source_commit"),
        _nested_source(reports["stock_authority"], "source_commit"),
        _nested_source(reports["cache"], "authority", "source_commit"),
    ]
    gate1_reports = reports["gate1"].get("reports")
    if not isinstance(gate1_reports, Mapping) or set(gate1_reports) != {
        "b0",
        "b1",
        "b2",
        "b3",
    }:
        raise ValueError("IBER-BE Gate-1 prerequisite reports are incomplete")
    for arm in ("b0", "b1", "b2", "b3"):
        report = gate1_reports[arm]
        if not isinstance(report, Mapping):
            raise ValueError("IBER-BE Gate-1 prerequisite report is invalid")
        commits.append(_nested_source(report, "cache_authority", "source_commit"))
    if any(commit != expected for commit in commits):
        raise ValueError("IBER-BE prerequisite source commit mismatch")
    return expected


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            raise ValueError(f"publication ledger has blank row {line_number}")
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError("publication ledger rows must be objects")
        rows.append(row)
    return rows


def select_verified_resume(
    screen_root: str | Path, *, source_commit: str
) -> Path | None:
    """Return only the remotely verified, contiguous screen checkpoint tip."""
    root = Path(screen_root)
    expected_source = source_commit.lower()
    rows = _read_ledger(root / "publication-ledger.jsonl")
    for expected_epoch, row in enumerate(rows, 1):
        if row.get("completed_epoch") != expected_epoch:
            raise ValueError("publication ledger epochs are not contiguous")
        if row.get("verified") is not True:
            raise ValueError("publication ledger row is not verified")
        expected_identity = {
            "design_version": DESIGN_VERSION,
            "stage": "screen",
            "probe": "b3",
            "seed": 0,
            "source_commit": expected_source,
        }
        for name, value in expected_identity.items():
            actual = row.get(name)
            if name == "source_commit" and isinstance(actual, str):
                actual = actual.lower()
            if actual != value:
                label = "source commit" if name == "source_commit" else name
                raise ValueError(f"publication ledger {label} mismatch")
        checkpoint_authority = row.get("checkpoint")
        if not isinstance(checkpoint_authority, Mapping):
            raise ValueError("publication ledger checkpoint authority is missing")
        checkpoint = root / "checkpoints" / f"epoch-{expected_epoch:04d}.pt"
        if not checkpoint.is_file():
            raise ValueError("verified resume checkpoint is missing")
        actual_sha = file_sha256(checkpoint).lower()
        expected_sha = str(checkpoint_authority.get("sha256", "")).lower()
        if actual_sha != expected_sha:
            raise ValueError("verified resume checkpoint SHA256 mismatch")
    if len(rows) > SCREEN_EPOCHS:
        raise ValueError("publication ledger exceeds the 30-epoch screen")
    if not rows:
        return None
    return root / "checkpoints" / f"epoch-{len(rows):04d}.pt"


def _completed_screen_epoch(screen_root: Path) -> int:
    epochs = []
    for path in (screen_root / "checkpoints").glob("epoch-*.pt"):
        match = _CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is not None:
            epochs.append(int(match.group(1)))
    return max(epochs, default=0)


def _existing_reports(paths: PipelinePaths) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    authority = paths.run_root / "authority.json"
    if authority.is_file():
        reports["authority"] = _load_json(authority)
    gate0_reports = sorted((paths.run_root / "gate0").glob("attempt-*.json"))
    if gate0_reports:
        reports["gate0"] = _load_json(gate0_reports[-1])
    stock = paths.run_root / "stock-authority.json"
    if stock.is_file():
        reports["stock_authority"] = _load_json(stock)
    cache = paths.cache_root / "manifest.json"
    if cache.is_file():
        reports["cache"] = _load_json(cache)
    gate1 = paths.gate1_decision
    if gate1.is_file():
        reports["gate1"] = _load_json(gate1)
    return reports


def collect_pipeline_evidence(
    paths: PipelinePaths, *, source_commit: str
) -> PipelineEvidence:
    reports = _existing_reports(paths)
    source_consistent = True
    if set(reports) == {"authority", "gate0", "stock_authority", "cache", "gate1"}:
        try:
            validate_prerequisite_sources(reports, source_commit)
        except ValueError:
            source_consistent = False
    else:
        available_commits: list[str] = []
        extractors = {
            "authority": ("source_commit",),
            "gate0": ("authority", "source_commit"),
            "stock_authority": ("source_commit",),
            "cache": ("authority", "source_commit"),
        }
        try:
            for name, keys in extractors.items():
                if name in reports:
                    available_commits.append(_nested_source(reports[name], *keys))
            source_consistent = all(
                commit == source_commit.lower() for commit in available_commits
            )
        except ValueError:
            source_consistent = False

    resume = select_verified_resume(paths.screen_root, source_commit=source_commit)
    verified_epoch = 0 if resume is None else int(resume.stem.split("-")[-1])
    decision_path = paths.screen_root / "screen-decision.json"
    decision = _load_json(decision_path) if decision_path.is_file() else None
    cache_report = reports.get("cache")
    return PipelineEvidence(
        authority=_status(reports["authority"]) if "authority" in reports else None,
        gate0=_status(reports["gate0"]) if "gate0" in reports else None,
        stock_authority=(
            _status(reports["stock_authority"])
            if "stock_authority" in reports
            else None
        ),
        cache_complete=(
            cache_report is not None and cache_report.get("complete") is True
        ),
        gate1=_status(reports["gate1"]) if "gate1" in reports else None,
        screen_completed_epoch=_completed_screen_epoch(paths.screen_root),
        screen_verified_epoch=verified_epoch,
        screen_decision=_status(decision) if decision is not None else None,
        source_consistent=source_consistent,
    )


def build_phase_command(
    phase: str,
    paths: PipelinePaths,
    *,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    """Build a frozen child command; phases cannot be selected from the CLI."""
    python = sys.executable
    scripts = REPOSITORY_ROOT / "scripts"
    common = [
        "--baseline-checkpoint",
        str(paths.baseline_checkpoint),
        "--dataset-root",
        str(paths.dataset_root),
    ]
    if phase == "gate0":
        gate0_root = paths.run_root / "gate0"
        attempt = len(list(gate0_root.glob("attempt-*.json"))) + 1
        command = [
            python,
            str(scripts / "run_iber_canary.py"),
            *common,
            "--output",
            str(gate0_root / f"attempt-{attempt:03d}.json"),
            "--device",
            paths.device,
        ]
    elif phase == "stock_authority":
        command = [
            python,
            str(scripts / "evaluate_iber_stock.py"),
            *common,
            "--output",
            str(paths.run_root / "stock-authority.json"),
        ]
    elif phase == "cache":
        command = [
            python,
            str(scripts / "cache_iber_evidence.py"),
            *common,
            "--output-root",
            str(paths.cache_root),
            "--device",
            paths.device,
        ]
    elif phase == "probe":
        command = [
            python,
            str(scripts / "run_iber_probe.py"),
            "--cache-root",
            str(paths.cache_root),
            "--output-root",
            str(paths.run_root / "probe"),
            "--device",
            paths.device,
        ]
    elif phase == "screen30":
        command = [
            python,
            str(scripts / "train_iber.py"),
            *common,
            "--gate1-decision",
            str(paths.gate1_decision),
            "--publication-config",
            str(paths.publication_config),
            "--output-root",
            str(paths.screen_root),
            "--device",
            paths.device,
        ]
        if resume_checkpoint is not None:
            command.extend(("--resume-checkpoint", str(resume_checkpoint)))
    else:
        raise ValueError(f"IBER-BE phase has no subprocess command: {phase}")
    _assert_credential_free(command)
    return command


def finalize_screen_decision(
    evaluation_path: str | Path,
    destination: str | Path,
    *,
    source_commit: str,
) -> dict[str, Any]:
    """Validate and freeze the Gate-2 decision already evaluated at epoch 30."""
    evaluation = _load_json(Path(evaluation_path))
    expected = {
        "stage": "screen",
        "probe": "b3",
        "seed": 0,
        "epoch": SCREEN_EPOCHS,
    }
    for name, value in expected.items():
        if evaluation.get(name) != value:
            label = "epoch30" if name == "epoch" else name
            raise ValueError(f"IBER-BE screen decision {label} authority mismatch")
    actual_source = str(evaluation.get("source_commit", "")).lower()
    if actual_source != source_commit.lower():
        raise ValueError("IBER-BE screen decision source commit mismatch")
    decision = evaluation.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("IBER-BE epoch30 evaluation decision is missing")
    status = decision.get("status")
    if status not in {"passed", "scientific_failed", "engineering_invalid"}:
        raise ValueError("IBER-BE epoch30 evaluation decision status is invalid")
    payload = {
        "format_version": 1,
        "design_version": DESIGN_VERSION,
        "stage": "screen_decision",
        "status": status,
        "source_commit": actual_source,
        "evaluation_epoch": SCREEN_EPOCHS,
        "evaluation": {
            "path": str(Path(evaluation_path).resolve()),
            "sha256": file_sha256(evaluation_path),
        },
        "decision": dict(decision),
    }
    write_immutable_report(destination, payload)
    return payload


def _authority_report(paths: PipelinePaths, *, source_commit: str) -> dict[str, Any]:
    actual: dict[str, Any] = {"source_commit": source_commit}
    try:
        train_images = sorted(
            path
            for path in (paths.dataset_root / "images" / "train").glob("**/*")
            if path.is_file()
        )
        subset = select_hashed_subset(
            train_images,
            root=paths.dataset_root,
            fraction=0.10,
        )
        actual.update(
            {
                "baseline_sha256": file_sha256(paths.baseline_checkpoint),
                "dataset_sha256": str(
                    dataset_signature(paths.dataset_root)["sha256"]
                ),
                "subset_sha256": subset_signature(
                    subset,
                    root=paths.dataset_root,
                ),
                "category_sha256": category_mapping_sha256(CATEGORY_NAMES),
                "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
            }
        )
        expected = {
            "baseline_sha256": EXPECTED_BASELINE_SHA256,
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "subset_sha256": EXPECTED_SUBSET_SHA256,
            "category_sha256": EXPECTED_CATEGORY_SHA256,
            "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
        }
        violations = {
            name: {"expected": value, "actual": actual.get(name)}
            for name, value in expected.items()
            if str(actual.get(name, "")).upper() != str(value).upper()
        }
        status = (
            "engineering_invalid"
            if violations
            else "passed_with_runtime_amendment"
        )
        return {
            "format_version": 1,
            "design_version": DESIGN_VERSION,
            "stage": "authority",
            "status": status,
            **actual,
            "violations": violations,
        }
    except Exception as error:
        return {
            "format_version": 1,
            "design_version": DESIGN_VERSION,
            "stage": "authority",
            "status": "engineering_invalid",
            **actual,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--publication-config", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser.parse_args(argv)


def _transition(
    state: dict[str, Any], state_path: Path, phase: str, **details: Any
) -> None:
    if state.get("phase") == phase:
        return
    history = state.setdefault("history", [])
    history.append(
        {
            "sequence": len(history) + 1,
            "phase": phase,
            "time_ns": time.time_ns(),
            **details,
        }
    )
    state["phase"] = phase
    atomic_write_state(state_path, state)


def _write_pid(run_root: Path) -> None:
    destination = run_root / "pipeline.pid"
    temporary = destination.with_suffix(".pid.tmp")
    temporary.write_text(f"{os.getpid()}\n", encoding="ascii", newline="\n")
    os.replace(temporary, destination)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = PipelinePaths(
        baseline_checkpoint=args.baseline_checkpoint.resolve(),
        dataset_root=args.dataset_root.resolve(),
        run_root=args.run_root.resolve(),
        cache_root=args.cache_root.resolve(),
        publication_config=args.publication_config.resolve(),
        device=str(args.device),
    )
    paths.run_root.mkdir(parents=True, exist_ok=True)
    paths.cache_root.mkdir(parents=True, exist_ok=True)
    source_commit = _source_commit()
    state_path = paths.run_root / "pipeline-state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {
            "format_version": 1,
            "design_version": DESIGN_VERSION,
            "source_commit": source_commit,
            "supervisor_pid": os.getpid(),
            "supervisor_process_group_id": os.getpgrp()
            if hasattr(os, "getpgrp")
            else os.getpid(),
            "history": [],
        }
    )
    if state.get("design_version") != DESIGN_VERSION or state.get(
        "source_commit"
    ) != source_commit:
        _transition(
            state,
            state_path,
            "engineering_invalid",
            reason="pipeline_authority_drift",
        )
        return 1
    _write_pid(paths.run_root)

    while True:
        try:
            evidence = collect_pipeline_evidence(
                paths,
                source_commit=source_commit,
            )
            phase = next_pipeline_phase(evidence)
        except Exception as error:
            _transition(
                state,
                state_path,
                "engineering_invalid",
                failed_action="evidence_validation",
                error_type=type(error).__name__,
                error=str(error),
            )
            return 1
        _transition(state, state_path, phase)
        if phase in TERMINAL_PHASES:
            if phase == "screen_complete":
                return 0
            return 2 if phase == "scientific_failed" else 1

        if phase == "authority":
            write_immutable_report(
                paths.run_root / "authority.json",
                _authority_report(paths, source_commit=source_commit),
            )
            continue

        if phase == "screen_decision":
            try:
                finalize_screen_decision(
                    paths.screen_root / "evaluations" / "epoch-0030.json",
                    paths.screen_root / "screen-decision.json",
                    source_commit=source_commit,
                )
            except Exception as error:
                _transition(
                    state,
                    state_path,
                    "engineering_invalid",
                    failed_action=phase,
                    error_type=type(error).__name__,
                    error=str(error),
                )
                return 1
            continue

        resume = None
        if phase == "screen30":
            prerequisite_reports = _existing_reports(paths)
            validate_prerequisite_sources(
                prerequisite_reports,
                source_commit,
            )
            resume = select_verified_resume(
                paths.screen_root,
                source_commit=source_commit,
            )
        command = build_phase_command(
            phase,
            paths,
            resume_checkpoint=resume,
        )
        return_code = run_tracked_subprocess(
            command,
            phase=phase,
            run_root=paths.run_root,
            state_path=state_path,
            state=state,
        )
        if return_code != 0:
            try:
                refreshed = next_pipeline_phase(
                    collect_pipeline_evidence(
                        paths,
                        source_commit=source_commit,
                    )
                )
            except Exception:
                refreshed = "engineering_invalid"
            if refreshed == "scientific_failed":
                continue
            _transition(
                state,
                state_path,
                "engineering_invalid",
                failed_action=phase,
                return_code=return_code,
            )
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
