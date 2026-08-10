"""Fail-closed recovery audit for one interrupted RA-GLGM experiment arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint_recovery import validate_checkpoint  # noqa: E402
from src.ra_experiment_protocol import (  # noqa: E402
    RA_STAGES,
    RA_VARIANTS,
    continuous_epochs,
    file_sha256,
    finite_number,
    load_ra_authority,
    read_json,
    read_jsonl,
    validate_runtime_identity,
)
from src.fdr_protocol import canonical_json_bytes  # noqa: E402
from src.ra_learnability_probe import validate_learnability_report  # noqa: E402


STAGE_LIMITS = {"smoke": 2, "screen10": 10, "screen": 30, "formal": 100}
STANDARD_METRICS = ("map", "map50", "map75", "precision", "recall", "cuda_peak_mib")
FIXED_AMP_SCALE = 128.0
COMMON_GRADIENT_FIELDS = ("gradient_norm", "fdr_gradient_norm")
RECOVERY_LINEAGE_FILENAME = "optimizer-recovery-lineage.jsonl"


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _optimizer_raw_lines(path: Path, expected_rows: int) -> tuple[bytes, list[bytes]]:
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    if (
        len(lines) != expected_rows
        or any(not line.strip() for line in lines)
        or any(not line.endswith((b"\n", b"\r")) for line in lines)
    ):
        raise ValueError("optimizer evidence must be one complete nonempty JSON row per line")
    return data, lines


def _validate_recovery_lineage(
    path: Path,
    *,
    optimizer_path: Path,
    optimizer_rows: list[dict[str, Any]],
    run_id: str,
    variant: str,
    stage: str,
    completed_epochs: int,
) -> dict[str, Any]:
    """Rebuild recovery generations and the discarded optimizer-attempt lineage."""

    _, raw_lines = _optimizer_raw_lines(optimizer_path, len(optimizer_rows))
    if not path.exists():
        records: list[dict[str, Any]] = []
    else:
        try:
            records = read_jsonl(path)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("optimizer recovery lineage is unreadable") from error

    discarded: set[int] = set()
    previous_event_sha = "0" * 64
    previous_prefix_attempts = 0
    for generation, record in enumerate(records, 1):
        payload = dict(record)
        recorded_event_sha = payload.pop("event_sha256", None)
        expected_event_sha = hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()
        if (
            record.get("generation") != generation
            or record.get("previous_event_sha256") != previous_event_sha
            or recorded_event_sha != expected_event_sha
            or record.get("run_id") != run_id
            or record.get("variant") != variant
            or record.get("stage") != stage
        ):
            raise ValueError(f"optimizer recovery lineage authority/hash failure at generation {generation}")
        prefix_attempts = record.get("optimizer_attempts_before")
        prefix_bytes = record.get("optimizer_evidence_bytes_before")
        recovery_epoch = record.get("completed_epoch")
        if (
            not _positive_int(prefix_attempts)
            or int(prefix_attempts) < previous_prefix_attempts
            or int(prefix_attempts) > len(optimizer_rows)
            or not _positive_int(prefix_bytes)
            or not _positive_int(recovery_epoch)
            or int(recovery_epoch) > completed_epochs
        ):
            raise ValueError(f"optimizer recovery lineage sequence is invalid at generation {generation}")
        prefix = b"".join(raw_lines[: int(prefix_attempts)])
        if (
            len(prefix) != int(prefix_bytes)
            or hashlib.sha256(prefix).hexdigest().upper()
            != str(record.get("optimizer_evidence_prefix_sha256", "")).upper()
        ):
            raise ValueError(f"optimizer evidence prefix drift at recovery generation {generation}")
        for attempt in range(previous_prefix_attempts + 1, int(prefix_attempts) + 1):
            if optimizer_rows[attempt - 1].get("recovery_generation") != generation - 1:
                raise ValueError(
                    f"optimizer attempt {attempt} is bound to the wrong recovery generation"
                )

        expected_discarded = [
            attempt
            for attempt in range(1, int(prefix_attempts) + 1)
            if attempt not in discarded
            and int(optimizer_rows[attempt - 1].get("completed_epoch", -1))
            > int(recovery_epoch)
        ]
        first = record.get("discarded_attempt_first")
        last = record.get("discarded_attempt_last")
        count = record.get("discarded_attempt_count")
        if expected_discarded:
            if (
                expected_discarded != list(range(expected_discarded[0], expected_discarded[-1] + 1))
                or first != expected_discarded[0]
                or last != expected_discarded[-1]
                or count != len(expected_discarded)
            ):
                raise ValueError(f"discarded optimizer lineage mismatch at generation {generation}")
            discarded.update(expected_discarded)
        elif first is not None or last is not None or count != 0:
            raise ValueError(f"empty discarded optimizer lineage mismatch at generation {generation}")

        checkpoint = Path(str(record.get("checkpoint", ""))).resolve()
        if (
            not checkpoint.is_file()
            or file_sha256(checkpoint) != str(record.get("checkpoint_sha256", "")).upper()
        ):
            raise ValueError(f"recovery checkpoint drift at generation {generation}")
        previous_event_sha = str(recorded_event_sha)
        previous_prefix_attempts = int(prefix_attempts)

    for attempt in range(previous_prefix_attempts + 1, len(optimizer_rows) + 1):
        if optimizer_rows[attempt - 1].get("recovery_generation") != len(records):
            raise ValueError(f"optimizer attempt {attempt} is bound to the wrong recovery generation")
    return {
        "recovery_generation": len(records),
        "discarded_optimizer_attempt_numbers": discarded,
        "discarded_optimizer_attempts": len(discarded),
        "recovery_lineage_sha256": file_sha256(path) if records else None,
        "recovery_lineage_records": records,
    }


def validate_optimizer_evidence(
    path: str | Path,
    *,
    run_id: str,
    variant: str,
    stage: str,
    completed_epochs: int,
    allow_trailing_uncommitted_epoch: bool = False,
    recovery_lineage_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate every optimizer attempt and return Smoke-grade evidence facts."""
    evidence_path = Path(path).resolve()
    try:
        rows = read_jsonl(evidence_path)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise ValueError(f"optimizer evidence is missing or unreadable: {evidence_path}") from error
    if not rows:
        raise ValueError("optimizer evidence is empty")

    observed_epochs: list[int] = []
    for attempt, row in enumerate(rows, 1):
        epoch = row.get("completed_epoch")
        if (
            row.get("optimizer_attempt") != attempt
            or not _positive_int(row.get("optimizer_attempt"))
            or not _positive_int(epoch)
            or int(epoch)
            > completed_epochs + (1 if allow_trailing_uncommitted_epoch else 0)
        ):
            raise ValueError(f"optimizer evidence sequence is invalid at attempt {attempt}")
        if (
            row.get("run_id") != run_id
            or row.get("variant") != variant
            or row.get("stage") != stage
        ):
            raise ValueError(f"optimizer evidence authority is invalid at attempt {attempt}")
        if (
            row.get("amp_scale_before") != FIXED_AMP_SCALE
            or row.get("amp_scale_after") != FIXED_AMP_SCALE
            or row.get("amp_step_skipped") is not False
            or row.get("gradient_norm_finite") is not True
        ):
            raise ValueError(f"optimizer AMP/gradient evidence is invalid at attempt {attempt}")
        for field in COMMON_GRADIENT_FIELDS:
            value = row.get(field)
            if not finite_number(value) or float(value) < 0.0:
                raise ValueError(
                    f"optimizer {field} is non-finite or invalid at attempt {attempt}"
                )
        if variant == "ra_glgm":
            value = row.get("ra_glgm_gradient_norm")
            if not finite_number(value) or float(value) < 0.0:
                raise ValueError(
                    f"optimizer RA private gradient is non-finite at attempt {attempt}"
                )
        observed_epochs.append(int(epoch))

    if observed_epochs != sorted(observed_epochs):
        raise ValueError("optimizer evidence epoch sequence is not monotonic")
    lineage_path = (
        Path(recovery_lineage_path).resolve()
        if recovery_lineage_path is not None
        else evidence_path.with_name(RECOVERY_LINEAGE_FILENAME)
    )
    lineage = _validate_recovery_lineage(
        lineage_path,
        optimizer_path=evidence_path,
        optimizer_rows=rows,
        run_id=run_id,
        variant=variant,
        stage=stage,
        completed_epochs=completed_epochs,
    )
    discarded = lineage["discarded_optimizer_attempt_numbers"]
    active_attempts = [
        (attempt, epoch)
        for attempt, epoch in enumerate(observed_epochs, 1)
        if attempt not in discarded
    ]
    committed_observed = {
        epoch for _, epoch in active_attempts if epoch <= completed_epochs
    }
    if committed_observed != set(range(1, completed_epochs + 1)):
        raise ValueError("optimizer evidence does not cover every completed epoch exactly")
    expected_epochs = set(range(1, completed_epochs + 1))
    active_public_nonzero = {
        observed_epochs[attempt - 1]
        for attempt, row in enumerate(rows, 1)
        if attempt not in discarded and float(row["gradient_norm"]) > 0.0
    }
    active_fdr_nonzero = {
        observed_epochs[attempt - 1]
        for attempt, row in enumerate(rows, 1)
        if attempt not in discarded and float(row["fdr_gradient_norm"]) > 0.0
    }
    active_ra_nonzero = {
        observed_epochs[attempt - 1]
        for attempt, row in enumerate(rows, 1)
        if attempt not in discarded
        and variant == "ra_glgm"
        and float(row["ra_glgm_gradient_norm"]) > 0.0
    }
    if active_public_nonzero & expected_epochs != expected_epochs:
        raise ValueError("optimizer evidence has no nonzero public gradient in every epoch")
    if active_fdr_nonzero & expected_epochs != expected_epochs:
        raise ValueError("optimizer evidence has no nonzero FDR gradient in every epoch")
    if variant == "ra_glgm" and active_ra_nonzero & expected_epochs != expected_epochs:
        raise ValueError("optimizer evidence has no nonzero RA private gradient in every epoch")
    trailing_attempts = [
        attempt for attempt, epoch in active_attempts if epoch > completed_epochs
    ]
    return {
        "optimizer_attempts": len(rows),
        "active_optimizer_attempts": len(rows) - len(discarded),
        "trailing_uncommitted_optimizer_attempts": len(trailing_attempts),
        "trailing_uncommitted_attempt_first": trailing_attempts[0] if trailing_attempts else None,
        "trailing_uncommitted_attempt_last": trailing_attempts[-1] if trailing_attempts else None,
        "optimizer_evidence_sha256": file_sha256(evidence_path),
        "amp_scale": FIXED_AMP_SCALE,
        "amp_skipped_steps": 0,
        "public_gradient_finite": True,
        "fdr_gradient_finite": True,
        "public_gradient_nonzero": True,
        "fdr_gradient_nonzero": True,
        "ra_private_gradient_nonzero": True if variant == "ra_glgm" else None,
        **{key: value for key, value in lineage.items() if key != "discarded_optimizer_attempt_numbers" and key != "recovery_lineage_records"},
    }


def record_optimizer_recovery_generation(
    run_dir: str | Path, decision: Mapping[str, Any]
) -> dict[str, Any]:
    """Persist one recovery boundary before launching from an audited checkpoint."""

    if decision.get("decision") != "resume":
        raise ValueError("optimizer recovery generation requires a resume decision")
    run = Path(run_dir).resolve()
    optimizer_path = run / "optimizer-evidence.jsonl"
    rows = read_jsonl(optimizer_path)
    data, _ = _optimizer_raw_lines(optimizer_path, len(rows))
    lineage_path = run / RECOVERY_LINEAGE_FILENAME
    existing = read_jsonl(lineage_path) if lineage_path.exists() else []
    expected_lineage_sha = decision.get("recovery_lineage_sha256")
    actual_lineage_sha = file_sha256(lineage_path) if existing else None
    if (
        decision.get("recovery_generation") != len(existing)
        or expected_lineage_sha != actual_lineage_sha
        or decision.get("optimizer_evidence_sha256") != file_sha256(optimizer_path)
    ):
        raise ValueError("resume decision became stale before recovery generation recording")
    generation = len(existing) + 1
    first = decision.get("trailing_uncommitted_attempt_first")
    last = decision.get("trailing_uncommitted_attempt_last")
    count = int(decision.get("trailing_uncommitted_optimizer_attempts", -1))
    if count < 0 or (count == 0) != (first is None and last is None):
        raise ValueError("resume decision has invalid discarded optimizer attempt range")
    if count and (not _positive_int(first) or not _positive_int(last) or int(last) - int(first) + 1 != count):
        raise ValueError("resume decision discarded optimizer attempt range is not contiguous")
    checkpoint = Path(str(decision.get("checkpoint", ""))).resolve()
    if (
        not checkpoint.is_file()
        or file_sha256(checkpoint) != str(decision.get("checkpoint_sha256", "")).upper()
    ):
        raise ValueError("resume checkpoint changed before recovery generation was recorded")
    payload: dict[str, Any] = {
        "generation": generation,
        "previous_event_sha256": existing[-1]["event_sha256"] if existing else "0" * 64,
        "run_id": decision["run_id"],
        "variant": decision["variant"],
        "stage": decision["stage"],
        "completed_epoch": decision["completed_epoch"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": decision["checkpoint_sha256"],
        "optimizer_attempts_before": len(rows),
        "optimizer_evidence_bytes_before": len(data),
        "optimizer_evidence_prefix_sha256": hashlib.sha256(data).hexdigest().upper(),
        "discarded_attempt_first": first,
        "discarded_attempt_last": last,
        "discarded_attempt_count": count,
    }
    payload["event_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()
    if lineage_path.is_symlink():
        raise ValueError("optimizer recovery lineage may not be a symlink")
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lineage_path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "ab", closefd=False) as stream:
            stream.write(
                json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
                + b"\n"
            )
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def _load_gate(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    from scripts.evaluate_ra_glgm_gate import validate_screen_gate_report

    root = path.resolve().parent
    gate = validate_screen_gate_report(
        path,
        baseline_run=root / "screen-seed0-baseline-ra-glgm-v1.1",
        method_run=root / "screen-seed0-ra_glgm-ra-glgm-v1.1",
    )
    return gate, file_sha256(path)


def _load_screen10_gate(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    from scripts.evaluate_ra_glgm_gate import validate_screen10_gate_report

    root = path.resolve().parent
    gate = validate_screen10_gate_report(
        path,
        baseline_run=root / "screen10-seed0-baseline-ra-glgm-v1.1",
        method_run=root / "screen10-seed0-ra_glgm-ra-glgm-v1.1",
    )
    return gate, file_sha256(path)


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".reconcile.tmp")
    temporary.write_text(
        "".join(
            json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _quarantine_checkpoint(run: Path, checkpoint: Path) -> str:
    """Move one provably uncommitted checkpoint out of the authoritative weights set."""

    digest = file_sha256(checkpoint)
    quarantine = run / "recovery-quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    destination = quarantine / f"{checkpoint.stem}.{digest[:16]}.uncommitted.pt"
    if destination.exists():
        if file_sha256(destination) != digest:
            raise ValueError("checkpoint quarantine contains contradictory bytes")
        checkpoint.unlink()
    else:
        os.replace(checkpoint, destination)
    return str(destination)


def reconcile_epoch_artifacts(
    run: Path,
    *,
    run_id: str,
    variant: str,
    stage: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Repair only a single interrupted tail transaction; reject contradictory history."""

    evidence_path = run / "ra-epochs.jsonl"
    queue_path = run / "publication-queue.jsonl"
    evidence = read_jsonl(evidence_path) if evidence_path.exists() else []
    queue = read_jsonl(queue_path) if queue_path.exists() else []
    repairs: list[dict[str, Any]] = []

    def sequential(rows: list[dict[str, Any]], name: str) -> None:
        epochs = [int(row.get("completed_epoch", -1)) for row in rows]
        if epochs != list(range(1, len(rows) + 1)) or len(rows) > limit:
            raise ValueError(f"{name} is not a continuous in-range epoch prefix")

    sequential(evidence, "epoch evidence")
    sequential(queue, "publication queue")
    if abs(len(evidence) - len(queue)) > 1:
        raise ValueError("epoch evidence/publication queue differ by more than one tail record")
    for epoch in range(1, min(len(evidence), len(queue)) + 1):
        evidence_row = evidence[epoch - 1]
        queue_row = queue[epoch - 1]
        checkpoint = (run / "weights" / f"epoch{epoch - 1}.pt").resolve()
        if (
            evidence_row.get("run_id") != run_id
            or evidence_row.get("variant") != variant
            or evidence_row.get("stage") != stage
            or queue_row.get("run_id") != run_id
            or queue_row.get("variant") != variant
            or queue_row.get("stage") != stage
            or queue_row.get("status") != "pending"
            or Path(str(queue_row.get("checkpoint", ""))).resolve() != checkpoint
            or not checkpoint.is_file()
            or str(queue_row.get("checkpoint_sha256", "")).upper()
            != file_sha256(checkpoint)
        ):
            raise ValueError(f"checkpoint/queue contradictory committed artifact prefix at epoch {epoch}")

    if len(evidence) == len(queue) + 1:
        epoch = len(evidence)
        row = evidence[-1]
        checkpoint = (run / "weights" / f"epoch{epoch - 1}.pt").resolve()
        if (
            row.get("run_id") != run_id
            or row.get("variant") != variant
            or row.get("stage") != stage
        ):
            raise ValueError("unpaired epoch evidence tail has foreign authority")
        if checkpoint.is_file():
            valid, reason = validate_checkpoint(checkpoint)
            if not valid or int(reason.split("=", 1)[1]) != epoch - 1:
                raise ValueError("unpaired epoch evidence tail has a contradictory checkpoint")
            queue.append(
                {
                    "run_id": run_id,
                    "variant": variant,
                    "stage": stage,
                    "completed_epoch": epoch,
                    "status": "pending",
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "artifacts": [
                        str(evidence_path),
                        str(run / "ra-epochs.csv"),
                        str(run / "ra-run.json"),
                    ],
                }
            )
            _atomic_jsonl(queue_path, queue)
            repairs.append({"action": "backfill_queue", "completed_epoch": epoch})
        else:
            evidence.pop()
            _atomic_jsonl(evidence_path, evidence)
            repairs.append({"action": "rollback_evidence", "completed_epoch": epoch})
    elif len(queue) == len(evidence) + 1:
        epoch = len(queue)
        row = queue[-1]
        checkpoint = (run / "weights" / f"epoch{epoch - 1}.pt").resolve()
        if (
            row.get("run_id") != run_id
            or row.get("variant") != variant
            or row.get("stage") != stage
            or Path(str(row.get("checkpoint", ""))).resolve() != checkpoint
            or not checkpoint.is_file()
            or str(row.get("checkpoint_sha256", "")).upper() != file_sha256(checkpoint)
        ):
            raise ValueError("unpaired publication queue tail is contradictory")
        queue.pop()
        _atomic_jsonl(queue_path, queue)
        moved = _quarantine_checkpoint(run, checkpoint)
        repairs.append(
            {"action": "rollback_queue_and_quarantine_checkpoint", "completed_epoch": epoch, "moved_to": moved}
        )

    completed = len(evidence)
    extras = sorted(
        path
        for path in (run / "weights").glob("epoch*.pt")
        if path.name not in {f"epoch{epoch}.pt" for epoch in range(completed)}
    )
    if extras:
        permitted = run / "weights" / f"epoch{completed}.pt"
        if len(extras) != 1 or extras[0].resolve() != permitted.resolve():
            raise ValueError("checkpoint directory contains contradictory future snapshots")
        moved = _quarantine_checkpoint(run, extras[0])
        repairs.append(
            {"action": "quarantine_uncommitted_checkpoint", "completed_epoch": completed + 1, "moved_to": moved}
        )
    return repairs


def validate_resume(
    run_dir: str | Path,
    *,
    variant: str,
    stage: str,
    protocol_manifest: str | Path,
    learnability_report: str | Path,
    screen_gate: str | Path | None = None,
    screen10_gate: str | Path | None = None,
) -> dict[str, Any]:
    """Return an audited resume decision; never search another experiment authority."""
    if variant not in RA_VARIANTS or stage not in RA_STAGES:
        raise ValueError("unknown RA recovery arm/stage")
    run = Path(run_dir).resolve()
    authority = load_ra_authority(protocol_manifest, repository_root=ROOT)
    runtime = read_json(run / "ra-run.json")
    identity = validate_runtime_identity(runtime, variant=variant, stage=stage)
    expected_identity = authority.get("run_identities", {}).get(f"{variant}_{stage}")
    if identity != expected_identity:
        raise ValueError("runtime identity does not match the selected RA authority")
    if runtime.get("protocol_sha256") != authority.get("protocol_sha256"):
        raise ValueError("runtime protocol differs from RA authority")
    if runtime.get("initial_state") != authority.get("initial_state"):
        raise ValueError("runtime paired initialization differs from RA authority")
    if runtime.get("source") != authority.get("source"):
        raise ValueError("runtime source differs from RA authority")
    if runtime.get("dataset_authority") != authority.get("dataset_authority"):
        raise ValueError("runtime dataset authority differs from RA authority")
    if runtime.get("locked_evaluator_sha256") != authority.get("locked_evaluator", {}).get(
        "sha256"
    ):
        raise ValueError("runtime locked evaluator differs from RA authority")
    if runtime.get("initialization_mode") != "fresh_paired_scratch":
        raise ValueError("runtime was not launched from fresh paired scratch initialization")
    if runtime.get("parent_checkpoint") is not None:
        raise ValueError("runtime illegally inherits a parent checkpoint")
    learnability_path = Path(learnability_report).resolve()
    validate_learnability_report(learnability_path, protocol_manifest=authority)
    learnability_sha = file_sha256(learnability_path)
    if runtime.get("learnability_report_sha256") != learnability_sha:
        raise ValueError("runtime learnability report differs from recovery authority")
    if str(runtime.get("gpu_uuid", "")) != str(authority.get("gpu_uuid", "")):
        raise ValueError("runtime physical GPU differs from paired authority")
    if int(runtime.get("schedule_epochs", -1)) != (
        50 if stage in {"screen10", "screen"} else STAGE_LIMITS[stage]
    ):
        raise ValueError("runtime scheduler length differs from frozen protocol")
    expected_cutoff = 10 if stage == "screen10" else 30 if stage == "screen" else None
    if runtime.get("cutoff_epoch") != expected_cutoff:
        raise ValueError("runtime cutoff differs from frozen protocol")

    gate, gate_sha = _load_gate(Path(screen_gate).resolve() if screen_gate else None)
    screen10, screen10_sha = _load_screen10_gate(
        Path(screen10_gate).resolve() if screen10_gate else None
    )
    if stage == "screen":
        if screen10 is None:
            raise ValueError("Screen30 recovery requires the passing Screen10 gate")
        if runtime.get("screen10_gate_sha256") != screen10_sha:
            raise ValueError("Screen30 runtime is not bound to the supplied Screen10 gate")
        if runtime.get("initialization_mode") != "fresh_paired_scratch":
            raise ValueError("Screen30 was not launched from fresh paired scratch initialization")
        if runtime.get("parent_checkpoint") is not None:
            raise ValueError("Screen30 may not inherit a Smoke or Screen10 checkpoint")
        if runtime.get("screen_gate_sha256") is not None:
            raise ValueError("Screen30 may not inherit a Screen30 gate")
    if stage == "formal":
        if gate is None:
            raise ValueError("Formal100 recovery requires the passing Screen30 gate")
        if runtime.get("screen_gate_sha256") != gate_sha:
            raise ValueError("Formal100 runtime is not bound to the supplied Screen30 gate")
        if runtime.get("initialization_mode") != "fresh_paired_scratch":
            raise ValueError("Formal100 was not launched from fresh paired scratch initialization")
        if runtime.get("parent_checkpoint") is not None:
            raise ValueError("Formal100 may not inherit a Smoke or Screen checkpoint")
        if runtime.get("screen10_gate_sha256") is not None:
            raise ValueError("Formal100 must be bound only to the Screen30 gate")
    if stage in {"smoke", "screen10"} and (
        runtime.get("screen_gate_sha256") is not None
        or runtime.get("screen10_gate_sha256") is not None
    ):
        raise ValueError(f"{stage} may not inherit an upstream gate")

    reconciliation = reconcile_epoch_artifacts(
        run,
        run_id=str(identity["run_id"]),
        variant=variant,
        stage=stage,
        limit=STAGE_LIMITS[stage],
    )
    evidence = read_jsonl(run / "ra-epochs.jsonl")
    completed = len(evidence)
    limit = STAGE_LIMITS[stage]
    if completed == 0:
        raise ValueError("run has no completed epoch and no auditable recovery point")
    if completed > limit or not continuous_epochs(evidence, completed):
        raise ValueError(f"epoch evidence is not continuous 1..{completed}")
    for epoch, row in enumerate(evidence, 1):
        if (
            row.get("run_id") != identity["run_id"]
            or row.get("variant") != variant
            or row.get("stage") != stage
            or not all(finite_number(row.get(name)) for name in STANDARD_METRICS)
        ):
            raise ValueError(f"foreign or non-finite epoch evidence at epoch {epoch}")

    optimizer = validate_optimizer_evidence(
        run / "optimizer-evidence.jsonl",
        run_id=identity["run_id"],
        variant=variant,
        stage=stage,
        completed_epochs=completed,
        allow_trailing_uncommitted_epoch=completed < limit,
        recovery_lineage_path=run / RECOVERY_LINEAGE_FILENAME,
    )

    queue = read_jsonl(run / "publication-queue.jsonl")
    if len(queue) != completed:
        raise ValueError("publication queue count differs from completed epoch count")
    for epoch, row in enumerate(queue, 1):
        checkpoint = (run / "weights" / f"epoch{epoch - 1}.pt").resolve()
        if (
            row.get("run_id") != identity["run_id"]
            or int(row.get("completed_epoch", -1)) != epoch
            or row.get("status") != "pending"
            or Path(str(row.get("checkpoint", ""))).resolve() != checkpoint
            or not checkpoint.is_file()
            or str(row.get("checkpoint_sha256", "")).upper() != file_sha256(checkpoint)
        ):
            raise ValueError(f"checkpoint/queue authority failure at epoch {epoch}")

    latest = (run / "weights" / f"epoch{completed - 1}.pt").resolve()
    valid, reason = validate_checkpoint(latest)
    if not valid:
        raise ValueError(f"latest exact epoch checkpoint is not resumable: {reason}")
    checkpoint_epoch = int(reason.split("=", 1)[1])
    if checkpoint_epoch != completed - 1:
        raise ValueError("checkpoint internal epoch differs from evidence sequence")
    extra = [
        path.name
        for path in (run / "weights").glob("epoch*.pt")
        if path.name not in {f"epoch{epoch}.pt" for epoch in range(completed)}
    ]
    if extra:
        raise ValueError(f"checkpoint directory contains uncommitted future snapshots: {extra[:3]}")
    return {
        "format_version": 1,
        "decision": "resume" if completed < limit else "complete",
        "variant": variant,
        "stage": stage,
        "run_id": identity["run_id"],
        "completed_epoch": completed,
        "target_epoch": limit,
        "checkpoint": str(latest),
        "checkpoint_sha256": file_sha256(latest),
        "screen_gate_sha256": gate_sha,
        "screen10_gate_sha256": screen10_sha,
        "learnability_report_sha256": learnability_sha,
        "authority": "same-run exact-epoch only",
        "artifact_reconciliation": reconciliation,
        **optimizer,
    }


def write_create_only(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=RA_VARIANTS, required=True)
    parser.add_argument("--stage", choices=RA_STAGES, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--learnability-report", type=Path, required=True)
    parser.add_argument("--screen-gate", type=Path)
    parser.add_argument("--screen10-gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    decision = validate_resume(
        args.run_dir,
        variant=args.variant,
        stage=args.stage,
        protocol_manifest=args.protocol_manifest,
        learnability_report=args.learnability_report,
        screen_gate=args.screen_gate,
        screen10_gate=args.screen10_gate,
    )
    write_create_only(args.output, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
