"""Fail-closed recovery audit for one interrupted RA-GLGM experiment arm."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint_recovery import validate_checkpoint  # noqa: E402
from src.ra_experiment_protocol import (  # noqa: E402
    RA_EXPERIMENT_PROTOCOL_SHA256,
    RA_STAGES,
    RA_VARIANTS,
    continuous_epochs,
    file_sha256,
    finite_number,
    read_json,
    read_jsonl,
    validate_runtime_identity,
)
from src.ra_learnability_probe import validate_learnability_report  # noqa: E402


STAGE_LIMITS = {"smoke": 2, "screen": 30, "formal": 100}
STANDARD_METRICS = ("map", "map50", "map75", "precision", "recall", "cuda_peak_mib")
FIXED_AMP_SCALE = 128.0
COMMON_GRADIENT_FIELDS = ("gradient_norm", "fdr_gradient_norm")


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_optimizer_evidence(
    path: str | Path,
    *,
    run_id: str,
    variant: str,
    stage: str,
    completed_epochs: int,
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
    public_nonzero_epochs: set[int] = set()
    fdr_nonzero_epochs: set[int] = set()
    ra_nonzero_epochs: set[int] = set()
    for attempt, row in enumerate(rows, 1):
        epoch = row.get("completed_epoch")
        if (
            row.get("optimizer_attempt") != attempt
            or not _positive_int(row.get("optimizer_attempt"))
            or not _positive_int(epoch)
            or int(epoch) > completed_epochs
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
            if float(value) > 0.0:
                destination = (
                    public_nonzero_epochs
                    if field == "gradient_norm"
                    else fdr_nonzero_epochs
                )
                destination.add(int(epoch))
        if variant == "ra_glgm":
            value = row.get("ra_glgm_gradient_norm")
            if not finite_number(value) or float(value) < 0.0:
                raise ValueError(
                    f"optimizer RA private gradient is non-finite at attempt {attempt}"
                )
            if float(value) > 0.0:
                ra_nonzero_epochs.add(int(epoch))
        observed_epochs.append(int(epoch))

    if observed_epochs != sorted(observed_epochs):
        raise ValueError("optimizer evidence epoch sequence is not monotonic")
    if set(observed_epochs) != set(range(1, completed_epochs + 1)):
        raise ValueError("optimizer evidence does not cover every completed epoch exactly")
    expected_epochs = set(range(1, completed_epochs + 1))
    if public_nonzero_epochs != expected_epochs:
        raise ValueError("optimizer evidence has no nonzero public gradient in every epoch")
    if fdr_nonzero_epochs != expected_epochs:
        raise ValueError("optimizer evidence has no nonzero FDR gradient in every epoch")
    if variant == "ra_glgm" and ra_nonzero_epochs != expected_epochs:
        raise ValueError("optimizer evidence has no nonzero RA private gradient in every epoch")
    return {
        "optimizer_attempts": len(rows),
        "optimizer_evidence_sha256": file_sha256(evidence_path),
        "amp_scale": FIXED_AMP_SCALE,
        "amp_skipped_steps": 0,
        "public_gradient_finite": True,
        "fdr_gradient_finite": True,
        "public_gradient_nonzero": True,
        "fdr_gradient_nonzero": True,
        "ra_private_gradient_nonzero": True if variant == "ra_glgm" else None,
    }


def _load_gate(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    from scripts.evaluate_ra_glgm_gate import validate_screen_gate_report

    root = path.resolve().parent
    gate = validate_screen_gate_report(
        path,
        baseline_run=root / "screen-seed0-baseline-ra-glgm-v1",
        method_run=root / "screen-seed0-ra_glgm-ra-glgm-v1",
    )
    return gate, file_sha256(path)


def validate_resume(
    run_dir: str | Path,
    *,
    variant: str,
    stage: str,
    protocol_manifest: str | Path,
    learnability_report: str | Path,
    screen_gate: str | Path | None = None,
) -> dict[str, Any]:
    """Return an audited resume decision; never search another experiment authority."""
    if variant not in RA_VARIANTS or stage not in RA_STAGES:
        raise ValueError("unknown RA recovery arm/stage")
    run = Path(run_dir).resolve()
    authority = read_json(protocol_manifest)
    if authority.get("protocol_sha256") != RA_EXPERIMENT_PROTOCOL_SHA256:
        raise ValueError("protocol authority is not the frozen RA-GLGM protocol")
    runtime = read_json(run / "ra-run.json")
    identity = validate_runtime_identity(runtime, variant=variant, stage=stage)
    expected_identity = authority.get("run_identities", {}).get(f"{variant}_{stage}")
    if identity != expected_identity:
        raise ValueError("runtime identity does not match the selected RA authority")
    if runtime.get("initial_state") != authority.get("initial_state"):
        raise ValueError("runtime paired initialization differs from RA authority")
    if runtime.get("source") != authority.get("source"):
        raise ValueError("runtime source differs from RA authority")
    if runtime.get("dataset_authority") != authority.get("dataset_authority"):
        raise ValueError("runtime dataset authority differs from RA authority")
    learnability_path = Path(learnability_report).resolve()
    validate_learnability_report(learnability_path, protocol_manifest=authority)
    learnability_sha = file_sha256(learnability_path)
    if runtime.get("learnability_report_sha256") != learnability_sha:
        raise ValueError("runtime learnability report differs from recovery authority")
    if str(runtime.get("gpu_uuid", "")) != str(authority.get("gpu_uuid", "")):
        raise ValueError("runtime physical GPU differs from paired authority")
    if int(runtime.get("schedule_epochs", -1)) != (
        50 if stage == "screen" else STAGE_LIMITS[stage]
    ):
        raise ValueError("runtime scheduler length differs from frozen protocol")
    if runtime.get("cutoff_epoch") != (30 if stage == "screen" else None):
        raise ValueError("runtime cutoff differs from frozen protocol")

    gate, gate_sha = _load_gate(Path(screen_gate).resolve() if screen_gate else None)
    if stage == "formal":
        if gate is None:
            raise ValueError("Formal100 recovery requires the passing Screen30 gate")
        if runtime.get("screen_gate_sha256") != gate_sha:
            raise ValueError("Formal100 runtime is not bound to the supplied Screen30 gate")
        if runtime.get("initialization_mode") != "fresh_paired_scratch":
            raise ValueError("Formal100 was not launched from fresh paired scratch initialization")
        if runtime.get("parent_checkpoint") is not None:
            raise ValueError("Formal100 may not inherit a Smoke or Screen checkpoint")

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
        "learnability_report_sha256": learnability_sha,
        "authority": "same-run exact-epoch only",
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
    )
    write_create_only(args.output, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
