"""Fail-closed audit of one FDR+BPDD+RA-GLGM run and rolling resume point."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint_recovery import validate_checkpoint  # noqa: E402
from src.fdr_bpdd_ra_glgm_protocol import (  # noqa: E402
    COMBO_PARAMETERS,
    COMBO_PROTOCOL_SHA256,
    COMBO_VARIANT,
    load_combo_authority,
)
from src.ra_experiment_protocol import file_sha256, read_json, read_jsonl  # noqa: E402


STAGE_LIMITS = {"smoke": 2, "formal": 100}
MILESTONE_PERIOD = {"smoke": 1, "formal": 5}
FINITE_FIELDS = (
    "precision",
    "recall",
    "map50",
    "map",
    "map75",
    "loss_bpdd",
    "loss_ra_support",
    "loss_ra_scale",
    "bpdd_active_edge_ratio",
    "bpdd_mean_reliability",
    "bpdd_matched_queries",
    "bpdd_eligible_edges",
    "gradient_norm",
    "fdr_gradient_norm",
    "ra_glgm_gradient_norm",
    "learning_rate",
    "amp_scale",
    "cuda_peak_mib",
    "epoch_wall_seconds",
)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _checkpoint_state(source: Any) -> Mapping[str, torch.Tensor]:
    if callable(getattr(source, "state_dict", None)):
        state = source.state_dict()
    elif isinstance(source, Mapping):
        state = source
    else:
        raise TypeError("combo checkpoint model does not expose a state dict")
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("combo checkpoint state is invalid")
    return state


def _validate_checkpoint_contract(path: Path, expected_epoch: int) -> None:
    valid, reason = validate_checkpoint(path)
    if not valid or reason != f"epoch={expected_epoch}":
        raise ValueError(f"combo checkpoint epoch/optimizer contract failed: {reason}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    model = artifact.get("ema") if artifact.get("ema") is not None else artifact.get("model")
    state = _checkpoint_state(model)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("combo checkpoint must preserve the strict model object")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != COMBO_PARAMETERS:
        raise ValueError(f"combo checkpoint parameter count differs: {parameters}")
    if not isinstance(artifact.get("optimizer"), Mapping):
        raise ValueError("combo checkpoint optimizer state is invalid")
    if not isinstance(artifact.get("scaler"), Mapping):
        raise ValueError("combo checkpoint AMP scaler state is invalid")
    ra_names = [name for name in state if ".ra_glgm." in name]
    if not ra_names:
        raise ValueError("combo checkpoint has no RA private state")
    # BPDD is parameter-free; accepting a BPDD-named tensor would silently change deployment.
    if any("bpdd" in name.lower() for name in state):
        raise ValueError("BPDD unexpectedly entered combo checkpoint state")


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
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


def _rewrite_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("combo recovery cannot rewrite an empty committed prefix")
    fields = tuple(rows[0])
    temporary = path.with_suffix(path.suffix + ".reconcile.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: row.get(name) for name in fields} for row in rows)
    os.replace(temporary, path)


def _quarantine_future_milestones(run: Path, *, completed: int, period: int) -> None:
    expected = {f"epoch{epoch - 1}.pt" for epoch in range(period, completed + 1, period)}
    extras = [
        path
        for path in (run / "weights").glob("epoch*.pt")
        if path.name not in expected
    ]
    for checkpoint in extras:
        valid, reason = validate_checkpoint(checkpoint)
        if not valid:
            raise ValueError(f"unexpected unreadable combo checkpoint: {checkpoint}: {reason}")
        internal_epoch = int(reason.split("=", 1)[1]) + 1
        if internal_epoch != completed + 1 or internal_epoch % period != 0:
            raise ValueError(f"contradictory future combo checkpoint: {checkpoint}")
        digest = file_sha256(checkpoint)
        quarantine = run / "recovery-quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        destination = quarantine / f"{checkpoint.stem}.{digest[:16]}.uncommitted.pt"
        if destination.exists() and file_sha256(destination) != digest:
            raise ValueError("combo recovery quarantine contains contradictory bytes")
        if destination.exists():
            checkpoint.unlink()
        else:
            os.replace(checkpoint, destination)


def validate_combo_run(
    run_dir: str | Path,
    *,
    stage: str,
    authority_path: str | Path,
) -> dict[str, Any]:
    if stage not in STAGE_LIMITS:
        raise ValueError(f"unknown combo recovery stage: {stage}")
    run = Path(run_dir).resolve()
    authority = load_combo_authority(authority_path, repository_root=ROOT)
    runtime = read_json(run / "combo-run.json")
    identity = authority["run_identities"][stage]
    expected_runtime = {
        "format_version": 1,
        "protocol_sha256": COMBO_PROTOCOL_SHA256,
        "source": authority["source"],
        "run_identity": identity,
        "initial_state": authority["initial_state"],
        "dataset_authority": authority["dataset_authority"],
        "gpu_uuid": authority["gpu_uuid"],
        "schedule_epochs": STAGE_LIMITS[stage],
        "model_parameters": COMBO_PARAMETERS,
        "bpdd_parameters": 0,
        "initialization_mode": "fresh_seed0_scratch",
        "parent_checkpoint": None,
        "milestone_period": MILESTONE_PERIOD[stage],
        "rolling_last_every_epoch": True,
    }
    for name, expected in expected_runtime.items():
        if runtime.get(name) != expected:
            raise ValueError(f"combo runtime authority differs at {name}")
    if not isinstance(runtime.get("data"), str) or not runtime["data"]:
        raise ValueError("combo runtime data YAML is missing")

    evidence_path = run / "combo-epochs.jsonl"
    queue_path = run / "local-checkpoint-queue.jsonl"
    evidence = read_jsonl(evidence_path)
    queue = read_jsonl(queue_path)
    if not evidence or abs(len(evidence) - len(queue)) > 1:
        raise ValueError("combo evidence/queue transaction is not recoverable")
    if len(evidence) == len(queue) + 1:
        evidence = evidence[:-1]
        if not evidence:
            raise ValueError("combo has no committed epoch after tail rollback")
        _atomic_jsonl(evidence_path, evidence)
        _rewrite_csv(run / "combo-epochs.csv", evidence)
    elif len(queue) == len(evidence) + 1:
        queue = queue[:-1]
        _atomic_jsonl(queue_path, queue)
    if len(evidence) != len(queue):
        raise ValueError("combo evidence/queue reconciliation failed")
    completed = len(evidence)
    if completed > STAGE_LIMITS[stage]:
        raise ValueError("combo completed epochs exceed the frozen stage")
    for epoch, (row, queued) in enumerate(zip(evidence, queue, strict=True), 1):
        if (
            row.get("completed_epoch") != epoch
            or row.get("stage") != stage
            or row.get("run_id") != identity["run_id"]
            or queued.get("completed_epoch") != epoch
            or queued.get("stage") != stage
            or queued.get("variant") != COMBO_VARIANT
            or queued.get("run_id") != identity["run_id"]
            or queued.get("status") != "local-only"
            or any(not _finite(row.get(name)) for name in FINITE_FIELDS)
            or row.get("amp_scale") != 128.0
            or row.get("amp_skipped_steps") != 0
        ):
            raise ValueError(f"invalid combo evidence/queue at epoch {epoch}")
        milestone = epoch % MILESTONE_PERIOD[stage] == 0
        checkpoint = run / "weights" / f"epoch{epoch - 1}.pt"
        if milestone:
            digest = file_sha256(checkpoint) if checkpoint.is_file() else None
            if (
                str(row.get("milestone_checkpoint", "")) != str(checkpoint)
                or row.get("milestone_checkpoint_sha256") != digest
                or str(queued.get("milestone_checkpoint", "")) != str(checkpoint)
                or queued.get("milestone_checkpoint_sha256") != digest
            ):
                raise ValueError(f"combo milestone checkpoint differs at epoch {epoch}")
        elif (
            row.get("milestone_checkpoint") is not None
            or row.get("milestone_checkpoint_sha256") is not None
            or queued.get("milestone_checkpoint") is not None
            or queued.get("milestone_checkpoint_sha256") is not None
            or checkpoint.exists()
        ):
            raise ValueError(f"unexpected non-milestone checkpoint at epoch {epoch}")

    rolling_path = run / "rolling-checkpoint.json"
    rolling = read_json(rolling_path)
    committed = (run / "weights" / "committed.pt").resolve()
    valid_committed, committed_reason = validate_checkpoint(committed)
    if not valid_committed:
        raise ValueError(f"combo committed checkpoint is invalid: {committed_reason}")
    committed_epoch = int(committed_reason.split("=", 1)[1]) + 1
    if completed > committed_epoch:
        evidence = evidence[:committed_epoch]
        queue = queue[:committed_epoch]
        completed = committed_epoch
        _atomic_jsonl(evidence_path, evidence)
        _atomic_jsonl(queue_path, queue)
        _rewrite_csv(run / "combo-epochs.csv", evidence)
    elif completed < committed_epoch:
        raise ValueError("combo committed checkpoint is ahead of lightweight evidence")
    _quarantine_future_milestones(
        run,
        completed=completed,
        period=MILESTONE_PERIOD[stage],
    )
    committed_sha = file_sha256(committed)
    if (
        rolling.get("completed_epoch") != completed
        or Path(str(rolling.get("checkpoint", ""))).resolve() != committed
        or rolling.get("sha256") != committed_sha
    ):
        rolling = {
            "run_id": identity["run_id"],
            "completed_epoch": completed,
            "checkpoint": str(committed),
            "sha256": committed_sha,
            "bytes": committed.stat().st_size,
        }
        temporary = rolling_path.with_suffix(".json.reconcile.tmp")
        temporary.write_text(json.dumps(rolling, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, rolling_path)
    if (
        rolling.get("run_id") != identity["run_id"]
        or rolling.get("completed_epoch") != completed
        or Path(str(rolling.get("checkpoint", ""))).resolve() != committed
        or not committed.is_file()
        or rolling.get("sha256") != committed_sha
        or evidence[-1].get("last_checkpoint_sha256") != rolling.get("sha256")
        or evidence[-1].get("last_checkpoint_bytes") != committed.stat().st_size
    ):
        raise ValueError("combo rolling checkpoint authority differs from committed evidence")
    _validate_checkpoint_contract(committed, completed - 1)

    optimizer = read_jsonl(run / "optimizer-evidence.jsonl")
    if not optimizer:
        raise ValueError("combo optimizer evidence is missing")
    observed_epochs: set[int] = set()
    for attempt, row in enumerate(optimizer, 1):
        epoch = row.get("completed_epoch")
        if (
            row.get("optimizer_attempt") != attempt
            or row.get("run_id") != identity["run_id"]
            or row.get("variant") != COMBO_VARIANT
            or row.get("stage") != stage
            or not isinstance(epoch, int)
            or epoch < 1
            or epoch > completed + (1 if completed < STAGE_LIMITS[stage] else 0)
            or row.get("amp_scale_before") != 128.0
            or row.get("amp_scale_after") != 128.0
            or row.get("amp_step_skipped") is not False
            or row.get("gradient_norm_finite") is not True
            or any(
                not _finite(row.get(name)) or float(row[name]) <= 0.0
                for name in ("gradient_norm", "fdr_gradient_norm", "ra_glgm_gradient_norm")
            )
        ):
            raise ValueError(f"invalid combo optimizer evidence at attempt {attempt}")
        if epoch <= completed:
            observed_epochs.add(epoch)
    if observed_epochs != set(range(1, completed + 1)):
        raise ValueError("combo optimizer evidence does not cover each committed epoch")
    return {
        "format_version": 1,
        "decision": "complete" if completed == STAGE_LIMITS[stage] else "resume",
        "stage": stage,
        "variant": COMBO_VARIANT,
        "run_id": identity["run_id"],
        "completed_epoch": completed,
        "target_epoch": STAGE_LIMITS[stage],
        "checkpoint": str(committed),
        "checkpoint_sha256": committed_sha,
        "milestone_checkpoints": completed // MILESTONE_PERIOD[stage],
        "optimizer_attempts": len(optimizer),
        "amp_skipped_steps": 0,
        "authority": "same combination run rolling last only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_LIMITS), required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_combo_run(args.run_dir, stage=args.stage, authority_path=args.authority)
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
