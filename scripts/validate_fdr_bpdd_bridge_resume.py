"""Fail-closed audit of the B-arm run and its only legal resume checkpoint."""

from __future__ import annotations

import argparse
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
from src.fdr_bpdd_bridge_protocol import (  # noqa: E402
    BRIDGE_PROTOCOL_SHA256,
    BRIDGE_VARIANT,
    load_bridge_authority,
)
from src.ra_experiment_protocol import (  # noqa: E402
    BASELINE_PARAMETERS,
    file_sha256,
    read_json,
    read_jsonl,
)


STAGE_LIMITS = {"smoke": 2, "formal": 100}
MILESTONE_PERIOD = {"smoke": 1, "formal": 5}
FINITE_FIELDS = (
    "precision",
    "recall",
    "map50",
    "map",
    "map75",
    "loss_bpdd",
    "bpdd_active_edge_ratio",
    "bpdd_mean_reliability",
    "bpdd_matched_queries",
    "bpdd_eligible_edges",
    "gradient_norm",
    "fdr_gradient_norm",
    "learning_rate",
    "amp_scale",
    "cuda_peak_mib",
    "epoch_wall_seconds",
)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _checkpoint_state(source: Any) -> Mapping[str, torch.Tensor]:
    if callable(getattr(source, "state_dict", None)):
        state = source.state_dict()
    elif isinstance(source, Mapping):
        state = source
    else:
        raise TypeError("bridge checkpoint model does not expose a state dict")
    if not state or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise TypeError("bridge checkpoint state is invalid")
    return state


def _validate_checkpoint_contract(path: Path, expected_epoch: int) -> None:
    valid, reason = validate_checkpoint(path)
    if not valid or reason != f"epoch={expected_epoch}":
        raise ValueError(f"bridge checkpoint epoch/optimizer contract failed: {reason}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    model = (
        artifact.get("ema")
        if artifact.get("ema") is not None
        else artifact.get("model")
    )
    state = _checkpoint_state(model)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("bridge checkpoint must preserve the strict model object")
    if (
        sum(parameter.numel() for parameter in model.parameters())
        != BASELINE_PARAMETERS
    ):
        raise ValueError("bridge checkpoint parameter count differs")
    if not isinstance(artifact.get("optimizer"), Mapping):
        raise ValueError("bridge checkpoint optimizer state is invalid")
    if not isinstance(artifact.get("scaler"), Mapping):
        raise ValueError("bridge checkpoint AMP scaler state is invalid")
    if any("ra_glgm" in name or "bpdd" in name.lower() for name in state):
        raise ValueError("bridge checkpoint contains forbidden RA/BPDD state")


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".reconcile.tmp")
    temporary.write_text(
        "".join(
            json.dumps(
                dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_bridge_run(
    run_dir: str | Path,
    *,
    stage: str,
    authority_path: str | Path,
) -> dict[str, Any]:
    if stage not in STAGE_LIMITS:
        raise ValueError(f"unknown bridge recovery stage: {stage}")
    run = Path(run_dir).resolve()
    authority = load_bridge_authority(authority_path, repository_root=ROOT)
    runtime = read_json(run / "bridge-run.json")
    identity = authority["run_identities"][stage]
    expected_runtime = {
        "format_version": 1,
        "protocol_sha256": BRIDGE_PROTOCOL_SHA256,
        "source": authority["source"],
        "run_identity": identity,
        "initial_state": authority["initial_state"],
        "dataset_authority": authority["dataset_authority"],
        "reference_snapshots": authority["reference_snapshots"],
        "gpu_uuid": authority["gpu_uuid"],
        "schedule_epochs": STAGE_LIMITS[stage],
        "model_parameters": BASELINE_PARAMETERS,
        "bpdd_parameters": 0,
        "ra_parameters": 0,
        "initialization_mode": "fresh_seed0_scratch",
        "parent_checkpoint": None,
        "milestone_period": MILESTONE_PERIOD[stage],
        "rolling_last_every_epoch": True,
    }
    for name, expected in expected_runtime.items():
        if runtime.get(name) != expected:
            raise ValueError(f"bridge runtime authority differs at {name}")
    if not isinstance(runtime.get("data"), str) or not runtime["data"]:
        raise ValueError("bridge runtime data YAML is missing")

    evidence_path = run / "bridge-epochs.jsonl"
    queue_path = run / "local-checkpoint-queue.jsonl"
    evidence = read_jsonl(evidence_path)
    queue = read_jsonl(queue_path)
    if not evidence or abs(len(evidence) - len(queue)) > 1:
        raise ValueError("bridge evidence/queue transaction is not recoverable")
    if len(evidence) == len(queue) + 1:
        evidence = evidence[:-1]
        if not evidence:
            raise ValueError("bridge has no committed epoch after tail rollback")
        _atomic_jsonl(evidence_path, evidence)
    elif len(queue) == len(evidence) + 1:
        queue = queue[:-1]
        _atomic_jsonl(queue_path, queue)
    if len(evidence) != len(queue):
        raise ValueError("bridge evidence/queue reconciliation failed")
    completed = len(evidence)
    if completed > STAGE_LIMITS[stage]:
        raise ValueError("bridge completed epochs exceed the frozen stage")

    for epoch, (row, queued) in enumerate(zip(evidence, queue, strict=True), 1):
        if (
            row.get("completed_epoch") != epoch
            or row.get("stage") != stage
            or row.get("run_id") != identity["run_id"]
            or queued.get("completed_epoch") != epoch
            or queued.get("stage") != stage
            or queued.get("variant") != BRIDGE_VARIANT
            or queued.get("run_id") != identity["run_id"]
            or queued.get("status") != "local-only"
            or any(not _finite(row.get(name)) for name in FINITE_FIELDS)
            or row.get("amp_scale") != 128.0
            or row.get("amp_skipped_steps") != 0
        ):
            raise ValueError(f"invalid bridge evidence/queue at epoch {epoch}")
        milestone = epoch % MILESTONE_PERIOD[stage] == 0
        checkpoint = run / "weights" / f"epoch{epoch - 1}.pt"
        digest = file_sha256(checkpoint) if milestone and checkpoint.is_file() else None
        if milestone and (
            str(row.get("milestone_checkpoint", "")) != str(checkpoint)
            or row.get("milestone_checkpoint_sha256") != digest
            or str(queued.get("milestone_checkpoint", "")) != str(checkpoint)
            or queued.get("milestone_checkpoint_sha256") != digest
        ):
            raise ValueError(f"bridge milestone checkpoint differs at epoch {epoch}")
        if not milestone and any(
            value is not None
            for value in (
                row.get("milestone_checkpoint"),
                row.get("milestone_checkpoint_sha256"),
                queued.get("milestone_checkpoint"),
                queued.get("milestone_checkpoint_sha256"),
            )
        ):
            raise ValueError(f"unexpected non-milestone checkpoint at epoch {epoch}")

    committed = (run / "weights" / "committed.pt").resolve()
    valid, reason = validate_checkpoint(committed)
    if not valid or reason != f"epoch={completed - 1}":
        raise ValueError(f"bridge committed checkpoint is invalid: {reason}")
    committed_sha = file_sha256(committed)
    rolling = read_json(run / "rolling-checkpoint.json")
    if (
        rolling.get("run_id") != identity["run_id"]
        or rolling.get("completed_epoch") != completed
        or Path(str(rolling.get("checkpoint", ""))).resolve() != committed
        or rolling.get("sha256") != committed_sha
        or evidence[-1].get("last_checkpoint_sha256") != committed_sha
        or evidence[-1].get("last_checkpoint_bytes") != committed.stat().st_size
    ):
        raise ValueError("bridge rolling checkpoint authority differs")
    _validate_checkpoint_contract(committed, completed - 1)

    optimizer = read_jsonl(run / "optimizer-evidence.jsonl")
    if not optimizer:
        raise ValueError("bridge optimizer evidence is missing")
    observed_epochs: set[int] = set()
    for attempt, row in enumerate(optimizer, 1):
        epoch = row.get("completed_epoch")
        if (
            row.get("optimizer_attempt") != attempt
            or row.get("run_id") != identity["run_id"]
            or row.get("variant") != BRIDGE_VARIANT
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
                for name in ("gradient_norm", "fdr_gradient_norm")
            )
        ):
            raise ValueError(f"invalid bridge optimizer evidence at attempt {attempt}")
        if epoch <= completed:
            observed_epochs.add(epoch)
    if observed_epochs != set(range(1, completed + 1)):
        raise ValueError(
            "bridge optimizer evidence does not cover each committed epoch"
        )
    return {
        "format_version": 1,
        "decision": "complete" if completed == STAGE_LIMITS[stage] else "resume",
        "stage": stage,
        "variant": BRIDGE_VARIANT,
        "run_id": identity["run_id"],
        "completed_epoch": completed,
        "target_epoch": STAGE_LIMITS[stage],
        "checkpoint": str(committed),
        "checkpoint_sha256": committed_sha,
        "milestone_checkpoints": completed // MILESTONE_PERIOD[stage],
        "optimizer_attempts": len(optimizer),
        "amp_skipped_steps": 0,
        "authority": "same B-arm run rolling last only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_LIMITS), required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_bridge_run(
        args.run_dir, stage=args.stage, authority_path=args.authority
    )
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
