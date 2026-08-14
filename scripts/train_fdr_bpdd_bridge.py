"""Train the missing synchronized FDR+BPDD B arm for Smoke2 or Formal100."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_rtdetr_fdr as fdr_train  # noqa: E402
from scripts.sync_experiment_checkpoint import write_json_atomic  # noqa: E402
from src.checkpoint_recovery import validate_checkpoint  # noqa: E402
from src.fdr_bpdd_bridge_protocol import (  # noqa: E402
    BRIDGE_PROTOCOL,
    BRIDGE_PROTOCOL_SHA256,
    BRIDGE_STAGES,
    BRIDGE_VARIANT,
    load_bridge_authority,
)
from src.lpr_protocol import dataset_signature  # noqa: E402
from src.ra_experiment_protocol import (  # noqa: E402
    BASELINE_PARAMETERS,
    file_sha256,
    ignore_sidecar_signature,
    read_json,
)
from src.ra_glgm_protocol import validate_ra_glgm_initial_state  # noqa: E402
from src.rtdetr_fdr_bpdd_bridge import (  # noqa: E402
    FDR_BPDD_BRIDGE_MODEL_CFG,
    FDRBPDDBridgeTrainer,
)


STAGE_EPOCHS = {"smoke": 2, "formal": 100}
MILESTONE_PERIOD = {"smoke": 1, "formal": 5}
MIN_FREE_DISK_BYTES = 8 * 1024**3
EVIDENCE_FIELDS = (
    "completed_epoch",
    "stage",
    "run_id",
    "precision",
    "recall",
    "map50",
    "map",
    "map75",
    "loss_giou",
    "loss_class",
    "loss_bbox",
    "loss_fgl",
    "loss_fgl_aux",
    "loss_bbox_pre",
    "loss_giou_pre",
    "loss_bpdd",
    "bpdd_active_edge_ratio",
    "bpdd_mean_reliability",
    "bpdd_mean_teacher_improvement",
    "bpdd_mixture_beats_final_ratio",
    "bpdd_mean_mixture_advantage_over_final",
    "bpdd_matched_queries",
    "bpdd_eligible_edges",
    "gradient_norm",
    "fdr_gradient_norm",
    "learning_rate",
    "amp_scale",
    "amp_skipped_steps",
    "cuda_peak_mib",
    "epoch_wall_seconds",
    "last_checkpoint_sha256",
    "last_checkpoint_bytes",
    "milestone_checkpoint",
    "milestone_checkpoint_sha256",
    "disk_free_bytes",
)


class _AuditedBridgeTrainer(FDRBPDDBridgeTrainer):
    optimizer_evidence_context: Mapping[str, Any]

    def _record_optimizer_evidence(self, record: dict[str, Any]) -> None:
        context = getattr(self, "optimizer_evidence_context", None)
        if not isinstance(context, Mapping):
            raise RuntimeError("bridge optimizer evidence authority is missing")
        super()._record_optimizer_evidence(
            {
                "run_id": context["run_id"],
                "variant": BRIDGE_VARIANT,
                "stage": context["stage"],
                "completed_epoch": int(self.epoch) + 1,
                **record,
            }
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=BRIDGE_STAGES, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _gpu_uuid() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(values) != 1:
        raise ValueError("bridge protocol requires exactly one visible physical GPU")
    return values[0]


def validate_initial_state(path: Path, authority: Mapping[str, Any]) -> dict[str, Any]:
    state_path = path.resolve()
    expected = authority.get("initial_state")
    if not isinstance(expected, Mapping):
        raise ValueError("bridge authority has no initial state")
    if state_path != Path(str(expected.get("path", ""))).resolve():
        raise ValueError("bridge initial-state path differs from authority")
    if state_path.is_symlink() or not state_path.is_file():
        raise FileNotFoundError("bridge initial-state artifact is missing")
    digest = file_sha256(state_path)
    if digest != str(expected.get("sha256", "")).upper():
        raise ValueError("bridge initial-state SHA256 differs from authority")
    artifact = torch.load(state_path, map_location="cpu", weights_only=False)
    validate_ra_glgm_initial_state(artifact)
    if artifact.get("fingerprints") != expected.get("fingerprints"):
        raise ValueError("bridge initial-state fingerprints differ from authority")
    metadata = artifact.get("metadata", {})
    if metadata.get("seed") != 0 or metadata.get("initialization") != "fresh_scratch":
        raise ValueError("bridge must use the fresh seed0 scratch artifact")
    return {**dict(expected), "path": str(state_path), "sha256": digest}


def prepare_data(
    dataset_root: Path,
    stage: str,
    authority_root: Path,
    authority: Mapping[str, Any],
) -> Path:
    data = authority.get("dataset_authority")
    root = dataset_root.resolve()
    if (
        not isinstance(data, Mapping)
        or root != Path(str(data.get("root", ""))).resolve()
    ):
        raise ValueError("runtime dataset root differs from bridge authority")
    if dataset_signature(root) != data.get("positive"):
        raise ValueError("runtime positive dataset differs from bridge authority")
    if ignore_sidecar_signature(root) != data.get("ignore"):
        raise ValueError("runtime ignore sidecars differ from bridge authority")
    source_stage = "screen" if stage == "smoke" else "formal"
    return fdr_train.prepare_data_yaml(root, source_stage, authority_root)


def build_settings(args: argparse.Namespace, data_yaml: Path) -> dict[str, Any]:
    return {
        **fdr_train.FROZEN_SETTINGS,
        "model": str(FDR_BPDD_BRIDGE_MODEL_CFG.resolve()),
        "data": str(data_yaml.resolve()),
        "epochs": STAGE_EPOCHS[args.stage],
        "batch": int(BRIDGE_PROTOCOL["training"]["batch"]),
        "workers": int(BRIDGE_PROTOCOL["training"]["workers"]),
        "nbs": int(BRIDGE_PROTOCOL["training"]["nbs"]),
        "seed": 0,
        "project": str(args.output_root.resolve()),
        "name": args.name or f"{args.stage}-seed0-{BRIDGE_VARIANT}",
        "exist_ok": False,
        "save_period": -1,
        **({"resume": str(args.resume.resolve())} if args.resume is not None else {}),
    }


def _model(trainer: Any) -> Any:
    return trainer.model.module if hasattr(trainer.model, "module") else trainer.model


def reset_epoch_state(trainer: Any) -> None:
    trainer._bridge_epoch_started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _model(trainer).reset_bpdd_epoch_statistics()


def _private_evidence(trainer: Any) -> dict[str, float | None]:
    model = _model(trainer)
    fdr = getattr(model, "last_fdr_losses", {})
    bpdd = model.bpdd_epoch_statistics()
    return {
        "loss_fgl": fdr_train._number(fdr.get("loss_fgl")),
        "loss_fgl_aux": fdr_train._number(fdr.get("loss_fgl_aux")),
        "loss_bbox_pre": fdr_train._number(fdr.get("loss_bbox_pre")),
        "loss_giou_pre": fdr_train._number(fdr.get("loss_giou_pre")),
        "loss_bpdd": fdr_train._number(fdr.get("loss_bpdd")),
        "bpdd_active_edge_ratio": fdr_train._number(bpdd.get("active_edge_ratio")),
        "bpdd_mean_reliability": fdr_train._number(bpdd.get("mean_reliability")),
        "bpdd_mean_teacher_improvement": fdr_train._number(
            bpdd.get("mean_teacher_improvement")
        ),
        "bpdd_mixture_beats_final_ratio": fdr_train._number(
            bpdd.get("mixture_beats_final_ratio")
        ),
        "bpdd_mean_mixture_advantage_over_final": fdr_train._number(
            bpdd.get("mean_mixture_advantage_over_final")
        ),
        "bpdd_matched_queries": fdr_train._number(bpdd.get("matched_queries")),
        "bpdd_eligible_edges": fdr_train._number(bpdd.get("eligible_edges")),
    }


def _copy_milestone(last: Path, destination: Path) -> str:
    if destination.exists() or destination.is_symlink():
        if file_sha256(destination) != file_sha256(last):
            raise ValueError(f"changed bridge milestone checkpoint: {destination}")
        return file_sha256(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"stale milestone temporary exists: {temporary}")
    shutil.copyfile(last, temporary)
    os.replace(temporary, destination)
    return file_sha256(destination)


def _stage_committed_checkpoint(last: Path, committed: Path) -> Path:
    temporary = committed.with_suffix(committed.suffix + ".next")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    shutil.copyfile(last, temporary)
    if file_sha256(temporary) != file_sha256(last):
        raise OSError("staged bridge recovery checkpoint differs from last.pt")
    return temporary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVIDENCE_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in EVIDENCE_FIELDS} for row in rows
        )
    os.replace(temporary, path)


def _learning_rate(trainer: Any) -> float | None:
    groups = getattr(getattr(trainer, "optimizer", None), "param_groups", [])
    values = [float(group["lr"]) for group in groups if "lr" in group]
    return (
        values[0] if values and all(math.isfinite(value) for value in values) else None
    )


def finalize_epoch(trainer: Any, context: Mapping[str, Any]) -> None:
    run = Path(trainer.save_dir).resolve()
    completed = int(trainer.epoch) + 1
    last = run / "weights" / "last.pt"
    valid, reason = validate_checkpoint(last)
    if not valid or reason != f"epoch={trainer.epoch}":
        raise ValueError(f"bridge rolling checkpoint is invalid: {reason}")
    last_sha = file_sha256(last)
    committed = run / "weights" / "committed.pt"
    staged_committed = _stage_committed_checkpoint(last, committed)
    period = MILESTONE_PERIOD[str(context["stage"])]
    milestone = run / "weights" / f"epoch{trainer.epoch}.pt"
    milestone_sha = (
        _copy_milestone(last, milestone) if completed % period == 0 else None
    )
    free = shutil.disk_usage(run).free
    if free < MIN_FREE_DISK_BYTES:
        raise OSError(f"bridge disk hard stop: free_bytes={free}")
    norms = getattr(trainer, "last_gradient_norms", {})
    record = {
        "completed_epoch": completed,
        "stage": context["stage"],
        "run_id": context["run_identity"]["run_id"],
        "precision": fdr_train._metric(trainer, "metrics/precision(B)"),
        "recall": fdr_train._metric(trainer, "metrics/recall(B)"),
        "map50": fdr_train._metric(trainer, "metrics/mAP50(B)"),
        "map": fdr_train._metric(trainer, "metrics/mAP50-95(B)"),
        "map75": fdr_train._map75(trainer),
        **fdr_train._stock_losses(trainer),
        **_private_evidence(trainer),
        "gradient_norm": fdr_train._number(norms.get("gradient_norm")),
        "fdr_gradient_norm": fdr_train._number(norms.get("fdr_gradient_norm")),
        "learning_rate": _learning_rate(trainer),
        "amp_scale": float(trainer.scaler.get_scale()),
        "amp_skipped_steps": 0,
        "cuda_peak_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "epoch_wall_seconds": round(
            time.perf_counter() - trainer._bridge_epoch_started, 3
        ),
        "last_checkpoint_sha256": last_sha,
        "last_checkpoint_bytes": last.stat().st_size,
        "milestone_checkpoint": str(milestone) if milestone_sha else None,
        "milestone_checkpoint_sha256": milestone_sha,
        "disk_free_bytes": free,
    }
    finite_required = (
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
    if any(
        record[name] is None or not math.isfinite(float(record[name]))
        for name in finite_required
    ):
        raise FloatingPointError("NONFINITE_BRIDGE_EPOCH_EVIDENCE")
    evidence_path = run / "bridge-epochs.jsonl"
    fdr_train._append_epoch_record(evidence_path, record)
    rows = fdr_train._read_jsonl(evidence_path)
    _write_csv(run / "bridge-epochs.csv", rows)
    queue = {
        "run_id": context["run_identity"]["run_id"],
        "variant": BRIDGE_VARIANT,
        "stage": context["stage"],
        "completed_epoch": completed,
        "status": "local-only",
        "rolling_checkpoint": str(committed),
        "rolling_checkpoint_sha256_at_commit": last_sha,
        "milestone_checkpoint": str(milestone) if milestone_sha else None,
        "milestone_checkpoint_sha256": milestone_sha,
        "artifacts": [
            str(evidence_path),
            str(run / "bridge-epochs.csv"),
            str(run / "bridge-run.json"),
        ],
    }
    fdr_train._append_queue_record(run / "local-checkpoint-queue.jsonl", queue)
    os.replace(staged_committed, committed)
    write_json_atomic(
        run / "rolling-checkpoint.json",
        {
            "run_id": context["run_identity"]["run_id"],
            "completed_epoch": completed,
            "checkpoint": str(committed),
            "sha256": last_sha,
            "bytes": committed.stat().st_size,
        },
    )


def write_runtime_manifest(
    trainer: Any,
    *,
    args: argparse.Namespace,
    authority: Mapping[str, Any],
    identity: Mapping[str, Any],
    data_yaml: Path,
) -> None:
    actual_gpu = _gpu_uuid()
    if actual_gpu != authority["gpu_uuid"]:
        raise ValueError("runtime GPU UUID differs from bridge authority")
    model = _model(trainer)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != BASELINE_PARAMETERS:
        raise ValueError(f"bridge parameter count differs: {parameters}")
    if any("ra_glgm" in name or "bpdd" in name.lower() for name in model.state_dict()):
        raise ValueError("B checkpoint graph contains private RA/BPDD state")
    payload = {
        "format_version": 1,
        "protocol_sha256": BRIDGE_PROTOCOL_SHA256,
        "source": authority["source"],
        "run_identity": dict(identity),
        "initial_state": authority["initial_state"],
        "dataset_authority": authority["dataset_authority"],
        "reference_snapshots": authority["reference_snapshots"],
        "data": str(data_yaml.resolve()),
        "gpu_uuid": actual_gpu,
        "schedule_epochs": STAGE_EPOCHS[args.stage],
        "model_parameters": parameters,
        "bpdd_parameters": 0,
        "ra_parameters": 0,
        "initialization_mode": "fresh_seed0_scratch",
        "parent_checkpoint": None,
        "milestone_period": MILESTONE_PERIOD[args.stage],
        "rolling_last_every_epoch": True,
    }
    destination = Path(trainer.save_dir).resolve() / "bridge-run.json"
    if destination.exists():
        if read_json(destination) != payload:
            raise ValueError("changed bridge runtime authority")
    else:
        write_json_atomic(destination, payload)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    for name in ("authority", "initial_state", "dataset_root", "output_root"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    if args.resume is not None:
        args.resume = args.resume.resolve()
    authority = load_bridge_authority(args.authority, repository_root=ROOT)
    state = validate_initial_state(args.initial_state, authority)
    identity = authority["run_identities"][args.stage]
    authority_root = args.output_root / "_bridge-authority"
    data_yaml = prepare_data(args.dataset_root, args.stage, authority_root, authority)
    settings = build_settings(args, data_yaml)
    if args.resume is not None:
        from scripts.validate_fdr_bpdd_bridge_resume import validate_bridge_run

        decision = validate_bridge_run(
            args.resume.parent.parent,
            stage=args.stage,
            authority_path=args.authority,
        )
        if (
            decision["decision"] != "resume"
            or Path(decision["checkpoint"]).resolve() != args.resume
        ):
            raise ValueError(
                "bridge resume is not the latest authorized rolling checkpoint"
            )
    summary = {
        "status": "dry-run-passed" if args.dry_run else "launching",
        "stage": args.stage,
        "run_identity": identity,
        "initial_state": state,
        "settings": settings,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary
    trainer = _AuditedBridgeTrainer(
        overrides=settings,
        initial_state_path=args.initial_state,
        experiment_seed=0,
    )
    trainer._oom_retries = 3
    trainer.optimizer_evidence_context = {
        "run_id": identity["run_id"],
        "stage": args.stage,
    }
    context = {"stage": args.stage, "run_identity": identity}
    trainer.add_callback(
        "on_train_start",
        lambda current: write_runtime_manifest(
            current,
            args=args,
            authority=authority,
            identity=identity,
            data_yaml=data_yaml,
        ),
    )
    trainer.add_callback("on_train_epoch_start", reset_epoch_state)
    trainer.add_callback(
        "on_model_save", lambda current: finalize_epoch(current, context)
    )
    trainer.train()
    return {**summary, "status": "training-finished", "save_dir": str(trainer.save_dir)}


def main() -> None:
    execute(build_parser().parse_args())


if __name__ == "__main__":
    main()
