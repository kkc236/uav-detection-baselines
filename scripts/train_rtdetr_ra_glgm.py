"""Train one strict FDR versus FDR+RA-GLGM arm under frozen authority."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_rtdetr_fdr as fdr_train  # noqa: E402
from scripts.sync_experiment_checkpoint import write_json_atomic  # noqa: E402
from src.ra_experiment_protocol import (  # noqa: E402
    RA_EXPERIMENT_PROTOCOL,
    RA_EXPERIMENT_PROTOCOL_SHA256,
    RA_STAGES,
    RA_VARIANTS,
    file_sha256,
    ignore_sidecar_signature,
    load_ra_authority,
    read_json,
    validate_ra_source_authority,
)
from src.ra_glgm_protocol import validate_ra_glgm_initial_state  # noqa: E402
from src.ra_learnability_probe import validate_learnability_report  # noqa: E402
from scripts.evaluate_ra_glgm_gate import validate_screen_gate_report  # noqa: E402
from src.rtdetr_ra_glgm import (  # noqa: E402
    RA_GLGM_CONTROL_CFG,
    RA_GLGM_MODEL_CFG,
    RAGLGMControlTrainer,
    RAGLGMTrainer,
)


STAGE_SCHEDULE = {"smoke": 2, "screen": 50, "formal": 100}
STAGE_CUTOFF = {"smoke": 2, "screen": 30, "formal": 100}
EVIDENCE_FIELDS = (
    "completed_epoch",
    "variant",
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
    "loss_ra_support",
    "ra_target_mean",
    "ra_valid_fraction",
    "gradient_norm",
    "fdr_gradient_norm",
    "ra_glgm_gradient_norm",
    "cuda_peak_mib",
)


class _BoundOptimizerEvidenceMixin:
    """Bind every optimizer attempt to this immutable RA run authority."""

    optimizer_evidence_context: Mapping[str, Any]

    def _record_optimizer_evidence(self, record: dict[str, Any]) -> None:
        context = getattr(self, "optimizer_evidence_context", None)
        if not isinstance(context, Mapping):
            raise RuntimeError("RA optimizer evidence authority is missing")
        required = ("run_id", "variant", "stage")
        if any(not isinstance(context.get(name), str) or not context[name] for name in required):
            raise RuntimeError("RA optimizer evidence authority is incomplete")
        generation = context.get("recovery_generation")
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
            raise RuntimeError("RA optimizer recovery generation is invalid")
        super()._record_optimizer_evidence(
            {
                "run_id": context["run_id"],
                "variant": context["variant"],
                "stage": context["stage"],
                "recovery_generation": generation,
                "completed_epoch": int(self.epoch) + 1,
                **record,
            }
        )


class _AuditedRAGLGMTrainer(_BoundOptimizerEvidenceMixin, RAGLGMTrainer):
    pass


class _AuditedRAGLGMControlTrainer(_BoundOptimizerEvidenceMixin, RAGLGMControlTrainer):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=RA_VARIANTS, required=True)
    parser.add_argument("--stage", choices=RA_STAGES, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--learnability-report", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--screen-gate", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_authority(path: str | Path) -> dict[str, Any]:
    return load_ra_authority(path, repository_root=ROOT)


def validate_source(manifest: Mapping[str, Any]) -> dict[str, str]:
    return validate_ra_source_authority(manifest, repository_root=ROOT)


def validate_initial_state(path: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    state_path = Path(path).resolve()
    record = manifest.get("initial_state")
    if not isinstance(record, Mapping) or state_path != Path(str(record.get("path", ""))).resolve():
        raise ValueError("paired initial-state path differs from RA authority")
    if state_path.is_symlink() or not state_path.is_file():
        raise FileNotFoundError("paired initial-state artifact is missing")
    digest = file_sha256(state_path)
    if digest != str(record.get("sha256", "")).upper():
        raise ValueError("paired initial-state SHA256 mismatch")
    artifact = torch.load(state_path, map_location="cpu", weights_only=False)
    validate_ra_glgm_initial_state(artifact)
    return {**dict(record), "path": str(state_path), "sha256": digest}


def _screen_gate(
    path: Path | None, output_root: Path
) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    root = output_root.resolve()
    report = validate_screen_gate_report(
        path.resolve(),
        baseline_run=root / "screen-seed0-baseline-ra-glgm-v1",
        method_run=root / "screen-seed0-ra_glgm-ra-glgm-v1",
    )
    return report, file_sha256(path)


def prepare_data(
    dataset_root: Path,
    stage: str,
    authority_root: Path,
    manifest: Mapping[str, Any],
) -> Path:
    dataset_authority = manifest.get("dataset_authority")
    if not isinstance(dataset_authority, Mapping):
        raise ValueError("RA manifest is missing dataset authority")
    if dataset_root.resolve() != Path(str(dataset_authority.get("root", ""))).resolve():
        raise ValueError("runtime dataset root differs from RA authority")
    positive = dataset_authority.get("positive")
    if not isinstance(positive, Mapping) or positive.get("sha256") != RA_EXPERIMENT_PROTOCOL["dataset"]["sha256"]:
        raise ValueError("RA manifest positive dataset authority is invalid")
    actual_ignore = ignore_sidecar_signature(dataset_root)
    if actual_ignore != dataset_authority.get("ignore"):
        raise ValueError("runtime ignore sidecar differs from RA authority")
    subset_stage = "screen" if stage in {"smoke", "screen"} else "formal"
    source = fdr_train.prepare_data_yaml(dataset_root, subset_stage, authority_root)
    if stage != "smoke":
        return source
    destination = authority_root / "smoke-data.yaml"
    payload = json.loads(source.read_text(encoding="utf-8"))
    fdr_train._atomic_text(
        destination,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )
    return destination


def build_settings(args: argparse.Namespace, data_yaml: Path) -> dict[str, Any]:
    model = RA_GLGM_CONTROL_CFG if args.variant == "baseline" else RA_GLGM_MODEL_CFG
    settings = {
        **fdr_train.FROZEN_SETTINGS,
        "model": str(model.resolve()),
        "data": str(data_yaml.resolve()),
        "epochs": STAGE_SCHEDULE[args.stage],
        "seed": 0,
        "project": str(args.output_root.resolve()),
        "name": args.name or f"{args.stage}-seed0-{args.variant}-ra-glgm-v1",
        "exist_ok": False,
    }
    if args.resume is not None:
        settings["resume"] = str(args.resume.resolve())
    return settings


def create_trainer(
    variant: str,
    settings: Mapping[str, Any],
    initial_state: Path,
    *,
    optimizer_evidence_context: Mapping[str, Any] | None = None,
):
    common = {"overrides": dict(settings), "initial_state_path": initial_state.resolve()}
    if variant == "baseline":
        trainer = _AuditedRAGLGMControlTrainer(**common, experiment_seed=0)
    elif variant == "ra_glgm":
        trainer = _AuditedRAGLGMTrainer(**common, experiment_seed=0)
    else:
        raise ValueError(f"unknown RA variant: {variant}")
    if optimizer_evidence_context is None:
        raise ValueError("RA optimizer evidence context is required")
    trainer.optimizer_evidence_context = dict(optimizer_evidence_context)
    return trainer


def _model(trainer: Any) -> Any:
    return trainer.model.module if hasattr(trainer.model, "module") else trainer.model


def _private_losses(trainer: Any, variant: str) -> dict[str, float | None]:
    model = _model(trainer)
    fdr = getattr(model, "last_fdr_losses", {})
    ra = getattr(model, "last_ra_glgm_losses", {}) if variant == "ra_glgm" else {}
    return {
        "loss_fgl": fdr_train._number(fdr.get("loss_fgl")),
        "loss_fgl_aux": fdr_train._number(fdr.get("loss_fgl_aux")),
        "loss_bbox_pre": fdr_train._number(fdr.get("loss_bbox_pre")),
        "loss_giou_pre": fdr_train._number(fdr.get("loss_giou_pre")),
        "loss_ra_support": fdr_train._number(ra.get("loss_ra_support")),
        "ra_target_mean": fdr_train._number(ra.get("target_mean")),
        "ra_valid_fraction": fdr_train._number(ra.get("valid_fraction")),
    }


def evidence_record(trainer: Any, context: Mapping[str, Any]) -> dict[str, Any]:
    norms = getattr(trainer, "last_gradient_norms", {})
    record = {
        "completed_epoch": int(trainer.epoch) + 1,
        "variant": context["variant"],
        "stage": context["stage"],
        "run_id": context["run_identity"]["run_id"],
        "precision": fdr_train._metric(trainer, "metrics/precision(B)"),
        "recall": fdr_train._metric(trainer, "metrics/recall(B)"),
        "map50": fdr_train._metric(trainer, "metrics/mAP50(B)"),
        "map": fdr_train._metric(trainer, "metrics/mAP50-95(B)"),
        "map75": fdr_train._map75(trainer),
        **fdr_train._stock_losses(trainer),
        **_private_losses(trainer, str(context["variant"])),
        "gradient_norm": fdr_train._number(norms.get("gradient_norm")),
        "fdr_gradient_norm": fdr_train._number(norms.get("fdr_gradient_norm")),
        "ra_glgm_gradient_norm": fdr_train._number(norms.get("ra_glgm_gradient_norm")),
        "cuda_peak_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }
    required = ("precision", "recall", "map50", "map", "map75", "cuda_peak_mib")
    if any(record[name] is None or not math.isfinite(float(record[name])) for name in required):
        raise FloatingPointError("NONFINITE_RA_EPOCH_EVIDENCE")
    return record


def _write_evidence_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVIDENCE_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in EVIDENCE_FIELDS} for row in rows
        )
    os.replace(temporary, path)


def finalize_epoch(trainer: Any, context: Mapping[str, Any]) -> None:
    run = Path(trainer.save_dir).resolve()
    record = evidence_record(trainer, context)
    evidence_path = run / "ra-epochs.jsonl"
    evidence = fdr_train._append_epoch_record(evidence_path, record)
    rows = fdr_train._read_jsonl(evidence_path)
    _write_evidence_csv(run / "ra-epochs.csv", rows)
    checkpoint = run / "weights" / f"epoch{int(trainer.epoch)}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"exact epoch checkpoint was not saved: {checkpoint}")
    queue_record = {
        "run_id": context["run_identity"]["run_id"],
        "variant": context["variant"],
        "stage": context["stage"],
        "completed_epoch": int(trainer.epoch) + 1,
        "status": "pending",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "artifacts": [str(evidence_path), str(run / "ra-epochs.csv"), str(run / "ra-run.json")],
    }
    queued = fdr_train._append_queue_record(run / "publication-queue.jsonl", queue_record)
    if queued["completed_epoch"] != evidence["completed_epoch"]:
        raise RuntimeError("epoch evidence/queue mismatch")
    if int(evidence["completed_epoch"]) == STAGE_CUTOFF[str(context["stage"])]:
        trainer.stop = True


def _gpu_uuid() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(values) != 1:
        raise ValueError("RA protocol requires exactly one visible physical GPU")
    return values[0]


def write_runtime_manifest(
    trainer: Any,
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    data_yaml: Path,
    gate_sha256: str | None,
    learnability_sha256: str,
) -> None:
    actual_gpu = _gpu_uuid()
    if actual_gpu != manifest["gpu_uuid"]:
        raise ValueError("runtime GPU UUID differs from paired authority")
    destination = Path(trainer.save_dir).resolve() / "ra-run.json"
    payload = {
        "format_version": 1,
        "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
        "source": manifest["source"],
        "run_identity": dict(identity),
        "initial_state": manifest["initial_state"],
        "data": str(data_yaml.resolve()),
        "dataset_authority": manifest["dataset_authority"],
        "gpu_uuid": actual_gpu,
        "schedule_epochs": STAGE_SCHEDULE[args.stage],
        "cutoff_epoch": 30 if args.stage == "screen" else None,
        "model_parameters": sum(parameter.numel() for parameter in _model(trainer).parameters()),
        "locked_evaluator_sha256": manifest["locked_evaluator"]["sha256"],
        "initialization_mode": "fresh_paired_scratch",
        "parent_checkpoint": None,
        "screen_gate_sha256": gate_sha256,
        "learnability_report_sha256": learnability_sha256,
    }
    if destination.exists():
        if read_json(destination) != payload:
            raise ValueError("changed RA runtime authority")
    else:
        write_json_atomic(destination, payload)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    for name in (
        "protocol_manifest",
        "initial_state",
        "learnability_report",
        "dataset_root",
        "output_root",
    ):
        setattr(args, name, Path(getattr(args, name)).resolve())
    if args.resume is not None:
        args.resume = args.resume.resolve()
    if args.screen_gate is not None:
        args.screen_gate = args.screen_gate.resolve()
    if args.stage == "formal" and args.resume is None and args.screen_gate is None:
        raise ValueError("Formal100 requires the immutable passing Screen30 Gate")
    if args.stage != "formal" and args.screen_gate is not None:
        raise ValueError("Screen Gate may only authorize Formal100")
    manifest = load_authority(args.protocol_manifest)
    validate_source(manifest)
    state = validate_initial_state(args.initial_state, manifest)
    validate_learnability_report(args.learnability_report, protocol_manifest=manifest)
    learnability_sha = file_sha256(args.learnability_report)
    _, gate_sha = _screen_gate(args.screen_gate, args.output_root)
    authority_root = args.output_root / "_ra-authority"
    data_yaml = prepare_data(args.dataset_root, args.stage, authority_root, manifest)
    identity = manifest["run_identities"][f"{args.variant}_{args.stage}"]
    recovery_generation = 0
    if args.resume is not None:
        from scripts.validate_ra_resume import validate_resume

        decision = validate_resume(
            args.resume.parent.parent,
            variant=args.variant,
            stage=args.stage,
            protocol_manifest=args.protocol_manifest,
            learnability_report=args.learnability_report,
            screen_gate=args.screen_gate,
        )
        if Path(decision["checkpoint"]).resolve() != args.resume:
            raise ValueError("resume path is not the audited latest exact checkpoint")
        if decision["trailing_uncommitted_optimizer_attempts"] != 0:
            raise ValueError("resume requires a persisted optimizer recovery generation")
        recovery_generation = int(decision["recovery_generation"])
        if recovery_generation < 1:
            raise ValueError("resume was not authorized by the audited supervisor recovery lineage")
    settings = build_settings(args, data_yaml)
    summary = {
        "status": "dry-run-passed" if args.dry_run else "launching",
        "variant": args.variant,
        "stage": args.stage,
        "run_identity": identity,
        "initial_state": state,
        "settings": settings,
        "screen_gate_sha256": gate_sha,
        "learnability_report_sha256": learnability_sha,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary
    context = {"variant": args.variant, "stage": args.stage, "run_identity": identity}
    trainer = create_trainer(
        args.variant,
        settings,
        args.initial_state,
        optimizer_evidence_context={
            "run_id": str(identity["run_id"]),
            "variant": args.variant,
            "stage": args.stage,
            "recovery_generation": recovery_generation,
        },
    )
    trainer.add_callback(
        "on_train_start",
        lambda current: write_runtime_manifest(
            current,
            args=args,
            manifest=manifest,
            identity=identity,
            data_yaml=data_yaml,
            gate_sha256=gate_sha,
            learnability_sha256=learnability_sha,
        ),
    )
    trainer.add_callback("on_train_epoch_start", fdr_train.reset_peak_memory)
    trainer.add_callback("on_model_save", lambda current: finalize_epoch(current, context))
    trainer.train()
    return {**summary, "status": "training-finished", "save_dir": str(trainer.save_dir)}


def main() -> None:
    execute(build_parser().parse_args())


if __name__ == "__main__":
    main()
