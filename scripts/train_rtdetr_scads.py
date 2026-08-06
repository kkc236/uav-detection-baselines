"""Train strict paired FDR/SCADS RT-DETR-L arms under one authority."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sync_experiment_checkpoint import write_json_atomic  # noqa: E402
from scripts.train_rtdetr_fdr import (  # noqa: E402
    FORMAL_EPOCHS,
    FROZEN_SETTINGS,
    SCREEN_CUTOFF_EPOCH,
    SCREEN_SCHEDULE_EPOCHS,
    _append_epoch_record,
    _append_queue_record,
    _file_sha256,
    _map75,
    _metric,
    _number,
    _read_jsonl,
    _stock_losses,
    _unwrap_model,
    current_source_identity,
    prepare_data_yaml,
    reset_peak_memory,
)
from src.scads_protocol import (  # noqa: E402
    SCADS_PROTOCOL,
    SCADS_PROTOCOL_SHA256,
    build_run_identity,
    canonical_json_bytes,
    public_state_sha256,
    validate_resume_authority,
    validate_scads_initial_state,
)


FDR_MODEL_CONFIG = ROOT / "configs" / "rtdetr-l-fdr.yaml"
SCADS_MODEL_CONFIG = ROOT / "configs" / "rtdetr-l-fdr-scads.yaml"
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
    "loss_scads_route",
    "gradient_norm",
    "fdr_gradient_norm",
    "scads_gradient_norm",
    "route_narrow_count_last_batch",
    "route_base_count_last_batch",
    "route_wide_count_last_batch",
    "route_overflow_count_last_batch",
    "route_positive_count_last_batch",
    "cuda_peak_mib",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one strict FDR/SCADS arm.")
    parser.add_argument("--variant", choices=("fdr", "scads"), required=True)
    parser.add_argument("--stage", choices=("screen", "formal"), required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--publication-queue", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_authority(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SCADS protocol manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("SCADS protocol manifest format must be 1")
    unhashed = dict(manifest)
    claimed_hash = unhashed.pop("manifest_sha256", None)
    actual_hash = hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest().upper()
    if claimed_hash != actual_hash:
        raise ValueError("SCADS protocol manifest SHA256 mismatch")
    if manifest.get("protocol") != SCADS_PROTOCOL:
        raise ValueError("SCADS protocol payload differs from frozen code")
    if manifest.get("protocol_sha256") != SCADS_PROTOCOL_SHA256:
        raise ValueError("SCADS protocol SHA256 differs from frozen code")
    source = manifest.get("source")
    identities = manifest.get("run_identities")
    if not isinstance(source, Mapping) or not isinstance(identities, Mapping):
        raise ValueError("SCADS source or run identities are missing")
    if manifest.get("source_sha256") != public_state_sha256(source):
        raise ValueError("SCADS source identity hash mismatch")
    for variant in ("fdr", "scads"):
        for stage in ("screen", "formal"):
            key = f"{variant}_{stage}"
            expected = build_run_identity(source, stage=stage, variant=variant, seed=0)
            if identities.get(key) != expected:
                raise ValueError(f"SCADS run identity mismatch: {key}")
    return manifest


def validate_source_authority(manifest: Mapping[str, Any]) -> dict[str, str]:
    expected = manifest.get("source")
    if not isinstance(expected, Mapping):
        raise ValueError("SCADS source authority is missing")
    actual = current_source_identity(ROOT)
    if dict(expected) != actual:
        raise ValueError(
            f"checked-out source differs from SCADS authority: expected={dict(expected)}, actual={actual}"
        )
    return actual


def validate_initial_state_file(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path).resolve()
    record = manifest.get("initial_state")
    if not isinstance(record, Mapping):
        raise ValueError("SCADS initial-state authority is missing")
    if path != Path(str(record.get("path", ""))).resolve():
        raise ValueError("SCADS initial-state path differs from manifest")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"SCADS initial-state not found: {path}")
    digest = _file_sha256(path)
    if digest != str(record.get("sha256", "")).upper():
        raise ValueError("SCADS initial-state SHA256 mismatch")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    validate_scads_initial_state(artifact)
    return {**dict(record), "path": str(path), "sha256": digest}


def build_settings(args: argparse.Namespace, data_yaml: Path) -> dict[str, Any]:
    epochs = SCREEN_SCHEDULE_EPOCHS if args.stage == "screen" else FORMAL_EPOCHS
    model = FDR_MODEL_CONFIG if args.variant == "fdr" else SCADS_MODEL_CONFIG
    settings = {
        **FROZEN_SETTINGS,
        "model": str(model),
        "data": str(Path(data_yaml).resolve()),
        "epochs": epochs,
        "seed": 0,
        "project": str(Path(args.output_root).resolve()),
        "name": args.name or f"{args.stage}-seed0-{args.variant}-scads-v1",
        "exist_ok": False,
    }
    if args.resume is not None:
        settings["resume"] = str(Path(args.resume).resolve())
    return settings


def create_trainer(variant: str, settings: dict[str, Any], initial_state: Path):
    from src.rtdetr_scads import SCADSPairedFDRTrainer, SCADSTrainer

    common = {
        "overrides": settings,
        "initial_state_path": Path(initial_state).resolve(),
        "experiment_seed": 0,
    }
    if variant == "fdr":
        return SCADSPairedFDRTrainer(**common)
    if variant == "scads":
        return SCADSTrainer(**common)
    raise ValueError(f"unknown SCADS variant: {variant}")


def validate_resume_checkpoint(checkpoint: Path, identity: Mapping[str, Any]) -> Path:
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file() or checkpoint.parent.name != "weights":
        raise FileNotFoundError("SCADS resume checkpoint is invalid")
    runtime_path = checkpoint.parent.parent / "scads-run.json"
    if not runtime_path.is_file():
        raise FileNotFoundError("SCADS resume run authority is missing")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    validate_resume_authority(runtime.get("run_identity", {}), identity)
    return checkpoint


def _method_losses(trainer: Any) -> dict[str, float | None]:
    names = (
        "loss_fgl",
        "loss_fgl_aux",
        "loss_bbox_pre",
        "loss_giou_pre",
        "loss_scads_route",
    )
    model = _unwrap_model(trainer.model)
    values = getattr(model, "last_fdr_losses", {})
    return {name: _number(values.get(name)) for name in names}


def _route_diagnostics(trainer: Any, variant: str) -> dict[str, int | None]:
    names = (
        "route_narrow_count_last_batch",
        "route_base_count_last_batch",
        "route_wide_count_last_batch",
        "route_overflow_count_last_batch",
        "route_positive_count_last_batch",
    )
    if variant != "scads":
        return {name: None for name in names}
    criterion = getattr(_unwrap_model(trainer.model), "criterion", None)
    counts = getattr(criterion, "last_route_target_counts", torch.zeros(3, dtype=torch.long))
    values = counts.detach().cpu().tolist() if isinstance(counts, torch.Tensor) else [0, 0, 0]
    return {
        names[0]: int(values[0]),
        names[1]: int(values[1]),
        names[2]: int(values[2]),
        names[3]: int(getattr(criterion, "last_route_overflow_count", 0)),
        names[4]: int(getattr(criterion, "last_route_positive_count", 0)),
    }


def evidence_record(trainer: Any, context: Mapping[str, Any]) -> dict[str, Any]:
    variant = str(context["variant"])
    norms = getattr(trainer, "last_gradient_norms", {})
    return {
        "completed_epoch": int(trainer.epoch) + 1,
        "variant": variant,
        "stage": str(context["stage"]),
        "run_id": str(context["run_identity"]["run_id"]),
        "precision": _metric(trainer, "metrics/precision(B)"),
        "recall": _metric(trainer, "metrics/recall(B)"),
        "map50": _metric(trainer, "metrics/mAP50(B)"),
        "map": _metric(trainer, "metrics/mAP50-95(B)"),
        "map75": _map75(trainer),
        **_stock_losses(trainer),
        **_method_losses(trainer),
        "gradient_norm": _number(norms.get("gradient_norm")),
        "fdr_gradient_norm": _number(norms.get("fdr_gradient_norm")),
        "scads_gradient_norm": (
            _number(norms.get("scads_gradient_norm")) if variant == "scads" else None
        ),
        **_route_diagnostics(trainer, variant),
        "cuda_peak_mib": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 2)
            if torch.cuda.is_available()
            else 0.0
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVIDENCE_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in EVIDENCE_FIELDS} for row in rows)
    os.replace(temporary, path)


def _checkpoint_for_epoch(trainer: Any) -> Path:
    checkpoint = Path(trainer.save_dir).resolve() / "weights" / f"epoch{int(trainer.epoch)}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SCADS epoch checkpoint was not saved: {checkpoint}")
    return checkpoint


def finalize_epoch(trainer: Any, context: Mapping[str, Any]) -> dict[str, Any]:
    run = Path(trainer.save_dir).resolve()
    record = _append_epoch_record(run / "scads-epochs.jsonl", evidence_record(trainer, context))
    _write_csv(run / "scads-epochs.csv", _read_jsonl(run / "scads-epochs.jsonl"))
    checkpoint = _checkpoint_for_epoch(trainer)
    queue_value = context.get("publication_queue")
    queue = Path(queue_value).resolve() if queue_value else run / "publication-queue.jsonl"
    publication = _append_queue_record(
        queue,
        {
            "run_id": str(context["run_identity"]["run_id"]),
            "variant": str(context["variant"]),
            "stage": str(context["stage"]),
            "completed_epoch": int(trainer.epoch) + 1,
            "status": "pending",
            "checkpoint": str(checkpoint),
            "checkpoint_size": checkpoint.stat().st_size,
            "checkpoint_sha256": _file_sha256(checkpoint),
            "artifacts": [
                str((run / "scads-epochs.jsonl").resolve()),
                str((run / "scads-epochs.csv").resolve()),
                str((run / "scads-run.json").resolve()),
            ],
        },
    )
    if int(publication["completed_epoch"]) != int(record["completed_epoch"]):
        raise RuntimeError("SCADS evidence/publication epoch mismatch")
    if context["stage"] == "screen" and int(record["completed_epoch"]) == SCREEN_CUTOFF_EPOCH:
        trainer.stop = True
    return {"evidence": record, "publication": publication}


def write_runtime_manifest(
    trainer: Any,
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    data_yaml: Path,
) -> None:
    destination = Path(trainer.save_dir).resolve() / "scads-run.json"
    payload = {
        "format_version": 1,
        "protocol_sha256": SCADS_PROTOCOL_SHA256,
        "source": manifest["source"],
        "run_identity": dict(identity),
        "initial_state": {
            "path": str(Path(args.initial_state).resolve()),
            "sha256": manifest["initial_state"]["sha256"],
        },
        "data": str(Path(data_yaml).resolve()),
        "screen_cutoff_epoch": SCREEN_CUTOFF_EPOCH if args.stage == "screen" else None,
        "publication_queue": str(
            Path(args.publication_queue).resolve()
            if args.publication_queue is not None
            else destination.parent / "publication-queue.jsonl"
        ),
    }
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != payload:
            raise ValueError("changed SCADS runtime authority")
        return
    write_json_atomic(destination, payload)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    for name in ("protocol_manifest", "initial_state", "dataset_root", "output_root"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    if args.resume is not None:
        args.resume = Path(args.resume).resolve()
    if args.publication_queue is not None:
        args.publication_queue = Path(args.publication_queue).resolve()

    manifest = load_authority(args.protocol_manifest)
    validate_source_authority(manifest)
    state = validate_initial_state_file(args.initial_state, manifest)
    authority_root = args.output_root / "_scads-authority"
    data_yaml = prepare_data_yaml(args.dataset_root, args.stage, authority_root)
    identity = manifest["run_identities"][f"{args.variant}_{args.stage}"]
    if args.resume is not None:
        validate_resume_checkpoint(args.resume, identity)
    settings = build_settings(args, data_yaml)
    summary = {
        "status": "dry-run-passed" if args.dry_run else "launching",
        "variant": args.variant,
        "stage": args.stage,
        "run_identity": identity,
        "initial_state": state,
        "settings": settings,
        "screen_cutoff_epoch": SCREEN_CUTOFF_EPOCH if args.stage == "screen" else None,
        "publication_queue": (
            str(args.publication_queue)
            if args.publication_queue is not None
            else "<run_dir>/publication-queue.jsonl"
        ),
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True))
        return summary

    trainer = create_trainer(args.variant, settings, args.initial_state)
    context = {
        "variant": args.variant,
        "stage": args.stage,
        "run_identity": identity,
        "publication_queue": args.publication_queue,
    }
    trainer.add_callback(
        "on_train_start",
        lambda current: write_runtime_manifest(
            current,
            args=args,
            manifest=manifest,
            identity=identity,
            data_yaml=data_yaml,
        ),
    )
    trainer.add_callback("on_train_epoch_start", reset_peak_memory)
    trainer.add_callback("on_model_save", lambda current: finalize_epoch(current, context))
    trainer.train()
    return {**summary, "status": "training-finished", "save_dir": str(trainer.save_dir)}


def main() -> None:
    execute(build_parser().parse_args())


if __name__ == "__main__":
    main()
