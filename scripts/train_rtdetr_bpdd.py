"""Train strict paired FDR/BPDD RT-DETR-L arms under one immutable protocol."""

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

from scripts import train_rtdetr_fdr as fdr_cli  # noqa: E402
from scripts.sync_experiment_checkpoint import write_json_atomic  # noqa: E402
from src.bpdd_protocol import (  # noqa: E402
    BPDD_PROTOCOL,
    BPDD_PROTOCOL_SHA256,
    FDR_INITIAL_STATE_SHA256,
    build_run_identity,
    canonical_json_bytes,
    public_state_sha256,
    validate_resume_authority,
)


FDR_MODEL_CONFIG = (ROOT / "configs" / "rtdetr-l-fdr.yaml").resolve()
BPDD_MODEL_CONFIG = (ROOT / "configs" / "rtdetr-l-fdr-bpdd.yaml").resolve()
SCREEN_SCHEDULE_EPOCHS = 50
SCREEN_CUTOFF_EPOCH = 30
FORMAL_EPOCHS = 100
FROZEN_SETTINGS = dict(fdr_cli.FROZEN_SETTINGS)

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
    "loss_bpdd",
    "bpdd_active_edge_ratio",
    "bpdd_mean_reliability",
    "bpdd_mean_teacher_improvement",
    "gradient_norm",
    "fdr_gradient_norm",
    "gradients_finite",
    "cuda_peak_mib",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one strict seed0 FDR/BPDD RT-DETR-L arm."
    )
    parser.add_argument("--variant", choices=("fdr", "fdr_bpdd"), required=True)
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_epoch_record(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    rows = _read_jsonl(path)
    epoch = int(record["completed_epoch"])
    same = [row for row in rows if int(row["completed_epoch"]) == epoch]
    if same:
        if len(same) != 1 or same[0] != record:
            raise ValueError(f"changed BPDD evidence for completed epoch {epoch}")
        return same[0]
    expected = int(rows[-1]["completed_epoch"]) + 1 if rows else 1
    if epoch != expected:
        raise ValueError(f"BPDD evidence gap: expected {expected}, got {epoch}")
    rows.append(record)
    _atomic_text(
        path,
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in rows
        ),
    )
    return record


def _append_queue_record(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    rows = _read_jsonl(path)
    key = (record["run_id"], int(record["completed_epoch"]))
    same = [
        row
        for row in rows
        if (row.get("run_id"), int(row.get("completed_epoch", -1))) == key
    ]
    if same:
        if len(same) != 1 or same[0] != record:
            raise ValueError(f"changed BPDD publication entry for {key}")
        return same[0]
    rows.append(record)
    _atomic_text(
        path,
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in rows
        ),
    )
    return record


def load_authority(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"BPDD protocol manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("BPDD protocol manifest format must be 1")
    manifest_hash = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    actual_hash = hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest().upper()
    if manifest_hash != actual_hash:
        raise ValueError("BPDD protocol manifest SHA256 mismatch")
    if manifest.get("protocol") != BPDD_PROTOCOL:
        raise ValueError("BPDD protocol payload does not match frozen authority")
    if manifest.get("protocol_sha256") != BPDD_PROTOCOL_SHA256:
        raise ValueError("BPDD protocol SHA256 mismatch")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("BPDD source authority is missing")
    if manifest.get("source_sha256") != public_state_sha256(source):
        raise ValueError("BPDD source SHA256 mismatch")
    initial = manifest.get("initial_state")
    if not isinstance(initial, Mapping) or initial.get("sha256") != FDR_INITIAL_STATE_SHA256:
        raise ValueError("BPDD initial-state SHA256 mismatch")
    identities = manifest.get("run_identities")
    if not isinstance(identities, Mapping):
        raise ValueError("BPDD run identities are missing")
    for variant in ("fdr", "fdr_bpdd"):
        for stage in ("screen", "formal"):
            key = f"{variant}_{stage}"
            expected = build_run_identity(source, stage=stage, variant=variant, seed=0)
            if identities.get(key) != expected:
                raise ValueError(f"BPDD run identity mismatch: {key}")
    return manifest


def current_source_identity(root: Path = ROOT) -> dict[str, str]:
    return fdr_cli.current_source_identity(root)


def validate_source_authority(
    manifest: Mapping[str, Any], root: Path = ROOT
) -> dict[str, str]:
    expected = manifest.get("source")
    if not isinstance(expected, Mapping):
        raise ValueError("BPDD source authority is missing")
    actual = current_source_identity(root)
    if dict(expected) != actual:
        raise ValueError(
            "checked-out source differs from BPDD authority: "
            f"expected={dict(expected)}, actual={actual}"
        )
    return actual


def validate_initial_state_file(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return fdr_cli.validate_initial_state_file(path, manifest)


def prepare_data_yaml(dataset_root: Path, stage: str, authority_root: Path) -> Path:
    return fdr_cli.prepare_data_yaml(dataset_root, stage, authority_root)


def build_settings(args: argparse.Namespace, data_yaml: Path) -> dict[str, Any]:
    if args.variant not in {"fdr", "fdr_bpdd"}:
        raise ValueError(f"unknown BPDD variant: {args.variant}")
    if args.stage not in {"screen", "formal"}:
        raise ValueError(f"unknown BPDD stage: {args.stage}")
    epochs = SCREEN_SCHEDULE_EPOCHS if args.stage == "screen" else FORMAL_EPOCHS
    name = args.name or f"{args.stage}-seed0-{args.variant}-bpdd-v1"
    settings = {
        **FROZEN_SETTINGS,
        "model": str(
            FDR_MODEL_CONFIG if args.variant == "fdr" else BPDD_MODEL_CONFIG
        ),
        "data": str(Path(data_yaml).resolve()),
        "epochs": epochs,
        "seed": 0,
        "project": str(Path(args.output_root).resolve()),
        "name": name,
        "exist_ok": False,
    }
    if args.resume is not None:
        settings["resume"] = str(Path(args.resume).resolve())
    return settings


def create_trainer(variant: str, settings: dict[str, Any], initial_state: Path):
    from src.rtdetr_fdr import FDRTrainer
    from src.rtdetr_fdr_bpdd import FDRBPDDTrainer

    common = {
        "overrides": settings,
        "initial_state_path": Path(initial_state).resolve(),
        "experiment_seed": 0,
    }
    if variant == "fdr":
        return FDRTrainer(**common)
    if variant == "fdr_bpdd":
        return FDRBPDDTrainer(**common)
    raise ValueError(f"unknown BPDD variant: {variant}")


def validate_resume_checkpoint(
    checkpoint: Path, expected_identity: Mapping[str, Any]
) -> Path:
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"BPDD resume checkpoint not found: {checkpoint}")
    if checkpoint.parent.name != "weights":
        raise ValueError("BPDD resume checkpoint must be in a weights directory")
    runtime_path = checkpoint.parent.parent / "bpdd-run.json"
    if not runtime_path.is_file():
        raise FileNotFoundError(f"BPDD resume authority not found: {runtime_path}")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    identity = runtime.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("BPDD resume authority is missing run_identity")
    validate_resume_authority(identity, expected_identity)
    return checkpoint


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().float().cpu().item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def _metric(trainer: Any, name: str) -> float | None:
    metrics = getattr(trainer, "metrics", {})
    return _number(metrics.get(name)) if isinstance(metrics, Mapping) else None


def _map75(trainer: Any) -> float | None:
    try:
        return _number(trainer.validator.metrics.box.map75)
    except AttributeError:
        return None


def _stock_losses(trainer: Any) -> dict[str, float | None]:
    values = getattr(trainer, "tloss", None)
    if isinstance(values, torch.Tensor):
        values = values.detach().float().reshape(-1).cpu().tolist()
    values = list(values) if values is not None else []
    return {
        name: _number(values[index]) if index < len(values) else None
        for index, name in enumerate(("loss_giou", "loss_class", "loss_bbox"))
    }


def _method_evidence(trainer: Any, variant: str) -> dict[str, Any]:
    model = _unwrap_model(trainer.model)
    fdr_losses = getattr(model, "last_fdr_losses", {})
    fdr = {
        name: _number(fdr_losses.get(name))
        for name in ("loss_fgl", "loss_fgl_aux", "loss_bbox_pre", "loss_giou_pre")
    }
    if variant == "fdr":
        return {
            **fdr,
            "loss_bpdd": None,
            "bpdd_active_edge_ratio": None,
            "bpdd_mean_reliability": None,
            "bpdd_mean_teacher_improvement": None,
        }
    stats = getattr(model, "last_bpdd_statistics", {})
    return {
        **fdr,
        "loss_bpdd": _number(fdr_losses.get("loss_bpdd")),
        "bpdd_active_edge_ratio": _number(stats.get("active_edge_ratio")),
        "bpdd_mean_reliability": _number(stats.get("mean_reliability")),
        "bpdd_mean_teacher_improvement": _number(
            stats.get("mean_teacher_improvement")
        ),
    }


def _evidence_record(trainer: Any, context: Mapping[str, Any]) -> dict[str, Any]:
    variant = str(context["variant"])
    norms = getattr(trainer, "last_gradient_norms", {})
    finite_value = norms.get("gradients_finite")
    if finite_value is None:
        numeric_norms = [
            _number(value)
            for key, value in norms.items()
            if key.endswith("gradient_norm")
        ]
        finite_value = bool(numeric_norms and all(value is not None for value in numeric_norms))
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
        **_method_evidence(trainer, variant),
        "gradient_norm": _number(norms.get("gradient_norm")),
        "fdr_gradient_norm": _number(norms.get("fdr_gradient_norm")),
        "gradients_finite": bool(finite_value),
        "cuda_peak_mib": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 2)
            if torch.cuda.is_available()
            else 0.0
        ),
    }


def _write_evidence_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVIDENCE_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in EVIDENCE_FIELDS} for row in rows)
    os.replace(temporary, path)


def write_epoch_evidence(
    trainer: Any, context: Mapping[str, Any]
) -> dict[str, Any]:
    run = Path(trainer.save_dir).resolve()
    jsonl = run / "bpdd-epochs.jsonl"
    result = _append_epoch_record(jsonl, _evidence_record(trainer, context))
    _write_evidence_csv(run / "bpdd-epochs.csv", _read_jsonl(jsonl))
    return result


def _checkpoint_for_epoch(trainer: Any) -> Path:
    checkpoint = (
        Path(trainer.save_dir).resolve()
        / "weights"
        / f"epoch{int(trainer.epoch)}.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"exact BPDD epoch checkpoint not saved: {checkpoint}")
    return checkpoint


def queue_epoch_publication(
    trainer: Any, context: Mapping[str, Any]
) -> dict[str, Any]:
    run = Path(trainer.save_dir).resolve()
    checkpoint = _checkpoint_for_epoch(trainer)
    queue_value = context.get("publication_queue")
    queue = Path(queue_value).resolve() if queue_value else run / "publication-queue.jsonl"
    record = {
        "run_id": str(context["run_identity"]["run_id"]),
        "variant": str(context["variant"]),
        "stage": str(context["stage"]),
        "completed_epoch": int(trainer.epoch) + 1,
        "status": "pending",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "artifacts": [
            str((run / "bpdd-epochs.jsonl").resolve()),
            str((run / "bpdd-epochs.csv").resolve()),
            str((run / "bpdd-run.json").resolve()),
        ],
    }
    return _append_queue_record(queue, record)


def finalize_epoch(trainer: Any, context: Mapping[str, Any]) -> dict[str, Any]:
    evidence = write_epoch_evidence(trainer, context)
    queued = queue_epoch_publication(trainer, context)
    if int(evidence["completed_epoch"]) != int(queued["completed_epoch"]):
        raise RuntimeError("BPDD evidence/publication queue mismatch")
    if context["stage"] == "screen" and int(evidence["completed_epoch"]) == SCREEN_CUTOFF_EPOCH:
        trainer.stop = True
    return {"evidence": evidence, "publication": queued}


def reset_peak_memory(_trainer: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _runtime_manifest(
    trainer: Any,
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    data_yaml: Path,
) -> None:
    destination = Path(trainer.save_dir).resolve() / "bpdd-run.json"
    payload = {
        "format_version": 1,
        "protocol_sha256": BPDD_PROTOCOL_SHA256,
        "fdr_protocol_sha256": BPDD_PROTOCOL["fdr_authority"]["protocol_sha256"],
        "source": manifest["source"],
        "run_identity": dict(run_identity),
        "initial_state": {
            "path": str(Path(args.initial_state).resolve()),
            "sha256": manifest["initial_state"]["sha256"],
        },
        "data": str(Path(data_yaml).resolve()),
        "screen_cutoff_epoch": SCREEN_CUTOFF_EPOCH if args.stage == "screen" else None,
        "publication_queue": (
            str(Path(args.publication_queue).resolve())
            if args.publication_queue is not None
            else str(destination.parent / "publication-queue.jsonl")
        ),
    }
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"changed BPDD run authority: {destination}")
        return
    write_json_atomic(destination, payload)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    args.protocol_manifest = Path(args.protocol_manifest).resolve()
    args.initial_state = Path(args.initial_state).resolve()
    args.dataset_root = Path(args.dataset_root).resolve()
    args.output_root = Path(args.output_root).resolve()
    if args.resume is not None:
        args.resume = Path(args.resume).resolve()
    if args.publication_queue is not None:
        args.publication_queue = Path(args.publication_queue).resolve()

    manifest = load_authority(args.protocol_manifest)
    validate_source_authority(manifest)
    state = validate_initial_state_file(args.initial_state, manifest)
    data_yaml = prepare_data_yaml(
        args.dataset_root,
        args.stage,
        args.output_root / "_bpdd-authority",
    )
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
        lambda current: _runtime_manifest(
            current,
            args=args,
            manifest=manifest,
            run_identity=identity,
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
