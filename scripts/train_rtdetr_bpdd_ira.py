"""Train the formal seed0 FDR+BPDD+IRA RT-DETR-L arm."""

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

from scripts import train_rtdetr_bpdd as bpdd_cli  # noqa: E402
from scripts import train_rtdetr_fdr as fdr_cli  # noqa: E402
from scripts.sync_experiment_checkpoint import write_json_atomic  # noqa: E402
from src.bpdd_formal_evaluation import state_sha256  # noqa: E402

from src.bpdd_ira_protocol import (  # noqa: E402
    BPDD_IRA_PROTOCOL,
    BPDD_IRA_PROTOCOL_SHA256,
    FDR_INITIAL_STATE_SHA256,
    build_run_identity,
    canonical_json_bytes,
    public_state_sha256,
    validate_resume_authority,
)


MODEL_CONFIG = (ROOT / "configs" / "rtdetr-l-fdr-bpdd-ira.yaml").resolve()
FORMAL_EPOCHS = 100
VARIANT = "fdr_bpdd_ira"
STAGE = "formal"
FROZEN_SETTINGS = dict(fdr_cli.FROZEN_SETTINGS)

EVIDENCE_FIELDS = (
    *bpdd_cli.EVIDENCE_FIELDS,
    "ira_gradient_norm",
    "ira_residual_scale",
    "checkpoint_sha256",
    "ema_state_sha256",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the immutable formal seed0 FDR+BPDD+IRA arm."
    )
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
            raise ValueError(f"changed BPDD IRA evidence for completed epoch {epoch}")
        return same[0]
    expected = int(rows[-1]["completed_epoch"]) + 1 if rows else 1
    if epoch != expected:
        raise ValueError(f"BPDD IRA evidence gap: expected {expected}, got {epoch}")
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
            raise ValueError(f"changed BPDD IRA publication entry for {key}")
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
        raise FileNotFoundError(f"BPDD IRA protocol manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("BPDD IRA protocol manifest format must be 1")
    unhashed = dict(manifest)
    manifest_hash = unhashed.pop("manifest_sha256", None)
    actual_hash = hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest().upper()
    if manifest_hash != actual_hash:
        raise ValueError("BPDD IRA protocol manifest SHA256 mismatch")
    if manifest.get("protocol") != BPDD_IRA_PROTOCOL:
        raise ValueError("BPDD IRA protocol payload does not match frozen authority")
    if manifest.get("protocol_sha256") != BPDD_IRA_PROTOCOL_SHA256:
        raise ValueError("BPDD IRA protocol SHA256 mismatch")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("BPDD IRA source authority is missing")
    if manifest.get("source_sha256") != public_state_sha256(source):
        raise ValueError("BPDD IRA source SHA256 mismatch")
    initial = manifest.get("initial_state")
    if not isinstance(initial, Mapping) or initial.get("sha256") != FDR_INITIAL_STATE_SHA256:
        raise ValueError("BPDD IRA initial-state SHA256 mismatch")
    identities = manifest.get("run_identities")
    if not isinstance(identities, Mapping) or set(identities) != {
        "fdr_bpdd_ira_formal"
    }:
        raise ValueError("BPDD IRA run identities must contain only the formal arm")
    expected = build_run_identity(source, stage=STAGE, variant=VARIANT, seed=0)
    if identities["fdr_bpdd_ira_formal"] != expected:
        raise ValueError("BPDD IRA formal run identity mismatch")
    return manifest


def current_source_identity(root: Path = ROOT) -> dict[str, str]:
    return fdr_cli.current_source_identity(root)


def validate_source_authority(
    manifest: Mapping[str, Any], root: Path = ROOT
) -> dict[str, str]:
    expected = manifest.get("source")
    if not isinstance(expected, Mapping):
        raise ValueError("BPDD IRA source authority is missing")
    actual = current_source_identity(root)
    if dict(expected) != actual:
        raise ValueError(
            "checked-out source differs from BPDD IRA authority: "
            f"expected={dict(expected)}, actual={actual}"
        )
    return actual


def validate_initial_state_file(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return fdr_cli.validate_initial_state_file(path, manifest)


def prepare_data_yaml(dataset_root: Path, authority_root: Path) -> Path:
    return fdr_cli.prepare_data_yaml(dataset_root, STAGE, authority_root)


def build_settings(args: argparse.Namespace, data_yaml: Path) -> dict[str, Any]:
    settings = {
        **FROZEN_SETTINGS,
        "model": str(MODEL_CONFIG),
        "data": str(Path(data_yaml).resolve()),
        "epochs": FORMAL_EPOCHS,
        "seed": 0,
        "project": str(Path(args.output_root).resolve()),
        "name": args.name or "formal-seed0-fdr_bpdd_ira-v1",
        "exist_ok": False,
    }
    if args.resume is not None:
        settings["resume"] = str(Path(args.resume).resolve())
    return settings


def _load_trainer_type():
    from src.rtdetr_fdr_bpdd_ira import FDRBPDDIRATrainer

    return FDRBPDDIRATrainer


def create_trainer(settings: dict[str, Any], initial_state: Path):
    return _load_trainer_type()(
        overrides=settings,
        initial_state_path=Path(initial_state).resolve(),
        experiment_seed=0,
    )


def validate_resume_checkpoint(
    checkpoint: Path, expected_identity: Mapping[str, Any]
) -> Path:
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"BPDD IRA resume checkpoint not found: {checkpoint}")
    if checkpoint.parent.name != "weights":
        raise ValueError("BPDD IRA resume checkpoint must be in a weights directory")
    runtime_path = checkpoint.parent.parent / "bpdd-ira-run.json"
    if not runtime_path.is_file():
        raise FileNotFoundError(f"BPDD IRA resume authority not found: {runtime_path}")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    identity = runtime.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("BPDD IRA resume authority is missing run_identity")
    validate_resume_authority(identity, expected_identity)

    queue_value = runtime.get("publication_queue")
    if not isinstance(queue_value, str):
        raise ValueError("BPDD IRA resume authority is missing publication_queue")
    queue = Path(queue_value)
    rows = _read_jsonl(queue)
    registered = [
        row
        for row in rows
        if Path(str(row.get("checkpoint", ""))).resolve() == checkpoint
        and row.get("run_id") == expected_identity.get("run_id")
    ]
    if len(registered) != 1:
        raise ValueError("resume checkpoint is not uniquely registered in publication queue")
    if registered[0].get("checkpoint_sha256") != _file_sha256(checkpoint):
        raise ValueError("BPDD IRA resume checkpoint SHA256 mismatch")
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


def _checkpoint_ema_state_sha256(checkpoint: Path) -> str:
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(artifact, Mapping) or artifact.get("ema") is None:
        raise RuntimeError("BPDD IRA epoch checkpoint requires saved EMA state")
    ema = artifact["ema"]
    state = ema if isinstance(ema, Mapping) else ema.state_dict()
    if not isinstance(state, Mapping):
        raise TypeError("BPDD IRA checkpoint EMA does not expose a state mapping")
    return state_sha256(state)


def _checkpoint_for_epoch(trainer: Any) -> Path:
    checkpoint = (
        Path(trainer.save_dir).resolve()
        / "weights"
        / f"epoch{int(trainer.epoch)}.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"exact BPDD IRA epoch checkpoint not saved: {checkpoint}")
    return checkpoint


def _ira_residual_scale(model: Any) -> float | None:
    ira = getattr(_unwrap_model(model), "ira", None)
    return _number(getattr(ira, "residual_scale", None))


def _evidence_record(trainer: Any, context: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = _checkpoint_for_epoch(trainer)
    model = _unwrap_model(trainer.model)
    fdr_losses = getattr(model, "last_fdr_losses", {})
    stats = getattr(model, "last_bpdd_statistics", {})
    norms = getattr(trainer, "last_gradient_norms", {})
    numeric_norms = [
        _number(norms.get(name))
        for name in ("gradient_norm", "fdr_gradient_norm", "ira_gradient_norm")
    ]
    return {
        "completed_epoch": int(trainer.epoch) + 1,
        "variant": VARIANT,
        "stage": STAGE,
        "run_id": str(context["run_identity"]["run_id"]),
        "precision": bpdd_cli._metric(trainer, "metrics/precision(B)"),
        "recall": bpdd_cli._metric(trainer, "metrics/recall(B)"),
        "map50": bpdd_cli._metric(trainer, "metrics/mAP50(B)"),
        "map": bpdd_cli._metric(trainer, "metrics/mAP50-95(B)"),
        "map75": bpdd_cli._map75(trainer),
        **bpdd_cli._stock_losses(trainer),
        "loss_fgl": _number(fdr_losses.get("loss_fgl")),
        "loss_fgl_aux": _number(fdr_losses.get("loss_fgl_aux")),
        "loss_bbox_pre": _number(fdr_losses.get("loss_bbox_pre")),
        "loss_giou_pre": _number(fdr_losses.get("loss_giou_pre")),
        "loss_bpdd": _number(fdr_losses.get("loss_bpdd")),
        "bpdd_active_edge_ratio": _number(stats.get("active_edge_ratio")),
        "bpdd_mean_reliability": _number(stats.get("mean_reliability")),
        "bpdd_mean_teacher_improvement": _number(
            stats.get("mean_teacher_improvement")
        ),
        "bpdd_mixture_beats_final_ratio": _number(
            stats.get("mixture_beats_final_ratio")
        ),
        "bpdd_mean_mixture_advantage_over_final": _number(
            stats.get("mean_mixture_advantage_over_final")
        ),
        "gradient_norm": numeric_norms[0],
        "fdr_gradient_norm": numeric_norms[1],
        "gradients_finite": all(value is not None for value in numeric_norms),
        "cuda_peak_mib": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 2)
            if torch.cuda.is_available()
            else 0.0
        ),
        "ira_gradient_norm": numeric_norms[2],
        "ira_residual_scale": _ira_residual_scale(trainer.model),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "ema_state_sha256": _checkpoint_ema_state_sha256(checkpoint),
    }


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


def write_epoch_evidence(
    trainer: Any, context: Mapping[str, Any]
) -> dict[str, Any]:
    run = Path(trainer.save_dir).resolve()
    jsonl = run / "fdr-epochs.jsonl"
    result = _append_epoch_record(jsonl, _evidence_record(trainer, context))
    _write_evidence_csv(run / "fdr-epochs.csv", _read_jsonl(jsonl))
    return result


def queue_epoch_publication(
    trainer: Any,
    context: Mapping[str, Any],
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run = Path(trainer.save_dir).resolve()
    checkpoint = _checkpoint_for_epoch(trainer)
    evidence = evidence or _evidence_record(trainer, context)
    queue_value = context.get("publication_queue")
    queue = Path(queue_value).resolve() if queue_value else run / "publication-queue.jsonl"
    record = {
        "run_id": str(context["run_identity"]["run_id"]),
        "variant": VARIANT,
        "stage": STAGE,
        "completed_epoch": int(trainer.epoch) + 1,
        "status": "pending",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": str(evidence["checkpoint_sha256"]),
        "ema_state_sha256": str(evidence["ema_state_sha256"]),
        "artifacts": [
            str((run / "fdr-epochs.jsonl").resolve()),
            str((run / "fdr-epochs.csv").resolve()),
            str((run / "bpdd-ira-run.json").resolve()),
            str((run / "optimizer-evidence.jsonl").resolve()),
        ],
    }
    return _append_queue_record(queue, record)


def finalize_epoch(trainer: Any, context: Mapping[str, Any]) -> dict[str, Any]:
    evidence = write_epoch_evidence(trainer, context)
    queued = queue_epoch_publication(trainer, context, evidence)
    if int(evidence["completed_epoch"]) != int(queued["completed_epoch"]):
        raise RuntimeError("BPDD IRA evidence/publication queue mismatch")
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
    destination = Path(trainer.save_dir).resolve() / "bpdd-ira-run.json"
    queue = (
        Path(args.publication_queue).resolve()
        if args.publication_queue is not None
        else destination.parent / "publication-queue.jsonl"
    )
    payload = {
        "format_version": 1,
        "protocol_sha256": BPDD_IRA_PROTOCOL_SHA256,
        "source": manifest["source"],
        "run_identity": dict(run_identity),
        "initial_state": {
            "path": str(Path(args.initial_state).resolve()),
            "sha256": manifest["initial_state"]["sha256"],
        },
        "data": str(Path(data_yaml).resolve()),
        "publication_queue": str(queue.resolve()),
    }
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"changed BPDD IRA run authority: {destination}")
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
    initial_state = validate_initial_state_file(args.initial_state, manifest)
    data_yaml = prepare_data_yaml(
        args.dataset_root, args.output_root / "_bpdd-ira-authority"
    )
    identity = manifest["run_identities"]["fdr_bpdd_ira_formal"]
    if args.resume is not None:
        validate_resume_checkpoint(args.resume, identity)
    settings = build_settings(args, data_yaml)
    summary = {
        "status": "dry-run-passed" if args.dry_run else "launching",
        "variant": VARIANT,
        "stage": STAGE,
        "run_identity": identity,
        "initial_state": initial_state,
        "settings": settings,
        "publication_queue": (
            str(args.publication_queue)
            if args.publication_queue is not None
            else "<run_dir>/publication-queue.jsonl"
        ),
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True))
        return summary

    trainer = create_trainer(settings, args.initial_state)
    context = {
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
    return {
        **summary,
        "status": "training-finished",
        "save_dir": str(trainer.save_dir),
    }


def main() -> None:
    execute(build_parser().parse_args())


if __name__ == "__main__":
    main()
