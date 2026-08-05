"""Train strict paired stock/FDR-only RT-DETR-L arms under one immutable protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
FDR_MODEL_CONFIG = ROOT / "configs" / "rtdetr-l-fdr.yaml"
sys.path.insert(0, str(ROOT))

from scripts.sync_experiment_checkpoint import write_json_atomic  # noqa: E402
from src.fdr_protocol import (  # noqa: E402
    FDR_PROTOCOL,
    FDR_PROTOCOL_SHA256,
    build_run_identity,
    canonical_json_bytes,
    public_state_sha256,
    validate_fdr_initial_state,
    validate_resume_authority,
)
from src.lpr_protocol import (  # noqa: E402
    CATEGORY_NAMES,
    dataset_signature,
    select_hashed_subset,
    subset_signature,
)


SCREEN_SCHEDULE_EPOCHS = 50
SCREEN_CUTOFF_EPOCH = 30
FORMAL_EPOCHS = 100

FROZEN_SETTINGS: dict[str, Any] = {
    "model": "rtdetr-l.yaml",
    "imgsz": 640,
    "batch": 8,
    "workers": 8,
    "device": "0",
    "pretrained": False,
    "cache": False,
    "amp": True,
    "deterministic": True,
    "nbs": 64,
    "nms": False,
    "max_det": 300,
    "save": True,
    "save_period": 1,
    "optimizer": "MuSGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.0,
    "cos_lr": False,
    "plots": True,
    "val": True,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "mixup": 0.0,
    "scale": 0.5,
    "translate": 0.1,
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "cutmix": 0.0,
    "copy_paste": 0.0,
}

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
    "gradient_norm",
    "fdr_gradient_norm",
    "cuda_peak_mib",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one strict seed0 control/FDR RT-DETR-L arm."
    )
    parser.add_argument("--variant", choices=("control", "fdr"), required=True)
    parser.add_argument("--stage", choices=("screen", "formal"), required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--publication-queue",
        type=Path,
        help="Optional append-only JSONL outbox consumed by an external publisher.",
    )
    parser.add_argument("--name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate authority/data and print the frozen launch without creating a trainer.",
    )
    return parser


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_text(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_epoch_record(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    rows = _read_jsonl(path)
    epoch = int(record["completed_epoch"])
    same = [row for row in rows if int(row["completed_epoch"]) == epoch]
    if same:
        if len(same) != 1 or same[0] != record:
            raise ValueError(f"changed evidence for completed epoch {epoch}")
        return same[0]
    expected = int(rows[-1]["completed_epoch"]) + 1 if rows else 1
    if epoch != expected:
        raise ValueError(f"epoch evidence gap: expected {expected}, got {epoch}")
    rows.append(record)
    _atomic_text(
        path,
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
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
            raise ValueError(f"changed publication queue entry for {key[0]} epoch {key[1]}")
        return same[0]
    rows.append(record)
    _atomic_text(
        path,
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for row in rows
        ),
    )
    return record


def load_authority(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FDR protocol manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("FDR protocol manifest format must be 1")
    manifest_hash = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    actual_manifest_hash = hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest().upper()
    if manifest_hash != actual_manifest_hash:
        raise ValueError("FDR protocol manifest SHA256 mismatch")
    if manifest.get("protocol") != FDR_PROTOCOL:
        raise ValueError("FDR protocol payload does not match frozen authority")
    if manifest.get("protocol_sha256") != FDR_PROTOCOL_SHA256:
        raise ValueError("FDR protocol SHA256 does not match frozen authority")
    source = manifest.get("source")
    identities = manifest.get("run_identities")
    if not isinstance(source, Mapping) or not isinstance(identities, Mapping):
        raise ValueError("FDR protocol source/run identities are missing")
    if manifest.get("source_sha256") != public_state_sha256(source):
        raise ValueError("FDR protocol source SHA256 mismatch")
    for variant in ("control", "fdr"):
        for stage in ("screen", "formal"):
            key = f"{variant}_{stage}"
            expected = build_run_identity(
                source, stage=stage, variant=variant, seed=0
            )
            if identities.get(key) != expected:
                raise ValueError(f"FDR run identity mismatch: {key}")
    return manifest


def current_source_identity(root: Path = ROOT) -> dict[str, str]:
    """Fingerprint the exact checked-out commit and every tracked source byte."""
    root = Path(root).resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    digest = hashlib.sha256()
    for raw in tracked.split(b"\0"):
        if not raw:
            continue
        path = root / raw.decode("utf-8")
        digest.update(raw + b"\0")
        digest.update(path.read_bytes())
    return {"git_commit": commit, "tree_sha256": digest.hexdigest().upper()}


def validate_source_authority(
    manifest: Mapping[str, Any], root: Path = ROOT
) -> dict[str, str]:
    """Fail closed if runtime source differs from the immutable manifest."""
    expected = manifest.get("source")
    if not isinstance(expected, Mapping):
        raise ValueError("FDR protocol source authority is missing")
    actual = current_source_identity(root)
    if dict(expected) != actual:
        raise ValueError(
            "checked-out source differs from FDR authority: "
            f"expected={dict(expected)}, actual={actual}"
        )
    return actual


def validate_initial_state_file(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path).resolve()
    record = manifest.get("initial_state")
    if not isinstance(record, Mapping):
        raise ValueError("FDR initial-state authority is missing")
    expected_path = Path(str(record.get("path", ""))).resolve()
    if path != expected_path:
        raise ValueError(
            f"initial-state path differs from manifest: expected={expected_path}, actual={path}"
        )
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"FDR initial-state artifact not found: {path}")
    digest = _file_sha256(path)
    if digest != str(record.get("sha256", "")).upper():
        raise ValueError(
            f"initial-state SHA256 mismatch: expected={record.get('sha256')}, actual={digest}"
        )
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    validate_fdr_initial_state(artifact)
    return {**dict(record), "path": str(path), "sha256": digest}


def _data_payload(dataset_root: Path, train: Path | str) -> dict[str, Any]:
    return {
        "path": str(dataset_root),
        "train": str(train),
        "val": str((dataset_root / "images" / "val").resolve()),
        "names": list(CATEGORY_NAMES),
        "nc": len(CATEGORY_NAMES),
    }


def prepare_data_yaml(dataset_root: Path, stage: str, authority_root: Path) -> Path:
    dataset_root = Path(dataset_root).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"VisDrone dataset root not found: {dataset_root}")
    signature = dataset_signature(dataset_root)
    expected = FDR_PROTOCOL["dataset"]
    if signature.get("sha256") != expected["sha256"]:
        raise ValueError(
            "VisDrone dataset SHA256 mismatch: "
            f"expected={expected['sha256']}, actual={signature.get('sha256')}"
        )
    train_images = sorted((dataset_root / "images" / "train").glob("*.jpg"))
    val_images = sorted((dataset_root / "images" / "val").glob("*.jpg"))
    if len(train_images) != int(expected["train_images"]) or len(val_images) != int(
        expected["val_images"]
    ):
        raise ValueError(
            f"VisDrone image-count mismatch: train={len(train_images)}, val={len(val_images)}"
        )

    authority_root = Path(authority_root).resolve()
    authority_root.mkdir(parents=True, exist_ok=True)
    if stage == "screen":
        selected = select_hashed_subset(train_images, root=dataset_root, fraction=0.10)
        if len(selected) != int(expected["screen_train_images"]):
            raise ValueError(f"screen subset count mismatch: {len(selected)}")
        digest = subset_signature(selected, root=dataset_root)
        if digest != expected["screen_sha256"]:
            raise ValueError(
                f"screen subset SHA256 mismatch: expected={expected['screen_sha256']}, actual={digest}"
            )
        train_list = authority_root / "screen-train.txt"
        _atomic_text(train_list, "".join(f"{path.resolve()}\n" for path in selected))
        train_source: Path | str = train_list.resolve()
    elif stage == "formal":
        train_source = (dataset_root / "images" / "train").resolve()
    else:
        raise ValueError(f"unknown FDR stage: {stage}")

    destination = authority_root / f"{stage}-data.yaml"
    _atomic_text(
        destination,
        json.dumps(
            _data_payload(dataset_root, train_source),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
    )
    return destination


def build_settings(args: argparse.Namespace, data_yaml: Path) -> dict[str, Any]:
    if args.variant not in {"control", "fdr"}:
        raise ValueError(f"unknown FDR variant: {args.variant}")
    if args.stage not in {"screen", "formal"}:
        raise ValueError(f"unknown FDR stage: {args.stage}")
    epochs = SCREEN_SCHEDULE_EPOCHS if args.stage == "screen" else FORMAL_EPOCHS
    name = args.name or f"{args.stage}-seed0-{args.variant}-fdr-v1"
    settings = {
        **FROZEN_SETTINGS,
        "model": (
            str(FDR_MODEL_CONFIG)
            if args.variant == "fdr"
            else FROZEN_SETTINGS["model"]
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


def _load_trainer_types():
    from src.rtdetr_fdr import FDRControlTrainer, FDRTrainer

    return FDRControlTrainer, FDRTrainer


def create_trainer(variant: str, settings: dict[str, Any], initial_state: Path):
    control_type, fdr_type = _load_trainer_types()
    common = {
        "overrides": settings,
        "initial_state_path": Path(initial_state).resolve(),
    }
    if variant == "control":
        return control_type(**common)
    if variant == "fdr":
        return fdr_type(**common, experiment_seed=0)
    raise ValueError(f"unknown FDR variant: {variant}")


def validate_resume_checkpoint(
    checkpoint: Path, expected_identity: Mapping[str, Any]
) -> Path:
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
    if checkpoint.parent.name != "weights":
        raise ValueError("resume checkpoint must be inside its run weights directory")
    runtime_path = checkpoint.parent.parent / "fdr-run.json"
    if not runtime_path.is_file():
        raise FileNotFoundError(f"resume authority not found: {runtime_path}")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    identity = runtime.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("resume authority is missing run_identity")
    validate_resume_authority(identity, expected_identity)
    return checkpoint


def _unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


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


def _metric(trainer: Any, name: str) -> float | None:
    metrics = getattr(trainer, "metrics", {})
    if isinstance(metrics, Mapping):
        return _number(metrics.get(name))
    return None


def _map75(trainer: Any) -> float | None:
    try:
        return _number(trainer.validator.metrics.box.map75)
    except AttributeError:
        return None


def _stock_losses(trainer: Any) -> dict[str, float | None]:
    values = getattr(trainer, "tloss", None)
    if isinstance(values, torch.Tensor):
        values = values.detach().float().reshape(-1).cpu().tolist()
    elif values is not None:
        values = list(values)
    else:
        values = []
    names = ("loss_giou", "loss_class", "loss_bbox")
    return {
        name: _number(values[index]) if index < len(values) else None
        for index, name in enumerate(names)
    }


def _fdr_losses(trainer: Any, variant: str) -> dict[str, float | None]:
    names = ("loss_fgl", "loss_fgl_aux", "loss_bbox_pre", "loss_giou_pre")
    if variant == "control":
        return {name: None for name in names}
    model = _unwrap_model(trainer.model)
    values = getattr(model, "last_fdr_losses", {})
    return {name: _number(values.get(name)) for name in names}


def _evidence_record(trainer: Any, context: Mapping[str, Any]) -> dict[str, Any]:
    epoch = int(trainer.epoch) + 1
    variant = str(context["variant"])
    norms = getattr(trainer, "last_gradient_norms", {})
    return {
        "completed_epoch": epoch,
        "variant": variant,
        "stage": str(context["stage"]),
        "run_id": str(context["run_identity"]["run_id"]),
        "precision": _metric(trainer, "metrics/precision(B)"),
        "recall": _metric(trainer, "metrics/recall(B)"),
        "map50": _metric(trainer, "metrics/mAP50(B)"),
        "map": _metric(trainer, "metrics/mAP50-95(B)"),
        "map75": _map75(trainer),
        **_stock_losses(trainer),
        **_fdr_losses(trainer, variant),
        "gradient_norm": _number(norms.get("gradient_norm")),
        "fdr_gradient_norm": (
            _number(norms.get("fdr_gradient_norm")) if variant == "fdr" else None
        ),
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
    jsonl = run / "fdr-epochs.jsonl"
    record = _evidence_record(trainer, context)
    result = _append_epoch_record(jsonl, record)
    _write_evidence_csv(run / "fdr-epochs.csv", _read_jsonl(jsonl))
    return result


def _checkpoint_for_epoch(trainer: Any) -> Path:
    checkpoint = (
        Path(trainer.save_dir).resolve()
        / "weights"
        / f"epoch{int(trainer.epoch)}.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"exact epoch checkpoint was not saved: {checkpoint}")
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
            str((run / "fdr-epochs.jsonl").resolve()),
            str((run / "fdr-epochs.csv").resolve()),
            str((run / "fdr-run.json").resolve()),
        ],
    }
    return _append_queue_record(queue, record)


def finalize_epoch(trainer: Any, context: Mapping[str, Any]) -> dict[str, Any]:
    evidence = write_epoch_evidence(trainer, context)
    queued = queue_epoch_publication(trainer, context)
    if int(queued["completed_epoch"]) != int(evidence["completed_epoch"]):
        raise RuntimeError("epoch evidence/publication queue mismatch")
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
    destination = Path(trainer.save_dir).resolve() / "fdr-run.json"
    payload = {
        "format_version": 1,
        "protocol_sha256": FDR_PROTOCOL_SHA256,
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
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"changed FDR run authority: {destination}")
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
    authority_root = args.output_root / "_fdr-authority"
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
