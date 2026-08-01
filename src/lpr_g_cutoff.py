"""Create an auditable exact-prefix view of an overshot LPR-G screen run."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

from src.github_checkpoint_sync import checkpoint_metadata
from src.lpr_g_publication import PublicationLedger


EPOCH_JSONL_FILES = {
    "lpr_g_diagnostics.jsonl": "epoch",
    "common_state_audit.jsonl": "epoch",
}
STATIC_FILES = ("lpr_g_protocol.json", "args.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing cutoff evidence: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _require_contiguous(records: list[dict[str, Any]], field: str, label: str) -> list[int]:
    epochs = [int(record[field]) for record in records]
    if epochs != list(range(1, len(records) + 1)):
        raise ValueError(f"{label} is not contiguous: {epochs}")
    return epochs


def _verify_existing_view(destination: Path, source: Path, cutoff_epoch: int) -> dict[str, Any]:
    manifest_path = destination / "cutoff-view.json"
    if not manifest_path.is_file():
        raise FileExistsError(f"refusing changed cutoff view without manifest: {destination}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity_matches = (
        manifest.get("format_version") == 1
        and manifest.get("source_run") == str(source.resolve())
        and int(manifest.get("cutoff_epoch", -1)) == cutoff_epoch
    )
    output_files = manifest.get("output_files", {})
    files_match = isinstance(output_files, dict) and all(
        (destination / relative).is_file()
        and _sha256(destination / relative) == expected_sha
        for relative, expected_sha in output_files.items()
    )
    if not identity_matches or not output_files or not files_match:
        raise FileExistsError(f"refusing changed cutoff view: {destination}")
    return manifest


def materialize_cutoff_view(
    source_run: str | Path,
    destination_run: str | Path,
    *,
    cutoff_epoch: int,
) -> dict[str, Any]:
    """Atomically preserve only verified epochs 1..cutoff from an overshot run."""
    source = Path(source_run).resolve()
    destination = Path(destination_run).resolve()
    if cutoff_epoch < 1:
        raise ValueError("cutoff epoch must be positive")
    if source == destination:
        raise ValueError("cutoff source and destination must differ")
    if destination.exists():
        return _verify_existing_view(destination, source, cutoff_epoch)

    ledger = PublicationLedger(source / "publication-ledger.jsonl").records()
    ledger_epochs = [int(record["completed_epoch"]) for record in ledger]
    if len(ledger) < cutoff_epoch:
        raise ValueError(
            f"source has no verified epoch {cutoff_epoch}: {ledger_epochs}"
        )
    expected_source_epochs = list(range(1, len(ledger) + 1))
    if ledger_epochs != expected_source_epochs:
        raise ValueError(f"source publication ledger is not contiguous: {ledger_epochs}")

    results_path = source / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"missing cutoff evidence: {results_path}")
    with results_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        results = list(reader)
    if not fieldnames:
        raise ValueError("results.csv has no header")
    result_epochs = _require_contiguous(results, "epoch", "results.csv")
    if result_epochs != ledger_epochs:
        raise ValueError("results.csv and publication ledger cover different epochs")

    epoch_records: dict[str, list[dict[str, Any]]] = {}
    for name, field in EPOCH_JSONL_FILES.items():
        rows = _read_jsonl(source / name)
        if _require_contiguous(rows, field, name) != ledger_epochs:
            raise ValueError(f"{name} and publication ledger cover different epochs")
        epoch_records[name] = rows

    checkpoint = source / "weights" / f"epoch{cutoff_epoch - 1}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"cutoff checkpoint is missing: {checkpoint}")
    checkpoint_info = checkpoint_metadata(checkpoint)
    expected_sha = str(ledger[cutoff_epoch - 1]["checkpoint"]["sha256"])
    if checkpoint_info.completed_epoch != cutoff_epoch or checkpoint_info.sha256 != expected_sha:
        raise ValueError(
            "cutoff checkpoint SHA256 or completed epoch does not match verified ledger"
        )
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    optimizer_updates = int(checkpoint_payload.get("updates", -1))
    if optimizer_updates < 1:
        raise ValueError("cutoff checkpoint has no positive optimizer update count")
    optimizer = _read_jsonl(source / "optimizer-evidence.jsonl")
    attempts = [int(record.get("optimizer_attempt", -1)) for record in optimizer]
    if attempts != list(range(1, len(optimizer) + 1)) or len(optimizer) < optimizer_updates:
        raise ValueError("optimizer evidence does not cover the cutoff checkpoint updates")

    source_files = {
        str(path.relative_to(source)).replace("\\", "/"): _sha256(path)
        for path in (
            results_path,
            source / "lpr_g_diagnostics.jsonl",
            source / "common_state_audit.jsonl",
            source / "publication-ledger.jsonl",
            source / "optimizer-evidence.jsonl",
            checkpoint,
        )
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.cutoff-{cutoff_epoch}.", dir=destination.parent)
    )
    try:
        with (temporary / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results[:cutoff_epoch])
        for name, rows in epoch_records.items():
            _write_jsonl(temporary / name, rows[:cutoff_epoch])
        _write_jsonl(temporary / "publication-ledger.jsonl", ledger[:cutoff_epoch])
        _write_jsonl(temporary / "optimizer-evidence.jsonl", optimizer[:optimizer_updates])
        for name in STATIC_FILES:
            source_path = source / name
            if source_path.is_file():
                shutil.copy2(source_path, temporary / name)
                source_files[name] = _sha256(source_path)
        weights = temporary / "weights"
        weights.mkdir()
        shutil.copy2(checkpoint, weights / checkpoint.name)

        output_paths = [path for path in temporary.rglob("*") if path.is_file()]
        output_files = {
            str(path.relative_to(temporary)).replace("\\", "/"): _sha256(path)
            for path in sorted(output_paths)
        }
        manifest = {
            "format_version": 1,
            "design": "50-epoch schedule / exact cutoff view",
            "source_run": str(source),
            "source_ledger_epochs": ledger_epochs,
            "cutoff_epoch": cutoff_epoch,
            "source_files": source_files,
            "output_files": output_files,
            "checkpoint": {
                "name": checkpoint.name,
                "sha256": checkpoint_info.sha256,
                "completed_epoch": checkpoint_info.completed_epoch,
                "optimizer_updates": optimizer_updates,
            },
        }
        (temporary / "cutoff-view.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
