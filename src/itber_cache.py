"""Immutable sharded evidence cache for the I-TBER P0-P3 Probe."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import torch


CACHE_FORMAT_VERSION = 1
DESIGN_VERSION = "itber-v1.1"
REQUIRED_RECORD_TENSORS = (
    "hidden",
    "box_l2",
    "box_l1",
    "stock_boxes",
    "stock_scores",
    "f3",
    "target_edges",
    "match_source",
    "match_target",
)


class CacheViolation(ValueError):
    """An immutable cache authority, shard, or record is invalid."""

    def __init__(self, violations: Mapping[str, Any]) -> None:
        self.violations = dict(violations)
        super().__init__("I-TBER cache violation: " + ", ".join(sorted(self.violations)))


@dataclass(frozen=True)
class EvidenceShard:
    split: str
    path: str
    start_index: int
    end_index: int
    count: int
    bytes: int
    sha256: str


@dataclass(frozen=True)
class CacheManifest:
    format_version: int
    design_version: str
    complete: bool
    authority: dict[str, str]
    split_counts: dict[str, int]
    shard_size: int
    shards: tuple[EvidenceShard, ...]


@dataclass(frozen=True)
class EvidenceCache:
    manifest: CacheManifest
    records: dict[str, tuple[dict[str, Any], ...]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _normalized_authority(authority: Mapping[str, Any]) -> dict[str, str]:
    required = ("baseline_sha256", "dataset_sha256", "category_sha256", "source_commit")
    missing = [name for name in required if not authority.get(name)]
    if missing:
        raise CacheViolation({"authority.missing": missing})
    normalized = {name: str(authority[name]) for name in required}
    for name in ("baseline_sha256", "dataset_sha256", "category_sha256"):
        normalized[name] = normalized[name].upper()
    normalized["source_commit"] = normalized["source_commit"].lower()
    if len(normalized["source_commit"]) != 40:
        raise CacheViolation({"authority.source_commit": normalized["source_commit"]})
    return normalized


def _validate_record(record: Mapping[str, Any], expected_index: int, split: str) -> None:
    violations: dict[str, Any] = {}
    if record.get("index") != expected_index:
        violations[f"{split}.index"] = {
            "expected": expected_index,
            "actual": record.get("index"),
        }
    if not isinstance(record.get("image_id"), str) or not record.get("image_id"):
        violations[f"{split}.image_id"] = record.get("image_id")
    for name in REQUIRED_RECORD_TENSORS:
        if not isinstance(record.get(name), torch.Tensor):
            violations[f"{split}.{expected_index}.{name}"] = "not_tensor"
    if not violations:
        source = record["match_source"]
        target = record["match_target"]
        if source.ndim != 1 or target.ndim != 1 or len(source) != len(target):
            violations[f"{split}.{expected_index}.match"] = "invalid_shape"
        if record["stock_boxes"].ndim != 2 or record["stock_boxes"].shape[-1] != 4:
            violations[f"{split}.{expected_index}.stock_boxes"] = "invalid_shape"
        if record["f3"].ndim != 3:
            violations[f"{split}.{expected_index}.f3"] = "invalid_shape"
    if violations:
        raise CacheViolation(violations)


def _prepare_records(records: Sequence[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for expected_index, record in enumerate(records):
        _validate_record(record, expected_index, split)
        copied = {"index": expected_index, "image_id": str(record["image_id"])}
        for name in REQUIRED_RECORD_TENSORS:
            copied[name] = record[name].detach().cpu().contiguous().clone()
        prepared.append(copied)
    return prepared


def _safe_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise CacheViolation({"shard.path": relative})
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise CacheViolation({"shard.path": relative})
    candidate = root.joinpath(*parsed.parts)
    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise CacheViolation({"shard.path": relative}) from error
    return candidate


def _manifest_payload(manifest: CacheManifest) -> dict[str, Any]:
    return {
        "format_version": manifest.format_version,
        "design_version": manifest.design_version,
        "complete": manifest.complete,
        "authority": manifest.authority,
        "split_counts": manifest.split_counts,
        "shard_size": manifest.shard_size,
        "shards": [asdict(shard) for shard in manifest.shards],
    }


def write_evidence_cache(
    root: str | Path,
    *,
    train_records: Sequence[Mapping[str, Any]],
    val_records: Sequence[Mapping[str, Any]],
    authority: Mapping[str, Any],
    shard_size: int,
) -> CacheManifest:
    """Write all shards first and atomically publish the completion manifest last."""
    if shard_size < 1:
        raise ValueError("shard size must be positive")
    normalized_authority = _normalized_authority(authority)
    prepared = {
        "train": _prepare_records(train_records, "train"),
        "val": _prepare_records(val_records, "val"),
    }
    train_ids = {record["image_id"] for record in prepared["train"]}
    val_ids = {record["image_id"] for record in prepared["val"]}
    overlap = sorted(train_ids & val_ids)
    if overlap:
        raise CacheViolation({"split.image_id_overlap": overlap[:10]})

    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty cache root: {root}")
    shard_root = root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shards: list[EvidenceShard] = []
    for split in ("train", "val"):
        records = prepared[split]
        for shard_index, start in enumerate(range(0, len(records), shard_size)):
            selected = records[start : start + shard_size]
            relative = f"shards/{split}-{shard_index:05d}.pt"
            destination = root / relative
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            torch.save(
                {
                    "format_version": CACHE_FORMAT_VERSION,
                    "design_version": DESIGN_VERSION,
                    "split": split,
                    "start_index": start,
                    "records": selected,
                },
                temporary,
            )
            os.replace(temporary, destination)
            shards.append(
                EvidenceShard(
                    split=split,
                    path=relative,
                    start_index=start,
                    end_index=start + len(selected) - 1,
                    count=len(selected),
                    bytes=destination.stat().st_size,
                    sha256=_sha256(destination),
                )
            )
    manifest = CacheManifest(
        format_version=CACHE_FORMAT_VERSION,
        design_version=DESIGN_VERSION,
        complete=True,
        authority=normalized_authority,
        split_counts={name: len(records) for name, records in prepared.items()},
        shard_size=shard_size,
        shards=tuple(shards),
    )
    manifest_path = root / "manifest.json"
    temporary_manifest = root / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(_manifest_payload(manifest), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest


def _parse_manifest(payload: Mapping[str, Any]) -> CacheManifest:
    try:
        shards = tuple(EvidenceShard(**record) for record in payload["shards"])
        return CacheManifest(
            format_version=int(payload["format_version"]),
            design_version=str(payload["design_version"]),
            complete=bool(payload["complete"]),
            authority=dict(payload["authority"]),
            split_counts={name: int(value) for name, value in payload["split_counts"].items()},
            shard_size=int(payload["shard_size"]),
            shards=shards,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CacheViolation({"manifest.schema": str(error)}) from error


def load_evidence_cache(
    root: str | Path,
    *,
    expected_authority: Mapping[str, Any],
) -> EvidenceCache:
    """Verify every byte and return deterministic CPU records by split."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CacheViolation({"manifest": str(error)}) from error
    manifest = _parse_manifest(payload)
    violations: dict[str, Any] = {}
    if manifest.format_version != CACHE_FORMAT_VERSION:
        violations["format_version"] = manifest.format_version
    if manifest.design_version != DESIGN_VERSION:
        violations["design_version"] = manifest.design_version
    if not manifest.complete:
        violations["complete"] = False
    expected = _normalized_authority(expected_authority)
    for name, expected_value in expected.items():
        actual = manifest.authority.get(name)
        if actual != expected_value:
            violations[f"authority.{name}"] = {
                "expected": expected_value,
                "actual": actual,
            }
    if violations:
        raise CacheViolation(violations)

    records: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    image_ids: set[str] = set()
    for shard_number, shard in enumerate(manifest.shards):
        if shard.split not in records:
            raise CacheViolation({f"shards.{shard_number}.split": shard.split})
        path = _safe_path(root, shard.path)
        if not path.is_file():
            raise CacheViolation({f"shards.{shard_number}.missing": shard.path})
        actual_bytes = path.stat().st_size
        actual_sha = _sha256(path)
        shard_violations = {}
        if actual_bytes != shard.bytes:
            shard_violations["bytes"] = {"expected": shard.bytes, "actual": actual_bytes}
        if actual_sha != shard.sha256:
            shard_violations["sha256"] = {"expected": shard.sha256, "actual": actual_sha}
        if shard_violations:
            raise CacheViolation(
                {
                    f"shards.{shard_number}.{name}": detail
                    for name, detail in shard_violations.items()
                }
            )
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if (
            artifact.get("format_version") != CACHE_FORMAT_VERSION
            or artifact.get("design_version") != DESIGN_VERSION
            or artifact.get("split") != shard.split
            or artifact.get("start_index") != shard.start_index
        ):
            raise CacheViolation({f"shards.{shard_number}.metadata": "mismatch"})
        selected = artifact.get("records")
        if not isinstance(selected, list) or len(selected) != shard.count:
            raise CacheViolation({f"shards.{shard_number}.count": "mismatch"})
        for record in selected:
            expected_index = len(records[shard.split])
            _validate_record(record, expected_index, shard.split)
            if record["image_id"] in image_ids:
                raise CacheViolation({"split.image_id_overlap": record["image_id"]})
            image_ids.add(record["image_id"])
            records[shard.split].append(record)
    actual_counts = {name: len(values) for name, values in records.items()}
    if actual_counts != manifest.split_counts:
        raise CacheViolation(
            {"split_counts": {"expected": manifest.split_counts, "actual": actual_counts}}
        )
    return EvidenceCache(
        manifest=manifest,
        records={name: tuple(values) for name, values in records.items()},
    )
