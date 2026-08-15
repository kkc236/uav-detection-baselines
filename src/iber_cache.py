"""Immutable boundary evidence cache for IBER-BE Probe training."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import torch


CACHE_FORMAT_VERSION = 1
DESIGN_VERSION = "iber-be-v1.0"
DEFAULT_SHARD_SIZE = 16
REQUIRED_RECORD_TENSORS = (
    "hidden",
    "stock_boxes",
    "stock_scores",
    "f3",
    "image_rgb",
    "target_edges",
    "match_source",
    "match_target",
)

_AUTHORITY_FIELDS = (
    "baseline_sha256",
    "dataset_sha256",
    "category_sha256",
    "subset_sha256",
    "source_commit",
    "runtime_amendment_sha256",
)
_HASH_FIELDS = tuple(name for name in _AUTHORITY_FIELDS if name != "source_commit")
_RECORD_FIELDS = frozenset(("index", "image_id", *REQUIRED_RECORD_TENSORS))
_MANIFEST_FIELDS = frozenset(
    (
        "format_version",
        "design_version",
        "complete",
        "authority",
        "split_counts",
        "shard_size",
        "shards",
    )
)
_SHARD_FIELDS = frozenset(
    ("split", "path", "start_index", "end_index", "count", "bytes", "sha256")
)
_ARTIFACT_FIELDS = frozenset(
    (
        "format_version",
        "design_version",
        "split",
        "start_index",
        "end_index",
        "count",
        "records",
    )
)


class CacheViolation(ValueError):
    """An immutable cache authority, manifest, shard, or record is invalid."""

    def __init__(self, violations: Mapping[str, Any]) -> None:
        self.violations = dict(violations)
        super().__init__("IBER cache violation: " + ", ".join(sorted(self.violations)))


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


def _schema_violation(name: str, actual: object, expected: set[str] | frozenset[str]) -> CacheViolation:
    actual_fields = set(actual) if isinstance(actual, Mapping) else set()
    return CacheViolation(
        {
            f"{name}.schema": {
                "missing": sorted(expected - actual_fields),
                "extra": sorted(actual_fields - expected),
            }
        }
    )


def _normalized_authority(authority: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(authority, Mapping) or set(authority) != set(_AUTHORITY_FIELDS):
        raise _schema_violation("authority", authority, frozenset(_AUTHORITY_FIELDS))
    normalized = {name: str(authority[name]) for name in _AUTHORITY_FIELDS}
    violations: dict[str, Any] = {}
    for name in _HASH_FIELDS:
        value = normalized[name].upper()
        normalized[name] = value
        if re.fullmatch(r"[0-9A-F]{64}", value) is None:
            violations[f"authority.{name}"] = value
    commit = normalized["source_commit"].lower()
    normalized["source_commit"] = commit
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        violations["authority.source_commit"] = commit
    if violations:
        raise CacheViolation(violations)
    return normalized


def _validate_record(record: Mapping[str, Any], expected_index: int, split: str) -> None:
    if not isinstance(record, Mapping) or set(record) != _RECORD_FIELDS:
        raise _schema_violation(
            f"{split}.{expected_index}.record", record, _RECORD_FIELDS
        )
    violations: dict[str, Any] = {}
    prefix = f"{split}.{expected_index}"
    if isinstance(record["index"], bool) or record["index"] != expected_index:
        violations[f"{split}.index"] = {
            "expected": expected_index,
            "actual": record["index"],
        }
    image_id = record["image_id"]
    if not isinstance(image_id, str) or not image_id:
        violations[f"{split}.image_id"] = image_id
    for name in REQUIRED_RECORD_TENSORS:
        if not isinstance(record[name], torch.Tensor):
            violations[f"{prefix}.{name}"] = "not_tensor"
    if violations:
        raise CacheViolation(violations)

    hidden = record["hidden"]
    stock_boxes = record["stock_boxes"]
    stock_scores = record["stock_scores"]
    f3 = record["f3"]
    image_rgb = record["image_rgb"]
    target_edges = record["target_edges"]
    source = record["match_source"]
    target = record["match_target"]
    if hidden.ndim != 2:
        violations[f"{prefix}.hidden"] = "invalid_shape"
    if stock_boxes.ndim != 2 or stock_boxes.shape[-1:] != (4,):
        violations[f"{prefix}.stock_boxes"] = "invalid_shape"
    if stock_scores.ndim != 2:
        violations[f"{prefix}.stock_scores"] = "invalid_shape"
    if (
        hidden.ndim == 2
        and stock_boxes.ndim == 2
        and stock_scores.ndim == 2
        and len({hidden.shape[0], stock_boxes.shape[0], stock_scores.shape[0]}) != 1
    ):
        violations[f"{prefix}.queries"] = "mismatch"
    if f3.ndim != 3:
        violations[f"{prefix}.f3"] = "invalid_shape"
    if (
        image_rgb.dtype is not torch.uint8
        or image_rgb.shape != (3, 640, 640)
        or not image_rgb.is_contiguous()
    ):
        violations[f"{prefix}.image_rgb"] = {
            "dtype": str(image_rgb.dtype),
            "shape": tuple(image_rgb.shape),
            "contiguous": image_rgb.is_contiguous(),
        }
    if target_edges.ndim != 2 or target_edges.shape[-1:] != (4,):
        violations[f"{prefix}.target_edges"] = "invalid_shape"
    if source.ndim != 1 or target.ndim != 1 or source.shape != target.shape:
        violations[f"{prefix}.match"] = "invalid_shape"
    if source.dtype is not torch.long or target.dtype is not torch.long:
        violations[f"{prefix}.match"] = "invalid_dtype"
    if violations:
        raise CacheViolation(violations)


def _prepare_records(
    records: Sequence[Mapping[str, Any]], split: str
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for expected_index, record in enumerate(records):
        _validate_record(record, expected_index, split)
        copied: dict[str, Any] = {
            "index": expected_index,
            "image_id": record["image_id"],
        }
        for name in REQUIRED_RECORD_TENSORS:
            copied[name] = record[name].detach().cpu().contiguous().clone()
        prepared.append(copied)
    return prepared


def _safe_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise CacheViolation({"shard.path": relative})
    raw_parts = relative.split("/")
    parsed = PurePosixPath(relative)
    if (
        parsed.is_absolute()
        or any(part in {"", ".", ".."} for part in raw_parts)
        or any(":" in part for part in raw_parts)
    ):
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
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> CacheManifest:
    """Write immutable shards before atomically publishing the complete manifest."""
    if isinstance(shard_size, bool) or not isinstance(shard_size, int) or shard_size < 1:
        raise ValueError("shard size must be a positive integer")
    normalized_authority = _normalized_authority(authority)
    prepared = {
        "train": _prepare_records(train_records, "train"),
        "val": _prepare_records(val_records, "val"),
    }
    train_ids = [record["image_id"] for record in prepared["train"]]
    val_ids = [record["image_id"] for record in prepared["val"]]
    duplicate_train = len(set(train_ids)) != len(train_ids)
    duplicate_val = len(set(val_ids)) != len(val_ids)
    overlap = sorted(set(train_ids) & set(val_ids))
    if duplicate_train or duplicate_val or overlap:
        raise CacheViolation(
            {
                "split.image_id_overlap": {
                    "duplicate_train": duplicate_train,
                    "duplicate_val": duplicate_val,
                    "cross_split": overlap[:10],
                }
            }
        )

    root = Path(root)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty cache root: {root}")
    shard_root = root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shards: list[EvidenceShard] = []
    for split in ("train", "val"):
        records = prepared[split]
        for shard_index, start in enumerate(range(0, len(records), shard_size)):
            selected = records[start : start + shard_size]
            end = start + len(selected) - 1
            relative = f"shards/{split}-{shard_index:05d}.pt"
            destination = _safe_path(root, relative)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            torch.save(
                {
                    "format_version": CACHE_FORMAT_VERSION,
                    "design_version": DESIGN_VERSION,
                    "split": split,
                    "start_index": start,
                    "end_index": end,
                    "count": len(selected),
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
                    end_index=end,
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
        split_counts={name: len(values) for name, values in prepared.items()},
        shard_size=shard_size,
        shards=tuple(shards),
    )
    temporary_manifest = root / "manifest.json.tmp"
    manifest_path = root / "manifest.json"
    try:
        temporary_manifest.write_text(
            json.dumps(
                _manifest_payload(manifest),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        if temporary_manifest.exists():
            temporary_manifest.unlink()
    return manifest


def _parse_manifest(payload: Any) -> CacheManifest:
    if not isinstance(payload, Mapping) or set(payload) != _MANIFEST_FIELDS:
        raise _schema_violation("manifest", payload, _MANIFEST_FIELDS)
    shards_payload = payload["shards"]
    if not isinstance(shards_payload, list):
        raise CacheViolation({"manifest.shards": "not_list"})
    shards: list[EvidenceShard] = []
    try:
        for index, item in enumerate(shards_payload):
            if not isinstance(item, Mapping) or set(item) != _SHARD_FIELDS:
                raise _schema_violation(f"shards.{index}", item, _SHARD_FIELDS)
            shards.append(EvidenceShard(**item))
        manifest = CacheManifest(
            format_version=payload["format_version"],
            design_version=payload["design_version"],
            complete=payload["complete"],
            authority=dict(payload["authority"]),
            split_counts=dict(payload["split_counts"]),
            shard_size=payload["shard_size"],
            shards=tuple(shards),
        )
    except (TypeError, ValueError) as error:
        raise CacheViolation({"manifest.schema": str(error)}) from error
    return manifest


def _validate_manifest(
    manifest: CacheManifest, expected_authority: Mapping[str, Any]
) -> None:
    violations: dict[str, Any] = {}
    if manifest.format_version != CACHE_FORMAT_VERSION:
        violations["format_version"] = manifest.format_version
    if manifest.design_version != DESIGN_VERSION:
        violations["design_version"] = manifest.design_version
    if manifest.complete is not True:
        violations["complete"] = manifest.complete
    if (
        isinstance(manifest.shard_size, bool)
        or not isinstance(manifest.shard_size, int)
        or manifest.shard_size < 1
    ):
        violations["shard_size"] = manifest.shard_size
    if set(manifest.split_counts) != {"train", "val"} or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in manifest.split_counts.values()
    ):
        violations["split_counts"] = manifest.split_counts
    actual_authority = _normalized_authority(manifest.authority)
    expected = _normalized_authority(expected_authority)
    for name, expected_value in expected.items():
        actual = actual_authority[name]
        if actual != expected_value:
            violations[f"authority.{name}"] = {
                "expected": expected_value,
                "actual": actual,
            }

    next_index = {"train": 0, "val": 0}
    seen_val = False
    for number, shard in enumerate(manifest.shards):
        prefix = f"shards.{number}"
        if shard.split not in next_index:
            violations[f"{prefix}.split"] = shard.split
            continue
        if shard.split == "val":
            seen_val = True
        elif seen_val:
            violations[f"{prefix}.split_order"] = shard.split
        expected_start = next_index[shard.split]
        if shard.start_index != expected_start:
            violations[f"{prefix}.start_index"] = {
                "expected": expected_start,
                "actual": shard.start_index,
            }
        if isinstance(shard.count, bool) or not isinstance(shard.count, int) or shard.count < 1:
            violations[f"{prefix}.count"] = shard.count
        else:
            expected_end = shard.start_index + shard.count - 1
            if shard.end_index != expected_end:
                violations[f"{prefix}.end_index"] = {
                    "expected": expected_end,
                    "actual": shard.end_index,
                }
            next_index[shard.split] = shard.start_index + shard.count
        if isinstance(shard.bytes, bool) or not isinstance(shard.bytes, int) or shard.bytes < 1:
            violations[f"{prefix}.bytes"] = shard.bytes
        if not isinstance(shard.sha256, str) or re.fullmatch(r"[0-9A-F]{64}", shard.sha256) is None:
            violations[f"{prefix}.sha256"] = shard.sha256
    if next_index != manifest.split_counts:
        violations["split_counts"] = {
            "expected": manifest.split_counts,
            "actual": next_index,
        }
    if violations:
        raise CacheViolation(violations)


def load_evidence_cache(
    root: str | Path,
    *,
    expected_authority: Mapping[str, Any],
) -> EvidenceCache:
    """Verify the entire immutable cache before loading or exposing any record."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CacheViolation({"manifest": str(error)}) from error
    manifest = _parse_manifest(payload)
    _validate_manifest(manifest, expected_authority)

    verified_paths: list[Path] = []
    preflight_violations: dict[str, Any] = {}
    for number, shard in enumerate(manifest.shards):
        path = _safe_path(root, shard.path)
        if not path.is_file():
            preflight_violations[f"shards.{number}.missing"] = shard.path
            continue
        actual_bytes = path.stat().st_size
        actual_sha = _sha256(path)
        if actual_bytes != shard.bytes:
            preflight_violations[f"shards.{number}.bytes"] = {
                "expected": shard.bytes,
                "actual": actual_bytes,
            }
        if actual_sha != shard.sha256:
            preflight_violations[f"shards.{number}.sha256"] = {
                "expected": shard.sha256,
                "actual": actual_sha,
            }
        verified_paths.append(path)
    if preflight_violations:
        raise CacheViolation(preflight_violations)

    records: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    image_ids: set[str] = set()
    for number, (shard, path) in enumerate(zip(manifest.shards, verified_paths)):
        try:
            artifact = torch.load(path, map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, RuntimeError, EOFError) as error:
            raise CacheViolation(
                {f"shards.{number}.load": type(error).__name__}
            ) from error
        if not isinstance(artifact, Mapping) or set(artifact) != _ARTIFACT_FIELDS:
            raise _schema_violation(f"shards.{number}.artifact", artifact, _ARTIFACT_FIELDS)
        metadata = {
            "format_version": CACHE_FORMAT_VERSION,
            "design_version": DESIGN_VERSION,
            "split": shard.split,
            "start_index": shard.start_index,
            "end_index": shard.end_index,
            "count": shard.count,
        }
        drift = {
            name: {"expected": expected, "actual": artifact[name]}
            for name, expected in metadata.items()
            if type(artifact[name]) is not type(expected) or artifact[name] != expected
        }
        if drift:
            raise CacheViolation({f"shards.{number}.metadata": drift})
        selected = artifact["records"]
        if not isinstance(selected, list) or len(selected) != shard.count:
            raise CacheViolation({f"shards.{number}.count": "mismatch"})
        for record in selected:
            expected_index = len(records[shard.split])
            _validate_record(record, expected_index, shard.split)
            image_id = record["image_id"]
            if image_id in image_ids:
                raise CacheViolation({"split.image_id_overlap": image_id})
            image_ids.add(image_id)
            records[shard.split].append(dict(record))
    actual_counts = {name: len(values) for name, values in records.items()}
    if actual_counts != manifest.split_counts:
        raise CacheViolation(
            {"split_counts": {"expected": manifest.split_counts, "actual": actual_counts}}
        )
    return EvidenceCache(
        manifest=manifest,
        records={name: tuple(values) for name, values in records.items()},
    )


def image_rgb_for_probe(record: Mapping[str, Any]) -> torch.Tensor:
    """Convert cached RGB bytes to the exact float input consumed by Probe."""
    image_rgb = record.get("image_rgb")
    if (
        not isinstance(image_rgb, torch.Tensor)
        or image_rgb.dtype is not torch.uint8
        or image_rgb.shape != (3, 640, 640)
        or not image_rgb.is_contiguous()
    ):
        raise CacheViolation({"record.image_rgb": "invalid"})
    return image_rgb.float().div(255)


__all__ = [
    "CACHE_FORMAT_VERSION",
    "DESIGN_VERSION",
    "DEFAULT_SHARD_SIZE",
    "REQUIRED_RECORD_TENSORS",
    "CacheManifest",
    "CacheViolation",
    "EvidenceCache",
    "EvidenceShard",
    "image_rgb_for_probe",
    "load_evidence_cache",
    "write_evidence_cache",
]
