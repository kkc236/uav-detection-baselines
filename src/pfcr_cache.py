"""Immutable resumable sharded evidence cache for the PFCR learnability probe."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from src.pfcr import pfcr_split
from src.rtdetr_complementarity_oracle import (
    ComplementarityOracleCacheViolation,
    _is_symlink_or_reparse,
    _validate_paired_record,
)


FORMAT_VERSION = 1
AUTHORITY_FIELDS = {
    "fdr_sha256",
    "frequencycm_sha256",
    "dataset_sha256",
    "evaluator_sha256",
    "feature_schema_sha256",
    "source_commit",
}


class PFCRCacheViolation(ValueError):
    """An immutable PFCR authority, shard, manifest, or record is invalid."""


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _authority(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != AUTHORITY_FIELDS:
        raise PFCRCacheViolation("authority schema mismatch")
    normalized: dict[str, str] = {}
    for name in sorted(AUTHORITY_FIELDS):
        item = value[name]
        if not isinstance(item, str):
            raise PFCRCacheViolation(f"authority {name} must be a string")
        if name == "source_commit":
            if re.fullmatch(r"[0-9a-fA-F]{40}", item) is None:
                raise PFCRCacheViolation("authority source_commit is invalid")
            normalized[name] = item.lower()
        else:
            if re.fullmatch(r"[0-9a-fA-F]{64}", item) is None:
                raise PFCRCacheViolation(f"authority {name} is invalid")
            normalized[name] = item.upper()
    return normalized


def _record(value: object) -> dict[str, Any]:
    try:
        checked = _validate_paired_record(value)
    except (ComplementarityOracleCacheViolation, TypeError, ValueError) as error:
        raise PFCRCacheViolation(str(error)) from error
    return checked


def _load_canonical_json(path: Path, *, label: str) -> dict[str, Any]:
    if _is_symlink_or_reparse(path) or not path.is_file():
        raise PFCRCacheViolation(f"{label} is not a regular file")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PFCRCacheViolation(f"{label} load failed") from error
    if not isinstance(payload, dict) or raw != _canonical_json(payload):
        raise PFCRCacheViolation(f"{label} is not canonical")
    return payload


def _save_torch(path: Path, payload: object) -> None:
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(_canonical_json(payload))
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_shard(root: Path, item: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    required = {"path", "split", "start_index", "count", "bytes", "sha256"}
    if not isinstance(item, Mapping) or set(item) != required:
        raise PFCRCacheViolation("shard schema mismatch")
    split = item["split"]
    if split not in {"train", "dev"}:
        raise PFCRCacheViolation("shard split mismatch")
    expected_path = f"shards/{split}-{int(item['start_index']):05d}"
    if item["path"] != expected_path:
        raise PFCRCacheViolation("shard path mismatch")
    directory = root / expected_path
    if _is_symlink_or_reparse(directory) or not directory.is_dir():
        raise PFCRCacheViolation("shard directory is invalid")
    if {path.name for path in directory.iterdir()} != {"manifest.json", "records.pt"}:
        raise PFCRCacheViolation("shard contents mismatch")
    local_manifest = _load_canonical_json(directory / "manifest.json", label="shard manifest")
    if local_manifest != dict(item):
        raise PFCRCacheViolation("shard manifest mismatch")
    payload_path = directory / "records.pt"
    if _sha256(payload_path) != item["sha256"]:
        raise PFCRCacheViolation("shard sha256 mismatch")
    if payload_path.stat().st_size != item["bytes"]:
        raise PFCRCacheViolation("shard bytes mismatch")
    try:
        artifact = torch.load(payload_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise PFCRCacheViolation("shard deserialization failed") from error
    if not isinstance(artifact, dict) or set(artifact) != {
        "format_version",
        "split",
        "start_index",
        "records",
    }:
        raise PFCRCacheViolation("shard artifact schema mismatch")
    if (
        artifact["format_version"] != FORMAT_VERSION
        or artifact["split"] != split
        or artifact["start_index"] != item["start_index"]
        or not isinstance(artifact["records"], list)
        or len(artifact["records"]) != item["count"]
    ):
        raise PFCRCacheViolation("shard artifact metadata mismatch")
    records = tuple(_record(record) for record in artifact["records"])
    if any(pfcr_split(record["image_id"]) != split for record in records):
        raise PFCRCacheViolation("shard record split mismatch")
    return records


def _existing_shards(root: Path) -> list[dict[str, Any]]:
    shard_root = root / "shards"
    if _is_symlink_or_reparse(shard_root) or not shard_root.is_dir():
        raise PFCRCacheViolation("shard root is invalid")
    items: list[dict[str, Any]] = []
    for directory in sorted(shard_root.iterdir(), key=lambda path: path.name):
        if directory.name.startswith("."):
            continue
        manifest = _load_canonical_json(directory / "manifest.json", label="shard manifest")
        _verify_shard(root, manifest)
        items.append(manifest)
    return sorted(items, key=lambda item: (item["split"], item["start_index"]))


class PFCRCacheWriter:
    """Append verified evidence shards and publish the completion manifest last."""

    def __init__(
        self, root: Path, authority: Mapping[str, str], *, shard_size: int = 64
    ) -> None:
        if type(shard_size) is not int or shard_size <= 0:
            raise ValueError("shard_size must be a positive integer")
        self.root = Path(root)
        self.authority = _authority(authority)
        self.shard_size = shard_size
        self._buffers: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
        self._seen: set[str] = set()
        self._shards: list[dict[str, Any]] = []
        if _is_symlink_or_reparse(self.root):
            raise PFCRCacheViolation("cache root is a symlink or reparse point")
        if not os.path.lexists(self.root):
            self.root.mkdir(parents=True)
            (self.root / "shards").mkdir()
            _save_json(self.root / "authority.json", self.authority)
            _fsync_directory(self.root)
        else:
            if not self.root.is_dir():
                raise PFCRCacheViolation("cache root is not a directory")
            if (self.root / "manifest.json").exists():
                raise FileExistsError(f"completed cache already exists: {self.root}")
            if {path.name for path in self.root.iterdir()} != {"authority.json", "shards"}:
                raise PFCRCacheViolation("incomplete cache root contents mismatch")
            existing_authority = _load_canonical_json(
                self.root / "authority.json", label="authority"
            )
            if existing_authority != self.authority:
                raise PFCRCacheViolation("authority mismatch while resuming cache")
            self._shards = _existing_shards(self.root)
            for item in self._shards:
                for record in _verify_shard(self.root, item):
                    image_id = record["image_id"]
                    if image_id in self._seen:
                        raise PFCRCacheViolation("duplicate image ID in existing cache")
                    self._seen.add(image_id)

    @property
    def completed_image_ids(self) -> frozenset[str]:
        return frozenset(self._seen)

    def append_many(self, records: Sequence[Mapping[str, Any]]) -> None:
        for value in records:
            checked = _record(value)
            image_id = checked["image_id"]
            if image_id in self._seen:
                raise PFCRCacheViolation(f"duplicate image ID: {image_id}")
            self._seen.add(image_id)
            self._buffers[pfcr_split(image_id)].append(checked)

    def _next_start(self, split: str) -> int:
        return sum(int(item["count"]) for item in self._shards if item["split"] == split)

    def _write_shard(self, split: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
        start = self._next_start(split)
        relative = f"shards/{split}-{start:05d}"
        target = self.root / relative
        if os.path.lexists(target):
            raise FileExistsError(f"shard target already exists: {target}")
        staging = Path(
            tempfile.mkdtemp(prefix=f".{split}-{start:05d}.staging-", dir=self.root / "shards")
        )
        published = False
        try:
            payload_path = staging / "records.pt"
            _save_torch(
                payload_path,
                {
                    "format_version": FORMAT_VERSION,
                    "split": split,
                    "start_index": start,
                    "records": selected,
                },
            )
            item = {
                "path": relative,
                "split": split,
                "start_index": start,
                "count": len(selected),
                "bytes": payload_path.stat().st_size,
                "sha256": _sha256(payload_path),
            }
            _save_json(staging / "manifest.json", item)
            _fsync_directory(staging)
            os.rename(staging, target)
            published = True
            _fsync_directory(self.root / "shards")
            return item
        finally:
            if not published and os.path.lexists(staging):
                shutil.rmtree(staging)

    def flush(self) -> int:
        written = 0
        for split in ("dev", "train"):
            buffer = self._buffers[split]
            while buffer:
                selected = buffer[: self.shard_size]
                del buffer[: len(selected)]
                item = self._write_shard(split, selected)
                self._shards.append(item)
                written += 1
        self._shards.sort(key=lambda item: (item["split"], item["start_index"]))
        return written

    def finalize(self) -> dict[str, Any]:
        self.flush()
        if (self.root / "manifest.json").exists():
            raise FileExistsError(f"cache manifest already exists: {self.root}")
        counts = {
            split: sum(int(item["count"]) for item in self._shards if item["split"] == split)
            for split in ("dev", "train")
        }
        manifest = {
            "format_version": FORMAT_VERSION,
            "complete": True,
            "authority": self.authority,
            "shard_size": self.shard_size,
            "counts": counts,
            "shards": self._shards,
        }
        _save_json(self.root / "manifest.json", manifest)
        _fsync_directory(self.root)
        return manifest


def load_pfcr_cache(
    root: Path, authority: Mapping[str, str]
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Verify every cache byte and schema before returning split records."""

    root = Path(root)
    if _is_symlink_or_reparse(root) or not root.is_dir():
        raise PFCRCacheViolation("cache root is invalid")
    expected_authority = _authority(authority)
    if {path.name for path in root.iterdir()} != {"authority.json", "manifest.json", "shards"}:
        raise PFCRCacheViolation("cache root contents mismatch")
    stored_authority = _load_canonical_json(root / "authority.json", label="authority")
    if stored_authority != expected_authority:
        raise PFCRCacheViolation("cache authority mismatch")
    manifest = _load_canonical_json(root / "manifest.json", label="manifest")
    if set(manifest) != {
        "format_version",
        "complete",
        "authority",
        "shard_size",
        "counts",
        "shards",
    }:
        raise PFCRCacheViolation("manifest schema mismatch")
    if (
        manifest["format_version"] != FORMAT_VERSION
        or manifest["complete"] is not True
        or manifest["authority"] != expected_authority
        or type(manifest["shard_size"]) is not int
        or manifest["shard_size"] <= 0
        or not isinstance(manifest["shards"], list)
    ):
        raise PFCRCacheViolation("manifest metadata mismatch")
    records: dict[str, list[dict[str, Any]]] = {"dev": [], "train": []}
    seen: set[str] = set()
    expected_start = {"dev": 0, "train": 0}
    for item in manifest["shards"]:
        split = item.get("split") if isinstance(item, dict) else None
        if split not in expected_start or item.get("start_index") != expected_start[split]:
            raise PFCRCacheViolation("shard sequence mismatch")
        selected = _verify_shard(root, item)
        expected_start[split] += len(selected)
        for checked in selected:
            if checked["image_id"] in seen:
                raise PFCRCacheViolation("duplicate image ID in cache")
            seen.add(checked["image_id"])
            records[split].append(checked)
    actual_counts = {split: len(values) for split, values in records.items()}
    if manifest["counts"] != actual_counts:
        raise PFCRCacheViolation("manifest counts mismatch")
    return {split: tuple(values) for split, values in records.items()}


__all__ = ["PFCRCacheViolation", "PFCRCacheWriter", "load_pfcr_cache"]
