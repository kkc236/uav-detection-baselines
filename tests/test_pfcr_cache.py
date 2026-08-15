from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.pfcr import pfcr_split
from src.pfcr_cache import (
    PFCRCacheViolation,
    PFCRCacheWriter,
    load_pfcr_cache,
)


def authority() -> dict[str, str]:
    return {
        "fdr_sha256": "A" * 64,
        "frequencycm_sha256": "B" * 64,
        "dataset_sha256": "C" * 64,
        "evaluator_sha256": "D" * 64,
        "feature_schema_sha256": "E" * 64,
        "source_commit": "1" * 40,
    }


def record(image_id: str) -> dict[str, object]:
    return {
        "image_id": image_id,
        "original_shape": (540, 960),
        "resized_shape": (640, 640),
        "fdr_boxes": torch.full((300, 4), 0.25, dtype=torch.float32),
        "fdr_logits": torch.zeros(300, 10, dtype=torch.float32),
        "frequencycm_boxes": torch.full((300, 4), 0.25, dtype=torch.float32),
        "frequencycm_logits": torch.zeros(300, 10, dtype=torch.float32),
        "target_boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]], dtype=torch.float32),
        "target_classes": torch.tensor([2], dtype=torch.long),
    }


def ids_for(split: str, count: int) -> list[str]:
    selected: list[str] = []
    index = 0
    while len(selected) < count:
        name = f"img-{index:05d}.jpg"
        if pfcr_split(name) == split:
            selected.append(name)
        index += 1
    return selected


def test_cache_is_create_only_authority_bound_and_sharded(tmp_path: Path):
    root = tmp_path / "cache"
    records = [record(name) for name in ids_for("train", 65)]
    writer = PFCRCacheWriter(root, authority(), shard_size=64)
    writer.append_many(records)
    manifest = writer.finalize()

    assert manifest["complete"] is True
    assert manifest["counts"] == {"dev": 0, "train": 65}
    assert len(manifest["shards"]) == 2
    loaded = load_pfcr_cache(root, authority())
    assert [item["image_id"] for item in loaded["train"]] == [
        item["image_id"] for item in records
    ]
    assert loaded["dev"] == ()
    with pytest.raises(FileExistsError):
        PFCRCacheWriter(root, authority(), shard_size=64)


def test_cache_writes_train_and_dev_shards_independently(tmp_path: Path):
    root = tmp_path / "cache"
    writer = PFCRCacheWriter(root, authority(), shard_size=2)
    writer.append_many(
        [record(name) for name in ids_for("train", 3) + ids_for("dev", 3)]
    )
    manifest = writer.finalize()
    assert manifest["counts"] == {"dev": 3, "train": 3}
    assert [(item["split"], item["count"]) for item in manifest["shards"]] == [
        ("dev", 2),
        ("dev", 1),
        ("train", 2),
        ("train", 1),
    ]


def test_incomplete_cache_resumes_only_verified_completed_shards(tmp_path: Path):
    root = tmp_path / "cache"
    first_ids = ids_for("train", 2)
    writer = PFCRCacheWriter(root, authority(), shard_size=2)
    writer.append_many([record(name) for name in first_ids])
    assert writer.flush() == 0  # full shards are persisted immediately for crash recovery
    assert len(list((root / "shards").glob("train-*"))) == 1

    resumed = PFCRCacheWriter(root, authority(), shard_size=2)
    assert resumed.completed_image_ids == frozenset(first_ids)
    remaining = ids_for("train", 3)[2:]
    resumed.append_many([record(name) for name in remaining])
    resumed.finalize()
    loaded = load_pfcr_cache(root, authority())
    assert [item["image_id"] for item in loaded["train"]] == first_ids + remaining


def test_cache_rejects_corrupted_shard_before_torch_load(tmp_path: Path, monkeypatch):
    root = tmp_path / "cache"
    writer = PFCRCacheWriter(root, authority(), shard_size=2)
    writer.append_many([record(name) for name in ids_for("train", 2)])
    manifest = writer.finalize()
    shard = root / manifest["shards"][0]["path"] / "records.pt"
    shard.write_bytes(b"corrupt")
    calls = []
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(PFCRCacheViolation, match="sha256"):
        load_pfcr_cache(root, authority())
    assert calls == []


def test_cache_rejects_authority_drift(tmp_path: Path):
    root = tmp_path / "cache"
    writer = PFCRCacheWriter(root, authority(), shard_size=2)
    writer.append_many([record(name) for name in ids_for("train", 1)])
    writer.finalize()
    changed = {**authority(), "feature_schema_sha256": "F" * 64}
    with pytest.raises(PFCRCacheViolation, match="authority"):
        load_pfcr_cache(root, changed)


def test_cache_rejects_duplicate_ids_and_schema_drift(tmp_path: Path):
    root = tmp_path / "cache"
    item = record(ids_for("train", 1)[0])
    writer = PFCRCacheWriter(root, authority(), shard_size=2)
    writer.append_many([item])
    with pytest.raises(PFCRCacheViolation, match="duplicate"):
        writer.append_many([item])

    wrong = record(ids_for("train", 2)[1])
    wrong["fdr_boxes"] = torch.zeros(299, 4)
    with pytest.raises(PFCRCacheViolation, match="shape"):
        writer.append_many([wrong])


def test_manifest_is_canonical_and_hashes_match(tmp_path: Path):
    root = tmp_path / "cache"
    writer = PFCRCacheWriter(root, authority(), shard_size=2)
    writer.append_many([record(name) for name in ids_for("dev", 1)])
    manifest = writer.finalize()
    raw = (root / "manifest.json").read_bytes()
    assert raw == (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    for shard in manifest["shards"]:
        payload = root / shard["path"] / "records.pt"
        assert hashlib.sha256(payload.read_bytes()).hexdigest().upper() == shard["sha256"]
