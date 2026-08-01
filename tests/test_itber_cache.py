from __future__ import annotations

import json

import pytest
import torch

from src.itber_cache import (
    CacheViolation,
    load_evidence_cache,
    write_evidence_cache,
)
from src.itber_protocol import (
    EXPECTED_BASELINE_SHA256,
    EXPECTED_CATEGORY_SHA256,
    EXPECTED_DATASET_SHA256,
)


def _authority() -> dict[str, str]:
    return {
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "category_sha256": EXPECTED_CATEGORY_SHA256,
        "source_commit": "a" * 40,
    }


def _record(index: int, image_id: str, value: float) -> dict:
    return {
        "index": index,
        "image_id": image_id,
        "hidden": torch.full((3, 4), value),
        "box_l2": torch.full((3, 4), 0.4),
        "box_l1": torch.full((3, 4), 0.4),
        "stock_boxes": torch.full((3, 4), 0.4),
        "stock_scores": torch.full((3, 2), value),
        "f3": torch.full((2, 5, 5), value, dtype=torch.float16),
        "target_edges": torch.tensor([[0.3, 0.3, 0.5, 0.5]]),
        "match_source": torch.tensor([0]),
        "match_target": torch.tensor([0]),
    }


def test_cache_roundtrip_has_contiguous_split_isolated_records_and_shard_hashes(tmp_path) -> None:
    root = tmp_path / "cache"
    train = [_record(i, f"train-{i}", float(i)) for i in range(3)]
    val = [_record(i, f"val-{i}", float(10 + i)) for i in range(2)]

    manifest = write_evidence_cache(
        root,
        train_records=train,
        val_records=val,
        authority=_authority(),
        shard_size=2,
    )
    loaded = load_evidence_cache(root, expected_authority=_authority())

    assert manifest.complete is True
    assert manifest.split_counts == {"train": 3, "val": 2}
    assert [shard.split for shard in manifest.shards] == ["train", "train", "val"]
    assert all(shard.bytes > 0 and len(shard.sha256) == 64 for shard in manifest.shards)
    assert [record["index"] for record in loaded.records["train"]] == [0, 1, 2]
    assert [record["image_id"] for record in loaded.records["val"]] == ["val-0", "val-1"]
    torch.testing.assert_close(loaded.records["train"][2]["hidden"], train[2]["hidden"])


@pytest.mark.parametrize("mode", ["gap", "duplicate_image", "wrong_split_index"])
def test_cache_rejects_noncontiguous_or_cross_split_identity(tmp_path, mode: str) -> None:
    train = [_record(0, "train-0", 0), _record(1, "train-1", 1)]
    val = [_record(0, "val-0", 2)]
    if mode == "gap":
        train[1]["index"] = 2
    elif mode == "duplicate_image":
        val[0]["image_id"] = "train-0"
    else:
        val[0]["index"] = 1

    with pytest.raises(CacheViolation):
        write_evidence_cache(
            tmp_path / "cache",
            train_records=train,
            val_records=val,
            authority=_authority(),
            shard_size=2,
        )


def test_loader_rejects_corruption_and_changed_baseline_authority(tmp_path) -> None:
    root = tmp_path / "cache"
    write_evidence_cache(
        root,
        train_records=[_record(0, "train-0", 0)],
        val_records=[_record(0, "val-0", 1)],
        authority=_authority(),
        shard_size=1,
    )
    changed = dict(_authority(), baseline_sha256="BAD")
    with pytest.raises(CacheViolation, match="baseline"):
        load_evidence_cache(root, expected_authority=changed)

    shard = next((root / "shards").glob("*.pt"))
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(CacheViolation, match="sha256|bytes"):
        load_evidence_cache(root, expected_authority=_authority())


def test_loader_rejects_unsafe_manifest_path(tmp_path) -> None:
    root = tmp_path / "cache"
    write_evidence_cache(
        root,
        train_records=[_record(0, "train-0", 0)],
        val_records=[_record(0, "val-0", 1)],
        authority=_authority(),
        shard_size=1,
    )
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["shards"][0]["path"] = "../escape.pt"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CacheViolation, match="path"):
        load_evidence_cache(root, expected_authority=_authority())


def test_completion_manifest_is_written_last(tmp_path, monkeypatch) -> None:
    root = tmp_path / "cache"

    def fail_save(*_args, **_kwargs):
        raise RuntimeError("injected shard failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="injected"):
        write_evidence_cache(
            root,
            train_records=[_record(0, "train-0", 0)],
            val_records=[_record(0, "val-0", 1)],
            authority=_authority(),
            shard_size=1,
        )
    assert not (root / "manifest.json").exists()
