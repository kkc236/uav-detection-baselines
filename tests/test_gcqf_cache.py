import json
from pathlib import Path

import pytest
import torch

from src.gcqf_cache import (
    CACHE_SCHEMA_VERSION,
    GCQFEvidenceRecord,
    VerifiedEvidenceCache,
    write_evidence_cache,
)
from src.gcte_types import QueryEvidence, ViewGeometry


def _evidence(query_count: int) -> QueryEvidence:
    return QueryEvidence(
        queries=torch.zeros(1, query_count, 2),
        logits=torch.zeros(1, query_count, 3),
        boxes=torch.full((1, query_count, 4), 0.25),
        quality=torch.full((1, query_count, 1), 0.5),
    )


def _record(image_id: str) -> GCQFEvidenceRecord:
    local_count = 4 * 300
    return GCQFEvidenceRecord(
        image_id=image_id,
        global_evidence=_evidence(300),
        local_evidence=_evidence(local_count),
        geometry=ViewGeometry(
            homography=torch.eye(3)
            .reshape(1, 1, 3, 3)
            .repeat(1, local_count, 1, 1),
            crop_metadata=torch.tensor(
                [[0.0, 0.0, 0.6, 0.6, 1.0, 1.0]]
            )
            .reshape(1, 1, 6)
            .repeat(1, local_count, 1),
            view_index=torch.arange(4)
            .repeat_interleave(300)
            .reshape(1, local_count),
            valid_mask=torch.ones(1, local_count, dtype=torch.bool),
        ),
        anchor_mask=torch.ones(1, local_count, 1, dtype=torch.bool),
        quality_targets=torch.zeros(1, local_count, 1),
        equivariance_pairs=torch.tensor([[0, 300]], dtype=torch.long),
        fixed_anchor_payload={"predictions": []},
    )


def _write(tmp_path: Path) -> Path:
    return write_evidence_cache(
        output=tmp_path / "cache",
        records=[_record("a.jpg"), _record("b.jpg")],
        baseline_sha256="A" * 64,
        dataset_signature="B" * 64,
        split="train10",
        records_per_shard=1,
    )


def test_cache_manifest_binds_schema_source_and_all_shards(tmp_path):
    manifest_path = _write(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == CACHE_SCHEMA_VERSION
    assert manifest["baseline_sha256"] == "A" * 64
    assert manifest["dataset_signature"] == "B" * 64
    assert manifest["record_count"] == 2
    assert manifest["queries"] == {
        "global": 300,
        "local_per_view": 300,
        "local_views": 4,
        "local_total": 1200,
    }
    assert len(manifest["shards"]) == 2
    assert all(len(item["sha256"]) == 64 for item in manifest["shards"])


def test_verified_cache_round_trips_records(tmp_path):
    manifest_path = _write(tmp_path)

    cache = VerifiedEvidenceCache(
        manifest_path,
        expected_baseline_sha256="A" * 64,
        expected_dataset_signature="B" * 64,
    )
    records = list(cache.iter_records())

    assert [record.image_id for record in records] == ["a.jpg", "b.jpg"]
    assert records[0].local_evidence.query_count == 1200
    assert records[0].equivariance_pairs.tolist() == [[0, 300]]


def test_cache_rejects_corrupted_shard_before_loading(tmp_path):
    manifest_path = _write(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard = manifest_path.parent / manifest["shards"][0]["file"]
    shard.write_bytes(shard.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="checksum"):
        VerifiedEvidenceCache(manifest_path)


def test_cache_rejects_unregistered_extra_shard(tmp_path):
    manifest_path = _write(tmp_path)
    (manifest_path.parent / "extra.pt").write_bytes(b"extra")

    with pytest.raises(ValueError, match="extra"):
        VerifiedEvidenceCache(manifest_path)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("expected_baseline_sha256", "C" * 64, "baseline"),
        ("expected_dataset_signature", "D" * 64, "dataset"),
    ),
)
def test_cache_rejects_authority_mismatch(
    tmp_path,
    keyword,
    value,
    message,
):
    manifest_path = _write(tmp_path)

    with pytest.raises(ValueError, match=message):
        VerifiedEvidenceCache(manifest_path, **{keyword: value})


def test_record_rejects_non_300_query_views():
    record = _record("bad.jpg")

    with pytest.raises(ValueError, match="1200"):
        GCQFEvidenceRecord(
            image_id=record.image_id,
            global_evidence=record.global_evidence,
            local_evidence=_evidence(1199),
            geometry=record.geometry,
            anchor_mask=record.anchor_mask,
            quality_targets=record.quality_targets,
            equivariance_pairs=record.equivariance_pairs,
            fixed_anchor_payload=record.fixed_anchor_payload,
        )
