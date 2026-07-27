"""Sealed, checksummed evidence cache for module-only GCQF training."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch

from src.gcte_types import QueryEvidence, ViewGeometry


CACHE_SCHEMA_VERSION = "gcte-gcqf-evidence/v1"
GLOBAL_QUERIES = 300
LOCAL_VIEWS = 4
LOCAL_QUERIES_PER_VIEW = 300
LOCAL_QUERIES = LOCAL_VIEWS * LOCAL_QUERIES_PER_VIEW


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _validate_digest(name: str, value: str) -> str:
    candidate = str(value).upper()
    if len(candidate) != 64 or any(
        character not in "0123456789ABCDEF" for character in candidate
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return candidate


@dataclass(frozen=True)
class GCQFEvidenceRecord:
    image_id: str
    global_evidence: QueryEvidence
    local_evidence: QueryEvidence
    geometry: ViewGeometry
    anchor_mask: torch.Tensor
    quality_targets: torch.Tensor
    equivariance_pairs: torch.Tensor
    fixed_anchor_payload: dict[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.image_id, str)
            or not self.image_id
            or ".." in Path(self.image_id).parts
        ):
            raise ValueError("image_id must be a canonical relative identity")
        if (
            self.global_evidence.batch_size != 1
            or self.global_evidence.query_count != GLOBAL_QUERIES
        ):
            raise ValueError("global evidence must contain exactly 300 queries")
        if (
            self.local_evidence.batch_size != 1
            or self.local_evidence.query_count != LOCAL_QUERIES
        ):
            raise ValueError("local evidence must contain exactly 1200 queries")
        if (
            self.global_evidence.query_dim != self.local_evidence.query_dim
            or self.global_evidence.num_classes
            != self.local_evidence.num_classes
        ):
            raise ValueError("global and local evidence dimensions must match")
        if (
            self.geometry.batch_size != 1
            or self.geometry.query_count != LOCAL_QUERIES
        ):
            raise ValueError("geometry must describe all 1200 local queries")
        expected = (1, LOCAL_QUERIES, 1)
        if (
            self.anchor_mask.shape != expected
            or self.anchor_mask.dtype != torch.bool
        ):
            raise ValueError("anchor_mask must be bool [1,1200,1]")
        if self.quality_targets.shape != expected:
            raise ValueError("quality_targets must be [1,1200,1]")
        if not self.quality_targets.is_floating_point() or not bool(
            torch.isfinite(self.quality_targets).all()
        ):
            raise ValueError("quality_targets must be finite floating point")
        if bool(
            (
                (self.quality_targets < 0.0)
                | (self.quality_targets > 1.0)
            ).any()
        ):
            raise ValueError("quality_targets must be in [0,1]")
        if (
            self.equivariance_pairs.ndim != 2
            or self.equivariance_pairs.shape[1:] != (2,)
            or self.equivariance_pairs.dtype != torch.long
        ):
            raise ValueError("equivariance_pairs must be long [P,2]")
        if self.equivariance_pairs.numel() and (
            bool((self.equivariance_pairs < 0).any())
            or bool((self.equivariance_pairs >= LOCAL_QUERIES).any())
        ):
            raise ValueError("equivariance pair index is out of range")
        if not isinstance(self.fixed_anchor_payload, dict):
            raise ValueError("fixed_anchor_payload must be a mapping")
        for view_index in range(LOCAL_VIEWS):
            count = int(
                (
                    (self.geometry.view_index == view_index)
                    & self.geometry.valid_mask
                ).sum()
            )
            if count != LOCAL_QUERIES_PER_VIEW:
                raise ValueError(
                    "each local view must contain exactly 300 valid queries"
                )

    def to_payload(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "global_evidence": _evidence_payload(self.global_evidence),
            "local_evidence": _evidence_payload(self.local_evidence),
            "geometry": {
                "homography": self.geometry.homography.detach().cpu(),
                "crop_metadata": self.geometry.crop_metadata.detach().cpu(),
                "view_index": self.geometry.view_index.detach().cpu(),
                "valid_mask": self.geometry.valid_mask.detach().cpu(),
            },
            "anchor_mask": self.anchor_mask.detach().cpu(),
            "quality_targets": self.quality_targets.detach().cpu(),
            "equivariance_pairs": self.equivariance_pairs.detach().cpu(),
            "fixed_anchor_payload": self.fixed_anchor_payload,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GCQFEvidenceRecord":
        required = {
            "image_id",
            "global_evidence",
            "local_evidence",
            "geometry",
            "anchor_mask",
            "quality_targets",
            "equivariance_pairs",
            "fixed_anchor_payload",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("evidence record schema drift")
        geometry = payload["geometry"]
        if not isinstance(geometry, dict) or set(geometry) != {
            "homography",
            "crop_metadata",
            "view_index",
            "valid_mask",
        }:
            raise ValueError("view geometry schema drift")
        return cls(
            image_id=payload["image_id"],
            global_evidence=_evidence_from_payload(
                payload["global_evidence"]
            ),
            local_evidence=_evidence_from_payload(
                payload["local_evidence"]
            ),
            geometry=ViewGeometry(**geometry),
            anchor_mask=payload["anchor_mask"],
            quality_targets=payload["quality_targets"],
            equivariance_pairs=payload["equivariance_pairs"],
            fixed_anchor_payload=payload["fixed_anchor_payload"],
        )


def _evidence_payload(evidence: QueryEvidence) -> dict[str, torch.Tensor]:
    return {
        # GCQF G0 always consumes decoder queries under CUDA autocast.  Seal
        # them in the exact compute dtype to avoid duplicating almost 1 GiB of
        # inactive FP32 mantissa across the train10 and validation caches.
        "queries": evidence.queries.detach().to(
            device="cpu",
            dtype=torch.float16,
        ),
        "logits": evidence.logits.detach().cpu(),
        "boxes": evidence.boxes.detach().cpu(),
        "quality": evidence.quality.detach().cpu(),
    }


def _evidence_from_payload(payload: Any) -> QueryEvidence:
    if not isinstance(payload, dict) or set(payload) != {
        "queries",
        "logits",
        "boxes",
        "quality",
    }:
        raise ValueError("query evidence schema drift")
    return QueryEvidence(**payload)


def write_evidence_cache(
    *,
    output: str | Path,
    records: Iterable[GCQFEvidenceRecord],
    baseline_sha256: str,
    dataset_signature: str,
    split: str,
    records_per_shard: int = 16,
) -> Path:
    """Write records once and seal every shard in a deterministic manifest."""

    if records_per_shard <= 0:
        raise ValueError("records_per_shard must be positive")
    if not split:
        raise ValueError("split must be nonempty")
    baseline = _validate_digest("baseline_sha256", baseline_sha256)
    dataset = _validate_digest("dataset_signature", dataset_signature)
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    shards: list[dict[str, Any]] = []
    batch: list[dict[str, Any]] = []
    record_count = 0

    def flush() -> None:
        nonlocal batch
        if not batch:
            return
        name = f"shard-{len(shards):05d}.pt"
        path = root / name
        torch.save({"records": batch}, path)
        shards.append(
            {
                "file": name,
                "sha256": _sha256_file(path),
                "record_count": len(batch),
            }
        )
        batch = []

    for record in records:
        if not isinstance(record, GCQFEvidenceRecord):
            raise TypeError("records must contain GCQFEvidenceRecord values")
        batch.append(record.to_payload())
        record_count += 1
        if len(batch) == records_per_shard:
            flush()
    flush()
    if record_count == 0:
        raise ValueError("cache requires at least one record")
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "baseline_sha256": baseline,
        "dataset_signature": dataset,
        "split": split,
        "record_count": record_count,
        "queries": {
            "global": GLOBAL_QUERIES,
            "local_per_view": LOCAL_QUERIES_PER_VIEW,
            "local_views": LOCAL_VIEWS,
            "local_total": LOCAL_QUERIES,
        },
        "shards": shards,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


class VerifiedEvidenceCache:
    """A cache that verifies its complete closed world before iteration."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        expected_baseline_sha256: str | None = None,
        expected_dataset_signature: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        self._validate_manifest(
            expected_baseline_sha256=expected_baseline_sha256,
            expected_dataset_signature=expected_dataset_signature,
        )
        self._validate_shards()

    def _validate_manifest(
        self,
        *,
        expected_baseline_sha256: str | None,
        expected_dataset_signature: str | None,
    ) -> None:
        manifest = self.manifest
        required = {
            "schema_version",
            "baseline_sha256",
            "dataset_signature",
            "split",
            "record_count",
            "queries",
            "shards",
        }
        if not isinstance(manifest, dict) or set(manifest) != required:
            raise ValueError("cache manifest schema drift")
        if manifest["schema_version"] != CACHE_SCHEMA_VERSION:
            raise ValueError("cache schema version mismatch")
        baseline = _validate_digest(
            "baseline_sha256",
            manifest["baseline_sha256"],
        )
        dataset = _validate_digest(
            "dataset_signature",
            manifest["dataset_signature"],
        )
        if expected_baseline_sha256 is not None and baseline != _validate_digest(
            "expected_baseline_sha256",
            expected_baseline_sha256,
        ):
            raise ValueError("cache baseline authority mismatch")
        if expected_dataset_signature is not None and dataset != _validate_digest(
            "expected_dataset_signature",
            expected_dataset_signature,
        ):
            raise ValueError("cache dataset authority mismatch")
        if manifest["queries"] != {
            "global": GLOBAL_QUERIES,
            "local_per_view": LOCAL_QUERIES_PER_VIEW,
            "local_views": LOCAL_VIEWS,
            "local_total": LOCAL_QUERIES,
        }:
            raise ValueError("cache query contract mismatch")
        if not isinstance(manifest["record_count"], int) or manifest[
            "record_count"
        ] <= 0:
            raise ValueError("cache record count must be positive")
        if not isinstance(manifest["shards"], list) or not manifest["shards"]:
            raise ValueError("cache must declare shards")

    def _validated_payloads(self) -> Iterator[list[dict[str, Any]]]:
        total = 0
        for item in self.manifest["shards"]:
            if not isinstance(item, dict) or set(item) != {
                "file",
                "sha256",
                "record_count",
            }:
                raise ValueError("cache shard manifest schema drift")
            path = self.root / item["file"]
            if not path.is_file():
                raise ValueError(f"cache shard missing: {item['file']}")
            observed = _sha256_file(path)
            if observed != _validate_digest("shard sha256", item["sha256"]):
                raise ValueError(f"cache shard checksum mismatch: {item['file']}")
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(payload, dict) or set(payload) != {"records"}:
                raise ValueError("cache shard payload schema drift")
            rows = payload["records"]
            if (
                not isinstance(rows, list)
                or len(rows) != item["record_count"]
            ):
                raise ValueError("cache shard record count mismatch")
            for row in rows:
                GCQFEvidenceRecord.from_payload(row)
            total += len(rows)
            yield rows
        if total != self.manifest["record_count"]:
            raise ValueError("cache manifest total record count mismatch")

    def _validate_shards(self) -> None:
        declared = {item["file"] for item in self.manifest["shards"]}
        observed = {path.name for path in self.root.glob("*.pt")}
        if observed - declared:
            raise ValueError(
                f"cache contains extra shards: {sorted(observed - declared)}"
            )
        if declared - observed:
            raise ValueError(
                f"cache is missing shards: {sorted(declared - observed)}"
            )
        for _ in self._validated_payloads():
            pass

    def iter_records(self) -> Iterator[GCQFEvidenceRecord]:
        for rows in self._validated_payloads():
            for row in rows:
                yield GCQFEvidenceRecord.from_payload(row)


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "GCQFEvidenceRecord",
    "VerifiedEvidenceCache",
    "write_evidence_cache",
]
