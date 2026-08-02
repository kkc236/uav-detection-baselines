"""Verify an IBER-BE transfer bundle against immutable authorities."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from src.iber_protocol import (
    DESIGN_VERSION,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    execution_environment,
)


SHA256_PATTERN = re.compile(r"^[0-9A-F]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
FOREIGN_PATH_PATTERN = re.compile(
    r"(?:^|[-_/])i[-_]?tber(?:$|[-_./])", re.IGNORECASE
)


class BundleViolation(ValueError):
    """A transfer manifest or one of its artifacts is invalid."""

    def __init__(self, violations: Mapping[str, Any]) -> None:
        self.violations = dict(violations)
        super().__init__("IBER-BE bundle violation: " + ", ".join(sorted(self.violations)))


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _safe_bundle_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise BundleViolation({"path": {"actual": relative}})
    if FOREIGN_PATH_PATTERN.search(relative):
        raise BundleViolation({"path.foreign_identity": {"actual": relative}})
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise BundleViolation({"path": {"actual": relative}})

    resolved_root = root.resolve(strict=True)
    candidate = resolved_root.joinpath(*parsed.parts)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise BundleViolation({"path": {"actual": relative}}) from error

    current = candidate
    while current != resolved_root:
        if current.exists() and current.is_symlink():
            raise BundleViolation({"path.symlink": {"actual": relative}})
        current = current.parent
    return candidate


def _expected_authority(source_commit: str) -> dict[str, Any]:
    return {
        "format_version": 1,
        "design_version": DESIGN_VERSION,
        "source_commit": source_commit,
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "dataset.sha256": EXPECTED_DATASET_SHA256,
        "subset.sha256": EXPECTED_SUBSET_SHA256,
        "execution_environment": execution_environment(),
    }


def verify_bundle(
    root: str | Path,
    manifest_path: str | Path,
    *,
    expected_source_commit: str,
) -> dict[str, Any]:
    """Verify identity, source commit, paths, sizes, and streaming SHA-256 values."""
    source_commit = str(expected_source_commit).lower()
    if COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise BundleViolation({"expected_source_commit": {"actual": expected_source_commit}})

    root_path = Path(root)
    manifest = Path(manifest_path)
    if manifest.is_symlink():
        raise BundleViolation({"manifest.symlink": {"actual": str(manifest)}})
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleViolation({"manifest": {"actual": str(error)}}) from error
    if not isinstance(payload, dict):
        raise BundleViolation({"manifest": {"expected": "object"}})

    expected = _expected_authority(source_commit)
    actual = {
        "format_version": payload.get("format_version"),
        "design_version": payload.get("design_version"),
        "source_commit": str(payload.get("source_commit", "")).lower(),
        "baseline_sha256": str(payload.get("baseline_sha256", "")).upper(),
        "dataset.sha256": str(payload.get("dataset", {}).get("sha256", "")).upper()
        if isinstance(payload.get("dataset"), dict)
        else None,
        "subset.sha256": str(payload.get("subset", {}).get("sha256", "")).upper()
        if isinstance(payload.get("subset"), dict)
        else None,
        "execution_environment": payload.get("execution_environment"),
    }
    violations: dict[str, Any] = {}
    for name, expected_value in expected.items():
        if actual[name] != expected_value:
            violations[name] = {"expected": expected_value, "actual": actual[name]}

    records = payload.get("files")
    if not isinstance(records, list) or not records:
        violations["files"] = {
            "expected": "non-empty list",
            "actual": type(records).__name__,
        }
        records = []

    seen: set[str] = set()
    total_bytes = 0
    for index, record in enumerate(records):
        prefix = f"files.{index}"
        if not isinstance(record, dict):
            violations[prefix] = {"expected": "object", "actual": type(record).__name__}
            continue
        relative = record.get("path")
        if isinstance(relative, str) and relative in seen:
            violations[f"{prefix}.duplicate_path"] = {"actual": relative}
            continue
        if isinstance(relative, str):
            seen.add(relative)
        try:
            artifact = _safe_bundle_path(root_path, relative)
        except BundleViolation as error:
            violations[f"{prefix}.path"] = error.violations
            continue

        expected_bytes = record.get("bytes")
        expected_sha = str(record.get("sha256", "")).upper()
        if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes < 0:
            violations[f"{prefix}.bytes"] = {"actual": expected_bytes}
            continue
        if SHA256_PATTERN.fullmatch(expected_sha) is None:
            violations[f"{prefix}.sha256"] = {"actual": expected_sha or None}
            continue
        if not artifact.is_file() or artifact.is_symlink():
            violations[f"{prefix}.missing"] = {"path": str(artifact)}
            continue

        actual_bytes = artifact.stat().st_size
        actual_sha = _stream_sha256(artifact)
        if actual_bytes != expected_bytes:
            violations[f"{prefix}.bytes"] = {
                "expected": expected_bytes,
                "actual": actual_bytes,
            }
        if actual_sha != expected_sha:
            violations[f"{prefix}.sha256"] = {
                "expected": expected_sha,
                "actual": actual_sha,
            }
        total_bytes += actual_bytes

    if violations:
        raise BundleViolation(violations)
    return {
        "status": "passed",
        "design_version": DESIGN_VERSION,
        "source_commit": source_commit,
        "root": str(root_path.resolve()),
        "manifest": str(manifest.resolve()),
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "subset_sha256": EXPECTED_SUBSET_SHA256,
        "file_count": len(records),
        "total_bytes": total_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    try:
        report = verify_bundle(
            args.root,
            args.manifest,
            expected_source_commit=args.source_commit,
        )
        status = 0
    except BundleViolation as error:
        report = {"status": "invalid", "violations": error.violations}
        status = 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
