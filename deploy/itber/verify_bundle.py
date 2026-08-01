"""Verify an I-TBER transfer bundle without following untrusted paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from src.itber_protocol import EXPECTED_BASELINE_SHA256


SHA256_PATTERN = re.compile(r"^[0-9A-F]{64}$")


class BundleViolation(ValueError):
    """A transfer manifest or one of its files is invalid."""

    def __init__(self, violations: Mapping[str, Any]) -> None:
        self.violations = dict(violations)
        super().__init__("bundle violation: " + ", ".join(sorted(self.violations)))


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _safe_bundle_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise BundleViolation({"path": {"actual": relative}})
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise BundleViolation({"path": {"actual": relative}})
    candidate = root.joinpath(*parsed.parts)
    resolved_root = root.resolve()
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise BundleViolation({"path": {"actual": relative}}) from error
    current = candidate
    while current != root:
        if current.exists() and current.is_symlink():
            raise BundleViolation({"path.symlink": {"actual": relative}})
        current = current.parent
    return candidate


def verify_bundle(root: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    """Verify format, authority, paths, sizes, and SHA-256 values."""
    root = Path(root)
    manifest_path = Path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleViolation({"manifest": {"actual": str(error)}}) from error
    violations: dict[str, Any] = {}
    if manifest.get("format_version") != 1:
        violations["format_version"] = {
            "expected": 1,
            "actual": manifest.get("format_version"),
        }
    baseline = str(manifest.get("baseline_sha256", "")).upper()
    if baseline != EXPECTED_BASELINE_SHA256:
        violations["baseline_sha256"] = {
            "expected": EXPECTED_BASELINE_SHA256,
            "actual": baseline or None,
        }
    records = manifest.get("files")
    if not isinstance(records, list):
        violations["files"] = {"expected": "list", "actual": type(records).__name__}
        records = []

    seen: set[str] = set()
    total_bytes = 0
    for index, record in enumerate(records):
        prefix = f"files.{index}"
        if not isinstance(record, dict):
            violations[prefix] = {"expected": "object", "actual": type(record).__name__}
            continue
        relative = record.get("path")
        if relative in seen:
            violations[f"{prefix}.duplicate_path"] = {"actual": relative}
            continue
        if isinstance(relative, str):
            seen.add(relative)
        try:
            path = _safe_bundle_path(root, relative)
        except BundleViolation as error:
            violations[f"{prefix}.path"] = error.violations
            continue
        expected_bytes = record.get("bytes")
        expected_sha = str(record.get("sha256", "")).upper()
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            violations[f"{prefix}.bytes"] = {"actual": expected_bytes}
            continue
        if not SHA256_PATTERN.fullmatch(expected_sha):
            violations[f"{prefix}.sha256"] = {"actual": expected_sha or None}
            continue
        if not path.is_file():
            violations[f"{prefix}.missing"] = {"path": str(path)}
            continue
        actual_bytes = path.stat().st_size
        actual_sha = _stream_sha256(path)
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
        "root": str(root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "file_count": len(records),
        "total_bytes": total_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify_bundle(args.root, args.manifest)
        status = 0
    except BundleViolation as error:
        report = {"status": "invalid", "violations": error.violations}
        status = 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
