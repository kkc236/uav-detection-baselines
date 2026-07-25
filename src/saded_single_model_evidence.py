"""Checksum and immutable-binding helpers for SADED formal evidence."""

from __future__ import annotations

from collections.abc import Mapping, Set
import hashlib
from pathlib import Path
import subprocess
from typing import Any


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_checksum_file(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        if (
            len(parts) != 2
            or len(parts[0]) != 64
            or any(character not in "0123456789abcdef" for character in parts[0])
            or not parts[1]
            or Path(parts[1]).name != parts[1]
            or parts[1] in records
        ):
            raise ValueError("invalid checksum closure syntax")
        records[parts[1]] = parts[0]
    return records


def verify_checksum_closure(
    directory: Path | str,
    *,
    expected_artifacts: Set[str] | set[str],
) -> dict[str, object]:
    root = Path(directory).resolve()
    checksum_path = root / "checksums.sha256"
    expected = set(expected_artifacts)
    actual_names = {
        path.name for path in root.iterdir() if path.is_file()
    }
    if actual_names != expected | {"checksums.sha256"}:
        raise ValueError("checksum closure artifact set mismatch")
    records = _parse_checksum_file(checksum_path)
    if set(records) != expected:
        raise ValueError("checksum closure target set mismatch")
    for name, expected_hash in records.items():
        if sha256_file(root / name) != expected_hash:
            raise ValueError(f"checksum mismatch: {name}")
    return {
        "passed": True,
        "artifact_count": len(expected),
        "checksums_sha256": sha256_file(checksum_path),
        "artifacts": dict(records),
    }


def validate_binding_hashes(
    paths: Mapping[str, Path | str],
    expected_hashes: Mapping[str, str],
) -> dict[str, str]:
    if set(paths) != set(expected_hashes):
        raise ValueError("binding label set mismatch")
    actual: dict[str, str] = {}
    for label, path in paths.items():
        value = sha256_file(path)
        if value != str(expected_hashes[label]).lower():
            raise ValueError(f"binding checksum mismatch: {label}")
        actual[label] = value
    return actual


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def source_state(
    repo: Path | str,
    source_files: tuple[str, ...],
) -> dict[str, Any]:
    root = Path(repo).resolve()
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("successor source is not clean")
    files: dict[str, str] = {}
    for relative in source_files:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"successor source file is missing: {relative}")
        _git(root, "ls-files", "--error-unmatch", relative)
        files[relative] = sha256_file(path)
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "clean": True,
        "files": files,
    }


def validate_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    args = checkpoint.get("train_args")
    results = checkpoint.get("train_results")
    if not isinstance(args, Mapping) or not isinstance(results, Mapping):
        raise ValueError("checkpoint training metadata is absent")
    expected = {
        "epochs": 100,
        "seed": 0,
        "pretrained": False,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "deterministic": True,
        "amp": True,
        "max_det": 300,
        "nms": False,
    }
    actual = {key: args.get(key) for key in expected}
    if actual != expected:
        raise ValueError("checkpoint training contract drift")
    epochs = results.get("epoch")
    if (
        not isinstance(epochs, list)
        or not epochs
        or epochs[-1] != 100
    ):
        raise ValueError("checkpoint is not the fixed epoch-100 endpoint")
    return {
        "passed": True,
        "fixed_endpoint_epoch": 100,
        "recorded_epoch_count": len(epochs),
        "train_args": actual,
    }


__all__ = [
    "sha256_file",
    "source_state",
    "validate_checkpoint_metadata",
    "validate_binding_hashes",
    "verify_checksum_closure",
]
