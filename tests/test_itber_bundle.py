from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deploy.itber.verify_bundle import BundleViolation, verify_bundle
from src.itber_protocol import EXPECTED_BASELINE_SHA256


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _write_manifest(tmp_path, files: dict[str, bytes]):
    root = tmp_path / "bundle"
    records = []
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        records.append(
            {"path": relative, "bytes": len(content), "sha256": _sha(content)}
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "baseline_sha256": EXPECTED_BASELINE_SHA256,
                "files": records,
            }
        ),
        encoding="utf-8",
    )
    return root, manifest


def test_valid_bundle_is_verified(tmp_path) -> None:
    root, manifest = _write_manifest(
        tmp_path, {"source/repository.bundle": b"git", "weights/baseline.pt": b"pt"}
    )

    report = verify_bundle(root, manifest)

    assert report["status"] == "passed"
    assert report["file_count"] == 2
    assert report["total_bytes"] == 5


@pytest.mark.parametrize("mutation", ["missing", "bytes", "sha", "baseline"])
def test_bundle_rejects_missing_changed_or_wrong_authority(tmp_path, mutation: str) -> None:
    root, manifest = _write_manifest(tmp_path, {"source/repository.bundle": b"git"})
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if mutation == "missing":
        (root / "source" / "repository.bundle").unlink()
    elif mutation == "bytes":
        payload["files"][0]["bytes"] = 4
    elif mutation == "sha":
        payload["files"][0]["sha256"] = "BAD"
    else:
        payload["baseline_sha256"] = "BAD"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BundleViolation):
        verify_bundle(root, manifest)


@pytest.mark.parametrize("bad_path", ["../escape", "/absolute", "a\\windows", "a/../../b"])
def test_bundle_rejects_path_traversal(tmp_path, bad_path: str) -> None:
    root, manifest = _write_manifest(tmp_path, {"safe": b"ok"})
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"][0]["path"] = bad_path
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BundleViolation, match="path"):
        verify_bundle(root, manifest)


def test_bundle_rejects_duplicate_paths(tmp_path) -> None:
    root, manifest = _write_manifest(tmp_path, {"safe": b"ok"})
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"].append(dict(payload["files"][0]))
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BundleViolation, match="duplicate"):
        verify_bundle(root, manifest)


def test_lock_contains_exact_scientific_runtime() -> None:
    lock = Path("requirements-itber.lock")
    content = lock.read_text(encoding="utf-8")
    for requirement in (
        "torch==2.5.1+cu121",
        "torchvision==0.20.1+cu121",
        "ultralytics==8.4.90",
        "opencv-python-headless==4.12.0.88",
        "pytest==9.1.1",
    ):
        assert requirement in content
