from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest

from src.saded_single_model_evidence import (
    source_state,
    validate_checkpoint_metadata,
    validate_binding_hashes,
    verify_checksum_closure,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checksum_closure_requires_exact_artifact_set(tmp_path):
    (tmp_path / "a.json").write_text("a", encoding="utf-8")
    (tmp_path / "b.json").write_text("b", encoding="utf-8")
    (tmp_path / "checksums.sha256").write_text(
        f"{_sha(tmp_path / 'a.json')}  a.json\n"
        f"{_sha(tmp_path / 'b.json')}  b.json\n",
        encoding="ascii",
    )

    result = verify_checksum_closure(
        tmp_path,
        expected_artifacts={"a.json", "b.json"},
    )

    assert result["passed"] is True
    assert result["artifact_count"] == 2


def test_checksum_closure_rejects_extra_or_changed_artifact(tmp_path):
    (tmp_path / "a.json").write_text("a", encoding="utf-8")
    (tmp_path / "b.json").write_text("b", encoding="utf-8")
    (tmp_path / "checksums.sha256").write_text(
        f"{_sha(tmp_path / 'a.json')}  a.json\n"
        f"{_sha(tmp_path / 'b.json')}  b.json\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="artifact set"):
        verify_checksum_closure(
            tmp_path,
            expected_artifacts={"a.json"},
        )

    (tmp_path / "a.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksum_closure(
            tmp_path,
            expected_artifacts={"a.json", "b.json"},
        )


def test_binding_hashes_require_exact_labels_and_values(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    paths = {"first": first, "second": second}
    expected = {"first": _sha(first), "second": _sha(second)}

    assert validate_binding_hashes(paths, expected) == expected

    with pytest.raises(ValueError, match="binding label set"):
        validate_binding_hashes(paths, {"first": expected["first"]})
    with pytest.raises(ValueError, match="binding checksum mismatch"):
        validate_binding_hashes(
            paths,
            {"first": expected["first"], "second": "0" * 64},
        )


def test_source_state_requires_clean_tracked_files(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "tracked.py"
    tracked.write_text("x = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tracked.py"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "fixture"],
        cwd=tmp_path,
        check=True,
    )

    state = source_state(tmp_path, ("tracked.py",))

    assert state["clean"] is True
    assert state["files"]["tracked.py"] == _sha(tracked)
    (tmp_path / "untracked").write_text("drift", encoding="utf-8")
    with pytest.raises(ValueError, match="not clean"):
        source_state(tmp_path, ("tracked.py",))


def test_checkpoint_metadata_proves_fixed_epoch_100_contract():
    metadata = {
        "train_args": {
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
        },
        "train_results": {"epoch": list(range(12, 101))},
    }

    result = validate_checkpoint_metadata(metadata)

    assert result["passed"] is True
    assert result["fixed_endpoint_epoch"] == 100
    metadata["train_results"]["epoch"][-1] = 99
    with pytest.raises(ValueError, match="epoch-100"):
        validate_checkpoint_metadata(metadata)
