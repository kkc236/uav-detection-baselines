from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from src.fdr_bpdd_ra_glgm_protocol import (
    COMBO_PROTOCOL,
    COMBO_PROTOCOL_SHA256,
    build_combo_run_identity,
    load_combo_authority,
)
from src.fdr_protocol import canonical_json_bytes


def test_combo_protocol_hash_is_canonical_and_stage_identity_isolated() -> None:
    assert COMBO_PROTOCOL_SHA256 == hashlib.sha256(
        canonical_json_bytes(COMBO_PROTOCOL)
    ).hexdigest().upper()
    source = {"git_commit": "a" * 40, "tree_sha256": "B" * 64}
    smoke = build_combo_run_identity(source, stage="smoke", gpu_uuid="GPU-test")
    formal = build_combo_run_identity(source, stage="formal", gpu_uuid="GPU-test")
    assert smoke["run_id"] != formal["run_id"]
    assert smoke["protocol_sha256"] == formal["protocol_sha256"]


def _authority(tmp_path: Path) -> Path:
    source = {"git_commit": "a" * 40, "tree_sha256": "B" * 64}
    initial = tmp_path / "initial.pt"
    torch.save({"fingerprints": {"common": "x", "private": "y"}}, initial)
    payload = {
        "format_version": 1,
        "source": source,
        "source_sha256": build_combo_run_identity(
            source, stage="smoke", gpu_uuid="GPU-test"
        )["source_sha256"],
        "protocol": COMBO_PROTOCOL,
        "protocol_sha256": COMBO_PROTOCOL_SHA256,
        "gpu_uuid": "GPU-test",
        "initial_state": {
            "path": str(initial),
            "sha256": "unused",
            "fingerprints": {"common": "x", "private": "y"},
        },
        "dataset_authority": {
            "root": str(tmp_path),
            "positive": {"file_count": 0, "sha256": "unused"},
            "ignore": {},
        },
        "run_identities": {
            stage: build_combo_run_identity(source, stage=stage, gpu_uuid="GPU-test")
            for stage in ("smoke", "formal")
        },
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_combo_authority_rejects_manifest_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _authority(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["gpu_uuid"] = "GPU-tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "src.ra_experiment_protocol.current_source_identity", lambda _root: payload["source"]
    )
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        load_combo_authority(path, repository_root=tmp_path)


def test_combo_authority_rejects_positive_dataset_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _authority(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        "src.ra_experiment_protocol.current_source_identity", lambda _root: payload["source"]
    )
    monkeypatch.setattr("src.ra_experiment_protocol.file_sha256", lambda _path: "UNUSED")
    monkeypatch.setattr("src.ra_glgm_protocol.validate_ra_glgm_initial_state", lambda _a: None)
    monkeypatch.setattr(
        "src.lpr_protocol.dataset_signature",
        lambda _root: {"file_count": 1, "sha256": "drift"},
    )
    monkeypatch.setattr("src.ra_experiment_protocol.ignore_sidecar_signature", lambda _root: {})
    with pytest.raises(ValueError, match="positive dataset differs"):
        load_combo_authority(path, repository_root=tmp_path)


def test_combo_authority_rejects_protocol_payload_drift(tmp_path: Path) -> None:
    path = _authority(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol"] = deepcopy(payload["protocol"])
    payload["protocol"]["seed"] = 1
    payload["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in payload.items() if key != "manifest_sha256"})
    ).hexdigest().upper()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol payload differs"):
        load_combo_authority(path, repository_root=tmp_path)
