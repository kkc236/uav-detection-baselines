import hashlib
import json
from pathlib import Path

import torch
import yaml

from scripts.validate_tsgr_e0_promotion import (
    EXPECTED_E1_TRAINING,
    EXPECTED_TSGR_CONFIG,
    _atomic_write_locked,
    validate_authoritative_artifacts,
    validate_production_preflight,
)
from src.ebc_qp_protocol import dataset_signature, state_fingerprint, subset_signature


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _json_sha(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(data).hexdigest().upper()


def _build_authority(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    for relative, content in (
        ("images/train/a.jpg", b"image-a"),
        ("images/val/b.jpg", b"image-b"),
        ("labels/train/a.txt", b"0 0.5 0.5 0.1 0.1\n"),
        ("labels/val/b.txt", b"0 0.5 0.5 0.1 0.1\n"),
    ):
        path = dataset_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    selected = [dataset_root / "images/train/a.jpg"]
    subset_path = tmp_path / "d2-train-10pct.txt"
    subset_path.write_text(f"{selected[0].resolve()}\n", encoding="utf-8")
    subset = {
        "count": 1,
        "fraction": 0.1,
        "path": str(subset_path.resolve()),
        "sha256": subset_signature(selected, root=dataset_root),
    }
    names = {0: "object"}
    data_path = tmp_path / "data.yaml"
    data_path.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root.resolve()),
                "train": str(subset_path.resolve()),
                "val": "images/val",
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    source_paths = {}
    for name in ("head.py", "tasks.py", "rtdetr-l.yaml"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        source_paths[name] = path
    source_sha = {name: _file_sha(path) for name, path in source_paths.items()}
    environment = {"python": "3.10", "torch": "2.5", "cuda": "12.1", "ultralytics": "8.4.90", "gpu": "GPU"}
    dataset = dataset_signature(dataset_root)
    experiment_payload = {
        "dataset": dataset,
        "category_mapping_sha256": _json_sha(names),
        "subset": {key: value for key, value in subset.items() if key != "path"},
        "data_sha256": _file_sha(data_path),
        "source_sha256": source_sha,
        "git_commit": "TRAINING-COMMIT",
        "environment": environment,
        "e1_training": json.loads(json.dumps(EXPECTED_E1_TRAINING)),
        "tsgr_config": json.loads(json.dumps(EXPECTED_TSGR_CONFIG)),
    }
    experiment_signature = _json_sha(experiment_payload)
    frozen_path = tmp_path / "e1-experiment-signature.json"
    frozen_path.write_text(
        json.dumps({"format_version": 1, "experiment_signature": experiment_signature, "payload": experiment_payload}),
        encoding="utf-8",
    )
    metadata = {
        "seed": 0,
        "dataset": dataset,
        "category_mapping_sha256": _json_sha(names),
        "subset": {key: value for key, value in subset.items() if key != "path"},
        "source_sha256": source_sha,
        "git_commit": "TRAINING-COMMIT",
        "environment": environment,
        "experiment_signature": experiment_signature,
    }
    common_state = {"model.weight": torch.tensor([1.0])}
    innovation_state = {"model.p2_adapter.weight": torch.tensor([2.0])}
    initial_path = tmp_path / "initial.pt"
    torch.save(
        {
            "metadata": metadata,
            "common_state": common_state,
            "innovation_state": innovation_state,
            "fingerprints": {
                "common": state_fingerprint(common_state),
                "innovation": state_fingerprint(innovation_state),
            },
        },
        initial_path,
    )
    manifest = {
        "format_version": 1,
        "seed": 0,
        "experiment_signature": experiment_signature,
        "dataset": dataset,
        "category_mapping_sha256": _json_sha(names),
        "subset": subset,
        "data": {"path": str(data_path.resolve()), "sha256": _file_sha(data_path)},
        "initial_state": {"path": str(initial_path.resolve()), "sha256": _file_sha(initial_path)},
        "source_sha256": source_sha,
        "git_commit": "TRAINING-COMMIT",
        "environment": environment,
    }
    manifest["signature"] = _json_sha(manifest)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(manifest), encoding="utf-8")
    return protocol_path, frozen_path, data_path, initial_path, subset_path, source_paths


def test_promotion_authority_recomputes_every_signed_artifact(tmp_path):
    args = _build_authority(tmp_path)
    result = validate_authoritative_artifacts(*args)
    assert result["evidence_valid"] is True
    assert result["errors"] == []
    assert result["dataset"]["file_count"] == 4

    args[2].write_text("path: changed\n", encoding="utf-8")
    result = validate_authoritative_artifacts(*args)
    assert result["evidence_valid"] is False
    assert "data SHA mismatch" in result["errors"]


def test_promotion_authority_rejects_semantic_contract_and_source_drift(tmp_path):
    args = _build_authority(tmp_path)
    frozen = json.loads(args[1].read_text(encoding="utf-8"))
    frozen["payload"]["e1_training"]["expected_optimizer_attempts"] = 144
    frozen["experiment_signature"] = _json_sha(frozen["payload"])
    args[1].write_text(json.dumps(frozen), encoding="utf-8")
    result = validate_authoritative_artifacts(*args)
    assert result["evidence_valid"] is False
    assert "frozen E1 training contract mismatch" in result["errors"]

    args = _build_authority(tmp_path / "missing-source")
    source_paths = dict(args[5])
    source_paths.pop("tasks.py")
    result = validate_authoritative_artifacts(*args[:5], source_paths)
    assert result["evidence_valid"] is False
    assert "Ultralytics source keyset mismatch" in result["errors"]


def test_promotion_authority_rejects_subset_and_initial_metadata_drift(tmp_path):
    args = _build_authority(tmp_path)
    replacement = tmp_path / "dataset" / "images" / "val" / "b.jpg"
    args[4].write_text(f"{replacement.resolve()}\n", encoding="utf-8")
    result = validate_authoritative_artifacts(*args)
    assert result["evidence_valid"] is False
    assert any("subset semantic" in error for error in result["errors"])

    args = _build_authority(tmp_path / "initial-drift")
    artifact = torch.load(args[3], map_location="cpu", weights_only=False)
    artifact["metadata"]["experiment_signature"] = "OTHER"
    torch.save(artifact, args[3])
    result = validate_authoritative_artifacts(*args)
    assert result["evidence_valid"] is False
    assert "initial-state metadata mismatch: experiment_signature" in result["errors"]


def test_promotion_sidecar_has_recomputable_signature_and_external_checksum(tmp_path):
    output = tmp_path / "promotion.json"
    payload = {"format_version": 1, "promotion_ready": False, "classification": "PENDING"}
    payload["signature"] = _json_sha(payload)
    checksum = _atomic_write_locked(output, payload)

    saved = json.loads(output.read_text(encoding="utf-8"))
    signature = saved.pop("signature")
    assert signature == _json_sha(saved)
    assert checksum == _file_sha(output)
    assert output.with_suffix(".json.sha256").read_text(encoding="ascii").startswith(checksum)


def test_production_preflight_requires_full_nonaborting_monitor_trace(tmp_path):
    evidence_path = tmp_path / "optimizer-evidence.jsonl"
    records = [
        {
            "optimizer_attempt": index,
            "amp_scale_before": 128.0,
            "amp_scale_after": 128.0,
            "amp_step_skipped": False,
            "nonfinite_fields": [],
            "runtime_violation": None,
            "update_monitor_ratio": 0.5,
            "update_monitor_abort": False,
            "p2_entry_count": 0,
            "ordinary_query_count": 300,
        }
        for index in range(1, 146)
    ]
    evidence_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    manifest = {
        "format_version": 1,
        "stage": "e1",
        "arm": "tsgr-p2",
        "seed": 0,
        "git_commit_start": "FINAL-COMMIT",
        "git_commit_end": "FINAL-COMMIT",
        "ebc_config": EXPECTED_TSGR_CONFIG,
        "controlled_amp": {
            "init_scale": 128.0,
            "growth_interval": 2**31 - 1,
            "expected_optimizer_attempts": 145,
            "optimizer_attempts": 145,
            "skipped_attempts": 0,
        },
        "artifacts": {
            "optimizer_evidence": {
                "path": str(evidence_path.resolve()),
                "bytes": evidence_path.stat().st_size,
                "sha256": _file_sha(evidence_path),
            }
        },
    }
    manifest["signature"] = _json_sha(manifest)
    manifest_path = tmp_path / "e1-run-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_production_preflight(manifest_path, expected_commit="FINAL-COMMIT")
    assert result["verified"] is True
    assert result["update_monitor_abort_count"] == 0

    records[73]["update_monitor_abort"] = True
    evidence_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    manifest["artifacts"]["optimizer_evidence"].update(
        bytes=evidence_path.stat().st_size,
        sha256=_file_sha(evidence_path),
    )
    manifest.pop("signature")
    manifest["signature"] = _json_sha(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_production_preflight(manifest_path, expected_commit="FINAL-COMMIT")
    assert result["verified"] is False
    assert any("attempt 74" in error for error in result["errors"])
