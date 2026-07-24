from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.adjudicate_ascv_loc import load_preflight_summaries, load_records
from src.ascv_loc_adjudicator import build_preflight_gate, replay_preflight_gate
from src.ascv_loc_protocol import sha256_file


def _metrics(path: Path, value: float = 0.1) -> None:
    view = {
        "mAP50-95": value,
        "AP-tiny-SBR": value,
        "tiny_recall": value,
        "AP75": value,
        "AP-large-SBR": value,
    }
    path.write_text(json.dumps({"A": view, "C": view}))


def test_load_records_requires_exact_seed_arm_files_and_preserves_ac(tmp_path: Path) -> None:
    paths = {}
    for seed in (0, 1, 2):
        for arm in ("control", "ascv"):
            path = tmp_path / f"s{seed}-{arm}.json"
            _metrics(path, 0.1 + seed / 100)
            paths[(seed, arm)] = path

    records, inputs = load_records(paths)

    assert set(records) == {"0", "1", "2"}
    assert set(records["0"]) == {"control", "ascv"}
    assert set(records["0"]["control"]) == {"A", "C"}
    assert len(inputs) == 6
    assert all(record["sha256"] for record in inputs)


def test_load_records_rejects_test_dev_and_non_ac_payloads(tmp_path: Path) -> None:
    forbidden = tmp_path / "test-dev" / "metrics.json"
    forbidden.parent.mkdir()
    _metrics(forbidden)
    with pytest.raises(ValueError, match="test-dev"):
        load_records({(0, "control"): forbidden})

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"A": {}, "B": {}}))
    with pytest.raises(ValueError, match="exact A/C"):
        load_records({(0, "control"): invalid})


def _preflight_summary(path: Path, arm: str) -> Path:
    checkpoint = path / arm / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"checkpoint-{arm}".encode())
    summary = {
        "schema_version": "ascv-loc-training-summary/v2",
        "stage": "PREFLIGHT_1",
        "arm": arm,
        "seed": 0,
        "protocol_manifest_sha256": "manifest",
        "protocol_source_commit": "a" * 40,
        "source_repo_bundle_sha256": "repo",
        "source_upstream_bundle_sha256": "upstream",
        "initial_state_sha256": "initial",
        "initial_state_common_fingerprint": "fingerprint",
        "data_sha256": "data",
        "subset_binding": {
            "count": 647,
            "semantic_sha256": "semantic",
            "file_sha256": "file",
        },
        "batch_canaries": [{"epoch": 0, "batch": 1, "sha256": "A" * 64}],
        "batch": 8,
        "workers": 8,
        "observed_tensor_batch_sizes": [8],
        "loader": {
            "trainer_batch_size": 8,
            "per_rank_batch_size": 8,
            "loader_batch_size": 8,
            "loader_num_workers": 8,
        },
        "optimizer": {
            "class": "MuSGD",
            "requested_lr0": 0.01,
            "requested_momentum": 0.937,
            "groups": [{"lr": 0.01, "momentum": 0.937, "weight_decay": 0.0005}],
        },
        "amp": True,
        "amp_scale": 128.0,
        "amp_scale_min": 128.0,
        "amp_scale_max": 128.0,
        "successful_batches": 1,
        "optimizer_attempts": 1,
        "internal_validation_bypass_count": 1,
        "test_loader_is_none": True,
        "hardware": {"gpu": "NVIDIA GeForce RTX 4090", "device": "cuda:0"},
        "cuda_peak_reserved_mib": 20000.0 if arm == "ascv" else 12000.0,
        "local_forward_calls": 2 if arm == "ascv" else 0,
        "local_forward_call_histogram": {"1": 0, "2": 1 if arm == "ascv" else 0},
        "local_bn_preserved_batches": 1 if arm == "ascv" else 0,
        "checkpoint": {
            "kind": "last.pt",
            "path": checkpoint.resolve().as_posix(),
            "sha256": sha256_file(checkpoint),
        },
    }
    summary_path = path / arm / "ascv_training_summary.json"
    summary_path.write_text(json.dumps(summary))
    return summary_path


def test_preflight_gate_replays_and_rehashes_both_summaries_and_checkpoints(tmp_path: Path) -> None:
    paths = {
        "control": _preflight_summary(tmp_path, "control"),
        "ascv": _preflight_summary(tmp_path, "ascv"),
    }
    summaries, inputs = load_preflight_summaries(paths)
    gate = build_preflight_gate(summaries, inputs)

    assert gate["decision"] == "PREFLIGHT_GO"
    assert replay_preflight_gate(gate)["decision"] == "PREFLIGHT_GO"

    Path(gate["inputs"][1]["checkpoint"]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint checksum"):
        replay_preflight_gate(gate)
