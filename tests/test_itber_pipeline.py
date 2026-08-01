from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.run_itber_pipeline import (
    PipelineEvidence,
    atomic_write_state,
    build_train_command,
    next_pipeline_phase,
)
from src.itber_protocol import (
    BASELINE_TRAINING_CONTRACT_SHA256,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
)


CACHE_SHA = "C" * 64


def _checkpoint(path: Path, *, stage: str, epoch: int) -> None:
    torch.save(
        {
            "format_version": 1,
            "design_version": "itber-v1.1",
            "stage": stage,
            "probe": "p3",
            "seed": 0,
            "epoch": epoch,
            "baseline_sha256": EXPECTED_BASELINE_SHA256,
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "cache_manifest_sha256": CACHE_SHA,
            "baseline_training_contract_sha256": BASELINE_TRAINING_CONTRACT_SHA256,
            "refiner": {"weight": torch.ones(1)},
            "optimizer": {"state": {}, "param_groups": []},
            "scaler": {"scale": 128.0},
            "rng": {"torch": torch.get_rng_state()},
        },
        path,
    )


def _evidence(**updates) -> PipelineEvidence:
    values = {
        "authority": None,
        "gate0": None,
        "cache_complete": False,
        "gate1": None,
        "screen": None,
        "formal": None,
    }
    values.update(updates)
    return PipelineEvidence(**values)


def test_state_machine_orders_all_gates_and_stops_failures() -> None:
    assert next_pipeline_phase(_evidence()) == "authority"
    assert next_pipeline_phase(_evidence(authority="passed")) == "gate0"
    assert next_pipeline_phase(_evidence(authority="passed", gate0="passed")) == "cache"
    assert next_pipeline_phase(_evidence(authority="passed", gate0="passed", cache_complete=True)) == "probe"
    assert next_pipeline_phase(
        _evidence(authority="passed", gate0="passed", cache_complete=True, gate1="passed")
    ) == "screen"
    assert next_pipeline_phase(
        _evidence(
            authority="passed",
            gate0="passed",
            cache_complete=True,
            gate1="passed",
            screen="passed",
        )
    ) == "formal"
    assert next_pipeline_phase(
        _evidence(
            authority="passed",
            gate0="passed",
            cache_complete=True,
            gate1="passed",
            screen="passed",
            formal="passed",
        )
    ) == "formal_complete"

    assert next_pipeline_phase(_evidence(authority="engineering_invalid")) == "engineering_invalid"
    assert next_pipeline_phase(
        _evidence(authority="passed", gate0="engineering_invalid")
    ) == "engineering_invalid"
    assert next_pipeline_phase(
        _evidence(authority="passed", gate0="passed", cache_complete=True, gate1="scientific_failed")
    ) == "scientific_failed"


def test_train_command_has_no_scientific_overrides_and_accepts_same_stage_resume(tmp_path: Path) -> None:
    resume = tmp_path / "epoch-0004.pt"
    _checkpoint(resume, stage="screen", epoch=4)
    command = build_train_command(
        stage="screen",
        baseline_checkpoint=tmp_path / "baseline.pt",
        dataset_root=tmp_path / "data",
        cache_manifest=tmp_path / "manifest.json",
        output_root=tmp_path / "run",
        publication_config=tmp_path / "publication.json",
        resume_checkpoint=resume,
        cache_manifest_sha256=CACHE_SHA,
    )

    assert "--resume-checkpoint" in command
    for forbidden in ("--epochs", "--seed", "--batch", "--workers", "--imgsz", "--optimizer", "--lr"):
        assert forbidden not in command


def test_formal_never_resumes_a_screen_checkpoint(tmp_path: Path) -> None:
    resume = tmp_path / "screen.pt"
    _checkpoint(resume, stage="screen", epoch=12)
    with pytest.raises(ValueError, match="stage"):
        build_train_command(
            stage="formal",
            baseline_checkpoint=tmp_path / "baseline.pt",
            dataset_root=tmp_path / "data",
            cache_manifest=tmp_path / "manifest.json",
            output_root=tmp_path / "formal",
            publication_config=tmp_path / "publication.json",
            resume_checkpoint=resume,
            cache_manifest_sha256=CACHE_SHA,
        )


def test_atomic_state_preserves_append_only_history(tmp_path: Path) -> None:
    path = tmp_path / "pipeline-state.json"
    atomic_write_state(path, {"phase": "gate0", "history": [{"phase": "authority"}]})
    atomic_write_state(
        path,
        {"phase": "cache", "history": [{"phase": "authority"}, {"phase": "gate0"}]},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["phase"] == "cache"
    assert [row["phase"] for row in payload["history"]] == ["authority", "gate0"]
