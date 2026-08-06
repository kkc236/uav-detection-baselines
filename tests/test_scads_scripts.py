from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.build_scads_initial_state import write_artifact
from scripts.prepare_scads_protocol import prepare_manifest
from scripts import train_rtdetr_scads
from src.scads_protocol import build_run_identity, build_scads_initial_state


def _tiny_artifact(source_commit: str) -> dict:
    common = {"shared.weight": torch.tensor([1.0])}
    scads = {
        **common,
        "model.28.decoder.support_router.weight": torch.tensor([2.0]),
    }
    return build_scads_initial_state(
        common,
        scads,
        metadata={"source_commit": source_commit, "seed": 0},
    )


def test_initial_state_writer_is_create_only_and_emits_sha_summary(tmp_path: Path) -> None:
    output = tmp_path / "paired-seed0.pt"
    artifact = _tiny_artifact("a" * 40)

    summary = write_artifact(output, artifact)

    assert summary["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest().upper()
    assert summary["common_tensor_count"] == 1
    assert summary["scads_private_tensor_count"] == 1
    assert json.loads(output.with_suffix(".pt.json").read_text())["sha256"] == summary["sha256"]
    with pytest.raises(FileExistsError):
        write_artifact(output, artifact)


def test_protocol_manifest_binds_source_initial_state_and_all_run_ids(tmp_path: Path) -> None:
    commit = "a" * 40
    state = tmp_path / "paired.pt"
    torch.save(_tiny_artifact(commit), state)
    output = tmp_path / "protocol.json"

    manifest = prepare_manifest(
        source_commit=commit,
        source_tree_sha256="b" * 64,
        initial_state=state,
        output=output,
    )

    assert manifest["initial_state"]["sha256"] == hashlib.sha256(state.read_bytes()).hexdigest().upper()
    assert set(manifest["run_identities"]) == {
        "fdr_screen",
        "scads_screen",
        "fdr_formal",
        "scads_formal",
    }
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    with pytest.raises(FileExistsError):
        prepare_manifest(
            source_commit=commit,
            source_tree_sha256="b" * 64,
            initial_state=state,
            output=output,
        )


def test_training_dry_run_validates_authority_without_creating_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {"git_commit": "a" * 40, "tree_sha256": "B" * 64}
    identities = {
        f"{variant}_{stage}": build_run_identity(
            source, stage=stage, variant=variant, seed=0
        )
        for stage in ("screen", "formal")
        for variant in ("fdr", "scads")
    }
    manifest = {
        "source": source,
        "initial_state": {"sha256": "C" * 64},
        "run_identities": identities,
    }
    data_yaml = tmp_path / "screen-data.yaml"
    data_yaml.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(train_rtdetr_scads, "load_authority", lambda _path: manifest)
    monkeypatch.setattr(train_rtdetr_scads, "validate_source_authority", lambda _value: source)
    monkeypatch.setattr(
        train_rtdetr_scads,
        "validate_initial_state_file",
        lambda path, _value: {"path": str(path), "sha256": "C" * 64},
    )
    monkeypatch.setattr(
        train_rtdetr_scads,
        "prepare_data_yaml",
        lambda _root, _stage, _authority: data_yaml,
    )
    monkeypatch.setattr(
        train_rtdetr_scads,
        "create_trainer",
        lambda *_args, **_kwargs: pytest.fail("dry-run created a trainer"),
    )
    args = argparse.Namespace(
        variant="scads",
        stage="screen",
        protocol_manifest=tmp_path / "protocol.json",
        initial_state=tmp_path / "paired.pt",
        dataset_root=tmp_path / "VisDrone",
        output_root=tmp_path / "runs",
        resume=None,
        publication_queue=tmp_path / "queue.jsonl",
        name=None,
        dry_run=True,
    )

    result = train_rtdetr_scads.execute(args)

    assert result["status"] == "dry-run-passed"
    assert result["variant"] == "scads"
    assert result["settings"]["save_period"] == 1
    assert result["settings"]["batch"] == 8
    assert result["settings"]["model"].endswith("rtdetr-l-fdr-scads.yaml")
