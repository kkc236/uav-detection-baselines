from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.train_rtdetr_iber_formal as training
from src.iber_formal_protocol import FORMAL_FROZEN_PROTOCOL
from src.iber_formal_publication import FormalPublicationIdentity


def _manifest(tmp_path: Path) -> dict:
    return {
        "format_version": 3,
        "design_version": "iber-be-v1.0-signed-formal100",
        "seed": 0,
        "dataset_root": str(tmp_path / "VisDrone"),
        "dataset": {
            "sha256": "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB",
            "train_images": 6471,
            "val_images": 548,
            "classes": 10,
        },
        "data": {"formal": {"path": str(tmp_path / "formal.yaml")}},
        "source_sha256": training.EXPECTED_SOURCE_SHA256,
    }


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        seed=0,
        protocol_manifest=tmp_path / "protocol.json",
        initial_state=tmp_path / "initial.pt",
        resume=None,
        project=tmp_path / "runs",
        name="formal-seed0-iber-be",
        token_file=tmp_path / "token",
        repo="owner/repo",
        repo_url="https://github.com/owner/repo.git",
        tag="iber-be-v1-formal-live",
        source_branch="codex/iber-be",
        results_branch="iber-be-v1-results",
        results_repo=tmp_path / "results",
        asset_prefix="iber-be-v1.0-formal-seed0-b3",
        retain=3,
    )


def test_formal_cli_has_no_mutable_scientific_hyperparameters() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/train_rtdetr_iber_formal.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for required in ("--protocol-manifest", "--initial-state", "--resume", "--token-file"):
        assert required in result.stdout
    for forbidden in (
        "--epochs",
        "--batch",
        "--workers",
        "--optimizer",
        "--lr0",
        "--mosaic",
        "--pretrained",
        "--device",
    ):
        assert forbidden not in result.stdout


def test_formal_settings_are_exact_and_seed0_only(tmp_path: Path) -> None:
    args = _args(tmp_path)
    settings = training.build_settings(args, _manifest(tmp_path))

    for name, value in FORMAL_FROZEN_PROTOCOL.items():
        if name not in {"amp_scale", "query_count"}:
            assert settings[name] == value
    assert settings["data"] == str(tmp_path / "formal.yaml")
    args.seed = 1
    with pytest.raises(ValueError, match="seed0"):
        training.build_settings(args, _manifest(tmp_path))


def test_resume_requires_the_same_immutable_runtime_authority(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.protocol_manifest.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "run" / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {"train_args": {**FORMAL_FROZEN_PROTOCOL, "data": str(tmp_path / "formal.yaml")}},
        checkpoint,
    )
    args.resume = checkpoint
    authority = training.runtime_authority(args, _manifest(tmp_path), "a" * 64)
    (checkpoint.parent.parent / "iber_formal_protocol.json").write_text(
        json.dumps(authority), encoding="utf-8"
    )

    training.validate_resume_authority(args, authority)
    changed = dict(authority)
    changed["source_commit"] = "b" * 40
    (checkpoint.parent.parent / "iber_formal_protocol.json").write_text(
        json.dumps(changed), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="resume authority"):
        training.validate_resume_authority(args, authority)


def test_resume_rejects_checkpoint_training_protocol_drift(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.protocol_manifest.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "run" / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    args.resume = checkpoint
    authority = training.runtime_authority(args, _manifest(tmp_path), "a" * 64)
    (checkpoint.parent.parent / "iber_formal_protocol.json").write_text(
        json.dumps(authority), encoding="utf-8"
    )
    torch.save(
        {
            "train_args": {
                **FORMAL_FROZEN_PROTOCOL,
                "data": str(tmp_path / "formal.yaml"),
                "momentum": 0.8,
            }
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="checkpoint.*momentum"):
        training.validate_resume_authority(args, authority)


def test_epoch_publication_happens_before_local_pruning(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    checkpoint = tmp_path / "run" / "weights" / "epoch0.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    trainer = SimpleNamespace(epoch=0, save_dir=checkpoint.parent.parent)
    args = _args(tmp_path)

    monkeypatch.setattr(
        training,
        "publish_with_retry",
        lambda *_a, **_k: events.append("publish")
        or {"completed_epoch": 1, "verified": True},
    )
    monkeypatch.setattr(
        training,
        "prune_local_epoch_checkpoints",
        lambda *_a, **_k: events.append("prune"),
    )

    record = training.publish_current_epoch(trainer, args=args, identity=object())

    assert record["verified"] is True
    assert events == ["publish", "prune"]


def test_signed_diagnostics_record_finite_private_activity(tmp_path: Path) -> None:
    model = SimpleNamespace(
        last_iber_output=SimpleNamespace(
            gates=torch.tensor([0.2, 0.8]),
            residuals=torch.tensor([-0.1, 0.3]),
            f3_boundary_evidence=torch.tensor([0.4, -0.2]),
            rgb_boundary_evidence=torch.tensor([0.1, 0.6]),
        ),
        last_iber_losses={"box_l1": torch.tensor(0.4), "box_giou": torch.tensor(0.7)},
    )
    trainer = SimpleNamespace(
        epoch=0,
        args=SimpleNamespace(epochs=100),
        model=model,
        tloss=torch.tensor([1.0, 2.0, 3.0]),
        last_gradient_norms={"gradient_norm": 2.5, "iber_gradient_norm": 0.5},
        validator=SimpleNamespace(metrics=SimpleNamespace(box=SimpleNamespace(map=0.1, map50=0.2, map75=0.05))),
        save_dir=tmp_path,
    )

    row = training.write_formal_diagnostics(trainer)

    assert row["epoch"] == 1
    assert row["iber_gradient_norm"] == 0.5
    assert row["f3_boundary_rms"] > 0
    assert row["rgb_boundary_rms"] > 0
    assert (tmp_path / "iber_formal_diagnostics.jsonl").is_file()


def test_signed_diagnostics_are_idempotent_and_reject_changed_replay(
    tmp_path: Path,
) -> None:
    model = SimpleNamespace(
        last_iber_output=SimpleNamespace(
            gates=torch.tensor([0.2]),
            residuals=torch.tensor([0.1]),
            f3_boundary_evidence=torch.tensor([0.4]),
            rgb_boundary_evidence=torch.tensor([0.6]),
        ),
        last_iber_losses={"box_l1": torch.tensor(0.4), "box_giou": torch.tensor(0.7)},
    )
    trainer = SimpleNamespace(
        epoch=0,
        args=SimpleNamespace(epochs=100),
        model=model,
        tloss=torch.tensor([1.0, 2.0, 3.0]),
        last_gradient_norms={"gradient_norm": 2.5, "iber_gradient_norm": 0.5},
        validator=SimpleNamespace(
            metrics=SimpleNamespace(box=SimpleNamespace(map=0.1, map50=0.2, map75=0.05))
        ),
        save_dir=tmp_path,
    )

    training.write_formal_diagnostics(trainer)
    training.write_formal_diagnostics(trainer)
    rows = (tmp_path / "iber_formal_diagnostics.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(rows) == 1

    trainer.validator.metrics.box.map = 0.11
    with pytest.raises(ValueError, match="changed formal diagnostic"):
        training.write_formal_diagnostics(trainer)


def test_signed_diagnostics_reject_missing_private_activity(tmp_path: Path) -> None:
    trainer = SimpleNamespace(
        epoch=0,
        args=SimpleNamespace(epochs=100),
        model=SimpleNamespace(last_iber_output=None, last_iber_losses={}),
        tloss=torch.tensor([1.0, 2.0, 3.0]),
        last_gradient_norms={"gradient_norm": 2.5, "iber_gradient_norm": 0.5},
        validator=SimpleNamespace(
            metrics=SimpleNamespace(box=SimpleNamespace(map=0.1, map50=0.2, map75=0.05))
        ),
        save_dir=tmp_path,
    )

    with pytest.raises(RuntimeError, match="private activity"):
        training.write_formal_diagnostics(trainer)


def test_resume_catches_up_saved_unpublished_epochs_before_training(
    monkeypatch, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    weights = run / "weights"
    weights.mkdir(parents=True)
    checkpoints = []
    for zero_based_epoch in (0, 1):
        path = weights / f"epoch{zero_based_epoch}.pt"
        path.write_bytes(str(zero_based_epoch).encode())
        checkpoints.append(path)
    trainer = SimpleNamespace(save_dir=run)
    args = _args(tmp_path)
    identity = FormalPublicationIdentity(
        source_commit="1" * 40,
        protocol_sha256="2" * 64,
        initial_state_sha256="3" * 64,
    )
    events: list[str] = []

    class Ledger:
        def __init__(self, *_args, **_kwargs):
            self.last_completed_epoch = 0

        def records(self):
            return []

    monkeypatch.setattr(training, "FormalPublicationLedger", Ledger)
    monkeypatch.setattr(
        training,
        "pending_epoch_checkpoints",
        lambda *_args: [(1, checkpoints[0]), (2, checkpoints[1])],
    )
    monkeypatch.setattr(
        training,
        "publish_with_retry",
        lambda _run, checkpoint, _config: events.append(Path(checkpoint).name)
        or {"completed_epoch": len(events), "verified": True},
    )
    monkeypatch.setattr(
        training,
        "prune_local_epoch_checkpoints",
        lambda *_args, **_kwargs: events.append("prune"),
    )

    records = training.recover_pending_publications(
        trainer, args=args, identity=identity
    )

    assert [record["completed_epoch"] for record in records] == [1, 2]
    assert events == ["epoch0.pt", "epoch1.pt", "prune"]
