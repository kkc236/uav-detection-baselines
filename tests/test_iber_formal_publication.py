from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import src.iber_formal_publication as publication
from src.iber_formal_publication import (
    FORMAL_ASSET_PREFIX,
    FormalPublicationConfig,
    FormalPublicationIdentity,
    FormalPublicationLedger,
    pending_epoch_checkpoints,
)


SOURCE_COMMIT = "1" * 40
PROTOCOL_SHA256 = "2" * 64
INITIAL_STATE_SHA256 = "3" * 64


def _identity() -> FormalPublicationIdentity:
    return FormalPublicationIdentity(
        source_commit=SOURCE_COMMIT,
        protocol_sha256=PROTOCOL_SHA256,
        initial_state_sha256=INITIAL_STATE_SHA256,
    )


def _record(epoch: int) -> dict:
    return {
        **_identity().as_dict(),
        "completed_epoch": epoch,
        "checkpoint": {
            "asset_name": f"{FORMAL_ASSET_PREFIX}-epoch-{epoch:04d}.pt",
            "bytes": 10,
            "sha256": "4" * 64,
        },
        "result_commit_sha": "5" * 40,
        "verified": True,
    }


def _checkpoint(path: Path, completed_epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": completed_epoch - 1,
            "optimizer": {"state": {}, "param_groups": []},
            "model": {"weight": torch.tensor([1.0])},
        },
        path,
    )


def test_formal_ledger_is_identity_bound_append_only_and_limited_to_100(
    tmp_path: Path,
) -> None:
    ledger = FormalPublicationLedger(tmp_path / "publication-ledger.jsonl", _identity())
    ledger.append_verified(_record(1))
    ledger.append_verified(_record(2))

    assert ledger.last_completed_epoch == 2
    assert [row["completed_epoch"] for row in ledger.records()] == [1, 2]
    with pytest.raises(ValueError, match="gap"):
        ledger.append_verified(_record(4))
    changed = _record(2)
    changed["checkpoint"] = {**changed["checkpoint"], "sha256": "6" * 64}
    with pytest.raises(ValueError, match="changed"):
        ledger.append_verified(changed)
    with pytest.raises(ValueError, match="1..100"):
        ledger.append_verified(_record(101))


def test_pending_formal_checkpoints_follow_internal_epoch_not_filename_order(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "weights"
    _checkpoint(weights / "epoch0.pt", 1)
    _checkpoint(weights / "epoch1.pt", 2)
    ledger = FormalPublicationLedger(tmp_path / "ledger.jsonl", _identity())

    assert [(epoch, path.name) for epoch, path in pending_epoch_checkpoints(weights, ledger)] == [
        (1, "epoch0.pt"),
        (2, "epoch1.pt"),
    ]

    ledger.append_verified(_record(1))
    assert [(epoch, path.name) for epoch, path in pending_epoch_checkpoints(weights, ledger)] == [
        (2, "epoch1.pt")
    ]


def test_formal_ledger_rejects_tampered_identity_on_read(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    row = _record(1)
    row["source_commit"] = "a" * 40
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        FormalPublicationLedger(path, _identity()).records()


def test_publish_commits_the_verified_ledger_to_results_branch_last(
    monkeypatch, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    checkpoint = run / "weights" / "epoch0.pt"
    _checkpoint(checkpoint, 1)
    (run / "iber_formal_protocol.json").write_text("{}\n", encoding="utf-8")
    (run / "iber_formal_diagnostics.jsonl").write_text(
        '{"epoch":1}\n', encoding="utf-8"
    )
    token = tmp_path / "token"
    token.write_text("secret", encoding="utf-8")
    results = tmp_path / "results"
    config = FormalPublicationConfig(
        repo="owner/repo",
        repo_url="https://github.com/owner/repo.git",
        source_branch="codex/iber-be",
        tag="formal-live",
        run_name="formal-seed0",
        token_file=token,
        results_repo=results,
        identity=_identity(),
    )
    commits: list[int] = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"sha": str(len(commits)) * 40}

    class Session:
        def get(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(publication, "github_session", lambda _token: Session())
    monkeypatch.setattr(
        publication,
        "get_or_create_release",
        lambda *_a, **_k: {"upload_url": "unused", "assets": []},
    )
    monkeypatch.setattr(
        publication,
        "upload_asset",
        lambda _session, *, release, path, asset_name: {
            "id": 1,
            "name": asset_name,
            "url": f"https://api.github.invalid/{asset_name}",
            "size": Path(path).stat().st_size,
        },
    )
    monkeypatch.setattr(
        publication,
        "verify_remote_asset",
        lambda _session, _asset, path: {
            "bytes": Path(path).stat().st_size,
            "sha256": "a" * 64,
        },
    )

    def checkout(path, **_kwargs):
        path.mkdir(parents=True, exist_ok=True)
        return {}

    monkeypatch.setattr(publication, "ensure_results_checkout", checkout)
    monkeypatch.setattr(
        publication,
        "commit_and_push_results",
        lambda *_a, **_k: commits.append(len(commits) + 1),
    )

    class Completed:
        @property
        def stdout(self):
            return str(len(commits)) * 40

    monkeypatch.setattr(publication, "_run", lambda *_a, **_k: Completed())

    record = publication.publish_exact_epoch(run, checkpoint, config)

    assert record["verified"] is True
    assert commits == [1, 2]
    remote_ledger = results / "results" / config.run_name / "publication-ledger.jsonl"
    assert remote_ledger.is_file()
    assert json.loads(remote_ledger.read_text(encoding="utf-8"))["completed_epoch"] == 1
    assert (remote_ledger.parent / "iber_formal_protocol.json").is_file()
    assert (remote_ledger.parent / "iber_formal_diagnostics.jsonl").is_file()


def test_publish_retry_repairs_a_transaction_interrupted_after_local_ledger_append(
    monkeypatch, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    checkpoint = run / "weights" / "epoch0.pt"
    _checkpoint(checkpoint, 1)
    (run / "iber_formal_protocol.json").write_text("{}\n", encoding="utf-8")
    token = tmp_path / "token"
    token.write_text("secret", encoding="utf-8")
    results = tmp_path / "results"
    config = FormalPublicationConfig(
        repo="owner/repo",
        repo_url="https://github.com/owner/repo.git",
        source_branch="codex/iber-be",
        tag="formal-live",
        run_name="formal-seed0",
        token_file=token,
        results_repo=results,
        identity=_identity(),
    )
    state = {"commit_calls": 0, "head": "0" * 40}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"sha": state["head"]}

    class Session:
        def get(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(publication, "github_session", lambda _token: Session())
    monkeypatch.setattr(
        publication,
        "get_or_create_release",
        lambda *_a, **_k: {"upload_url": "unused", "assets": []},
    )
    monkeypatch.setattr(
        publication,
        "upload_asset",
        lambda _session, *, release, path, asset_name: {
            "id": 1,
            "name": asset_name,
            "url": f"https://api.github.invalid/{asset_name}",
            "size": Path(path).stat().st_size,
        },
    )
    monkeypatch.setattr(
        publication,
        "verify_remote_asset",
        lambda _session, _asset, path: {
            "bytes": Path(path).stat().st_size,
            "sha256": "a" * 64,
        },
    )

    def checkout(path, **_kwargs):
        path.mkdir(parents=True, exist_ok=True)
        return {}

    def commit(*_args, **_kwargs):
        state["commit_calls"] += 1
        if state["commit_calls"] == 2:
            raise RuntimeError("simulated failure after local ledger append")
        state["head"] = str(state["commit_calls"]) * 40

    class Completed:
        @property
        def stdout(self):
            return state["head"]

    monkeypatch.setattr(publication, "ensure_results_checkout", checkout)
    monkeypatch.setattr(publication, "commit_and_push_results", commit)
    monkeypatch.setattr(publication, "_run", lambda *_a, **_k: Completed())
    monkeypatch.setattr(publication.time, "sleep", lambda _delay: None)

    record = publication.publish_with_retry(
        run, checkpoint, config, attempts=2, delay=0
    )

    assert record["completed_epoch"] == 1
    assert record["verified"] is True
    assert state["commit_calls"] == 3
    assert FormalPublicationLedger(
        run / "publication-ledger.jsonl", config.identity
    ).last_completed_epoch == 1


def test_failed_remote_hash_verification_deletes_the_asset_before_retry(
    monkeypatch, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    checkpoint = run / "weights" / "epoch0.pt"
    _checkpoint(checkpoint, 1)
    token = tmp_path / "token"
    token.write_text("secret", encoding="utf-8")
    config = FormalPublicationConfig(
        repo="owner/repo",
        repo_url="https://github.com/owner/repo.git",
        source_branch="codex/iber-be",
        tag="formal-live",
        run_name="formal-seed0",
        token_file=token,
        results_repo=tmp_path / "results",
        identity=_identity(),
    )
    state = {"verification_calls": 0, "deletes": 0, "head": "1" * 40}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"sha": state["head"]}

    class Session:
        def get(self, *_args, **_kwargs):
            return Response()

        def delete(self, *_args, **_kwargs):
            state["deletes"] += 1
            return Response()

    monkeypatch.setattr(publication, "github_session", lambda _token: Session())
    monkeypatch.setattr(
        publication,
        "get_or_create_release",
        lambda *_a, **_k: {"upload_url": "unused", "assets": []},
    )
    monkeypatch.setattr(
        publication,
        "upload_asset",
        lambda _session, *, release, path, asset_name: {
            "id": 1,
            "name": asset_name,
            "url": f"https://api.github.invalid/{asset_name}",
            "size": Path(path).stat().st_size,
        },
    )

    def verify(_session, _asset, path):
        state["verification_calls"] += 1
        if state["verification_calls"] == 1:
            raise RuntimeError("simulated same-size hash mismatch")
        return {"bytes": Path(path).stat().st_size, "sha256": "a" * 64}

    def checkout(path, **_kwargs):
        path.mkdir(parents=True, exist_ok=True)
        return {}

    class Completed:
        stdout = state["head"]

    monkeypatch.setattr(publication, "verify_remote_asset", verify)
    monkeypatch.setattr(publication, "ensure_results_checkout", checkout)
    monkeypatch.setattr(publication, "commit_and_push_results", lambda *_a, **_k: None)
    monkeypatch.setattr(publication, "_run", lambda *_a, **_k: Completed())
    monkeypatch.setattr(publication.time, "sleep", lambda _delay: None)

    record = publication.publish_with_retry(
        run, checkpoint, config, attempts=2, delay=0
    )

    assert record["verified"] is True
    assert state["deletes"] == 1
