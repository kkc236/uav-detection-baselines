from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import src.iber_publication as publication
from src.iber_protocol import (
    DESIGN_VERSION,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    PROTOCOL_SHA256,
    RUNTIME_AMENDMENT_SHA256,
)
from src.iber_publication import (
    ASSET_PREFIX,
    DEFAULT_TAG,
    RESULTS_BRANCH,
    PublicationConfig,
    PublicationIdentity,
    PublicationLedger,
    assets_to_prune,
    pending_epoch_checkpoints,
    private_checkpoint_metadata,
    publish_exact_epoch,
    publish_with_retry,
    read_token_file,
    verify_remote_asset,
)


CATEGORY_SHA = "A" * 64
GATE1_SHA = "B" * 64
SOURCE_COMMIT = "c" * 40


def _identity() -> PublicationIdentity:
    return PublicationIdentity(
        design_version=DESIGN_VERSION,
        stage="screen",
        probe="b3",
        seed=0,
        baseline_sha256=EXPECTED_BASELINE_SHA256,
        dataset_sha256=EXPECTED_DATASET_SHA256,
        subset_sha256=EXPECTED_SUBSET_SHA256,
        category_sha256=CATEGORY_SHA,
        protocol_sha256=PROTOCOL_SHA256,
        runtime_amendment_sha256=RUNTIME_AMENDMENT_SHA256,
        gate1_decision_sha256=GATE1_SHA,
        source_commit=SOURCE_COMMIT,
    )


def _checkpoint(path: Path, epoch: int) -> None:
    torch.save(
        {
            "format_version": 1,
            **_identity().as_dict(),
            "private_seed": 10_000,
            "epoch": epoch,
            "detector_sha_before": "D" * 64,
            "detector_sha_after": "D" * 64,
            "refiner": {"weight": torch.ones(1)},
            "optimizer": {"state": {}, "param_groups": []},
            "scaler": {"scale": 128.0},
            "rng": {"torch": torch.get_rng_state()},
        },
        path,
    )


def _record(epoch: int, *, identity: PublicationIdentity | None = None) -> dict:
    digest = f"{epoch:064x}"
    return {
        **(identity or _identity()).as_dict(),
        "completed_epoch": epoch,
        "checkpoint": {"bytes": 10, "sha256": digest},
        "remote_verification": {
            "checkpoint": {"bytes": 10, "sha256": digest},
            "manifest": {"bytes": 20, "sha256": "a" * 64},
        },
        "result_commit_sha": "e" * 40,
        "result_commit_verified": True,
        "verified": True,
    }


def test_publication_identity_and_namespace_are_frozen() -> None:
    assert DEFAULT_TAG == "iber-be-v1-rtdetr-l-live"
    assert ASSET_PREFIX == "iber-be-v1.0-screen-seed0-b3"
    assert RESULTS_BRANCH == "iber-be-v1-results"
    assert _identity().as_dict()["design_version"] == "iber-be-v1.0"

    for field, value in (
        ("design_version", "itber-v1.1"),
        ("stage", "formal"),
        ("probe", "b2"),
        ("seed", 1),
    ):
        values = _identity().__dict__.copy()
        values[field] = value
        with pytest.raises(ValueError, match=field.replace("design_version", "design version")):
            PublicationIdentity(**values)


def test_publication_config_locks_branch_tag_prefix_and_retention(tmp_path: Path) -> None:
    config = PublicationConfig(
        repo="owner/repo",
        repo_url="https://github.com/owner/repo.git",
        source_branch="codex/iber-be",
        run_name="iber-be-screen-seed0",
        token_file=tmp_path / "token",
        results_repo=tmp_path / "results-checkout",
        identity=_identity(),
    )
    assert config.results_branch == RESULTS_BRANCH
    assert config.tag == DEFAULT_TAG
    assert config.asset_prefix == ASSET_PREFIX
    assert config.retain == 3

    with pytest.raises(ValueError, match="credentials"):
        PublicationConfig(
            repo="owner/repo",
            repo_url="https://secret@github.com/owner/repo.git",
            source_branch="codex/iber-be",
            run_name="iber-be-screen-seed0",
            token_file=tmp_path / "token",
            results_repo=tmp_path / "results-checkout",
            identity=_identity(),
        )


def test_private_checkpoint_metadata_requires_exact_resumable_screen_checkpoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "epoch-0001.pt"
    _checkpoint(path, 1)
    metadata = private_checkpoint_metadata(path, identity=_identity())
    assert metadata.completed_epoch == 1
    assert metadata.bytes == path.stat().st_size
    assert metadata.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    artifact = torch.load(path, map_location="cpu", weights_only=True)
    artifact["detector"] = {"forbidden": torch.ones(1)}
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="detector"):
        private_checkpoint_metadata(path, identity=_identity())


@pytest.mark.parametrize("epoch", [0, 31, True])
def test_private_checkpoint_epoch_is_limited_to_contiguous_screen_range(
    tmp_path: Path, epoch: int
) -> None:
    path = tmp_path / "bad.pt"
    _checkpoint(path, epoch)
    with pytest.raises(ValueError, match="epoch"):
        private_checkpoint_metadata(path, identity=_identity())


def test_ledger_is_append_only_exact_identity_and_contiguous_1_to_30(tmp_path: Path) -> None:
    ledger = PublicationLedger(tmp_path / "publication-ledger.jsonl", _identity())
    ledger.append_verified(_record(1))
    ledger.append_verified(_record(1))
    assert ledger.last_completed_epoch == 1

    with pytest.raises(ValueError, match="gap"):
        ledger.append_verified(_record(3))
    with pytest.raises(ValueError, match="identity"):
        foreign = _identity().__dict__.copy()
        foreign["source_commit"] = "d" * 40
        ledger.append_verified(_record(2, identity=PublicationIdentity(**foreign)))
    with pytest.raises(ValueError, match="verified"):
        ledger.append_verified({**_record(2), "verified": False})
    with pytest.raises(ValueError, match="result commit"):
        ledger.append_verified({**_record(2), "result_commit_verified": False})
    with pytest.raises(ValueError, match="changed"):
        changed = _record(1)
        changed["checkpoint"] = {"bytes": 10, "sha256": "f" * 64}
        changed["remote_verification"] = {
            **changed["remote_verification"],
            "checkpoint": {"bytes": 10, "sha256": "f" * 64},
        }
        ledger.append_verified(changed)

    for epoch in range(2, 31):
        ledger.append_verified(_record(epoch))
    with pytest.raises(ValueError, match="30"):
        ledger.append_verified(_record(31))

    lines = ledger.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 30
    assert [json.loads(line)["completed_epoch"] for line in lines] == list(range(1, 31))


def test_pending_checkpoints_reject_gaps_and_foreign_filename_epoch(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    _checkpoint(checkpoint_root / "epoch-0001.pt", 1)
    _checkpoint(checkpoint_root / "epoch-0002.pt", 2)
    ledger = PublicationLedger(tmp_path / "ledger.jsonl", _identity())

    assert [epoch for epoch, _ in pending_epoch_checkpoints(
        checkpoint_root, ledger, identity=_identity()
    )] == [1, 2]

    (checkpoint_root / "epoch-0002.pt").unlink()
    _checkpoint(checkpoint_root / "epoch-0003.pt", 3)
    with pytest.raises(RuntimeError, match="missing completed epoch 2"):
        pending_epoch_checkpoints(checkpoint_root, ledger, identity=_identity())

    _checkpoint(checkpoint_root / "epoch-0099.pt", 2)
    with pytest.raises(ValueError, match="filename"):
        pending_epoch_checkpoints(checkpoint_root, ledger, identity=_identity())


class _Response:
    def __init__(self, payload: bytes):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        assert chunk_size > 0
        yield self.payload[:2]
        yield self.payload[2:]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Session:
    def __init__(self, payload: bytes):
        self.payload = payload

    def get(self, *_args, **_kwargs):
        assert _kwargs["headers"] == {"Accept": "application/octet-stream"}
        return _Response(self.payload)


def test_remote_asset_verification_reads_remote_bytes_and_sha(tmp_path: Path) -> None:
    payload = b"exact remote checkpoint bytes"
    expected = tmp_path / "checkpoint.pt"
    expected.write_bytes(payload)
    asset = {
        "url": "https://api.github.invalid/assets/1",
        "size": len(payload),
        "name": f"{ASSET_PREFIX}-epoch-0001.pt",
    }
    receipt = verify_remote_asset(_Session(payload), asset, expected)
    assert receipt == {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

    with pytest.raises(RuntimeError, match="SHA-256"):
        verify_remote_asset(_Session(payload + b"changed"), asset, expected)


def test_retention_prunes_only_complete_pairs_older_than_latest_three() -> None:
    assets = []
    for epoch in range(1, 6):
        assets.extend(
            [
                {"id": epoch * 10, "name": f"{ASSET_PREFIX}-epoch-{epoch:04d}.pt"},
                {"id": epoch * 10 + 1, "name": f"{ASSET_PREFIX}-epoch-{epoch:04d}.json"},
            ]
        )
    assets.extend(
        [
            {"id": 99, "name": f"{ASSET_PREFIX}-epoch-0006.pt"},
            {"id": 100, "name": "itber-v1.1-screen-seed0-p3-epoch-0001.pt"},
        ]
    )

    expired = assets_to_prune(assets, prefix=ASSET_PREFIX, retain=3)
    assert [asset["id"] for asset in expired] == [10, 11, 20, 21]


def test_token_file_is_mode_600_and_retry_never_chains_or_echoes_secret(
    monkeypatch, tmp_path: Path
) -> None:
    token_file = tmp_path / "github-token"
    secret = "credential-must-never-escape"
    token_file.write_text(secret + "\n", encoding="utf-8")
    if os.name != "nt":
        token_file.chmod(0o644)
        with pytest.raises(PermissionError) as error:
            read_token_file(token_file)
        assert secret not in str(error.value)
        token_file.chmod(0o600)
    assert read_token_file(token_file) == secret

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"remote rejected {secret}")

    monkeypatch.setattr(publication, "publish_exact_epoch", fail)
    with pytest.raises(RuntimeError, match="failed after 2 attempts") as error:
        publish_with_retry(tmp_path, tmp_path / "epoch-0001.pt", object(), attempts=2, delay=0)
    assert secret not in str(error.value)
    assert error.value.__cause__ is None


def test_publish_cli_exposes_only_paths_not_scientific_or_namespace_overrides() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/publish_iber_epoch.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--run-dir" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--config" in result.stdout
    for forbidden in ("--stage", "--probe", "--seed", "--tag", "--retain", "--token"):
        assert forbidden not in result.stdout


def test_publication_uploads_restorable_manifest_and_appends_ledger_last(
    monkeypatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "epoch-0001.pt"
    checkpoint.parent.mkdir(parents=True)
    _checkpoint(checkpoint, 1)
    token_file = tmp_path / "token"
    token_file.write_text("secret", encoding="utf-8")
    if os.name != "nt":
        token_file.chmod(0o600)
    results_repo = tmp_path / "results"
    config = PublicationConfig(
        repo="owner/repo",
        repo_url="https://github.com/owner/repo.git",
        source_branch="codex/iber-be",
        run_name="iber-be-screen-seed0",
        token_file=token_file,
        results_repo=results_repo,
        identity=_identity(),
    )
    events: list[str] = []
    uploaded_manifest: dict = {}
    release = {
        "url": "https://api.github.invalid/release/1",
        "html_url": "https://github.invalid/release/1",
        "upload_url": "https://uploads.github.invalid/1{?name}",
        "assets": [],
    }

    class Session:
        pass

    monkeypatch.setattr(publication, "github_session", lambda _token: Session())
    monkeypatch.setattr(publication, "get_or_create_release", lambda *_a, **_k: release)

    def upload(_session, *, release, path, asset_name):
        del release
        if asset_name.endswith(".json"):
            uploaded_manifest.update(json.loads(path.read_text(encoding="utf-8")))
            events.append("manifest-upload")
        return {
            "id": 1 if asset_name.endswith(".pt") else 2,
            "name": asset_name,
            "url": f"https://api.github.invalid/assets/{asset_name}",
            "size": path.stat().st_size,
        }

    monkeypatch.setattr(publication, "upload_asset", upload)
    monkeypatch.setattr(
        publication,
        "verify_remote_asset",
        lambda _s, _a, path: {
            "bytes": Path(path).stat().st_size,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        },
    )
    monkeypatch.setattr(
        publication,
        "_refresh_release",
        lambda *_a, **_k: {
            "assets": [
                {
                    "id": 1,
                    "name": f"{ASSET_PREFIX}-epoch-0001.pt",
                    "url": "https://api.github.invalid/assets/1",
                },
                {
                    "id": 2,
                    "name": f"{ASSET_PREFIX}-epoch-0001.json",
                    "url": "https://api.github.invalid/assets/2",
                },
            ]
        },
    )

    def checkout(path, **_kwargs):
        path.mkdir(parents=True, exist_ok=True)
        return {}

    monkeypatch.setattr(publication, "ensure_results_checkout", checkout)
    monkeypatch.setattr(
        publication,
        "commit_and_push_results",
        lambda *_a, **_k: events.append("result-push"),
    )
    commits = iter(("1" * 40, "2" * 40))

    class Completed:
        def __init__(self, stdout: str):
            self.stdout = stdout

    monkeypatch.setattr(publication, "_run", lambda *_a, **_k: Completed(next(commits)))
    monkeypatch.setattr(
        publication,
        "_verify_result_branch",
        lambda *_a, **_k: events.append("result-verified"),
    )
    original_append = PublicationLedger.append_verified

    def append_last(ledger, record):
        events.append("ledger-append")
        return original_append(ledger, record)

    monkeypatch.setattr(PublicationLedger, "append_verified", append_last)

    record = publish_exact_epoch(run_dir, checkpoint, config)

    assert uploaded_manifest["result_commit_sha"] == "1" * 40
    assert uploaded_manifest["result_commit_verified"] is True
    assert uploaded_manifest["verified"] is True
    assert events.index("manifest-upload") > events.index("result-verified")
    assert events[-1] == "ledger-append"
    assert record["result_commit_sha"] == "1" * 40
    remote_ledger = (
        results_repo
        / "results"
        / config.run_name
        / "publication-ledger.jsonl"
    )
    assert remote_ledger.is_file()
    assert json.loads(remote_ledger.read_text(encoding="utf-8"))["completed_epoch"] == 1


def test_git_subprocesses_force_http11_without_persisting_or_exposing_token(
    monkeypatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def capture(command, **_kwargs):
        commands.append(command)
        return Completed()

    monkeypatch.setattr(publication, "_run", capture)
    publication.run_git_http11(
        ["ls-remote", "--heads", "origin", RESULTS_BRANCH],
        cwd=tmp_path,
        environment={"GIT_ASKPASS": "askpass.sh"},
    )

    assert commands == [
        [
            "git",
            "-c",
            "http.version=HTTP/1.1",
            "ls-remote",
            "--heads",
            "origin",
            RESULTS_BRANCH,
        ]
    ]
    rendered = " ".join(commands[0])
    assert "credential-must-never-escape" not in rendered
    assert "git config" not in rendered


def test_publish_config_accepts_and_validates_frozen_deployment_template(
    tmp_path: Path,
) -> None:
    from scripts.publish_iber_epoch import load_publication_config

    gate1 = tmp_path / "gate1-decision.json"
    gate1.write_text('{"status":"passed"}\n', encoding="utf-8")
    gate1_sha = hashlib.sha256(gate1.read_bytes()).hexdigest().upper()
    checkpoint = tmp_path / "epoch-0001.pt"
    _checkpoint(checkpoint, 1)
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
    artifact["gate1_decision_sha256"] = gate1_sha
    torch.save(artifact, checkpoint)
    token_file = tmp_path / "github-token"
    token_file.write_text("secret", encoding="utf-8")
    config_path = tmp_path / "publication-screen.json"
    payload = {
        "format_version": 1,
        "design_version": DESIGN_VERSION,
        "stage": "screen",
        "probe": "b3",
        "seed": 0,
        "expected_private_epochs": 30,
        "repo": "owner/repo",
        "repo_url": "https://github.com/owner/repo.git",
        "source_branch": "codex/iber-be",
        "source_commit": SOURCE_COMMIT,
        "results_branch": RESULTS_BRANCH,
        "tag": DEFAULT_TAG,
        "asset_prefix": ASSET_PREFIX,
        "run_name": "iber-be-screen-seed0",
        "token_file": str(token_file),
        "results_repo": str(tmp_path / "results"),
        "gate1_decision": str(gate1),
        "retain": 3,
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    if os.name != "nt":
        token_file.chmod(0o600)
        config_path.chmod(0o600)

    config = load_publication_config(config_path, checkpoint)
    assert config.identity.gate1_decision_sha256 == gate1_sha
    assert config.source_branch == "codex/iber-be"

    config_path.write_text(json.dumps({**payload, "tag": "foreign"}), encoding="utf-8")
    if os.name != "nt":
        config_path.chmod(0o600)
    with pytest.raises(ValueError, match="tag"):
        load_publication_config(config_path, checkpoint)
