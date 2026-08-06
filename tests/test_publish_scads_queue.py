from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.publish_scads_queue import (
    append_ledger,
    asset_names,
    publish_record,
    prune_verified_epochs,
)


class FakeReleaseClient:
    def __init__(self) -> None:
        self.release_url = "https://github.example/releases/scads"
        self.remote: dict[str, bytes] = {}

    def ensure_release(self) -> dict:
        return {"url": self.release_url}

    def assets(self):
        return self.release_url, {
            name: {"name": name, "size": len(content)}
            for name, content in self.remote.items()
        }

    def upload(self, path: Path, asset_name: str) -> None:
        self.remote[asset_name] = Path(path).read_bytes()

    def download(self, asset_name: str, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=False)
        output = destination / asset_name
        output.write_bytes(self.remote[asset_name])
        return output


def _queued(tmp_path: Path, *, epoch: int = 1) -> dict:
    run = tmp_path / "run"
    weights = run / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    checkpoint = weights / f"epoch{epoch - 1}.pt"
    checkpoint.write_bytes(f"checkpoint-{epoch}".encode())
    evidence = run / "scads-epochs.jsonl"
    evidence.write_text(json.dumps({"completed_epoch": epoch}) + "\n", encoding="utf-8")
    return {
        "run_id": "scads-screen-seed0-abc-def",
        "variant": "scads",
        "stage": "screen",
        "completed_epoch": epoch,
        "status": "pending",
        "checkpoint": str(checkpoint),
        "checkpoint_size": checkpoint.stat().st_size,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper(),
        "artifacts": [str(evidence)],
    }


def test_asset_names_are_stable_and_reject_unsafe_run_ids(tmp_path: Path) -> None:
    record = _queued(tmp_path)
    assert asset_names(record) == (
        "scads-screen-seed0-abc-def-epoch-0001.pt",
        "scads-screen-seed0-abc-def-epoch-0001.json",
    )
    record["run_id"] = "../unsafe"
    with pytest.raises(ValueError, match="not safe"):
        asset_names(record)


def test_publish_verifies_remote_size_and_downloaded_sha_sidecar(tmp_path: Path) -> None:
    record = _queued(tmp_path)
    client = FakeReleaseClient()
    ledger = tmp_path / "publication-ledger.jsonl"

    published = publish_record(record, client=client, ledger_path=ledger)

    checkpoint_asset, sidecar_asset = asset_names(record)
    assert published["status"] == "published-verified"
    assert published["checkpoint_asset"] == checkpoint_asset
    assert set(client.remote) == {checkpoint_asset, sidecar_asset}
    sidecar = json.loads(client.remote[sidecar_asset])
    assert sidecar["checkpoint"]["sha256"] == record["checkpoint_sha256"]
    assert sidecar["checkpoint"]["bytes"] == record["checkpoint_size"]
    assert sidecar["evidence_artifacts"][0]["name"] == "scads-epochs.jsonl"
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1


def test_publication_is_resume_safe_when_remote_assets_already_exist(tmp_path: Path) -> None:
    record = _queued(tmp_path)
    first_client = FakeReleaseClient()
    ledger = tmp_path / "publication-ledger.jsonl"
    first = publish_record(record, client=first_client, ledger_path=ledger)
    preserved = dict(first_client.remote)

    second_client = FakeReleaseClient()
    second_client.remote = preserved
    second = publish_record(record, client=second_client, ledger_path=ledger)

    assert second == first
    assert second_client.remote == preserved
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1


def test_remote_size_conflict_fails_closed(tmp_path: Path) -> None:
    record = _queued(tmp_path)
    checkpoint_asset, _sidecar_asset = asset_names(record)
    client = FakeReleaseClient()
    client.remote[checkpoint_asset] = b"wrong"
    with pytest.raises(ValueError, match="immutable remote asset size conflict"):
        publish_record(record, client=client, ledger_path=tmp_path / "ledger.jsonl")


def test_prune_removes_only_verified_old_epoch_checkpoints(tmp_path: Path) -> None:
    queue = [_queued(tmp_path, epoch=epoch) for epoch in (1, 2, 3)]
    ledger = []
    for row in queue:
        ledger.append(
            {
                **row,
                "status": "published-verified",
                "checkpoint_asset": asset_names(row)[0],
                "sidecar_asset": asset_names(row)[1],
                "release_url": "https://github.example/release",
            }
        )
    last = Path(queue[-1]["checkpoint"])
    best = last.parent / "best.pt"
    shutil.copy2(last, best)

    removed = prune_verified_epochs(queue, ledger, retain=1)

    assert len(removed) == 2
    assert last.is_file()
    assert best.is_file()


def test_append_ledger_rejects_changed_immutable_entry(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    base = {
        "run_id": "run",
        "variant": "scads",
        "stage": "screen",
        "completed_epoch": 1,
        "checkpoint_sha256": "A" * 64,
        "checkpoint_size": 1,
        "checkpoint_asset": "run-epoch-0001.pt",
        "sidecar_asset": "run-epoch-0001.json",
        "release_url": "https://github.example/release",
        "status": "published-verified",
    }
    append_ledger(path, base)
    changed = {**base, "checkpoint_sha256": "B" * 64}
    with pytest.raises(ValueError, match="changed publication ledger"):
        append_ledger(path, changed)
