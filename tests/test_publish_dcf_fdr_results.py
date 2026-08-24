from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.publish_dcf_fdr_results import (
    ASSET_NAMES,
    build_git_environment,
    checkpoint_assets,
    copy_lightweight_evidence,
    read_private_token,
    sanitized_error,
    update_release_manifest,
    upload_asset,
    verify_assets,
    verify_private_repository,
)
from src.dcf_fdr_publication import StagedEvidence


ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self, repo_payload: dict | None = None) -> None:
        self.repo_payload = repo_payload or {"private": True}
        self.deleted: list[str] = []
        self.uploaded: list[tuple[str, int]] = []

    def get(self, url: str, timeout: int) -> FakeResponse:
        return FakeResponse(200, self.repo_payload)

    def delete(self, url: str, timeout: int) -> FakeResponse:
        self.deleted.append(url)
        return FakeResponse(204)

    def post(self, url: str, *, params: dict, headers: dict, data, timeout) -> FakeResponse:
        payload = data.read()
        self.uploaded.append((params["name"], len(payload)))
        return FakeResponse(201, {"name": params["name"], "size": len(payload)})


def test_token_file_must_be_private_and_nonempty(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text("secret-value", encoding="utf-8")
    with patch("scripts.publish_dcf_fdr_results.stat.S_IMODE", return_value=0o644):
        with pytest.raises(PermissionError, match="0600"):
            read_private_token(token, enforce_mode=True)

    token.write_text("", encoding="utf-8")
    with patch("scripts.publish_dcf_fdr_results.stat.S_IMODE", return_value=0o600):
        with pytest.raises(RuntimeError, match="empty"):
            read_private_token(token, enforce_mode=True)


def test_git_askpass_environment_never_contains_token_value(tmp_path: Path) -> None:
    token_file = tmp_path / "github_token"
    token_file.write_text("super-secret-token", encoding="utf-8")
    askpass = tmp_path / "github-askpass.sh"
    environment = build_git_environment(askpass, token_file)
    script = askpass.read_text(encoding="utf-8")

    assert "super-secret-token" not in script
    assert "super-secret-token" not in json.dumps(environment)
    assert str(token_file) == environment["DCF_GITHUB_TOKEN_FILE"]
    if os.name != "nt":
        assert askpass.stat().st_mode & 0o777 == 0o700


def test_private_repository_gate_rejects_public_target() -> None:
    verify_private_repository(FakeSession({"private": True}), "owner/private")
    with pytest.raises(RuntimeError, match="not private"):
        verify_private_repository(FakeSession({"private": False}), "owner/public")


def test_release_asset_upload_is_idempotent_and_replaces_wrong_size(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "checkpoint.pt"
    asset.write_bytes(b"12345")
    upload_url = "https://uploads.example/{?name}"

    same = FakeSession()
    release = {
        "upload_url": upload_url,
        "assets": [{"name": "clean.pt", "size": 5, "url": "asset/1"}],
    }
    assert upload_asset(same, release, asset, "clean.pt") == "skipped"
    assert same.deleted == []
    assert same.uploaded == []

    wrong = FakeSession()
    release["assets"][0]["size"] = 4
    assert upload_asset(wrong, release, asset, "clean.pt") == "replaced"
    assert wrong.deleted == ["asset/1"]
    assert wrong.uploaded == [("clean.pt", 5)]


def test_checkpoint_assets_have_four_distinct_formal_names(tmp_path: Path) -> None:
    paths = {}
    for arm in ("clean", "dcf"):
        for kind in ("best", "last"):
            path = tmp_path / arm / f"{kind}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{arm}-{kind}".encode())
            paths[(arm, kind)] = path

    assets = checkpoint_assets(paths)
    assert set(assets) == set(ASSET_NAMES.values())
    assert len(assets) == 4
    assert all(path.is_file() for path in assets.values())


def test_manifest_binds_large_assets_and_remote_sizes_must_match(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "artifact-manifest.json"
    manifest.write_text('{"release_assets": {}}\n', encoding="utf-8")
    asset = tmp_path / "clean-best.pt"
    asset.write_bytes(b"checkpoint")
    update_release_manifest(manifest, {"clean-best.pt": asset})

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["release_assets"]["clean-best.pt"]["bytes"] == len(b"checkpoint")
    assert len(payload["release_assets"]["clean-best.pt"]["sha256"]) == 64
    assert verify_assets(
        {"clean-best.pt": len(b"checkpoint")},
        [{"name": "clean-best.pt", "size": len(b"checkpoint")}],
    )
    assert not verify_assets(
        {"clean-best.pt": len(b"checkpoint")},
        [{"name": "clean-best.pt", "size": 1}],
    )


def test_lightweight_checkout_excludes_release_bundle(tmp_path: Path) -> None:
    root = tmp_path / "stage"
    root.mkdir()
    manifest = root / "artifact-manifest.json"
    bundle = root / "lightweight-evidence.tar.gz"
    results = root / "comparison.json"
    manifest.write_text("{}\n", encoding="utf-8")
    bundle.write_bytes(b"bundle")
    results.write_text("{}\n", encoding="utf-8")
    staged = StagedEvidence(root, manifest, bundle, {"decision": "failed_negative"})
    destination = tmp_path / "checkout" / "experiment"

    copy_lightweight_evidence(staged, destination)

    assert (destination / "comparison.json").is_file()
    assert (destination / "artifact-manifest.json").is_file()
    assert not (destination / "lightweight-evidence.tar.gz").exists()


def test_sanitized_error_redacts_known_token() -> None:
    assert sanitized_error("request failed for secret-token", "secret-token") == (
        "request failed for <redacted>"
    )


def test_watcher_is_non_destructive_and_waits_for_dcf_completion() -> None:
    text = (ROOT / "scripts" / "watch_and_publish_dcf_fdr.sh").read_text(
        encoding="utf-8"
    )
    assert "publish_dcf_fdr_results.py" in text
    assert "sleep 60" in text
    assert "train_dcf_fdr.py --arm dcf" in text
    assert "publication-succeeded.json" in text
    assert "publication-failed.json" in text
    assert "/data/uav/runs/dcf-fdr-ec4e2a46-clean" in text
    assert "/data/uav/runs/dcf-fdr-ec4e2a46-dcf" in text
    assert "shutdown" not in text.lower()
    assert "poweroff" not in text.lower()
    assert "rm -rf" not in text
