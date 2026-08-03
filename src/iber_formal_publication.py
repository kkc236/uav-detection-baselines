"""Transactional publication authority for full-model IBER-BE formal100."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scripts.sync_experiment_checkpoint import (
    _run,
    collect_lightweight_artifacts,
    validate_token_file,
    write_json_atomic,
)
from src.github_checkpoint_sync import (
    checkpoint_metadata,
    get_or_create_release,
    github_session,
    upload_asset,
)
from src.iber_formal_protocol import (
    EXPECTED_DATASET_SHA256,
    FORMAL_DESIGN_VERSION,
    FORMAL_EPOCHS,
)
from src.iber_publication import (
    commit_and_push_results,
    ensure_results_checkout,
    verify_remote_asset,
)


FORMAL_ASSET_PREFIX = "iber-be-v1.0-formal-seed0-b3"
FORMAL_RESULTS_BRANCH = "iber-be-v1-results"
FORMAL_RELEASE_NAME = "IBER-BE v1.0 formal100 live checkpoints"
FORMAL_RELEASE_BODY = (
    "Rolling remotely verified seed0 full-model IBER-BE checkpoints; "
    "the newest complete checkpoint/manifest pairs are retained."
)
_HEX64 = re.compile(r"[0-9A-Fa-f]{64}")
_GIT_SHA = re.compile(r"[0-9A-Fa-f]{40}(?:[0-9A-Fa-f]{24})?")


def _require_hash(name: str, value: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"formal publication {name} is invalid")
    return value.lower()


@dataclass(frozen=True)
class FormalPublicationIdentity:
    source_commit: str
    protocol_sha256: str
    initial_state_sha256: str

    def __post_init__(self) -> None:
        _require_hash("source commit", self.source_commit, _GIT_SHA)
        _require_hash("protocol SHA-256", self.protocol_sha256, _HEX64)
        _require_hash("initial-state SHA-256", self.initial_state_sha256, _HEX64)

    def as_dict(self) -> dict[str, Any]:
        return {
            "design_version": FORMAL_DESIGN_VERSION,
            "stage": "formal",
            "seed": 0,
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "source_commit": self.source_commit.lower(),
            "protocol_sha256": self.protocol_sha256.upper(),
            "initial_state_sha256": self.initial_state_sha256.upper(),
        }


@dataclass(frozen=True)
class FormalPublicationConfig:
    repo: str
    repo_url: str
    source_branch: str
    tag: str
    run_name: str
    token_file: Path
    results_repo: Path
    identity: FormalPublicationIdentity
    results_branch: str = FORMAL_RESULTS_BRANCH
    asset_prefix: str = FORMAL_ASSET_PREFIX
    retain: int = 3

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repo):
            raise ValueError("formal publication repo must be owner/name")
        if not self.repo_url.startswith("https://github.com/") or "@" in self.repo_url:
            raise ValueError("formal publication repo URL must be credential-free HTTPS GitHub")
        if self.retain < 1:
            raise ValueError("formal publication retention must be positive")
        if self.asset_prefix != FORMAL_ASSET_PREFIX:
            raise ValueError("formal publication asset prefix is frozen")


def _canonical_row(row: Mapping[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


class FormalPublicationLedger:
    """Append-only identity-bound authority for remotely verified epochs 1..100."""

    def __init__(self, path: str | Path, identity: FormalPublicationIdentity):
        self.path = Path(path)
        self.identity = identity

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        epochs = [row.get("completed_epoch") for row in rows]
        if epochs != list(range(1, len(rows) + 1)) or len(rows) > FORMAL_EPOCHS:
            raise ValueError(f"formal publication ledger is not contiguous in 1..100: {epochs}")
        identity = self.identity.as_dict()
        for row in rows:
            if row.get("verified") is not True:
                raise ValueError("formal publication ledger contains an unverified row")
            if any(row.get(name) != value for name, value in identity.items()):
                raise ValueError("formal publication ledger identity mismatch")
            if _GIT_SHA.fullmatch(str(row.get("result_commit_sha", ""))) is None:
                raise ValueError("formal publication result commit is invalid")
        return rows

    def append_verified(self, record: dict[str, Any]) -> None:
        if record.get("verified") is not True:
            raise ValueError("formal publication record is not remotely verified")
        if any(record.get(name) != value for name, value in self.identity.as_dict().items()):
            raise ValueError("formal publication record identity mismatch")
        epoch = record.get("completed_epoch")
        if type(epoch) is not int or not 1 <= epoch <= FORMAL_EPOCHS:
            raise ValueError("formal completed epoch must be in 1..100")
        rows = self.records()
        if epoch <= len(rows):
            if rows[epoch - 1] != record:
                raise ValueError(f"changed formal publication for completed epoch {epoch}")
            return
        if epoch != len(rows) + 1:
            raise ValueError(f"formal publication gap before completed epoch {epoch}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text("".join(_canonical_row(row) for row in [*rows, record]), encoding="utf-8")
        os.replace(temporary, self.path)

    @property
    def last_completed_epoch(self) -> int:
        rows = self.records()
        return int(rows[-1]["completed_epoch"]) if rows else 0


def pending_epoch_checkpoints(
    weights: str | Path,
    ledger: FormalPublicationLedger,
) -> list[tuple[int, Path]]:
    pending: list[tuple[int, Path]] = []
    last = ledger.last_completed_epoch
    for path in Path(weights).glob("epoch*.pt"):
        metadata = checkpoint_metadata(path)
        if metadata.completed_epoch > last:
            pending.append((metadata.completed_epoch, path))
    pending.sort(key=lambda item: item[0])
    if pending:
        actual = [epoch for epoch, _ in pending]
        expected = list(range(last + 1, pending[-1][0] + 1))
        if actual != expected:
            raise RuntimeError(f"formal publication checkpoint gap: {actual}")
    return pending


def _asset_name(prefix: str, epoch: int, suffix: str) -> str:
    return f"{prefix}-epoch-{epoch:04d}.{suffix}"


def _upload_verified_asset(
    session: Any,
    release: Mapping[str, Any],
    *,
    path: Path,
    asset_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Delete a same-name corrupt asset so an outer retry can replace it."""
    asset = upload_asset(
        session,
        release=dict(release),
        path=path,
        asset_name=asset_name,
    )
    try:
        receipt = verify_remote_asset(session, asset, path)
    except Exception:
        response = session.delete(str(asset["url"]), timeout=30)
        response.raise_for_status()
        raise
    return asset, receipt


def _collect_formal_artifacts(
    run_dir: Path,
    result_dir: Path,
    manifest: dict[str, Any],
) -> None:
    collect_lightweight_artifacts(run_dir, result_dir, manifest)
    for name in ("iber_formal_protocol.json", "iber_formal_diagnostics.jsonl"):
        source = run_dir / name
        if source.is_file():
            result_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, result_dir / name)


def _publish_results_snapshot(
    run_dir: Path,
    config: FormalPublicationConfig,
    manifest: dict[str, Any],
    session: Any,
) -> str:
    """Push one lightweight snapshot and verify the exact remote branch head."""
    environment = ensure_results_checkout(
        config.results_repo,
        repo_url=config.repo_url,
        branch=config.results_branch,
        token_file=config.token_file,
    )
    result_dir = config.results_repo / "results" / config.run_name
    _collect_formal_artifacts(run_dir, result_dir, manifest)
    commit_and_push_results(
        config.results_repo,
        result_directory=result_dir,
        completed_epoch=int(manifest["completed_epoch"]),
        branch=config.results_branch,
        environment=environment,
    )
    commit_sha = _run(
        ["git", "rev-parse", "HEAD"], cwd=config.results_repo
    ).stdout.strip()
    if _GIT_SHA.fullmatch(commit_sha) is None:
        raise RuntimeError("formal result commit SHA is invalid")
    remote = session.get(
        f"https://api.github.com/repos/{config.repo}/commits/{config.results_branch}",
        timeout=30,
    )
    remote.raise_for_status()
    if remote.json().get("sha") != commit_sha:
        raise RuntimeError("formal result branch did not reach the published snapshot")
    return commit_sha


def _validate_replayed_checkpoint(
    record: Mapping[str, Any],
    metadata: Any,
    config: FormalPublicationConfig,
) -> None:
    expected = record.get("checkpoint", {})
    actual = {
        "asset_name": _asset_name(
            config.asset_prefix, metadata.completed_epoch, "pt"
        ),
        "bytes": metadata.bytes,
        "sha256": metadata.sha256,
    }
    if (
        expected.get("asset_name") != actual["asset_name"]
        or expected.get("bytes") != actual["bytes"]
        or str(expected.get("sha256", "")).lower() != actual["sha256"].lower()
    ):
        raise ValueError(
            f"changed formal checkpoint replay for completed epoch {metadata.completed_epoch}"
        )


def publish_exact_epoch(
    run_dir: str | Path,
    checkpoint: str | Path,
    config: FormalPublicationConfig,
) -> dict[str, Any]:
    """Publish and remotely verify one exact formal checkpoint before ledger commit."""
    run_dir = Path(run_dir).resolve()
    checkpoint = Path(checkpoint).resolve()
    ledger = FormalPublicationLedger(run_dir / "publication-ledger.jsonl", config.identity)
    metadata = checkpoint_metadata(checkpoint)
    rows = ledger.records()
    if metadata.completed_epoch <= len(rows):
        record = rows[metadata.completed_epoch - 1]
        _validate_replayed_checkpoint(record, metadata, config)
        token = validate_token_file(config.token_file)
        session = github_session(token)
        _publish_results_snapshot(run_dir, config, record, session)
        return record

    expected_epoch = len(rows) + 1
    if metadata.completed_epoch != expected_epoch or metadata.completed_epoch > FORMAL_EPOCHS:
        raise RuntimeError(
            f"expected formal completed epoch {expected_epoch}, got {metadata.completed_epoch}"
        )
    token = validate_token_file(config.token_file)
    session = github_session(token)
    release = get_or_create_release(
        session,
        repo=config.repo,
        tag=config.tag,
        branch=config.source_branch,
        release_name=FORMAL_RELEASE_NAME,
        release_body=FORMAL_RELEASE_BODY,
    )
    checkpoint_asset, checkpoint_receipt = _upload_verified_asset(
        session,
        release,
        path=checkpoint,
        asset_name=_asset_name(config.asset_prefix, metadata.completed_epoch, "pt"),
    )
    manifest = {
        **config.identity.as_dict(),
        "completed_epoch": metadata.completed_epoch,
        "run_name": config.run_name,
        "checkpoint": {
            "asset_name": checkpoint_asset["name"],
            "bytes": metadata.bytes,
            "sha256": metadata.sha256,
        },
        "remote_verification": {"checkpoint": checkpoint_receipt},
    }
    manifest_path = run_dir / "weights" / f".{config.asset_prefix}-{metadata.completed_epoch:04d}.json"
    write_json_atomic(manifest_path, manifest)
    try:
        manifest_asset, manifest_receipt = _upload_verified_asset(
            session,
            release,
            path=manifest_path,
            asset_name=_asset_name(config.asset_prefix, metadata.completed_epoch, "json"),
        )
        manifest["remote_verification"]["manifest"] = manifest_receipt
        commit_sha = _publish_results_snapshot(run_dir, config, manifest, session)
        record = {**manifest, "result_commit_sha": commit_sha, "verified": True}
        ledger.append_verified(record)
        _publish_results_snapshot(run_dir, config, record, session)
        return record
    finally:
        manifest_path.unlink(missing_ok=True)


def publish_with_retry(
    run_dir: str | Path,
    checkpoint: str | Path,
    config: FormalPublicationConfig,
    *,
    attempts: int = 10,
    delay: int = 30,
) -> dict[str, Any]:
    if attempts < 1:
        raise ValueError("formal publication attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            return publish_exact_epoch(run_dir, checkpoint, config)
        except Exception:
            if attempt == attempts:
                break
            time.sleep(delay)
    raise RuntimeError(f"formal epoch publication failed after {attempts} attempts") from None


__all__ = [
    "FORMAL_ASSET_PREFIX",
    "FormalPublicationConfig",
    "FormalPublicationIdentity",
    "FormalPublicationLedger",
    "pending_epoch_checkpoints",
    "publish_exact_epoch",
    "publish_with_retry",
]
