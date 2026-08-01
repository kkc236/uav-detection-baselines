"""Transactional per-epoch GitHub publication for I-TBER v1.1."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from scripts.sync_experiment_checkpoint import (
    _run,
    commit_and_push_results,
    ensure_results_checkout,
    validate_token_file,
)
from src.github_checkpoint_sync import (
    get_or_create_release,
    github_session,
    sha256_file,
    upload_asset,
)
from src.itber_protocol import (
    BASELINE_TRAINING_CONTRACT_SHA256,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    RUNTIME_AMENDMENT_SHA256,
)


RELEASE_NAME = "I-TBER v1.1 RTX 4090 live checkpoints"
RELEASE_BODY = (
    "Rolling verified seed0 I-TBER private checkpoints; the newest three "
    "checkpoint/manifest pairs per stage are retained."
)
DEFAULT_TAG = "itber-v1.1-rtdetr-l-live"


@dataclass(frozen=True)
class PublicationIdentity:
    design_version: str
    stage: str
    probe: str
    seed: int
    baseline_sha256: str
    dataset_sha256: str
    cache_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.design_version != "itber-v1.1":
            raise ValueError("I-TBER publication design version must be itber-v1.1")
        if self.stage not in {"screen", "formal"}:
            raise ValueError("I-TBER publication stage must be screen or formal")
        if self.probe != "p3" or self.seed != 0:
            raise ValueError("I-TBER publication identity must be seed0 P3")
        if self.baseline_sha256.upper() != EXPECTED_BASELINE_SHA256:
            raise ValueError("I-TBER publication baseline authority mismatch")
        if self.dataset_sha256.upper() != EXPECTED_DATASET_SHA256:
            raise ValueError("I-TBER publication dataset authority mismatch")
        if not re.fullmatch(r"[0-9A-Fa-f]{64}", self.cache_manifest_sha256):
            raise ValueError("I-TBER cache manifest SHA-256 is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "design_version": self.design_version,
            "stage": self.stage,
            "probe": self.probe,
            "seed": self.seed,
            "baseline_sha256": self.baseline_sha256.upper(),
            "dataset_sha256": self.dataset_sha256.upper(),
            "cache_manifest_sha256": self.cache_manifest_sha256.upper(),
            "baseline_training_contract_sha256": BASELINE_TRAINING_CONTRACT_SHA256,
            "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
        }


@dataclass(frozen=True)
class PrivateCheckpointMetadata:
    source: Path
    completed_epoch: int
    bytes: int
    sha256: str


@dataclass(frozen=True)
class PublicationConfig:
    repo: str
    repo_url: str
    source_branch: str
    results_branch: str
    tag: str
    asset_prefix: str
    run_name: str
    token_file: Path
    results_repo: Path
    identity: PublicationIdentity
    retain: int = 3

    def __post_init__(self) -> None:
        expected_prefix = (
            f"{self.identity.design_version}-{self.identity.stage}-"
            f"seed{self.identity.seed}-{self.identity.probe}"
        )
        if self.asset_prefix != expected_prefix:
            raise ValueError(f"I-TBER asset prefix must be exactly {expected_prefix!r}")
        if self.retain < 1:
            raise ValueError("I-TBER publication retention must be positive")


def read_token_file(path: str | Path) -> str:
    """Read the mode-600 token without logging or embedding its value."""
    return validate_token_file(Path(path))


def private_checkpoint_metadata(
    path: str | Path,
    *,
    identity: PublicationIdentity,
) -> PrivateCheckpointMetadata:
    """Validate one resumable private checkpoint and its exact identity."""
    source = Path(path).resolve()
    artifact = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict):
        raise ValueError("I-TBER private checkpoint is not a dictionary")
    expected = {"format_version": 1, **identity.as_dict()}
    for name, value in expected.items():
        actual = artifact.get(name)
        if name.endswith("sha256") and isinstance(actual, str):
            actual = actual.upper()
        if actual != value:
            raise ValueError(f"I-TBER private checkpoint {name} identity mismatch")
    epoch = artifact.get("epoch")
    if not isinstance(epoch, int) or epoch < 1:
        raise ValueError("I-TBER private checkpoint epoch is invalid")
    for name in ("refiner", "optimizer", "scaler", "rng"):
        if name not in artifact or artifact[name] is None:
            raise ValueError(f"I-TBER private checkpoint {name} state is missing")
    if not isinstance(artifact["refiner"], Mapping) or not artifact["refiner"]:
        raise ValueError("I-TBER private checkpoint refiner state is empty")
    return PrivateCheckpointMetadata(
        source=source,
        completed_epoch=epoch,
        bytes=source.stat().st_size,
        sha256=sha256_file(source),
    )


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


class PublicationLedger:
    """Append-only authority containing only remotely verified epochs."""

    def __init__(self, path: str | Path, identity: PublicationIdentity):
        self.path = Path(path)
        self.identity = identity

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]
        epochs = [int(row.get("completed_epoch", -1)) for row in rows]
        if epochs != list(range(1, len(rows) + 1)):
            raise ValueError(f"I-TBER publication ledger is not contiguous: {epochs}")
        expected_identity = self.identity.as_dict()
        for row in rows:
            if row.get("verified") is not True:
                raise ValueError("I-TBER publication ledger contains an unverified record")
            if any(row.get(name) != value for name, value in expected_identity.items()):
                raise ValueError("I-TBER publication ledger identity mismatch")
        return rows

    def append_verified(self, record: dict[str, Any]) -> None:
        if record.get("verified") is not True:
            raise ValueError("I-TBER publication record is not verified")
        expected_identity = self.identity.as_dict()
        if any(record.get(name) != value for name, value in expected_identity.items()):
            raise ValueError("I-TBER publication record identity mismatch")
        rows = self.records()
        epoch = int(record.get("completed_epoch", -1))
        if epoch <= len(rows):
            if epoch < 1 or rows[epoch - 1] != record:
                raise ValueError(f"changed I-TBER publication for completed epoch {epoch}")
            return
        if epoch != len(rows) + 1:
            raise ValueError(f"I-TBER publication gap before completed epoch {epoch}")
        _write_jsonl_atomic(self.path, [*rows, record])

    @property
    def last_completed_epoch(self) -> int:
        rows = self.records()
        return int(rows[-1]["completed_epoch"]) if rows else 0


def pending_epoch_checkpoints(
    checkpoint_root: str | Path,
    ledger: PublicationLedger,
    *,
    identity: PublicationIdentity,
) -> list[tuple[int, Path]]:
    """Return unpublished private checkpoints, rejecting every epoch gap."""
    pending: list[tuple[int, Path]] = []
    last = ledger.last_completed_epoch
    for path in Path(checkpoint_root).glob("epoch-*.pt"):
        metadata = private_checkpoint_metadata(path, identity=identity)
        if metadata.completed_epoch > last:
            pending.append((metadata.completed_epoch, path))
    pending.sort(key=lambda item: item[0])
    if pending:
        expected = list(range(last + 1, pending[-1][0] + 1))
        actual = [epoch for epoch, _ in pending]
        if actual != expected:
            missing = next(epoch for epoch in expected if epoch not in actual)
            raise RuntimeError(f"missing completed epoch {missing} before I-TBER publication")
    return pending


def _asset_name(prefix: str, epoch: int, suffix: str) -> str:
    return f"{prefix}-epoch-{epoch:04d}.{suffix}"


def _refresh_release(session: Any, release: Mapping[str, Any]) -> dict[str, Any]:
    response = session.get(str(release["url"]), timeout=30)
    response.raise_for_status()
    return response.json()


def _prune_verified_pairs(session: Any, release: Mapping[str, Any], *, prefix: str, retain: int) -> None:
    pattern = re.compile(rf"^{re.escape(prefix)}-epoch-(\d+)\.(pt|json)$")
    pairs: dict[int, dict[str, dict[str, Any]]] = {}
    for asset in release.get("assets", []):
        match = pattern.match(str(asset.get("name", "")))
        if match:
            pairs.setdefault(int(match.group(1)), {})[match.group(2)] = asset
    complete_epochs = sorted(epoch for epoch, pair in pairs.items() if {"pt", "json"} <= pair.keys())
    for epoch in complete_epochs[:-retain]:
        for suffix in ("pt", "json"):
            response = session.delete(str(pairs[epoch][suffix]["url"]), timeout=30)
            response.raise_for_status()


def _collect_lightweight(run_dir: Path, destination: Path, record: Mapping[str, Any]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    excluded_parts = {"checkpoints", "weights", "cache", ".git"}
    for pattern in ("*.json", "*.jsonl", "*.csv", "*.yaml", "*.yml"):
        for source in run_dir.rglob(pattern):
            relative = source.relative_to(run_dir)
            if excluded_parts.intersection(relative.parts) or not source.is_file():
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    latest = destination / "latest.json"
    temporary = latest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(dict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, latest)


def publish_exact_epoch(
    run_dir: str | Path,
    checkpoint: str | Path,
    config: PublicationConfig,
) -> dict[str, Any]:
    """Publish checkpoint + manifest, verify both, then append the ledger."""
    run_root = Path(run_dir).resolve()
    ledger = PublicationLedger(run_root / "publication-ledger.jsonl", config.identity)
    metadata = private_checkpoint_metadata(checkpoint, identity=config.identity)
    if metadata.completed_epoch != ledger.last_completed_epoch + 1:
        raise RuntimeError(
            f"expected I-TBER completed epoch {ledger.last_completed_epoch + 1}, "
            f"got {metadata.completed_epoch}"
        )
    session = github_session(read_token_file(config.token_file))
    release = get_or_create_release(
        session,
        repo=config.repo,
        tag=config.tag,
        branch=config.source_branch,
        release_name=RELEASE_NAME,
        release_body=RELEASE_BODY,
    )
    checkpoint_name = _asset_name(config.asset_prefix, metadata.completed_epoch, "pt")
    checkpoint_asset = upload_asset(
        session,
        release=release,
        path=metadata.source,
        asset_name=checkpoint_name,
    )
    manifest = {
        "format_version": 1,
        **config.identity.as_dict(),
        "completed_epoch": metadata.completed_epoch,
        "run_name": config.run_name,
        "release_url": str(release["html_url"]),
        "checkpoint": {
            "asset_id": int(checkpoint_asset["id"]),
            "asset_name": checkpoint_name,
            "bytes": metadata.bytes,
            "sha256": metadata.sha256,
        },
    }
    manifest_path = run_root / "checkpoints" / f".{config.asset_prefix}-{metadata.completed_epoch:04d}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        manifest_name = _asset_name(config.asset_prefix, metadata.completed_epoch, "json")
        manifest_asset = upload_asset(
            session,
            release=release,
            path=manifest_path,
            asset_name=manifest_name,
        )
        refreshed = _refresh_release(session, release)
        remote = {asset["name"]: asset for asset in refreshed.get("assets", [])}
        if checkpoint_name not in remote or manifest_name not in remote:
            raise RuntimeError("I-TBER remote checkpoint/manifest pair is incomplete")
        if int(remote[checkpoint_name]["size"]) != metadata.bytes:
            raise RuntimeError("I-TBER remote checkpoint bytes mismatch")

        environment = ensure_results_checkout(
            config.results_repo,
            repo_url=config.repo_url,
            branch=config.results_branch,
            token_file=config.token_file,
        )
        result_directory = config.results_repo / "results" / config.run_name
        provisional = {
            **manifest,
            "manifest_asset_id": int(manifest_asset["id"]),
            "verified": True,
        }
        _collect_lightweight(run_root, result_directory, provisional)
        commit_and_push_results(
            config.results_repo,
            result_directory=result_directory,
            completed_epoch=metadata.completed_epoch,
            branch=config.results_branch,
            environment=environment,
        )
        commit_sha = _run(["git", "rev-parse", "HEAD"], cwd=config.results_repo).stdout.strip()
        branch_response = session.get(
            f"https://api.github.com/repos/{config.repo}/commits/{config.results_branch}", timeout=30
        )
        branch_response.raise_for_status()
        if branch_response.json().get("sha") != commit_sha:
            raise RuntimeError("I-TBER result branch verification failed")
        record = {**provisional, "result_commit_sha": commit_sha}
        ledger.append_verified(record)
        _collect_lightweight(run_root, result_directory, record)
        commit_and_push_results(
            config.results_repo,
            result_directory=result_directory,
            completed_epoch=metadata.completed_epoch,
            branch=config.results_branch,
            environment=environment,
        )
        _prune_verified_pairs(
            session,
            _refresh_release(session, release),
            prefix=config.asset_prefix,
            retain=config.retain,
        )
        return record
    finally:
        manifest_path.unlink(missing_ok=True)


def publish_with_retry(
    run_dir: str | Path,
    checkpoint: str | Path,
    config: PublicationConfig,
    *,
    attempts: int = 10,
    delay: int = 30,
) -> dict[str, Any]:
    """Retry one transaction without rendering exception details or credentials."""
    if attempts < 1:
        raise ValueError("I-TBER publication attempts must be positive")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return publish_exact_epoch(run_dir, checkpoint, config)
        except Exception as error:
            last_error = error
            if attempt < attempts:
                time.sleep(delay)
    raise RuntimeError(f"I-TBER epoch publication failed after {attempts} attempts") from last_error
