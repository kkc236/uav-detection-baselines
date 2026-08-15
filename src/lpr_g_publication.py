"""Transactional per-epoch GitHub publication for LPR-G v2."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from scripts.sync_experiment_checkpoint import (
    _run,
    collect_lightweight_artifacts,
    commit_and_push_results,
    ensure_results_checkout,
    validate_token_file,
    write_json_atomic,
)
from src.github_checkpoint_sync import (
    checkpoint_metadata,
    get_or_create_release,
    github_session,
    publish_checkpoint,
    upload_asset,
)


RELEASE_NAME = "LPR-G v2 RTX 4090 live checkpoints"
RELEASE_BODY = (
    "Rolling validated seed0 control/LPR-G checkpoints; newest three per arm are retained."
)


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
    variant: str
    stage: str
    retain: int = 3


def write_jsonl_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


class PublicationLedger:
    """Append-only authority containing only remotely verified epochs."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        epochs = [int(row["completed_epoch"]) for row in rows]
        if epochs != list(range(1, len(rows) + 1)):
            raise ValueError(f"publication ledger is not contiguous: {epochs}")
        if any(row.get("verified") is not True for row in rows):
            raise ValueError("publication ledger contains an unverified record")
        return rows

    def append_verified(self, record: dict) -> None:
        if record.get("verified") is not True:
            raise ValueError("publication record is not remotely verified")
        rows = self.records()
        epoch = int(record["completed_epoch"])
        if epoch <= len(rows):
            if rows[epoch - 1] != record:
                raise ValueError(f"changed publication for completed epoch {epoch}")
            return
        if epoch != len(rows) + 1:
            raise ValueError(f"publication gap before completed epoch {epoch}")
        existing = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
        write_jsonl_atomic(
            self.path,
            existing + json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        )

    @property
    def last_completed_epoch(self) -> int:
        rows = self.records()
        return int(rows[-1]["completed_epoch"]) if rows else 0


def pending_epoch_checkpoints(
    weights: Path,
    ledger: PublicationLedger,
) -> list[tuple[int, Path]]:
    """Return unpublished checkpoints ordered by checkpoint-internal epoch."""
    pending = []
    last_completed_epoch = ledger.last_completed_epoch
    for path in Path(weights).glob("epoch*.pt"):
        metadata = checkpoint_metadata(path)
        if metadata.completed_epoch > last_completed_epoch:
            pending.append((metadata.completed_epoch, path))
    pending.sort(key=lambda item: item[0])
    expected = list(
        range(last_completed_epoch + 1, last_completed_epoch + len(pending) + 1)
    )
    actual = [epoch for epoch, _ in pending]
    if actual != expected:
        missing = next(epoch for epoch in expected if epoch not in actual)
        raise RuntimeError(f"missing completed epoch {missing} before publication")
    return pending


def _validate_config(config: PublicationConfig) -> None:
    if config.variant not in {"control", "lprg"}:
        raise ValueError(f"unknown LPR-G publication variant: {config.variant}")
    if config.stage not in {"screen", "formal"}:
        raise ValueError(f"unknown LPR-G publication stage: {config.stage}")
    if config.retain < 1:
        raise ValueError("publication retention must be at least one")
    expected_identity = f"{config.stage}-seed0-{config.variant}"
    if expected_identity not in config.asset_prefix:
        raise ValueError(
            f"asset prefix must isolate the arm as {expected_identity!r}: {config.asset_prefix!r}"
        )


def publish_exact_epoch(
    run_dir: Path,
    checkpoint: Path,
    config: PublicationConfig,
) -> dict:
    """Publish checkpoint, manifest, and light artifacts before committing the ledger."""
    _validate_config(config)
    run_dir = Path(run_dir).resolve()
    ledger = PublicationLedger(run_dir / "publication-ledger.jsonl")
    metadata = checkpoint_metadata(checkpoint)
    expected_epoch = ledger.last_completed_epoch + 1
    if metadata.completed_epoch != expected_epoch:
        raise RuntimeError(
            f"expected completed epoch {expected_epoch}, got {metadata.completed_epoch}"
        )

    token = validate_token_file(config.token_file)
    session = github_session(token)
    manifest = publish_checkpoint(
        session,
        repo=config.repo,
        tag=config.tag,
        branch=config.source_branch,
        checkpoint=checkpoint,
        retain=config.retain,
        asset_prefix=config.asset_prefix,
        release_name=RELEASE_NAME,
        release_body=RELEASE_BODY,
    )
    manifest.update(
        {
            "design_version": "lpr-g-v2",
            "variant": config.variant,
            "stage": config.stage,
            "seed": 0,
            "run_name": config.run_name,
        }
    )
    manifest_path = (
        run_dir
        / "weights"
        / f".{config.asset_prefix}-{metadata.completed_epoch:04d}.json"
    )
    write_json_atomic(manifest_path, manifest)
    try:
        release = get_or_create_release(
            session,
            repo=config.repo,
            tag=config.tag,
            branch=config.source_branch,
            release_name=RELEASE_NAME,
            release_body=RELEASE_BODY,
        )
        upload_asset(
            session,
            release=release,
            path=manifest_path,
            asset_name=f"{config.asset_prefix}-epoch-{metadata.completed_epoch:04d}.json",
        )

        environment = ensure_results_checkout(
            config.results_repo,
            repo_url=config.repo_url,
            branch=config.results_branch,
            token_file=config.token_file,
        )
        result_dir = config.results_repo / "results" / config.run_name
        collect_lightweight_artifacts(run_dir, result_dir, manifest)
        commit_and_push_results(
            config.results_repo,
            result_directory=result_dir,
            completed_epoch=metadata.completed_epoch,
            branch=config.results_branch,
            environment=environment,
        )
        artifact_sha = _run(
            ["git", "rev-parse", "HEAD"], cwd=config.results_repo
        ).stdout.strip()
        remote = session.get(
            f"https://api.github.com/repos/{config.repo}/commits/{config.results_branch}",
            timeout=30,
        )
        remote.raise_for_status()
        if remote.json().get("sha") != artifact_sha:
            raise RuntimeError("GitHub result branch did not reach the epoch artifact commit")

        record = {
            **manifest,
            "completed_epoch": metadata.completed_epoch,
            "result_commit_sha": artifact_sha,
            "verified": True,
        }
        ledger.append_verified(record)
        collect_lightweight_artifacts(run_dir, result_dir, record)
        commit_and_push_results(
            config.results_repo,
            result_directory=result_dir,
            completed_epoch=metadata.completed_epoch,
            branch=config.results_branch,
            environment=environment,
        )
        return record
    finally:
        manifest_path.unlink(missing_ok=True)


def publish_with_retry(
    run_dir: Path,
    checkpoint: Path,
    config: PublicationConfig,
    attempts: int = 10,
    delay: int = 30,
) -> dict:
    """Retry one exact publication transaction without exposing credentials."""
    if attempts < 1:
        raise ValueError("publication attempts must be at least one")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return publish_exact_epoch(run_dir, checkpoint, config)
        except Exception as error:
            last_error = error
            if attempt < attempts:
                time.sleep(delay)
    raise RuntimeError(f"epoch publication failed after {attempts} attempts") from last_error
