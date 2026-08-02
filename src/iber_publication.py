"""Transactional publication authority for the IBER-BE v1.0 screen."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

import torch

from scripts.sync_experiment_checkpoint import (
    _run,
    validate_token_file,
)
from src.github_checkpoint_sync import (
    get_or_create_release,
    github_session,
    upload_asset,
)
from src.iber_protocol import (
    DESIGN_VERSION,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    PROTOCOL_SHA256,
    RUNTIME_AMENDMENT_SHA256,
    SCREEN_EPOCHS,
    file_sha256,
)


DEFAULT_TAG = "iber-be-v1-rtdetr-l-live"
ASSET_PREFIX = "iber-be-v1.0-screen-seed0-b3"
RESULTS_BRANCH = "iber-be-v1-results"
RETAINED_PAIRS = 3
RELEASE_NAME = "IBER-BE v1.0 RTX 4090 live checkpoints"
RELEASE_BODY = (
    "Rolling remotely verified seed0 B3 screen checkpoints. The newest three "
    "complete checkpoint/manifest pairs are retained."
)
_HEX64 = re.compile(r"[0-9A-Fa-f]{64}")
_GIT_SHA = re.compile(r"[0-9A-Fa-f]{40}(?:[0-9A-Fa-f]{24})?")


def _require_hex64(name: str, value: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"IBER-BE {name} SHA-256 is invalid")
    return value.upper()


@dataclass(frozen=True)
class PublicationIdentity:
    design_version: str
    stage: str
    probe: str
    seed: int
    baseline_sha256: str
    dataset_sha256: str
    subset_sha256: str
    category_sha256: str
    protocol_sha256: str
    runtime_amendment_sha256: str
    gate1_decision_sha256: str
    source_commit: str

    def __post_init__(self) -> None:
        if self.design_version != DESIGN_VERSION:
            raise ValueError(f"IBER-BE design version must be {DESIGN_VERSION}")
        if self.stage != "screen":
            raise ValueError("IBER-BE publication stage must be screen")
        if self.probe != "b3":
            raise ValueError("IBER-BE publication probe must be b3")
        if type(self.seed) is not int or self.seed != 0:
            raise ValueError("IBER-BE publication seed must be seed0")
        fixed = {
            "baseline": (self.baseline_sha256, EXPECTED_BASELINE_SHA256),
            "dataset": (self.dataset_sha256, EXPECTED_DATASET_SHA256),
            "subset": (self.subset_sha256, EXPECTED_SUBSET_SHA256),
            "protocol": (self.protocol_sha256, PROTOCOL_SHA256),
            "runtime amendment": (
                self.runtime_amendment_sha256,
                RUNTIME_AMENDMENT_SHA256,
            ),
        }
        for name, (actual, expected) in fixed.items():
            if _require_hex64(name, actual) != expected.upper():
                raise ValueError(f"IBER-BE publication {name} identity mismatch")
        _require_hex64("category", self.category_sha256)
        _require_hex64("Gate-1 decision", self.gate1_decision_sha256)
        if not isinstance(self.source_commit, str) or _GIT_SHA.fullmatch(self.source_commit) is None:
            raise ValueError("IBER-BE publication source commit is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "design_version": DESIGN_VERSION,
            "stage": "screen",
            "probe": "b3",
            "seed": 0,
            "baseline_sha256": EXPECTED_BASELINE_SHA256,
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "subset_sha256": EXPECTED_SUBSET_SHA256,
            "category_sha256": self.category_sha256.upper(),
            "protocol_sha256": PROTOCOL_SHA256,
            "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
            "gate1_decision_sha256": self.gate1_decision_sha256.upper(),
            "source_commit": self.source_commit.lower(),
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
    run_name: str
    token_file: Path
    results_repo: Path
    identity: PublicationIdentity
    results_branch: str = field(default=RESULTS_BRANCH, init=False)
    tag: str = field(default=DEFAULT_TAG, init=False)
    asset_prefix: str = field(default=ASSET_PREFIX, init=False)
    retain: int = field(default=RETAINED_PAIRS, init=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repo):
            raise ValueError("IBER-BE GitHub repo must be owner/name")
        parsed = urlsplit(self.repo_url)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("IBER-BE repo URL must not embed credentials")
        if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
            raise ValueError("IBER-BE repo URL must be an HTTPS github.com URL")
        if not self.source_branch or not self.run_name:
            raise ValueError("IBER-BE source branch and run name are required")


def read_token_file(path: str | Path) -> str:
    """Read a token only from a permission-restricted file."""
    return validate_token_file(Path(path))


def run_git_http11(
    arguments: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
    check: bool = True,
):
    """Run one Git subprocess with a non-persistent HTTP/1.1 override."""
    return _run(
        ["git", "-c", "http.version=HTTP/1.1", *arguments],
        cwd=cwd,
        env=environment,
        check=check,
    )


def _write_askpass(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  *Username*) printf '%s\\n' 'x-access-token' ;;\n"
        "  *) cat \"$IBER_GITHUB_TOKEN_FILE\" ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    path.chmod(0o700)


def _git_environment(askpass: Path, token_file: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_ASKPASS": str(askpass),
            "GIT_TERMINAL_PROMPT": "0",
            "IBER_GITHUB_TOKEN_FILE": str(token_file),
        }
    )
    return environment


def ensure_results_checkout(
    results_repo: Path,
    *,
    repo_url: str,
    branch: str,
    token_file: Path,
) -> dict[str, str]:
    """Create/update the dedicated results checkout using HTTP/1.1 Git commands."""
    results_repo = Path(results_repo)
    bootstrap_askpass = results_repo.parent / f".{results_repo.name}-iber-askpass.sh"
    final_askpass = results_repo / ".git" / "iber-github-askpass.sh"
    if not (results_repo / ".git").is_dir():
        results_repo.parent.mkdir(parents=True, exist_ok=True)
        _write_askpass(bootstrap_askpass)
        bootstrap_environment = _git_environment(bootstrap_askpass, token_file)
        try:
            run_git_http11(
                ["clone", repo_url, str(results_repo)],
                cwd=results_repo.parent,
                environment=bootstrap_environment,
            )
        finally:
            bootstrap_askpass.unlink(missing_ok=True)
    _write_askpass(final_askpass)
    environment = _git_environment(final_askpass, token_file)
    run_git_http11(["config", "user.name", "uav-training-bot"], cwd=results_repo)
    run_git_http11(
        ["config", "user.email", "uav-training-bot@users.noreply.github.com"],
        cwd=results_repo,
    )
    local_branch = run_git_http11(
        ["branch", "--list", branch], cwd=results_repo
    ).stdout.strip()
    remote = run_git_http11(
        ["ls-remote", "--heads", "origin", branch],
        cwd=results_repo,
        environment=environment,
    ).stdout
    if local_branch:
        run_git_http11(["switch", branch], cwd=results_repo)
        if remote.strip():
            run_git_http11(
                ["fetch", "origin", branch],
                cwd=results_repo,
                environment=environment,
            )
            run_git_http11(["rebase", "FETCH_HEAD"], cwd=results_repo)
        return environment
    if remote.strip():
        run_git_http11(
            ["fetch", "origin", branch],
            cwd=results_repo,
            environment=environment,
        )
        run_git_http11(["switch", "-c", branch, "FETCH_HEAD"], cwd=results_repo)
    else:
        run_git_http11(["switch", "-c", branch], cwd=results_repo)
    return environment


def commit_and_push_results(
    results_repo: Path,
    *,
    result_directory: Path,
    completed_epoch: int,
    branch: str,
    environment: dict[str, str],
) -> None:
    """Commit and push one result snapshot with HTTP/1.1 on every Git command."""
    relative = result_directory.relative_to(results_repo)
    run_git_http11(["add", "--", str(relative)], cwd=results_repo)
    changed = run_git_http11(
        ["diff", "--cached", "--quiet"], cwd=results_repo, check=False
    )
    if changed.returncode != 0:
        run_git_http11(
            ["commit", "-m", f"Update protected IBER-BE epoch {completed_epoch}"],
            cwd=results_repo,
        )
    for _ in range(5):
        pushed = run_git_http11(
            ["push", "origin", f"HEAD:{branch}"],
            cwd=results_repo,
            environment=environment,
            check=False,
        )
        if pushed.returncode == 0:
            return
        if not any(
            marker in pushed.stderr.lower()
            for marker in ("non-fast-forward", "fetch first", "[rejected]")
        ):
            pushed.check_returncode()
        run_git_http11(
            ["fetch", "origin", branch],
            cwd=results_repo,
            environment=environment,
        )
        run_git_http11(["rebase", "FETCH_HEAD"], cwd=results_repo)
    run_git_http11(
        ["push", "origin", f"HEAD:{branch}"],
        cwd=results_repo,
        environment=environment,
    )


def private_checkpoint_metadata(
    path: str | Path,
    *,
    identity: PublicationIdentity,
) -> PrivateCheckpointMetadata:
    """Validate one private-only, resumable IBER-BE screen checkpoint."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    artifact = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(artifact, Mapping):
        raise ValueError("IBER-BE private checkpoint is not a mapping")
    if "detector" in artifact or "model" in artifact or "ema" in artifact:
        raise ValueError("IBER-BE private checkpoint must not serialize the detector")
    expected = {"format_version": 1, **identity.as_dict()}
    for name, value in expected.items():
        actual = artifact.get(name)
        if isinstance(actual, str) and name.endswith("sha256"):
            actual = actual.upper()
        if name == "source_commit" and isinstance(actual, str):
            actual = actual.lower()
        if actual != value:
            raise ValueError(f"IBER-BE private checkpoint {name} identity mismatch")
    epoch = artifact.get("epoch")
    if type(epoch) is not int or not 1 <= epoch <= SCREEN_EPOCHS:
        raise ValueError(f"IBER-BE private checkpoint epoch must be in 1..{SCREEN_EPOCHS}")
    for name in ("refiner", "optimizer", "scaler", "rng"):
        if name not in artifact or artifact[name] is None:
            raise ValueError(f"IBER-BE private checkpoint {name} state is missing")
    if not isinstance(artifact["refiner"], Mapping) or not artifact["refiner"]:
        raise ValueError("IBER-BE private checkpoint refiner state is empty")
    if artifact.get("detector_sha_before") != artifact.get("detector_sha_after"):
        raise ValueError("IBER-BE private checkpoint detector fingerprint changed")
    return PrivateCheckpointMetadata(
        source=source,
        completed_epoch=epoch,
        bytes=source.stat().st_size,
        sha256=file_sha256(source).lower(),
    )


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _valid_verified_record(record: Mapping[str, Any]) -> None:
    if record.get("verified") is not True:
        raise ValueError("IBER-BE publication record is not remotely verified")
    if record.get("result_commit_verified") is not True:
        raise ValueError("IBER-BE result commit is not remotely verified")
    commit = record.get("result_commit_sha")
    if not isinstance(commit, str) or _GIT_SHA.fullmatch(commit) is None:
        raise ValueError("IBER-BE result commit SHA is invalid")
    checkpoint = record.get("checkpoint")
    remote = record.get("remote_verification")
    if not isinstance(checkpoint, Mapping) or not isinstance(remote, Mapping):
        raise ValueError("IBER-BE remote verification receipt is missing")
    for name in ("checkpoint", "manifest"):
        receipt = remote.get(name)
        if not isinstance(receipt, Mapping):
            raise ValueError(f"IBER-BE remote {name} verification receipt is missing")
        if type(receipt.get("bytes")) is not int or receipt["bytes"] < 0:
            raise ValueError(f"IBER-BE remote {name} byte receipt is invalid")
        if not isinstance(receipt.get("sha256"), str) or _HEX64.fullmatch(receipt["sha256"]) is None:
            raise ValueError(f"IBER-BE remote {name} SHA-256 receipt is invalid")
    if (
        checkpoint.get("bytes") != remote["checkpoint"].get("bytes")
        or str(checkpoint.get("sha256", "")).lower()
        != str(remote["checkpoint"].get("sha256", "")).lower()
    ):
        raise ValueError("IBER-BE checkpoint and remote verification receipt differ")


class PublicationLedger:
    """Append-only authority for remotely verified epochs 1 through 30."""

    def __init__(self, path: str | Path, identity: PublicationIdentity):
        self.path = Path(path)
        self.identity = identity

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("IBER-BE publication ledger row is not an object")
                rows.append(row)
        epochs = [row.get("completed_epoch") for row in rows]
        if epochs != list(range(1, len(rows) + 1)) or len(rows) > SCREEN_EPOCHS:
            raise ValueError(f"IBER-BE publication ledger is not contiguous in 1..30: {epochs}")
        expected_identity = self.identity.as_dict()
        for row in rows:
            _valid_verified_record(row)
            if any(row.get(name) != value for name, value in expected_identity.items()):
                raise ValueError("IBER-BE publication ledger identity mismatch")
        return rows

    def append_verified(self, record: dict[str, Any]) -> None:
        _valid_verified_record(record)
        expected_identity = self.identity.as_dict()
        if any(record.get(name) != value for name, value in expected_identity.items()):
            raise ValueError("IBER-BE publication record identity mismatch")
        rows = self.records()
        epoch = record.get("completed_epoch")
        if type(epoch) is not int or not 1 <= epoch <= SCREEN_EPOCHS:
            raise ValueError("IBER-BE completed epoch must be in 1..30")
        if epoch <= len(rows):
            if rows[epoch - 1] != record:
                raise ValueError(f"changed IBER-BE publication for completed epoch {epoch}")
            return
        if epoch != len(rows) + 1:
            raise ValueError(f"IBER-BE publication gap before completed epoch {epoch}")
        payload = b"".join(_canonical_json(row) for row in [*rows, record])
        _write_bytes_atomic(self.path, payload)

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
    """Return the exact unpublished contiguous checkpoint sequence."""
    last = ledger.last_completed_epoch
    pending: list[tuple[int, Path]] = []
    for path in Path(checkpoint_root).glob("epoch-*.pt"):
        metadata = private_checkpoint_metadata(path, identity=identity)
        expected_name = f"epoch-{metadata.completed_epoch:04d}.pt"
        if path.name != expected_name:
            raise ValueError(
                f"IBER-BE checkpoint filename must be {expected_name}, got {path.name}"
            )
        if metadata.completed_epoch > last:
            pending.append((metadata.completed_epoch, path))
    pending.sort(key=lambda item: item[0])
    if pending:
        expected = list(range(last + 1, pending[-1][0] + 1))
        actual = [epoch for epoch, _ in pending]
        if actual != expected:
            missing = next(epoch for epoch in expected if epoch not in actual)
            raise RuntimeError(f"missing completed epoch {missing} before IBER-BE publication")
    return pending


def _asset_name(prefix: str, epoch: int, suffix: str) -> str:
    return f"{prefix}-epoch-{epoch:04d}.{suffix}"


def assets_to_prune(
    assets: Iterable[Mapping[str, Any]],
    *,
    prefix: str = ASSET_PREFIX,
    retain: int = RETAINED_PAIRS,
) -> list[Mapping[str, Any]]:
    """Return assets belonging to complete pairs older than the rolling window."""
    if retain < 1:
        raise ValueError("IBER-BE retention must be positive")
    pattern = re.compile(rf"^{re.escape(prefix)}-epoch-(\d{{4}})\.(pt|json)$")
    pairs: dict[int, dict[str, Mapping[str, Any]]] = {}
    for asset in assets:
        match = pattern.fullmatch(str(asset.get("name", "")))
        if match:
            epoch = int(match.group(1))
            if 1 <= epoch <= SCREEN_EPOCHS:
                pairs.setdefault(epoch, {})[match.group(2)] = asset
    complete = sorted(epoch for epoch, pair in pairs.items() if {"pt", "json"} <= pair.keys())
    expired: list[Mapping[str, Any]] = []
    for epoch in complete[:-retain]:
        expired.extend((pairs[epoch]["pt"], pairs[epoch]["json"]))
    return expired


def verify_remote_asset(
    session: Any,
    asset: Mapping[str, Any],
    expected_path: str | Path,
) -> dict[str, Any]:
    """Download an asset and compare its exact bytes and SHA-256 to local authority."""
    expected = Path(expected_path)
    expected_bytes = expected.stat().st_size
    expected_sha = file_sha256(expected).lower()
    digest = hashlib.sha256()
    remote_bytes = 0
    with session.get(
        str(asset["url"]),
        headers={"Accept": "application/octet-stream"},
        stream=True,
        timeout=(30, 3600),
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                remote_bytes += len(chunk)
                digest.update(chunk)
    remote_sha = digest.hexdigest()
    if (
        remote_sha != expected_sha
        or remote_bytes != expected_bytes
        or int(asset.get("size", -1)) != expected_bytes
    ):
        raise RuntimeError("IBER-BE remote asset bytes/SHA-256 mismatch")
    return {"bytes": remote_bytes, "sha256": remote_sha}


def _refresh_release(session: Any, release: Mapping[str, Any]) -> dict[str, Any]:
    response = session.get(str(release["url"]), timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("IBER-BE release response is not an object")
    return payload


def _upload_verified(
    session: Any,
    release: Mapping[str, Any],
    *,
    path: Path,
    asset_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
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


def _collect_lightweight(
    run_dir: Path,
    destination: Path,
    record: Mapping[str, Any],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    excluded = {"checkpoints", "weights", "cache", ".git", "secrets"}
    for pattern in ("*.json", "*.jsonl", "*.csv", "*.yaml", "*.yml"):
        for source in run_dir.rglob(pattern):
            relative = source.relative_to(run_dir)
            if (
                excluded.intersection(relative.parts)
                or relative.as_posix() == "publication-ledger.jsonl"
                or not source.is_file()
            ):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    _write_bytes_atomic(destination / "latest.json", _canonical_json(record))


def _verify_result_branch(
    session: Any,
    *,
    repo: str,
    branch: str,
    expected_sha: str,
) -> None:
    response = session.get(
        f"https://api.github.com/repos/{repo}/commits/{branch}", timeout=30
    )
    response.raise_for_status()
    if response.json().get("sha") != expected_sha:
        raise RuntimeError("IBER-BE result branch verification failed")


def publish_exact_epoch(
    run_dir: str | Path,
    checkpoint: str | Path,
    config: PublicationConfig,
) -> dict[str, Any]:
    """Publish one epoch transaction and append the ledger only after verification."""
    run_root = Path(run_dir).resolve()
    ledger = PublicationLedger(run_root / "publication-ledger.jsonl", config.identity)
    metadata = private_checkpoint_metadata(checkpoint, identity=config.identity)
    expected_epoch = ledger.last_completed_epoch + 1
    if metadata.completed_epoch != expected_epoch:
        raise RuntimeError(
            f"expected IBER-BE completed epoch {expected_epoch}, got {metadata.completed_epoch}"
        )
    token = read_token_file(config.token_file)
    session = github_session(token)
    del token
    release = get_or_create_release(
        session,
        repo=config.repo,
        tag=DEFAULT_TAG,
        branch=config.source_branch,
        release_name=RELEASE_NAME,
        release_body=RELEASE_BODY,
    )
    checkpoint_name = _asset_name(ASSET_PREFIX, metadata.completed_epoch, "pt")
    checkpoint_asset, checkpoint_receipt = _upload_verified(
        session,
        release,
        path=metadata.source,
        asset_name=checkpoint_name,
    )
    base_manifest = {
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
        "remote_verification": {"checkpoint": checkpoint_receipt},
    }
    manifest_path = run_root / "checkpoints" / (
        f".{ASSET_PREFIX}-epoch-{metadata.completed_epoch:04d}.json"
    )
    try:
        environment = ensure_results_checkout(
            config.results_repo,
            repo_url=config.repo_url,
            branch=RESULTS_BRANCH,
            token_file=config.token_file,
        )
        result_directory = config.results_repo / "results" / config.run_name
        _collect_lightweight(run_root, result_directory, base_manifest)
        commit_and_push_results(
            config.results_repo,
            result_directory=result_directory,
            completed_epoch=metadata.completed_epoch,
            branch=RESULTS_BRANCH,
            environment=environment,
        )
        evidence_commit = run_git_http11(
            ["rev-parse", "HEAD"], cwd=config.results_repo
        ).stdout.strip()
        _verify_result_branch(
            session,
            repo=config.repo,
            branch=RESULTS_BRANCH,
            expected_sha=evidence_commit,
        )

        manifest = {
            **base_manifest,
            "result_commit_sha": evidence_commit,
            "result_commit_verified": True,
            "verified": True,
        }
        _write_bytes_atomic(manifest_path, _canonical_json(manifest))
        manifest_name = _asset_name(ASSET_PREFIX, metadata.completed_epoch, "json")
        manifest_asset, manifest_receipt = _upload_verified(
            session,
            release,
            path=manifest_path,
            asset_name=manifest_name,
        )
        refreshed = _refresh_release(session, release)
        remote_names = {str(asset.get("name")) for asset in refreshed.get("assets", [])}
        if {checkpoint_name, manifest_name} - remote_names:
            raise RuntimeError("IBER-BE remote checkpoint/manifest pair is incomplete")

        record = {
            **manifest,
            "manifest_asset_id": int(manifest_asset["id"]),
            "remote_verification": {
                "checkpoint": checkpoint_receipt,
                "manifest": manifest_receipt,
            },
        }
        _collect_lightweight(run_root, result_directory, record)
        remote_ledger = PublicationLedger(
            result_directory / "publication-ledger.jsonl", config.identity
        )
        remote_ledger.append_verified(record)
        commit_and_push_results(
            config.results_repo,
            result_directory=result_directory,
            completed_epoch=metadata.completed_epoch,
            branch=RESULTS_BRANCH,
            environment=environment,
        )
        result_record_commit = run_git_http11(
            ["rev-parse", "HEAD"], cwd=config.results_repo
        ).stdout.strip()
        _verify_result_branch(
            session,
            repo=config.repo,
            branch=RESULTS_BRANCH,
            expected_sha=result_record_commit,
        )
        for expired in assets_to_prune(
            _refresh_release(session, release).get("assets", []),
            prefix=ASSET_PREFIX,
            retain=RETAINED_PAIRS,
        ):
            response = session.delete(str(expired["url"]), timeout=30)
            response.raise_for_status()
        ledger.append_verified(record)
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
    """Retry publication without preserving or rendering exception credentials."""
    if type(attempts) is not int or attempts < 1:
        raise ValueError("IBER-BE publication attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            return publish_exact_epoch(run_dir, checkpoint, config)
        except Exception:
            if attempt < attempts:
                time.sleep(delay)
    raise RuntimeError(
        f"IBER-BE epoch publication failed after {attempts} attempts"
    ) from None
