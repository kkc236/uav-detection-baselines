"""Restore the newest verified IBER-BE v1.0 seed0 B3 screen checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.github_checkpoint_sync import github_session  # noqa: E402
from src.iber_protocol import (  # noqa: E402
    DESIGN_VERSION,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    PROTOCOL_SHA256,
    RUNTIME_AMENDMENT_SHA256,
    SCREEN_EPOCHS,
    file_sha256,
)
from src.iber_publication import (  # noqa: E402
    ASSET_PREFIX,
    DEFAULT_TAG,
    PublicationIdentity,
    private_checkpoint_metadata,
    read_token_file,
)


def select_latest_pair(
    assets: Iterable[dict[str, Any]],
    *,
    prefix: str = ASSET_PREFIX,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Select the highest complete pair in the fixed 1..30 screen namespace."""
    if prefix != ASSET_PREFIX:
        raise ValueError(f"IBER-BE asset prefix must be exactly {ASSET_PREFIX!r}")
    pattern = re.compile(rf"^{re.escape(prefix)}-epoch-(\d{{4}})\.(pt|json)$")
    pairs: dict[int, dict[str, dict[str, Any]]] = {}
    for asset in assets:
        match = pattern.fullmatch(str(asset.get("name", "")))
        if match:
            epoch = int(match.group(1))
            pairs.setdefault(epoch, {})[match.group(2)] = asset
    complete = [(epoch, pair) for epoch, pair in pairs.items() if {"pt", "json"} <= pair.keys()]
    invalid = sorted(epoch for epoch, _ in complete if not 1 <= epoch <= SCREEN_EPOCHS)
    if invalid:
        raise ValueError(f"IBER-BE restore epochs must be in 1..30, got {invalid}")
    if not complete:
        raise FileNotFoundError(f"no complete IBER-BE pair with prefix {prefix!r}")
    epoch, pair = max(complete, key=lambda item: item[0])
    return pair["pt"], pair["json"], epoch


def _download_asset(session: Any, asset: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    try:
        with session.get(
            str(asset["url"]),
            headers={"Accept": "application/octet-stream"},
            stream=True,
            timeout=(30, 3600),
        ) as response:
            response.raise_for_status()
            with destination.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        stream.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        if written != int(asset.get("size", -1)):
            raise RuntimeError("downloaded IBER-BE asset byte count mismatch")
        remote_digest = asset.get("digest")
        if isinstance(remote_digest, str) and remote_digest.startswith("sha256:"):
            if digest.hexdigest() != remote_digest.removeprefix("sha256:").lower():
                raise RuntimeError("downloaded IBER-BE asset SHA-256 mismatch")
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def verify_downloaded_checkpoint(
    path: str | Path,
    manifest: Mapping[str, Any],
    *,
    expected_epoch: int,
    identity: PublicationIdentity,
    expected_prefix: str = ASSET_PREFIX,
):
    """Verify all scientific, resumability, remote, byte, and commit authority."""
    if expected_prefix != ASSET_PREFIX:
        raise RuntimeError("downloaded IBER-BE asset prefix identity mismatch")
    expected = {"format_version": 1, **identity.as_dict()}
    for name, value in expected.items():
        actual = manifest.get(name)
        if isinstance(actual, str) and name.endswith("sha256"):
            actual = actual.upper()
        if name == "source_commit" and isinstance(actual, str):
            actual = actual.lower()
        if actual != value:
            raise RuntimeError(
                f"downloaded IBER-BE {name.replace('_', ' ')} identity mismatch"
            )
    if manifest.get("verified") is not True:
        raise RuntimeError("downloaded IBER-BE verified receipt is missing")
    if manifest.get("result_commit_verified") is not True:
        raise RuntimeError("downloaded IBER-BE result commit verified receipt is missing")
    result_commit = manifest.get("result_commit_sha")
    if not isinstance(result_commit, str) or re.fullmatch(r"[0-9A-Fa-f]{40}(?:[0-9A-Fa-f]{24})?", result_commit) is None:
        raise RuntimeError("downloaded IBER-BE result commit SHA is invalid")
    if type(expected_epoch) is not int or not 1 <= expected_epoch <= SCREEN_EPOCHS:
        raise RuntimeError("downloaded IBER-BE checkpoint epoch must be in 1..30")
    if manifest.get("completed_epoch") != expected_epoch:
        raise RuntimeError("downloaded IBER-BE checkpoint epoch mismatch")

    metadata = private_checkpoint_metadata(path, identity=identity)
    checkpoint = manifest.get("checkpoint")
    remote = manifest.get("remote_verification")
    if not isinstance(checkpoint, Mapping) or not isinstance(remote, Mapping):
        raise RuntimeError("downloaded IBER-BE checkpoint verification receipt is missing")
    remote_checkpoint = remote.get("checkpoint")
    if not isinstance(remote_checkpoint, Mapping):
        raise RuntimeError("downloaded IBER-BE remote checkpoint receipt is missing")
    if metadata.completed_epoch != expected_epoch:
        raise RuntimeError("downloaded IBER-BE checkpoint epoch mismatch")
    if metadata.bytes != checkpoint.get("bytes") or metadata.bytes != remote_checkpoint.get("bytes"):
        raise RuntimeError("downloaded IBER-BE checkpoint bytes mismatch")
    expected_sha = str(checkpoint.get("sha256", "")).lower()
    remote_sha = str(remote_checkpoint.get("sha256", "")).lower()
    if metadata.sha256 != expected_sha or metadata.sha256 != remote_sha:
        raise RuntimeError("downloaded IBER-BE checkpoint SHA-256 mismatch")
    if checkpoint.get("asset_name") != f"{expected_prefix}-epoch-{expected_epoch:04d}.pt":
        raise RuntimeError("downloaded IBER-BE checkpoint asset prefix mismatch")
    return metadata


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def install_verified_checkpoint(
    downloaded: str | Path,
    destination: str | Path,
    *,
    expected_sha256: str,
) -> Path:
    """Atomically install verified bytes without replacing divergent local state."""
    source = Path(downloaded)
    target = Path(destination).resolve()
    try:
        actual_sha = file_sha256(source).lower()
        if actual_sha != str(expected_sha256).lower():
            raise RuntimeError("downloaded IBER-BE checkpoint SHA-256 mismatch before install")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_sha256(target).lower() != actual_sha:
                raise FileExistsError(
                    f"refusing to replace changed IBER-BE checkpoint: {target}"
                )
            return target
        # Windows FlushFileBuffers requires a writable descriptor.
        with source.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(source, target)
        _fsync_parent(target.parent)
        return target
    finally:
        source.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--repo", default="kkc236/uav-detection-baselines")
    parser.add_argument("--category-sha256", required=True)
    parser.add_argument("--gate1-decision-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


def restore_latest_checkpoint(args: argparse.Namespace) -> Path:
    identity = PublicationIdentity(
        design_version=DESIGN_VERSION,
        stage="screen",
        probe="b3",
        seed=0,
        baseline_sha256=EXPECTED_BASELINE_SHA256,
        dataset_sha256=EXPECTED_DATASET_SHA256,
        subset_sha256=EXPECTED_SUBSET_SHA256,
        category_sha256=args.category_sha256,
        protocol_sha256=PROTOCOL_SHA256,
        runtime_amendment_sha256=RUNTIME_AMENDMENT_SHA256,
        gate1_decision_sha256=args.gate1_decision_sha256,
        source_commit=args.source_commit,
    )
    session = github_session(read_token_file(args.token_file))
    response = session.get(
        f"https://api.github.com/repos/{args.repo}/releases/tags/{DEFAULT_TAG}",
        timeout=30,
    )
    response.raise_for_status()
    checkpoint_asset, manifest_asset, epoch = select_latest_pair(
        response.json().get("assets", []), prefix=ASSET_PREFIX
    )
    checkpoint_root = args.run_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    temporary_manifest = checkpoint_root / ".restore-iber-manifest.json.tmp"
    temporary_checkpoint = checkpoint_root / ".restore-iber-checkpoint.pt.tmp"
    try:
        _download_asset(session, manifest_asset, temporary_manifest)
        manifest = json.loads(temporary_manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise RuntimeError("downloaded IBER-BE manifest is not an object")
        _download_asset(session, checkpoint_asset, temporary_checkpoint)
        metadata = verify_downloaded_checkpoint(
            temporary_checkpoint,
            manifest,
            expected_epoch=epoch,
            identity=identity,
            expected_prefix=ASSET_PREFIX,
        )
        destination = checkpoint_root / f"epoch-{epoch:04d}.pt"
        installed = install_verified_checkpoint(
            temporary_checkpoint,
            destination,
            expected_sha256=metadata.sha256,
        )
        print(
            json.dumps(
                {
                    "checkpoint": str(installed),
                    "completed_epoch": epoch,
                    "sha256": metadata.sha256,
                    "design_version": DESIGN_VERSION,
                    "stage": "screen",
                    "probe": "b3",
                    "seed": 0,
                },
                sort_keys=True,
            )
        )
        return installed
    finally:
        temporary_manifest.unlink(missing_ok=True)
        temporary_checkpoint.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    args.token_file = args.token_file.resolve()
    restore_latest_checkpoint(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
