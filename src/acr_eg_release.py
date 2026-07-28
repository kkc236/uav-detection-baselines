"""Fail-closed GitHub Release publishing for formal ACR-EG checkpoints."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
import torch

from src.github_checkpoint_sync import (
    get_or_create_release,
    sha256_file,
    upload_asset,
)


ACR_EG_KEY_PREFIX = "acr_eg."
ACR_EG_KEY_COUNT = 48
FIXED_AMP_SCALE = 128.0
FIXED_AMP_GROWTH_INTERVAL = 2**31 - 1


@dataclass(frozen=True)
class ACREGCheckpointMetadata:
    source: Path
    checkpoint_epoch: int
    completed_epoch: int
    model_type: str
    state_key_count: int
    acr_eg_key_count: int
    optimizer_state_entries: int
    scaler_scale: float
    scaler_growth_interval: int
    updates: int
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ReleaseCoordinates:
    tag: str
    asset_name: str
    evidence_name: str


def _is_acr_eg_model(model: object) -> bool:
    from src.rtdetr_acr_eg import ACREGDetectionModel

    return isinstance(model, ACREGDetectionModel)


def _require_integral(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"ACR_EG_RELEASE_{name}_INVALID")
    result = int(value)
    if result < minimum:
        raise ValueError(f"ACR_EG_RELEASE_{name}_INVALID")
    return result


def inspect_acr_eg_checkpoint(
    path: str | Path,
    *,
    expected_completed_epoch: int | None = None,
) -> ACREGCheckpointMetadata:
    """Prove model identity and resumable training state before publication."""

    checkpoint_path = Path(path).resolve()
    before = checkpoint_path.stat()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("ACR_EG_RELEASE_CHECKPOINT_NOT_MAPPING")
    for key in ("ema", "optimizer", "scaler", "epoch", "updates"):
        if payload.get(key) is None:
            raise ValueError(f"ACR_EG_RELEASE_MISSING_{key.upper()}")

    model = payload["ema"]
    if not _is_acr_eg_model(model):
        raise ValueError("ACR_EG_RELEASE_MODEL_IDENTITY_MISMATCH")
    state = model.state_dict()
    acr_keys = [key for key in state if key.startswith(ACR_EG_KEY_PREFIX)]
    if len(acr_keys) != ACR_EG_KEY_COUNT:
        raise ValueError("ACR_EG_RELEASE_STATE_IDENTITY_MISMATCH")

    checkpoint_epoch = _require_integral("EPOCH", payload["epoch"])
    completed_epoch = checkpoint_epoch + 1
    if not 1 <= completed_epoch <= 100:
        raise ValueError("ACR_EG_RELEASE_EPOCH_INVALID")
    if (
        expected_completed_epoch is not None
        and completed_epoch != expected_completed_epoch
    ):
        raise ValueError("ACR_EG_RELEASE_EPOCH_MISMATCH")

    optimizer = payload["optimizer"]
    optimizer_state = optimizer.get("state") if isinstance(optimizer, dict) else None
    if not isinstance(optimizer_state, dict) or not optimizer_state:
        raise ValueError("ACR_EG_RELEASE_OPTIMIZER_STATE_EMPTY")

    scaler = payload["scaler"]
    if not isinstance(scaler, dict):
        raise ValueError("ACR_EG_RELEASE_SCALER_INVALID")
    scaler_scale = float(scaler.get("scale", 0.0))
    if scaler_scale != FIXED_AMP_SCALE:
        raise ValueError("ACR_EG_RELEASE_SCALER_SCALE_MISMATCH")
    scaler_growth_interval = int(scaler.get("growth_interval", -1))
    if scaler_growth_interval != FIXED_AMP_GROWTH_INTERVAL:
        raise ValueError("ACR_EG_RELEASE_SCALER_GROWTH_INTERVAL_MISMATCH")
    updates = _require_integral("UPDATES", payload["updates"])

    digest = sha256_file(checkpoint_path)
    after = checkpoint_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("ACR_EG_RELEASE_CHECKPOINT_CHANGED_DURING_INSPECTION")

    return ACREGCheckpointMetadata(
        source=checkpoint_path,
        checkpoint_epoch=checkpoint_epoch,
        completed_epoch=completed_epoch,
        model_type=type(model).__name__,
        state_key_count=len(state),
        acr_eg_key_count=len(acr_keys),
        optimizer_state_entries=len(optimizer_state),
        scaler_scale=scaler_scale,
        scaler_growth_interval=scaler_growth_interval,
        updates=updates,
        bytes=after.st_size,
        sha256=digest,
    )


def release_coordinates(source_commit: str, *, completed_epoch: int) -> ReleaseCoordinates:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("ACR_EG_RELEASE_SOURCE_COMMIT_INVALID")
    if not 1 <= completed_epoch <= 100:
        raise ValueError("ACR_EG_RELEASE_EPOCH_INVALID")
    short_commit = source_commit[:8]
    return ReleaseCoordinates(
        tag=f"gcte-acr-eg-{short_commit}-epoch-{completed_epoch:03d}",
        asset_name=f"epoch{completed_epoch - 1}.pt",
        evidence_name=f"epoch-{completed_epoch:03d}.json",
    )


def _downloaded_asset_digest(session: requests.Session, asset: dict[str, Any]) -> tuple[int, str]:
    response = session.get(
        str(asset["url"]),
        headers={"Accept": "application/octet-stream"},
        stream=True,
        allow_redirects=True,
        timeout=(30, 3600),
    )
    response.raise_for_status()
    digest = hashlib.sha256()
    size = 0
    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
        if chunk:
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def verify_remote_asset(
    session: requests.Session | None,
    *,
    asset: dict[str, Any],
    metadata: ACREGCheckpointMetadata,
) -> str:
    """Verify GitHub bytes and SHA256, downloading only when API digest is absent."""

    if int(asset.get("size", -1)) != metadata.bytes:
        raise RuntimeError("ACR_EG_RELEASE_REMOTE_SIZE_MISMATCH")
    api_digest = asset.get("digest")
    if isinstance(api_digest, str) and api_digest:
        expected = f"sha256:{metadata.sha256}"
        if api_digest.lower() != expected:
            raise RuntimeError("ACR_EG_RELEASE_REMOTE_DIGEST_MISMATCH")
        return api_digest.lower()
    if session is None:
        raise RuntimeError("ACR_EG_RELEASE_REMOTE_DIGEST_UNAVAILABLE")
    downloaded_size, downloaded_sha256 = _downloaded_asset_digest(session, asset)
    if downloaded_size != metadata.bytes:
        raise RuntimeError("ACR_EG_RELEASE_REMOTE_SIZE_MISMATCH")
    if downloaded_sha256 != metadata.sha256:
        raise RuntimeError("ACR_EG_RELEASE_REMOTE_DIGEST_MISMATCH")
    return f"sha256:{downloaded_sha256}"


def _delete_asset(session: requests.Session, asset: dict[str, Any]) -> None:
    response = session.delete(str(asset["url"]), timeout=30)
    response.raise_for_status()


def verify_release_target(
    session: requests.Session,
    *,
    repo: str,
    tag: str,
    source_commit: str,
) -> str:
    """Resolve lightweight or annotated tags and require the exact source commit."""

    api = f"https://api.github.com/repos/{repo}"
    response = session.get(
        f"{api}/git/ref/tags/{quote(tag, safe='')}",
        timeout=30,
    )
    response.raise_for_status()
    reference = response.json().get("object")
    for _ in range(5):
        if not isinstance(reference, dict):
            raise RuntimeError("ACR_EG_RELEASE_TARGET_INVALID")
        object_type = reference.get("type")
        object_sha = str(reference.get("sha", ""))
        if object_type == "commit":
            if object_sha != source_commit:
                raise RuntimeError("ACR_EG_RELEASE_TARGET_MISMATCH")
            return object_sha
        if object_type != "tag" or not re.fullmatch(r"[0-9a-f]{40}", object_sha):
            raise RuntimeError("ACR_EG_RELEASE_TARGET_INVALID")
        response = session.get(f"{api}/git/tags/{object_sha}", timeout=30)
        response.raise_for_status()
        reference = response.json().get("object")
    raise RuntimeError("ACR_EG_RELEASE_TARGET_DEPTH_EXCEEDED")


def publish_acr_eg_checkpoint(
    session: requests.Session,
    *,
    repo: str,
    source_commit: str,
    checkpoint: str | Path,
    expected_completed_epoch: int,
) -> dict[str, Any]:
    """Publish one immutable epoch Release and return verified evidence."""

    metadata = inspect_acr_eg_checkpoint(
        checkpoint,
        expected_completed_epoch=expected_completed_epoch,
    )
    coordinates = release_coordinates(
        source_commit,
        completed_epoch=metadata.completed_epoch,
    )
    release = get_or_create_release(
        session,
        repo=repo,
        tag=coordinates.tag,
        branch=source_commit,
        release_name=f"ACR-EG integrated checkpoint epoch {metadata.completed_epoch}",
        release_body=(
            "Verified resumable ACREGDetectionModel checkpoint from the formal "
            f"100-epoch continuation at source commit {source_commit}."
        ),
    )
    if str(release.get("tag_name")) != coordinates.tag:
        raise RuntimeError("ACR_EG_RELEASE_TAG_MISMATCH")
    resolved_commit = verify_release_target(
        session,
        repo=repo,
        tag=coordinates.tag,
        source_commit=source_commit,
    )

    existing = next(
        (
            item
            for item in release.get("assets", [])
            if item.get("name") == coordinates.asset_name
        ),
        None,
    )
    asset = existing
    if existing is not None:
        try:
            verify_remote_asset(session, asset=existing, metadata=metadata)
        except RuntimeError:
            _delete_asset(session, existing)
            asset = None
            release = {
                **release,
                "assets": [
                    item for item in release.get("assets", []) if item is not existing
                ],
            }
    if asset is None:
        asset = upload_asset(
            session,
            release=release,
            path=metadata.source,
            asset_name=coordinates.asset_name,
        )
    remote_digest = verify_remote_asset(session, asset=asset, metadata=metadata)

    metadata_payload = asdict(metadata)
    metadata_payload["source"] = str(metadata.source)
    return {
        "schema_version": 1,
        "state": "published_and_verified",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "source_short_commit": source_commit[:8],
        "model": {
            "type": metadata.model_type,
            "state_key_count": metadata.state_key_count,
            "acr_eg_key_count": metadata.acr_eg_key_count,
        },
        "continuity": {
            "checkpoint_epoch": metadata.checkpoint_epoch,
            "completed_epoch": metadata.completed_epoch,
            "optimizer_state_entries": metadata.optimizer_state_entries,
            "scaler_scale": metadata.scaler_scale,
            "scaler_growth_interval": metadata.scaler_growth_interval,
            "updates": metadata.updates,
        },
        "checkpoint": metadata_payload,
        "release": {
            "tag": coordinates.tag,
            "url": str(release["html_url"]),
            "target_commit": resolved_commit,
            "asset_id": int(asset["id"]),
            "asset_name": str(asset["name"]),
            "bytes": int(asset["size"]),
            "digest": remote_digest,
        },
    }


__all__ = [
    "ACREGCheckpointMetadata",
    "ReleaseCoordinates",
    "inspect_acr_eg_checkpoint",
    "publish_acr_eg_checkpoint",
    "release_coordinates",
    "verify_release_target",
    "verify_remote_asset",
]
