from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests
import torch


DEFAULT_ASSET_PREFIX = "btdse-last"


@dataclass(frozen=True)
class CheckpointMetadata:
    source: Path
    completed_epoch: int
    bytes: int
    sha256: str


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_metadata(path: str | Path) -> CheckpointMetadata:
    checkpoint_path = Path(path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Not an Ultralytics checkpoint: {checkpoint_path}")

    raw_epoch = checkpoint.get("epoch")
    if not isinstance(raw_epoch, int) or raw_epoch < 0:
        raise ValueError(f"Checkpoint has no completed epoch: {checkpoint_path}")
    if checkpoint.get("optimizer") is None:
        raise ValueError(f"Checkpoint optimizer state was stripped: {checkpoint_path}")
    if checkpoint.get("ema") is None and checkpoint.get("model") is None:
        raise ValueError(f"Checkpoint model state is missing: {checkpoint_path}")

    return CheckpointMetadata(
        source=checkpoint_path,
        completed_epoch=raw_epoch + 1,
        bytes=checkpoint_path.stat().st_size,
        sha256=sha256_file(checkpoint_path),
    )


def checkpoint_asset_name(completed_epoch: int, *, prefix: str = DEFAULT_ASSET_PREFIX) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", prefix):
        raise ValueError(f"Invalid checkpoint asset prefix: {prefix!r}")
    return f"{prefix}-epoch-{completed_epoch:04d}.pt"


def _asset_epoch(asset: dict[str, Any], *, prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}-epoch-(\d+)\.pt$")
    match = pattern.match(str(asset.get("name", "")))
    return int(match.group(1)) if match else -1


def matching_checkpoint_assets(
    assets: Iterable[dict[str, Any]], *, prefix: str = DEFAULT_ASSET_PREFIX
) -> list[dict[str, Any]]:
    return sorted(
        (asset for asset in assets if _asset_epoch(asset, prefix=prefix) >= 0),
        key=lambda asset: _asset_epoch(asset, prefix=prefix),
    )


def assets_to_delete(
    assets: Iterable[dict[str, Any]], retain: int, *, prefix: str = DEFAULT_ASSET_PREFIX
) -> list[dict[str, Any]]:
    if retain < 1:
        raise ValueError("retain must be at least one")
    matched = matching_checkpoint_assets(assets, prefix=prefix)
    return matched[:-retain]


def build_manifest(
    metadata: CheckpointMetadata,
    *,
    asset: dict[str, Any],
    release_url: str,
) -> dict[str, Any]:
    return {
        "completed_epoch": metadata.completed_epoch,
        "release_url": release_url,
        "checkpoint": {
            "asset_id": int(asset["id"]),
            "asset_name": str(asset["name"]),
            "bytes": metadata.bytes,
            "sha256": metadata.sha256,
        },
    }


def github_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    return session


def _checked(response: requests.Response) -> requests.Response:
    if response.ok:
        return response
    raise RuntimeError(f"GitHub API {response.status_code}: {response.text[:500]}")


def get_or_create_release(
    session: requests.Session,
    *,
    repo: str,
    tag: str,
    branch: str,
    release_name: str = "BTD-SE V2.5-S RTX 4090 Live Checkpoints",
    release_body: str = (
        "Rolling resumable checkpoints for scratch RT-DETR-L with BTD-SE V2.5-S. "
        "The newest three validated epochs are retained."
    ),
) -> dict[str, Any]:
    api = f"https://api.github.com/repos/{repo}"
    response = session.get(f"{api}/releases/tags/{tag}", timeout=30)
    if response.status_code == 404:
        response = session.post(
            f"{api}/releases",
            json={
                "tag_name": tag,
                "target_commitish": branch,
                "name": release_name,
                "body": release_body,
            },
            timeout=30,
        )
    return _checked(response).json()


def upload_asset(
    session: requests.Session,
    *,
    release: dict[str, Any],
    path: Path,
    asset_name: str,
) -> dict[str, Any]:
    existing = next((item for item in release.get("assets", []) if item["name"] == asset_name), None)
    if existing and int(existing["size"]) == path.stat().st_size:
        return existing
    if existing:
        _checked(session.delete(str(existing["url"]), timeout=30))

    upload_url = str(release["upload_url"]).split("{")[0]
    headers = {
        "Content-Type": "application/octet-stream",
        "Content-Length": str(path.stat().st_size),
    }
    with path.open("rb") as file:
        response = session.post(
            upload_url,
            params={"name": asset_name},
            headers=headers,
            data=file,
            timeout=(30, 3600),
        )
    asset = _checked(response).json()
    if int(asset["size"]) != path.stat().st_size:
        raise RuntimeError(f"Uploaded asset size mismatch for {asset_name}")
    return asset


def publish_checkpoint(
    session: requests.Session,
    *,
    repo: str,
    tag: str,
    branch: str,
    checkpoint: str | Path,
    retain: int = 3,
    asset_prefix: str = DEFAULT_ASSET_PREFIX,
    release_name: str = "BTD-SE V2.5-S RTX 4090 Live Checkpoints",
    release_body: str = (
        "Rolling resumable checkpoints for scratch RT-DETR-L with BTD-SE V2.5-S. "
        "The newest three validated epochs are retained."
    ),
) -> dict[str, Any]:
    metadata = checkpoint_metadata(checkpoint)
    release = get_or_create_release(
        session,
        repo=repo,
        tag=tag,
        branch=branch,
        release_name=release_name,
        release_body=release_body,
    )
    asset = upload_asset(
        session,
        release=release,
        path=metadata.source,
        asset_name=checkpoint_asset_name(metadata.completed_epoch, prefix=asset_prefix),
    )

    release = _checked(session.get(str(release["url"]), timeout=30)).json()
    for expired in assets_to_delete(release.get("assets", []), retain=retain, prefix=asset_prefix):
        _checked(session.delete(str(expired["url"]), timeout=30))

    return build_manifest(metadata, asset=asset, release_url=str(release["html_url"]))
