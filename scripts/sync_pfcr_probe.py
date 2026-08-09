"""Publish every complete PFCR epoch and the final report through GitHub Releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.github_checkpoint_sync import (  # noqa: E402
    get_or_create_release,
    github_session,
    upload_asset,
)


BRANCH = "codex/fdr-yaml-module"
RESCUE_BUDGETS = (15, 30, 60)
POLL_INTERVAL = 30
_EPOCH_PATTERN = re.compile(r"^epoch-(\d{2})$")


@dataclass(frozen=True)
class EpochEvidence:
    epoch: int
    checkpoint: Path
    metrics: Path
    checkpoint_sha256: str
    metrics_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON evidence is not a regular file: {path}")
    payload = json.loads(path.read_text("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence is not an object: {path}")
    return payload


def validate_epoch_pair(checkpoint: Path, metrics: Path) -> EpochEvidence:
    checkpoint = Path(checkpoint)
    metrics = Path(metrics)
    checkpoint_match = _EPOCH_PATTERN.fullmatch(checkpoint.stem)
    metrics_match = _EPOCH_PATTERN.fullmatch(metrics.stem)
    if checkpoint_match is None or metrics_match is None:
        raise ValueError("PFCR epoch evidence filename mismatch")
    epoch = int(checkpoint_match.group(1))
    if int(metrics_match.group(1)) != epoch or epoch < 1 or epoch > 20:
        raise ValueError("PFCR epoch evidence identity mismatch")
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("PFCR checkpoint is not a regular file")
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or artifact.get("epoch") != epoch:
        raise ValueError("PFCR checkpoint epoch mismatch")
    payload = _load_json(metrics)
    rows = payload.get("rows")
    if payload.get("epoch") != epoch or not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("PFCR metric epoch schema mismatch")
    budgets = tuple(row.get("slots") for row in rows if isinstance(row, dict))
    if budgets != RESCUE_BUDGETS:
        raise ValueError("PFCR metric rescue budget mismatch")
    if any(row.get("epoch") != epoch or row.get("split") != "dev" for row in rows):
        raise ValueError("PFCR metric row identity mismatch")
    return EpochEvidence(
        epoch=epoch,
        checkpoint=checkpoint,
        metrics=metrics,
        checkpoint_sha256=_sha256(checkpoint),
        metrics_sha256=_sha256(metrics),
    )


def discover_complete_epochs(run_root: Path) -> tuple[EpochEvidence, ...]:
    root = Path(run_root)
    checkpoints = root / "checkpoints"
    metrics = root / "metrics"
    if not checkpoints.is_dir() or not metrics.is_dir():
        return ()
    result: list[EpochEvidence] = []
    for epoch in range(1, 21):
        checkpoint = checkpoints / f"epoch-{epoch:02d}.pt"
        metric = metrics / f"epoch-{epoch:02d}.json"
        if not checkpoint.exists() or not metric.exists():
            break
        result.append(validate_epoch_pair(checkpoint, metric))
    return tuple(result)


def write_status(path: Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def _load_status(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        return {"state": "waiting", "published_epochs": [], "report_uploaded": False}
    payload = _load_json(Path(path))
    published = payload.get("published_epochs")
    if not isinstance(published, list) or any(type(value) is not int for value in published):
        raise ValueError("PFCR publication status is invalid")
    return payload


def _token(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("GitHub token file is missing")
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError("GitHub token file must have mode 600")
    value = path.read_text("utf-8").strip()
    if not value:
        raise ValueError("GitHub token file is empty")
    return value


def _manifest_file(item: EpochEvidence, directory: Path) -> Path:
    path = directory / f"pfcr-epoch-{item.epoch:02d}-manifest.json"
    write_status(
        path,
        {
            "format_version": 1,
            "epoch": item.epoch,
            "checkpoint": {
                "asset": f"pfcr-epoch-{item.epoch:02d}.pt",
                "bytes": item.checkpoint.stat().st_size,
                "sha256": item.checkpoint_sha256,
            },
            "metrics": {
                "asset": f"pfcr-epoch-{item.epoch:02d}.json",
                "bytes": item.metrics.stat().st_size,
                "sha256": item.metrics_sha256,
            },
        },
    )
    return path


def _upload_epoch(session, release: dict[str, Any], item: EpochEvidence) -> None:
    upload_asset(
        session,
        release=release,
        path=item.checkpoint,
        asset_name=f"pfcr-epoch-{item.epoch:02d}.pt",
    )
    upload_asset(
        session,
        release=release,
        path=item.metrics,
        asset_name=f"pfcr-epoch-{item.epoch:02d}.json",
    )
    with tempfile.TemporaryDirectory(prefix="pfcr-publish-") as temporary:
        manifest = _manifest_file(item, Path(temporary))
        upload_asset(
            session,
            release=release,
            path=manifest,
            asset_name=manifest.name,
        )


def _upload_report(session, release: dict[str, Any], report_root: Path) -> list[str]:
    root = Path(report_root)
    if not (root / "SHA256SUMS.txt").is_file():
        return []
    uploaded: list[str] = []
    for path in sorted(root.iterdir(), key=lambda value: value.name):
        if path.is_symlink() or not path.is_file():
            raise ValueError("PFCR report root contains a non-regular entry")
        upload_asset(session, release=release, path=path, asset_name=f"report-{path.name}")
        uploaded.append(path.name)
    return uploaded


def sync_once(args: argparse.Namespace) -> dict[str, Any]:
    status = _load_status(args.status_file)
    published = set(int(value) for value in status.get("published_epochs", []))
    session = github_session(_token(args.token_file))
    release = get_or_create_release(
        session,
        repo=args.repo,
        tag=args.tag,
        branch=BRANCH,
        release_name="PFCR v1 Live Learnability Evidence",
        release_body=(
            "Create-only epoch checkpoints, internal metrics, and final one-shot PFCR "
            "learnability decision. This is design-selection evidence, not a final detector."
        ),
    )
    for item in discover_complete_epochs(args.run_root):
        if item.epoch in published:
            continue
        _upload_epoch(session, release, item)
        published.add(item.epoch)
        status = {
            "state": "running",
            "published_epochs": sorted(published),
            "report_uploaded": False,
            "release_url": str(release["html_url"]),
        }
        write_status(args.status_file, status)
    report_files = _upload_report(session, release, args.report_root)
    if report_files:
        status = {
            "state": "complete",
            "published_epochs": sorted(published),
            "report_uploaded": True,
            "report_files": report_files,
            "release_url": str(release["html_url"]),
        }
        write_status(args.status_file, status)
    elif not published:
        status = {
            "state": "waiting",
            "published_epochs": [],
            "report_uploaded": False,
            "release_url": str(release["html_url"]),
        }
        write_status(args.status_file, status)
    return status


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL)
    args = parser.parse_args(argv)
    if args.interval < 10:
        parser.error("--interval must be at least 10 seconds")
    args.branch = BRANCH
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    while True:
        try:
            status = sync_once(args)
            print(json.dumps(status, sort_keys=True), flush=True)
            if status.get("state") == "complete":
                return 0
        except Exception as error:
            write_status(
                args.status_file,
                {
                    "state": "retrying",
                    "published_epochs": _load_status(args.status_file).get(
                        "published_epochs", []
                    ),
                    "report_uploaded": False,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            print(f"PFCR publication retry: {type(error).__name__}: {error}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
