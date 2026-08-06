"""Publish immutable SCADS/FDR epoch checkpoints from a local JSONL outbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ASSET_TOKEN = re.compile(r"^[A-Za-z0-9._-]+$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {number} is not an object: {source}")
        rows.append(value)
    return rows


def append_ledger(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    key = (str(record["run_id"]), int(record["completed_epoch"]))
    matches = [
        row
        for row in read_jsonl(destination)
        if (str(row.get("run_id")), int(row.get("completed_epoch", -1))) == key
    ]
    if matches:
        if len(matches) != 1:
            raise ValueError(f"duplicate publication ledger key: {key}")
        immutable = {
            name: record[name]
            for name in (
                "run_id",
                "variant",
                "stage",
                "completed_epoch",
                "checkpoint_sha256",
                "checkpoint_size",
                "checkpoint_asset",
                "sidecar_asset",
                "release_url",
            )
        }
        if any(matches[0].get(name) != value for name, value in immutable.items()):
            raise ValueError(f"changed publication ledger entry: {key}")
        return matches[0]
    encoded = json.dumps(
        dict(record), sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"
    with destination.open("a", encoding="utf-8", newline="") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(record)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def asset_names(record: Mapping[str, Any]) -> tuple[str, str]:
    run_id = str(record.get("run_id", ""))
    if not ASSET_TOKEN.fullmatch(run_id):
        raise ValueError(f"run_id is not safe for a release asset: {run_id!r}")
    epoch = int(record.get("completed_epoch", 0))
    if epoch < 1:
        raise ValueError("completed_epoch must be positive")
    stem = f"{run_id}-epoch-{epoch:04d}"
    return f"{stem}.pt", f"{stem}.json"


def validate_queue_record(record: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "run_id",
        "variant",
        "stage",
        "completed_epoch",
        "checkpoint",
        "checkpoint_sha256",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"publication queue row is missing: {sorted(missing)}")
    if record.get("status") != "pending":
        raise ValueError("publication queue status must remain pending")
    if record["variant"] not in {"fdr", "scads"}:
        raise ValueError(f"unknown publication variant: {record['variant']!r}")
    if record["stage"] not in {"screen", "formal"}:
        raise ValueError(f"unknown publication stage: {record['stage']!r}")
    checkpoint = Path(str(record["checkpoint"])).resolve()
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError(f"queued checkpoint is unavailable: {checkpoint}")
    expected_size = int(record.get("checkpoint_size", checkpoint.stat().st_size))
    if checkpoint.stat().st_size != expected_size:
        raise ValueError(f"queued checkpoint size changed: {checkpoint}")
    expected_sha = str(record["checkpoint_sha256"]).upper()
    if not re.fullmatch(r"[0-9A-F]{64}", expected_sha):
        raise ValueError("queued checkpoint SHA256 is invalid")
    actual_sha = file_sha256(checkpoint)
    if actual_sha != expected_sha:
        raise ValueError(f"queued checkpoint SHA256 changed: {checkpoint}")
    return {
        **dict(record),
        "checkpoint": str(checkpoint),
        "checkpoint_size": expected_size,
        "checkpoint_sha256": expected_sha,
    }


def build_sidecar(record: Mapping[str, Any], checkpoint_asset: str) -> dict[str, Any]:
    artifacts = []
    for raw in record.get("artifacts", []):
        path = Path(str(raw)).resolve()
        if path.is_file() and not path.is_symlink():
            artifacts.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return {
        "format_version": 1,
        "run_id": str(record["run_id"]),
        "variant": str(record["variant"]),
        "stage": str(record["stage"]),
        "completed_epoch": int(record["completed_epoch"]),
        "checkpoint": {
            "asset": checkpoint_asset,
            "bytes": int(record["checkpoint_size"]),
            "sha256": str(record["checkpoint_sha256"]).upper(),
        },
        "evidence_artifacts": artifacts,
    }


class GitHubReleaseClient:
    def __init__(self, *, repo: str, tag: str, target: str) -> None:
        self.repo = repo
        self.tag = tag
        self.target = target

    @staticmethod
    def _run(command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(command),
            check=check,
            capture_output=True,
            text=True,
            timeout=3600,
        )

    def require_auth(self) -> None:
        self._run(["gh", "auth", "status", "--hostname", "github.com"])

    def _view(self, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return self._run(
            [
                "gh",
                "release",
                "view",
                self.tag,
                "--repo",
                self.repo,
                "--json",
                "url,tagName,assets",
            ],
            check=check,
        )

    def ensure_release(self) -> dict[str, Any]:
        viewed = self._view(check=False)
        if viewed.returncode != 0:
            self._run(
                [
                    "gh",
                    "release",
                    "create",
                    self.tag,
                    "--repo",
                    self.repo,
                    "--target",
                    self.target,
                    "--title",
                    "SCADS/FDR paired epoch checkpoints",
                    "--notes",
                    "Immutable paired FDR versus SCADS checkpoints and SHA256 sidecars.",
                ]
            )
            viewed = self._view()
        payload = json.loads(viewed.stdout)
        if payload.get("tagName") != self.tag:
            raise ValueError("GitHub release tag differs from requested tag")
        return payload

    def assets(self) -> tuple[str, dict[str, dict[str, Any]]]:
        payload = json.loads(self._view().stdout)
        return str(payload["url"]), {
            str(asset["name"]): asset for asset in payload.get("assets", [])
        }

    def upload(self, path: Path, asset_name: str) -> None:
        self._run(
            [
                "gh",
                "release",
                "upload",
                self.tag,
                f"{Path(path).resolve()}#{asset_name}",
                "--repo",
                self.repo,
            ]
        )

    def download(self, asset_name: str, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=False)
        self._run(
            [
                "gh",
                "release",
                "download",
                self.tag,
                "--repo",
                self.repo,
                "--pattern",
                asset_name,
                "--dir",
                str(destination),
            ]
        )
        result = destination / asset_name
        if not result.is_file():
            raise FileNotFoundError(f"downloaded sidecar is missing: {result}")
        return result


def _ensure_asset(
    client: GitHubReleaseClient,
    assets: Mapping[str, Mapping[str, Any]],
    *,
    path: Path,
    name: str,
) -> None:
    existing = assets.get(name)
    if existing is not None:
        if int(existing.get("size", -1)) != path.stat().st_size:
            raise ValueError(f"immutable remote asset size conflict: {name}")
        return
    client.upload(path, name)


def publish_record(
    record: Mapping[str, Any],
    *,
    client: GitHubReleaseClient,
    ledger_path: Path,
) -> dict[str, Any]:
    checked = validate_queue_record(record)
    checkpoint = Path(checked["checkpoint"])
    checkpoint_asset, sidecar_asset = asset_names(checked)
    sidecar = build_sidecar(checked, checkpoint_asset)
    client.ensure_release()
    _, assets = client.assets()
    _ensure_asset(client, assets, path=checkpoint, name=checkpoint_asset)

    with tempfile.TemporaryDirectory(prefix="scads-sidecar-") as temporary:
        sidecar_path = Path(temporary) / sidecar_asset
        write_json_atomic(sidecar_path, sidecar)
        _, assets = client.assets()
        _ensure_asset(client, assets, path=sidecar_path, name=sidecar_asset)
        release_url, refreshed = client.assets()
        if int(refreshed.get(checkpoint_asset, {}).get("size", -1)) != checkpoint.stat().st_size:
            raise RuntimeError(f"remote checkpoint size verification failed: {checkpoint_asset}")
        if int(refreshed.get(sidecar_asset, {}).get("size", -1)) != sidecar_path.stat().st_size:
            raise RuntimeError(f"remote sidecar size verification failed: {sidecar_asset}")
        downloaded_dir = Path(temporary) / "verify"
        downloaded = client.download(sidecar_asset, downloaded_dir)
        remote_sidecar = json.loads(downloaded.read_text(encoding="utf-8"))
        if remote_sidecar != sidecar:
            raise RuntimeError(f"remote sidecar content verification failed: {sidecar_asset}")

    published = {
        "run_id": checked["run_id"],
        "variant": checked["variant"],
        "stage": checked["stage"],
        "completed_epoch": checked["completed_epoch"],
        "checkpoint": checked["checkpoint"],
        "checkpoint_sha256": checked["checkpoint_sha256"],
        "checkpoint_size": checked["checkpoint_size"],
        "checkpoint_asset": checkpoint_asset,
        "sidecar_asset": sidecar_asset,
        "release_url": release_url,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "status": "published-verified",
    }
    return append_ledger(ledger_path, published)


def prune_verified_epochs(
    queue: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    *,
    retain: int,
) -> list[str]:
    if retain < 1:
        raise ValueError("retain must be at least one")
    published = {
        (str(row["run_id"]), int(row["completed_epoch"]))
        for row in ledger
        if row.get("status") == "published-verified"
    }
    retained: set[tuple[str, int]] = set()
    for run_id in {key[0] for key in published}:
        epochs = sorted(key[1] for key in published if key[0] == run_id)
        retained.update((run_id, epoch) for epoch in epochs[-retain:])
    removed = []
    for row in queue:
        key = (str(row.get("run_id")), int(row.get("completed_epoch", -1)))
        if key not in published or key in retained:
            continue
        path = Path(str(row.get("checkpoint", ""))).resolve()
        if re.fullmatch(r"epoch\d+\.pt", path.name) and path.is_file() and not path.is_symlink():
            path.unlink()
            removed.append(str(path))
    return removed


def publish_pending(args: argparse.Namespace, client: GitHubReleaseClient) -> dict[str, Any]:
    queue = read_jsonl(args.queue)
    ledger = read_jsonl(args.ledger)
    completed = {
        (str(row["run_id"]), int(row["completed_epoch"]))
        for row in ledger
        if row.get("status") == "published-verified"
    }
    pending = [
        row
        for row in queue
        if row.get("stage") == args.stage
        and (str(row.get("run_id")), int(row.get("completed_epoch", -1))) not in completed
    ]
    published = []
    for row in pending:
        published.append(publish_record(row, client=client, ledger_path=args.ledger))
    current_ledger = read_jsonl(args.ledger)
    removed = prune_verified_epochs(queue, current_ledger, retain=args.retain_local)
    return {
        "state": "drained",
        "queue_rows": len(queue),
        "published_now": len(published),
        "verified_total": len(current_ledger),
        "removed_local_checkpoints": removed,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--repo", default="kkc236/uav-detection-baselines")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--target", default="codex/scads-fdr")
    parser.add_argument("--stage", choices=("screen", "formal"), required=True)
    parser.add_argument("--retain-local", type=int, default=1)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.queue = args.queue.resolve()
    args.ledger = args.ledger.resolve()
    args.status_file = (
        args.status_file.resolve()
        if args.status_file is not None
        else args.ledger.with_suffix(".status.json")
    )
    if args.retain_local < 1 or args.interval < 1:
        raise ValueError("retain-local and interval must be positive")
    client = GitHubReleaseClient(repo=args.repo, tag=args.tag, target=args.target)
    client.require_auth()
    while True:
        try:
            status = publish_pending(args, client)
            write_json_atomic(args.status_file, status)
            print(json.dumps(status, sort_keys=True), flush=True)
        except Exception as error:
            status = {
                "state": "retrying" if args.watch else "failed",
                "error": f"{type(error).__name__}: {error}",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json_atomic(args.status_file, status)
            if not args.watch:
                raise
            print(json.dumps(status, sort_keys=True), flush=True)
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
