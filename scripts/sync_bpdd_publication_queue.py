from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.github_checkpoint_sync import (  # noqa: E402
    checkpoint_asset_name,
    checkpoint_metadata,
    get_or_create_release,
    github_session,
    publish_checkpoint,
    upload_asset,
)


RELEASE_NAME = "BPDD exact per-epoch checkpoints"
RELEASE_BODY = (
    "Exact, ordered and resumable FDR/BPDD checkpoints with immutable per-epoch manifests."
)
REMOTE_RETAIN = 1_000_000


@dataclass(frozen=True)
class QueueEntry:
    run_id: str
    variant: str
    stage: str
    completed_epoch: int
    checkpoint: Path
    checkpoint_sha256: str
    queue_record_sha256: str
    raw: Mapping[str, Any]


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"BPDD publication JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid BPDD publication JSONL at line {line_number}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(f"BPDD publication row {line_number} is not an object")
        rows.append(row)
    return rows


def _checkpoint_path(value: Any, *, queue: Path) -> Path:
    raw = Path(str(value))
    return (raw if raw.is_absolute() else queue.parent / raw).resolve()


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def load_validated_queue(
    queue: str | Path,
    run_dir: str | Path,
    *,
    verified_through_epoch: int = 0,
) -> list[QueueEntry]:
    queue_path = Path(queue).resolve()
    run = Path(run_dir).resolve()
    rows = _read_jsonl(queue_path)
    entries: list[QueueEntry] = []
    identity: tuple[str, str, str] | None = None
    for expected_epoch, row in enumerate(rows, 1):
        required = {
            "run_id",
            "variant",
            "stage",
            "completed_epoch",
            "status",
            "checkpoint",
            "checkpoint_sha256",
        }
        missing = required - row.keys()
        if missing:
            raise ValueError(f"BPDD queue row {expected_epoch} is missing {sorted(missing)}")
        try:
            completed_epoch = int(row["completed_epoch"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid BPDD queue completed epoch at row {expected_epoch}") from error
        if completed_epoch != expected_epoch:
            raise ValueError(
                f"BPDD queue epoch gap/order error: expected {expected_epoch}, got {completed_epoch}"
            )
        if row["status"] != "pending":
            raise ValueError(f"BPDD queue epoch {completed_epoch} is not pending")
        current_identity = (
            str(row["run_id"]),
            str(row["variant"]),
            str(row["stage"]),
        )
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError(f"BPDD queue identity changed at epoch {completed_epoch}")
        checkpoint = _checkpoint_path(row["checkpoint"], queue=queue_path)
        if not _is_within(checkpoint, run):
            raise ValueError(
                f"BPDD queue checkpoint is outside the run directory at epoch {completed_epoch}"
            )
        expected_sha = str(row["checkpoint_sha256"]).upper()
        if not re.fullmatch(r"[0-9A-F]{64}", expected_sha):
            raise ValueError(f"invalid BPDD checkpoint SHA256 at epoch {completed_epoch}")
        if checkpoint.is_file():
            metadata = checkpoint_metadata(checkpoint)
            if metadata.completed_epoch != completed_epoch:
                raise ValueError(
                    "BPDD checkpoint completed epoch mismatch: "
                    f"queue={completed_epoch}, checkpoint={metadata.completed_epoch}"
                )
            if metadata.sha256.upper() != expected_sha:
                raise ValueError(f"BPDD checkpoint SHA256 mismatch at epoch {completed_epoch}")
        elif completed_epoch > verified_through_epoch:
            raise FileNotFoundError(
                f"BPDD queue checkpoint not found at epoch {completed_epoch}: {checkpoint}"
            )
        entries.append(
            QueueEntry(
                run_id=current_identity[0],
                variant=current_identity[1],
                stage=current_identity[2],
                completed_epoch=completed_epoch,
                checkpoint=checkpoint,
                checkpoint_sha256=expected_sha,
                queue_record_sha256=_sha256_bytes(_canonical_bytes(row)),
                raw=row,
            )
        )
    return entries


def prune_verified_local_checkpoints(
    entries: Sequence[QueueEntry],
    *,
    verified_through_epoch: int,
    local_retain: int,
) -> list[Path]:
    if local_retain <= 0:
        raise ValueError("BPDD local retain must be positive")
    if verified_through_epoch <= local_retain:
        return []

    verified = [
        entry for entry in entries if entry.completed_epoch <= verified_through_epoch
    ]
    retained_paths = {entry.checkpoint for entry in verified[-local_retain:]}
    unpublished_paths = {
        entry.checkpoint
        for entry in entries
        if entry.completed_epoch > verified_through_epoch
    }
    removed: list[Path] = []
    for entry in verified[:-local_retain]:
        checkpoint = entry.checkpoint
        if checkpoint.name.lower() in {"last.pt", "best.pt"}:
            continue
        if checkpoint in retained_paths or checkpoint in unpublished_paths:
            continue
        if checkpoint.is_file():
            checkpoint.unlink()
            removed.append(checkpoint)
    return removed


def load_validated_ledger(
    path: str | Path, entries: Sequence[QueueEntry]
) -> list[dict[str, Any]]:
    ledger_path = Path(path).resolve()
    if not ledger_path.exists():
        return []
    rows = _read_jsonl(ledger_path)
    if len(rows) > len(entries):
        raise ValueError("BPDD publication ledger is ahead of the queue")
    for expected_epoch, row in enumerate(rows, 1):
        entry = entries[expected_epoch - 1]
        if int(row.get("completed_epoch", -1)) != expected_epoch:
            raise ValueError("BPDD publication ledger is not contiguous")
        if row.get("verified") is not True:
            raise ValueError(f"BPDD publication ledger epoch {expected_epoch} is not verified")
        if (
            row.get("run_id"),
            row.get("variant"),
            row.get("stage"),
        ) != (entry.run_id, entry.variant, entry.stage):
            raise ValueError(f"BPDD publication ledger identity mismatch at epoch {expected_epoch}")
        if str(row.get("queue_record_sha256", "")).upper() != entry.queue_record_sha256:
            raise ValueError(f"BPDD publication ledger/queue mismatch at epoch {expected_epoch}")
        checkpoint = row.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"BPDD publication ledger checkpoint missing at epoch {expected_epoch}")
        if str(checkpoint.get("sha256", "")).upper() != entry.checkpoint_sha256:
            raise ValueError(f"BPDD publication ledger checkpoint SHA256 mismatch at epoch {expected_epoch}")
    return rows


def _append_ledger(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_bytes(record) + b"\n"
    with path.open("ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _asset_by_name(release: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    matches = [asset for asset in release.get("assets", []) if asset.get("name") == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one GitHub Release asset named {name}, got {len(matches)}")
    return matches[0]


def _verify_asset(asset: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if int(asset.get("id", -1)) != int(expected["asset_id"]):
        raise RuntimeError(f"GitHub Release asset id mismatch for {expected['asset_name']}")
    if str(asset.get("name")) != str(expected["asset_name"]):
        raise RuntimeError(f"GitHub Release asset name mismatch for {expected['asset_name']}")
    if "bytes" in expected and int(asset.get("size", -1)) != int(expected["bytes"]):
        raise RuntimeError(f"GitHub Release asset size mismatch for {expected['asset_name']}")
    digest = asset.get("digest")
    if digest and "sha256" in expected:
        actual = str(digest).removeprefix("sha256:").upper()
        if actual != str(expected["sha256"]).upper():
            raise RuntimeError(f"GitHub Release asset SHA256 mismatch for {expected['asset_name']}")


def verify_publication_assets(
    _session: Any, release: Mapping[str, Any], record: Mapping[str, Any]
) -> None:
    checkpoint = record["checkpoint"]
    manifest = record["manifest"]
    _verify_asset(_asset_by_name(release, str(checkpoint["asset_name"])), checkpoint)
    _verify_asset(_asset_by_name(release, str(manifest["asset_name"])), manifest)


def _release(session: Any, args: argparse.Namespace) -> dict[str, Any]:
    return get_or_create_release(
        session,
        repo=args.repo,
        tag=args.tag,
        branch=args.branch,
        release_name=RELEASE_NAME,
        release_body=RELEASE_BODY,
    )


def verify_ledger_assets(
    session: Any, records: Sequence[Mapping[str, Any]], args: argparse.Namespace
) -> None:
    if not records:
        return
    release = _release(session, args)
    for record in records:
        verify_publication_assets(session, release, record)


def _manifest_asset_name(prefix: str, completed_epoch: int) -> str:
    checkpoint_asset_name(completed_epoch, prefix=prefix)
    return f"{prefix}-epoch-{completed_epoch:04d}.json"


def publish_entry(
    session: Any, entry: QueueEntry, args: argparse.Namespace
) -> dict[str, Any]:
    checkpoint_manifest = publish_checkpoint(
        session,
        repo=args.repo,
        tag=args.tag,
        branch=args.branch,
        checkpoint=entry.checkpoint,
        retain=REMOTE_RETAIN,
        asset_prefix=args.asset_prefix,
        release_name=RELEASE_NAME,
        release_body=RELEASE_BODY,
    )
    if int(checkpoint_manifest.get("completed_epoch", -1)) != entry.completed_epoch:
        raise RuntimeError("published BPDD checkpoint epoch mismatch")
    checkpoint = checkpoint_manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("published BPDD checkpoint manifest is missing checkpoint metadata")
    if str(checkpoint.get("sha256", "")).upper() != entry.checkpoint_sha256:
        raise RuntimeError("published BPDD checkpoint SHA256 mismatch")
    expected_checkpoint_name = checkpoint_asset_name(
        entry.completed_epoch, prefix=args.asset_prefix
    )
    if checkpoint.get("asset_name") != expected_checkpoint_name:
        raise RuntimeError("published BPDD checkpoint asset name mismatch")

    remote_manifest = {
        "format_version": 1,
        "run_id": entry.run_id,
        "variant": entry.variant,
        "stage": entry.stage,
        "completed_epoch": entry.completed_epoch,
        "queue_record_sha256": entry.queue_record_sha256,
        "checkpoint": dict(checkpoint),
        "release_url": str(checkpoint_manifest["release_url"]),
    }
    manifest_bytes = _canonical_bytes(remote_manifest) + b"\n"
    manifest_sha = _sha256_bytes(manifest_bytes)
    manifest_path = args.run_dir / f".bpdd-publication-epoch-{entry.completed_epoch:04d}.json"
    manifest_path.write_bytes(manifest_bytes)
    try:
        release = _release(session, args)
        manifest_asset = upload_asset(
            session,
            release=release,
            path=manifest_path,
            asset_name=_manifest_asset_name(args.asset_prefix, entry.completed_epoch),
        )
    finally:
        manifest_path.unlink(missing_ok=True)
    record = {
        **remote_manifest,
        "manifest": {
            "asset_id": int(manifest_asset["id"]),
            "asset_name": str(manifest_asset["name"]),
            "bytes": int(manifest_asset["size"]),
            "sha256": manifest_sha.lower(),
        },
        "verified": True,
    }
    verify_publication_assets(session, _release(session, args), record)
    return record


def validate_token_file(path: str | Path) -> str:
    token_path = Path(path).resolve()
    if not token_path.is_file():
        raise FileNotFoundError(f"GitHub token file not found: {token_path}")
    if os.name != "nt" and stat.S_IMODE(token_path.stat().st_mode) & 0o077:
        raise PermissionError(f"GitHub token file must have mode 600: {token_path}")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"GitHub token file is empty: {token_path}")
    return token


def _write_status(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_bytes(dict(payload)) + b"\n")
    os.replace(temporary, path)


def sync_once(args: argparse.Namespace) -> list[dict[str, Any]]:
    ledger_path = args.run_dir / "publication-ledger.jsonl"
    ledger_records = _read_jsonl(ledger_path) if ledger_path.exists() else []
    entries = load_validated_queue(
        args.queue,
        args.run_dir,
        verified_through_epoch=len(ledger_records),
    )
    ledger = load_validated_ledger(ledger_path, entries)
    if not entries:
        _write_status(args.status_file, {"state": "waiting", "completed_epoch": 0})
        return []

    token = validate_token_file(args.token_file)
    session = github_session(token)
    verify_ledger_assets(session, ledger, args)
    prune_verified_local_checkpoints(
        entries,
        verified_through_epoch=len(ledger),
        local_retain=args.local_retain,
    )
    published: list[dict[str, Any]] = []
    for entry in entries[len(ledger) :]:
        expected_epoch = len(ledger) + len(published) + 1
        if entry.completed_epoch != expected_epoch:
            raise RuntimeError(
                f"BPDD publication cannot skip epoch {expected_epoch}"
            )
        record = publish_entry(session, entry, args)
        if int(record.get("completed_epoch", -1)) != expected_epoch:
            raise RuntimeError("BPDD publisher returned an out-of-order epoch")
        _append_ledger(ledger_path, record)
        published.append(record)
        prune_verified_local_checkpoints(
            entries,
            verified_through_epoch=len(ledger) + len(published),
            local_retain=args.local_retain,
        )
        _write_status(
            args.status_file,
            {
                "state": "published",
                "completed_epoch": expected_epoch,
                "queued_epochs": len(entries),
                "ledger_records": len(ledger) + len(published),
            },
        )
    if not published:
        _write_status(
            args.status_file,
            {
                "state": "verified",
                "completed_epoch": len(ledger),
                "queued_epochs": len(entries),
                "ledger_records": len(ledger),
            },
        )
    return published


def _sanitized_error(error: BaseException, args: argparse.Namespace) -> str:
    message = f"{type(error).__name__}: {error}"
    try:
        token = Path(args.token_file).read_text(encoding="utf-8").strip()
    except OSError:
        token = ""
    if token:
        message = message.replace(token, "[REDACTED]")
    return re.sub(r"(?i)(authorization|bearer)([ :=]+)[^\s,;]+", r"\1\2[REDACTED]", message)


def run_continuously(args: argparse.Namespace) -> None:
    while True:
        try:
            published = sync_once(args)
            if published:
                print(
                    f"Published BPDD epochs 1-{published[-1]['completed_epoch']} in exact order",
                    flush=True,
                )
        except Exception as error:
            safe = _sanitized_error(error, args)
            _write_status(args.status_file, {"state": "retrying", "error": safe})
            print(f"BPDD publication retrying: {safe}", file=sys.stderr, flush=True)
        time.sleep(args.interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish the append-only BPDD queue to GitHub Release in exact epoch order."
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--asset-prefix", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--local-retain", type=int, default=3)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--status-file", type=Path, default=Path("logs/bpdd-publication-sync.json")
    )
    return parser


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.queue = args.queue.resolve()
    args.token_file = args.token_file.resolve()
    args.run_dir = args.run_dir.resolve()
    args.status_file = args.status_file.resolve()
    if args.interval <= 0:
        raise ValueError("BPDD publication interval must be positive")
    if args.local_retain <= 0:
        raise ValueError("BPDD local retain must be positive")
    checkpoint_asset_name(1, prefix=args.asset_prefix)
    return args


def main() -> None:
    args = _resolve_args(build_parser().parse_args())
    if not args.once:
        run_continuously(args)
        return
    try:
        published = sync_once(args)
    except Exception as error:
        raise SystemExit(_sanitized_error(error, args)) from None
    completed = published[-1]["completed_epoch"] if published else 0
    print(json.dumps({"published": len(published), "completed_epoch": completed}))


if __name__ == "__main__":
    main()
