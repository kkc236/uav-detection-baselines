from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sync_experiment_checkpoint import (
    commit_and_push_results,
    ensure_results_checkout,
    validate_token_file,
    write_json_atomic,
)
from src.acr_eg_release import (
    publish_acr_eg_checkpoint,
    release_coordinates,
)
from src.github_checkpoint_sync import github_session


@dataclass(frozen=True)
class CheckpointObservation:
    bytes: int
    mtime_ns: int


def checkpoint_path_for_epoch(run_dir: str | Path, completed_epoch: int) -> Path:
    if not 1 <= completed_epoch <= 100:
        raise ValueError("ACR_EG_RELEASE_EPOCH_INVALID")
    return Path(run_dir) / "weights" / f"epoch{completed_epoch - 1}.pt"


def assert_source_checkout_commit(source_commit: str) -> None:
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    if current != source_commit:
        raise RuntimeError("ACR_EG_RELEASE_SOURCE_CHECKOUT_MISMATCH")


def observe_checkpoint(path: Path) -> CheckpointObservation | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return CheckpointObservation(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def checkpoint_is_stable(
    path: Path,
    *,
    previous: CheckpointObservation | None,
    stable_seconds: int,
) -> tuple[bool, CheckpointObservation | None]:
    current = observe_checkpoint(path)
    if current is None or previous != current:
        return False, current
    age_ns = time.time_ns() - current.mtime_ns
    return age_ns >= stable_seconds * 1_000_000_000, current


def _evidence_path(evidence_dir: Path, completed_epoch: int) -> Path:
    return evidence_dir / f"epoch-{completed_epoch:03d}.json"


def _already_published(path: Path, *, source_commit: str, completed_epoch: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("state") == "published_and_verified"
        and payload.get("source_commit") == source_commit
        and payload.get("continuity", {}).get("completed_epoch") == completed_epoch
    )


def _publish_results_evidence(
    args: argparse.Namespace,
    *,
    evidence: dict,
    evidence_name: str,
) -> None:
    environment = ensure_results_checkout(
        args.results_repo,
        repo_url=args.repo_url,
        branch=args.results_branch,
        token_file=args.token_file,
    )
    result_directory = args.results_repo / "results" / args.run_name / "checkpoints"
    result_directory.mkdir(parents=True, exist_ok=True)
    destination = result_directory / evidence_name
    write_json_atomic(destination, evidence)
    write_json_atomic(result_directory / "latest.json", evidence)
    commit_and_push_results(
        args.results_repo,
        result_directory=result_directory.parent,
        completed_epoch=int(evidence["continuity"]["completed_epoch"]),
        branch=args.results_branch,
        environment=environment,
    )


def publish_epoch(args: argparse.Namespace, completed_epoch: int) -> dict:
    checkpoint = checkpoint_path_for_epoch(args.run_dir, completed_epoch)
    token = validate_token_file(args.token_file)
    session = github_session(token)
    evidence = publish_acr_eg_checkpoint(
        session,
        repo=args.repo,
        source_commit=args.source_commit,
        checkpoint=checkpoint,
        expected_completed_epoch=completed_epoch,
    )
    evidence_path = _evidence_path(args.evidence_dir, completed_epoch)
    _publish_results_evidence(
        args,
        evidence=evidence,
        evidence_name=evidence_path.name,
    )
    write_json_atomic(evidence_path, evidence)
    write_json_atomic(args.status_file, evidence)
    return evidence


def run_continuously(args: argparse.Namespace) -> None:
    for completed_epoch in range(args.start_epoch, args.end_epoch + 1):
        coordinates = release_coordinates(
            args.source_commit,
            completed_epoch=completed_epoch,
        )
        evidence_path = _evidence_path(args.evidence_dir, completed_epoch)
        if _already_published(
            evidence_path,
            source_commit=args.source_commit,
            completed_epoch=completed_epoch,
        ):
            continue
        checkpoint = checkpoint_path_for_epoch(args.run_dir, completed_epoch)
        previous: CheckpointObservation | None = None
        while True:
            stable, previous = checkpoint_is_stable(
                checkpoint,
                previous=previous,
                stable_seconds=args.stable_seconds,
            )
            if stable:
                try:
                    evidence = publish_epoch(args, completed_epoch)
                    print(
                        f"Published and verified {coordinates.tag} "
                        f"({evidence['release']['digest']})",
                        flush=True,
                    )
                    break
                except Exception as error:
                    write_json_atomic(
                        args.status_file,
                        {
                            "state": "retrying",
                            "completed_epoch": completed_epoch,
                            "error": f"{type(error).__name__}: {error}",
                            "time": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    print(
                        f"ACR-EG checkpoint publication failed; retrying: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    previous = None
            time.sleep(args.interval)
    write_json_atomic(
        args.status_file,
        {
            "state": "complete",
            "source_commit": args.source_commit,
            "completed_epoch": args.end_epoch,
            "time": datetime.now(timezone.utc).isoformat(),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish and verify every formal ACR-EG epoch checkpoint."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--repo", default="kkc236/uav-detection-baselines")
    parser.add_argument(
        "--repo-url",
        default="https://github.com/kkc236/uav-detection-baselines.git",
    )
    parser.add_argument("--source-branch", default="codex/gcte-rtdetr-g0")
    parser.add_argument("--results-branch", default="training-results")
    parser.add_argument(
        "--results-repo",
        type=Path,
        default=Path("/mnt/uav/gcte-training-results"),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--start-epoch", type=int, default=10)
    parser.add_argument("--end-epoch", type=int, default=100)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--stable-seconds", type=int, default=30)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--completed-epoch", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.run_dir = args.run_dir.resolve()
    args.token_file = args.token_file.resolve()
    args.results_repo = args.results_repo.resolve()
    release_coordinates(args.source_commit, completed_epoch=args.start_epoch)
    assert_source_checkout_commit(args.source_commit)
    if not args.start_epoch <= args.end_epoch <= 100:
        raise ValueError("ACR_EG_RELEASE_EPOCH_RANGE_INVALID")
    args.run_name = args.run_name or f"gcte-acr-eg-{args.source_commit[:8]}"
    args.evidence_dir = (
        args.evidence_dir.resolve()
        if args.evidence_dir is not None
        else args.run_dir / "checkpoint-release-evidence"
    )
    args.status_file = (
        args.status_file.resolve()
        if args.status_file is not None
        else args.run_dir / "checkpoint-release-status.json"
    )
    if args.once:
        if args.completed_epoch is None:
            raise ValueError("--once requires --completed-epoch")
        evidence = publish_epoch(args, args.completed_epoch)
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return
    run_continuously(args)


if __name__ == "__main__":
    main()
