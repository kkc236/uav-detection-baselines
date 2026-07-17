from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint_recovery import find_resume_checkpoint
from src.github_checkpoint_sync import (
    checkpoint_metadata,
    get_or_create_release,
    github_session,
    matching_checkpoint_assets,
    publish_checkpoint,
    upload_asset,
)


LIGHTWEIGHT_ARTIFACTS = (
    "results.csv",
    "btdse_diagnostics.jsonl",
    "ioqc_sa_diagnostics.jsonl",
    "vsf_rmr_diagnostics.jsonl",
    "batch_history.jsonl",
    "adaptive_state.json",
    "args.yaml",
)


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{destination}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def validate_token_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"GitHub token file not found: {path}")
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError(f"GitHub token file must have mode 600: {path}")
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"GitHub token file is empty: {path}")
    return token


def collect_lightweight_artifacts(
    run_dir: str | Path,
    destination: str | Path,
    manifest: dict[str, Any],
) -> list[Path]:
    source = Path(run_dir)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in LIGHTWEIGHT_ARTIFACTS:
        source_path = source / name
        if source_path.is_file():
            destination_path = target / name
            shutil.copy2(source_path, destination_path)
            copied.append(destination_path)
    manifest_path = target / "latest.json"
    write_json_atomic(manifest_path, manifest)
    copied.append(manifest_path)
    return copied


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _git_environment(results_repo: Path, token_file: Path) -> dict[str, str]:
    askpass = results_repo / ".git" / "uav-github-askpass.sh"
    askpass.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  *Username*) printf '%s\\n' 'x-access-token' ;;\n"
        "  *) cat \"$UAV_GITHUB_TOKEN_FILE\" ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    askpass.chmod(0o700)
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_ASKPASS": str(askpass),
            "GIT_TERMINAL_PROMPT": "0",
            "UAV_GITHUB_TOKEN_FILE": str(token_file),
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
    if not (results_repo / ".git").is_dir():
        results_repo.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", repo_url, str(results_repo)], cwd=results_repo.parent)
    environment = _git_environment(results_repo, token_file)
    _run(["git", "config", "user.name", "uav-training-bot"], cwd=results_repo)
    _run(["git", "config", "user.email", "uav-training-bot@users.noreply.github.com"], cwd=results_repo)

    if _run(["git", "branch", "--list", branch], cwd=results_repo).stdout.strip():
        _run(["git", "switch", branch], cwd=results_repo)
        return environment
    remote = _run(["git", "ls-remote", "--heads", "origin", branch], cwd=results_repo, env=environment).stdout
    if remote.strip():
        _run(["git", "fetch", "origin", branch], cwd=results_repo, env=environment)
        _run(["git", "switch", "-c", branch, "FETCH_HEAD"], cwd=results_repo)
    else:
        _run(["git", "switch", "-c", branch], cwd=results_repo)
    return environment


def commit_and_push_results(
    results_repo: Path,
    *,
    result_directory: Path,
    completed_epoch: int,
    branch: str,
    environment: dict[str, str],
) -> None:
    relative = result_directory.relative_to(results_repo)
    _run(["git", "add", "--", str(relative)], cwd=results_repo)
    changed = _run(["git", "diff", "--cached", "--quiet"], cwd=results_repo, check=False)
    if changed.returncode != 0:
        _run(["git", "commit", "-m", f"Update protected training epoch {completed_epoch}"], cwd=results_repo)
    _run(["git", "push", "origin", f"HEAD:{branch}"], cwd=results_repo, env=environment)


def checkpoint_tree_fingerprint(run_dir: Path) -> tuple[tuple[str, int, int], ...]:
    weights = run_dir / "weights"
    return tuple(
        sorted(
            (path.name, path.stat().st_size, path.stat().st_mtime_ns)
            for path in weights.glob("*.pt")
            if path.is_file() and not path.name.startswith(".github-")
        )
    )


def prune_local_epoch_checkpoints(weights: Path, *, retain: int) -> list[Path]:
    if retain < 1:
        raise ValueError("retain must be at least one")
    pattern = re.compile(r"^epoch(\d+)\.pt$")
    candidates: list[tuple[int, Path]] = []
    for path in weights.glob("epoch*.pt"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    removed = [path for _, path in sorted(candidates)[:-retain]]
    for path in removed:
        path.unlink()
    return removed


def _manifest_name(prefix: str, completed_epoch: int) -> str:
    return f"{prefix}-epoch-{completed_epoch:04d}.json"


def _prune_remote_manifests(session, release: dict[str, Any], *, prefix: str, retain: int) -> None:
    pattern = re.compile(rf"^{re.escape(prefix)}-epoch-(\d+)\.json$")
    candidates = []
    for asset in release.get("assets", []):
        match = pattern.match(str(asset.get("name", "")))
        if match:
            candidates.append((int(match.group(1)), asset))
    for _, asset in sorted(candidates)[:-retain]:
        response = session.delete(str(asset["url"]), timeout=30)
        if not response.ok:
            raise RuntimeError(f"GitHub API {response.status_code}: {response.text[:500]}")


def sync_once(args: argparse.Namespace) -> dict[str, Any] | None:
    checkpoint = find_resume_checkpoint(args.run_dir)
    if checkpoint is None:
        return None

    staging = args.run_dir / "weights" / ".github-upload-checkpoint.pt"
    staging_temporary = Path(f"{staging}.tmp")
    shutil.copy2(checkpoint, staging_temporary)
    staging_temporary.replace(staging)
    manifest_staging = args.run_dir / "weights" / ".github-upload-manifest.json"
    try:
        metadata = checkpoint_metadata(staging)
        token = validate_token_file(args.token_file)
        session = github_session(token)
        manifest = publish_checkpoint(
            session,
            repo=args.repo,
            tag=args.tag,
            branch=args.source_branch,
            checkpoint=staging,
            retain=args.retain,
            asset_prefix=args.asset_prefix,
            release_name=args.release_name,
            release_body=args.release_body,
        )
        manifest.update(
            {
                "published_at": datetime.now(timezone.utc).isoformat(),
                "run_name": args.run_name or args.run_dir.name,
            }
        )
        write_json_atomic(manifest_staging, manifest)
        release = get_or_create_release(
            session,
            repo=args.repo,
            tag=args.tag,
            branch=args.source_branch,
            release_name=args.release_name,
            release_body=args.release_body,
        )
        upload_asset(
            session,
            release=release,
            path=manifest_staging,
            asset_name=_manifest_name(args.asset_prefix, metadata.completed_epoch),
        )
        refreshed = session.get(str(release["url"]), timeout=30)
        refreshed.raise_for_status()
        _prune_remote_manifests(session, refreshed.json(), prefix=args.asset_prefix, retain=args.retain)

        environment = ensure_results_checkout(
            args.results_repo,
            repo_url=args.repo_url,
            branch=args.results_branch,
            token_file=args.token_file,
        )
        result_directory = args.results_repo / "results" / (args.run_name or args.run_dir.name)
        collect_lightweight_artifacts(args.run_dir, result_directory, manifest)
        commit_and_push_results(
            args.results_repo,
            result_directory=result_directory,
            completed_epoch=metadata.completed_epoch,
            branch=args.results_branch,
            environment=environment,
        )
        prune_local_epoch_checkpoints(args.run_dir / "weights", retain=args.retain)
        write_json_atomic(args.status_file, {"state": "published", **manifest})
        return manifest
    finally:
        staging.unlink(missing_ok=True)
        staging_temporary.unlink(missing_ok=True)
        manifest_staging.unlink(missing_ok=True)


def run_continuously(args: argparse.Namespace) -> None:
    previous_fingerprint: tuple[tuple[str, int, int], ...] | None = None
    while True:
        fingerprint = checkpoint_tree_fingerprint(args.run_dir)
        if fingerprint and fingerprint != previous_fingerprint:
            try:
                manifest = sync_once(args)
                if manifest is not None:
                    print(
                        f"Published checkpoint epoch {manifest['completed_epoch']} to {manifest['release_url']}",
                        flush=True,
                    )
                previous_fingerprint = checkpoint_tree_fingerprint(args.run_dir)
            except Exception as error:
                write_json_atomic(
                    args.status_file,
                    {
                        "state": "retrying",
                        "error": f"{type(error).__name__}: {error}",
                        "time": datetime.now(timezone.utc).isoformat(),
                    },
                )
                print(f"Checkpoint sync failed; retrying: {error}", file=sys.stderr, flush=True)
                previous_fingerprint = None
        time.sleep(args.interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Protect resumable experiment checkpoints on GitHub.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--repo", default="kkc236/uav-detection-baselines")
    parser.add_argument("--repo-url", default="https://github.com/kkc236/uav-detection-baselines.git")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-branch", default="main")
    parser.add_argument("--results-branch", default="training-results")
    parser.add_argument("--results-repo", type=Path, default=Path.home() / "uav-training-results")
    parser.add_argument("--run-name")
    parser.add_argument("--retain", type=int, default=3)
    parser.add_argument("--asset-prefix", required=True)
    parser.add_argument("--release-name", required=True)
    parser.add_argument("--release-body", required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--status-file", type=Path, default=Path("logs/checkpoint_sync.json"))
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.run_dir = args.run_dir.resolve()
    args.token_file = args.token_file.resolve()
    args.results_repo = args.results_repo.resolve()
    args.status_file = args.status_file.resolve()
    if args.once:
        manifest = sync_once(args)
        if manifest is None:
            raise SystemExit("No resumable checkpoint is available yet")
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    run_continuously(args)


if __name__ == "__main__":
    main()

