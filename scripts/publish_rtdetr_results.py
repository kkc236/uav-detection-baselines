from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.result_publisher import (
    PublicationStatus,
    collect_result_artifacts,
    latest_completed_epoch,
    repo_slug_from_remote,
    sha256_file,
    verified_release_assets,
)


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, env=env, text=True).strip()


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


def checked(response: requests.Response) -> requests.Response:
    if response.ok:
        return response
    raise RuntimeError(f"GitHub API {response.status_code}: {response.text[:500]}")


def get_or_create_release(
    session: requests.Session,
    *,
    repo: str,
    tag: str,
    branch: str,
) -> dict[str, object]:
    api = f"https://api.github.com/repos/{repo}"
    response = session.get(f"{api}/releases/tags/{tag}", timeout=30)
    if response.status_code == 404:
        response = session.post(
            f"{api}/releases",
            json={
                "tag_name": tag,
                "target_commitish": branch,
                "name": "RT-DETR-L VisDrone Scratch Baseline",
                "body": "Adaptive batch scratch training on one RTX 5090. No external pretrained weights.",
            },
            timeout=30,
        )
    return checked(response).json()


def upload_release_asset(
    session: requests.Session,
    *,
    release: dict[str, object],
    path: Path,
) -> None:
    assets = {str(item["name"]): item for item in release.get("assets", [])}
    existing = assets.get(path.name)
    if existing and int(existing["size"]) == path.stat().st_size:
        return
    if existing:
        checked(session.delete(str(existing["url"]), timeout=30))

    upload_url = str(release["upload_url"]).split("{")[0]
    headers = {"Content-Type": "application/octet-stream", "Content-Length": str(path.stat().st_size)}
    with path.open("rb") as file:
        response = session.post(
            upload_url,
            params={"name": path.name},
            headers=headers,
            data=file,
            timeout=(30, 1800),
        )
    checked(response)


def push_results(repo_dir: Path, results_dir: Path, token_file: Path) -> tuple[str, str, str]:
    run(["git", "add", "--", str(results_dir)], cwd=repo_dir)
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo_dir,
        check=False,
    ).returncode
    if changed:
        subprocess.run(
            ["git", "commit", "-m", "Add completed adaptive RT-DETR baseline results"],
            cwd=repo_dir,
            check=True,
        )

    askpass = repo_dir / "logs" / "github_askpass.sh"
    askpass.parent.mkdir(parents=True, exist_ok=True)
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
    branch = run(["git", "branch", "--show-current"], cwd=repo_dir)
    run(["git", "push", "origin", branch], cwd=repo_dir, env=environment)
    remote = run(["git", "remote", "get-url", "origin"], cwd=repo_dir)
    sha = run(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    return repo_slug_from_remote(remote), branch, sha


def publish(args: argparse.Namespace) -> PublicationStatus:
    run_dir = args.run_dir.resolve()
    repo_dir = args.repo_dir.resolve()
    token_file = args.token_file.resolve()
    completed_epoch = latest_completed_epoch(run_dir / "results.csv")
    if completed_epoch < args.target_epoch:
        raise RuntimeError(f"Training incomplete: epoch {completed_epoch}/{args.target_epoch}")
    if not token_file.exists():
        raise FileNotFoundError(token_file)

    destination = repo_dir / args.results_dir
    collect_result_artifacts(run_dir, destination)
    weights = [run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"]
    for weight in weights:
        if not weight.exists():
            raise FileNotFoundError(weight)

    summary = {
        "completed_epoch": completed_epoch,
        "run_dir": str(run_dir),
        "weights": {
            weight.name: {"bytes": weight.stat().st_size, "sha256": sha256_file(weight)}
            for weight in weights
        },
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    artifacts_dir = repo_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    archive = artifacts_dir / f"{args.tag}-results.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(destination, arcname=destination.name)

    repo, branch, local_sha = push_results(repo_dir, destination, token_file)
    token = token_file.read_text(encoding="utf-8").strip()
    session = github_session(token)
    commit = checked(
        session.get(f"https://api.github.com/repos/{repo}/commits/{branch}", timeout=30)
    ).json()
    git_verified = commit.get("sha") == local_sha
    if not git_verified:
        raise RuntimeError("GitHub branch did not reach the local result commit")

    release = get_or_create_release(session, repo=repo, tag=args.tag, branch=branch)
    release_files = [*weights, archive]
    for path in release_files:
        upload_release_asset(session, release=release, path=path)
        release = checked(session.get(str(release["url"]), timeout=30)).json()

    expected = {path.name: path.stat().st_size for path in release_files}
    release_verified = verified_release_assets(expected, release.get("assets", []))
    status = PublicationStatus(
        completed_epoch=completed_epoch,
        git_push_verified=git_verified,
        release_verified=release_verified,
    )
    if not status.shutdown_allowed:
        raise RuntimeError("Publication verification failed; shutdown denied")
    print(json.dumps({"published": True, "release": release["html_url"], **summary}, indent=2))
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a completed RT-DETR baseline safely.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--token-file", type=Path, default=Path("/root/autodl-tmp/github_token"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/rtdetr-100ep-5090-adaptive"))
    parser.add_argument("--tag", default="rtdetr-100ep-5090-adaptive")
    parser.add_argument("--target-epoch", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    publish(parse_args())
