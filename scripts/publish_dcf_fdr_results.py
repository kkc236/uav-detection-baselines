"""Publish completed Clean FDR and DCF-FDR Formal100 evidence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dcf_fdr_publication import (  # noqa: E402
    ArmSpec,
    StagedEvidence,
    ValidatedArm,
    build_bundle,
    sha256_file,
    stage_evidence,
    validate_arm,
    write_json_atomic,
)


REPOSITORY = "kkc236/icassp2027-fdr-bpdd-fia-material"
BRANCH = "main"
TAG = "clean-dcf-fdr-formal100-seed0-20260824"
EXPERIMENT_DIR = "experiments/clean-dcf-fdr-formal100-seed0-20260824"
ASSET_NAMES = {
    ("clean", "best"): "clean-fdr-seed0-formal100-best.pt",
    ("clean", "last"): "clean-fdr-seed0-formal100-last.pt",
    ("dcf", "best"): "dcf-fdr-seed0-formal100-best.pt",
    ("dcf", "last"): "dcf-fdr-seed0-formal100-last.pt",
}


def checked(response: Any) -> Any:
    if response.ok:
        return response
    raise RuntimeError(
        f"GitHub API {response.status_code}: {str(response.text)[:500]}"
    )


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


def read_private_token(path: Path, *, enforce_mode: bool | None = None) -> str:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    if enforce_mode is None:
        enforce_mode = os.name != "nt"
    mode = stat.S_IMODE(path.stat().st_mode)
    if enforce_mode and mode != 0o600:
        raise PermissionError(f"GitHub token file must be a regular 0600 file: {path}")
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("GitHub token file is empty")
    return token


def build_git_environment(askpass: Path, token_file: Path) -> dict[str, str]:
    askpass = Path(askpass)
    askpass.parent.mkdir(parents=True, exist_ok=True)
    askpass.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  *Username*) printf '%s\\n' 'x-access-token' ;;\n"
        "  *) cat \"$DCF_GITHUB_TOKEN_FILE\" ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    askpass.chmod(0o700)
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_ASKPASS": str(askpass),
            "GIT_TERMINAL_PROMPT": "0",
            "DCF_GITHUB_TOKEN_FILE": str(Path(token_file).resolve()),
        }
    )
    return environment


def verify_private_repository(session: Any, repository: str) -> None:
    response = checked(
        session.get(f"https://api.github.com/repos/{repository}", timeout=30)
    )
    if response.json().get("private") is not True:
        raise RuntimeError(f"publication target is not private: {repository}")


def upload_asset(
    session: Any,
    release: Mapping[str, Any],
    path: Path,
    asset_name: str,
) -> str:
    path = Path(path)
    assets = {str(item["name"]): item for item in release.get("assets", [])}
    existing = assets.get(asset_name)
    action = "uploaded"
    if existing and int(existing["size"]) == path.stat().st_size:
        return "skipped"
    if existing:
        checked(session.delete(str(existing["url"]), timeout=30))
        action = "replaced"
    upload_url = str(release["upload_url"]).split("{")[0]
    headers = {
        "Content-Type": "application/octet-stream",
        "Content-Length": str(path.stat().st_size),
    }
    with path.open("rb") as stream:
        checked(
            session.post(
                upload_url,
                params={"name": asset_name},
                headers=headers,
                data=stream,
                timeout=(30, 3600),
            )
        )
    return action


def verify_assets(expected: Mapping[str, int], actual: list[Mapping[str, Any]]) -> bool:
    actual_sizes = {str(item["name"]): int(item["size"]) for item in actual}
    return all(actual_sizes.get(name) == size for name, size in expected.items())


def checkpoint_assets(paths: Mapping[tuple[str, str], Path]) -> dict[str, Path]:
    if set(paths) != set(ASSET_NAMES):
        raise ValueError("checkpoint mapping must contain clean/dcf best/last")
    assets = {ASSET_NAMES[key]: Path(path) for key, path in paths.items()}
    for name, path in assets.items():
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing release asset {name}: {path}")
    return assets


def update_release_manifest(
    manifest_path: Path, large_assets: Mapping[str, Path]
) -> Mapping[str, Any]:
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["release_assets"] = {
        name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in sorted(large_assets.items())
    }
    write_json_atomic(manifest_path, payload)
    return payload


def copy_lightweight_evidence(staged: StagedEvidence, destination: Path) -> None:
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for source in sorted(staged.root.rglob("*")):
        if not source.is_file() or source == staged.bundle_path:
            continue
        relative = source.relative_to(staged.root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def sanitized_error(message: str, token: str) -> str:
    return str(message).replace(token, "<redacted>") if token else str(message)


def _run_git(
    command: list[str], *, cwd: Path, environment: Mapping[str, str]
) -> str:
    result = subprocess.run(
        ["git", *command],
        cwd=cwd,
        env=dict(environment),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"git {' '.join(command[:2])} failed: {detail[:500]}")
    return result.stdout.strip()


def _prepare_checkout(
    checkout: Path,
    *,
    repository: str,
    branch: str,
    environment: Mapping[str, str],
) -> None:
    checkout = Path(checkout)
    remote = f"https://github.com/{repository}.git"
    if not (checkout / ".git").is_dir():
        if checkout.exists() and any(checkout.iterdir()):
            raise RuntimeError(f"material checkout path is nonempty and not Git: {checkout}")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run_git(
            ["clone", "--branch", branch, "--single-branch", remote, str(checkout)],
            cwd=checkout.parent,
            environment=environment,
        )
    actual_remote = _run_git(
        ["remote", "get-url", "origin"], cwd=checkout, environment=environment
    )
    if actual_remote != remote:
        raise RuntimeError(f"unexpected material repository remote: {actual_remote}")
    _run_git(["config", "user.name", "Codex Evidence Publisher"], cwd=checkout, environment=environment)
    _run_git(
        ["config", "user.email", "codex-evidence@users.noreply.github.com"],
        cwd=checkout,
        environment=environment,
    )


def _commit_and_push(
    checkout: Path,
    *,
    experiment_path: Path,
    branch: str,
    environment: Mapping[str, str],
) -> str:
    _run_git(["add", "--", str(experiment_path)], cwd=checkout, environment=environment)
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=checkout,
        env=dict(environment),
        check=False,
    ).returncode
    if changed not in {0, 1}:
        raise RuntimeError("git diff --cached failed")
    if changed == 1:
        _run_git(
            ["commit", "-m", "Add Clean FDR versus DCF-FDR Formal100 results"],
            cwd=checkout,
            environment=environment,
        )
    _run_git(["fetch", "origin", branch], cwd=checkout, environment=environment)
    _run_git(["rebase", f"origin/{branch}"], cwd=checkout, environment=environment)
    _run_git(["push", "origin", f"HEAD:{branch}"], cwd=checkout, environment=environment)
    return _run_git(["rev-parse", "HEAD"], cwd=checkout, environment=environment)


def _get_or_create_release(
    session: Any, *, repository: str, tag: str, branch: str
) -> Mapping[str, Any]:
    api = f"https://api.github.com/repos/{repository}"
    response = session.get(f"{api}/releases/tags/{tag}", timeout=30)
    if response.status_code == 404:
        response = session.post(
            f"{api}/releases",
            json={
                "tag_name": tag,
                "target_commitish": branch,
                "name": "Clean FDR vs DCF-FDR Seed0 Formal100",
                "body": (
                    "Formal100 evidence for the registered one-module DCF-FDR "
                    "internal ablation. Scientific failures remain preserved."
                ),
                "draft": False,
                "prerelease": True,
            },
            timeout=30,
        )
    return checked(response).json()


def _arm_spec(arm: str, output_root: Path) -> ArmSpec:
    return ArmSpec(
        arm=arm,
        output_root=Path(output_root).resolve(),
        run_name=f"formal-seed0-{arm}-fdr-v1",
    )


def _checkpoint_mapping(
    clean: ValidatedArm, dcf: ValidatedArm
) -> dict[tuple[str, str], Path]:
    return {
        ("clean", "best"): clean.artifacts["best.pt"],
        ("clean", "last"): clean.artifacts["last.pt"],
        ("dcf", "best"): dcf.artifacts["best.pt"],
        ("dcf", "last"): dcf.artifacts["last.pt"],
    }


def publish(args: argparse.Namespace) -> Mapping[str, Any]:
    clean = validate_arm(_arm_spec("clean", args.clean_root))
    dcf = validate_arm(_arm_spec("dcf", args.dcf_root))
    evidence_root = Path(args.staging_root).resolve() / "evidence"
    staged = stage_evidence(clean, dcf, evidence_root)
    large_assets = checkpoint_assets(_checkpoint_mapping(clean, dcf))
    update_release_manifest(staged.manifest_path, large_assets)
    build_bundle(staged.root, staged.bundle_path)

    ready = {
        "ready": True,
        "decision": staged.comparison["decision"],
        "source_commit": clean.summary.get("source_commit", None),
        "release_assets": {
            name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in sorted(large_assets.items())
        },
    }
    if args.check_only:
        print(json.dumps(ready, indent=2, sort_keys=True))
        return ready

    token_file = Path(args.token_file).resolve()
    token = read_private_token(token_file)
    session = github_session(token)
    verify_private_repository(session, args.repository)

    staging_root = Path(args.staging_root).resolve()
    askpass = staging_root / "private" / "github-askpass.sh"
    environment = build_git_environment(askpass, token_file)
    checkout = Path(args.material_checkout).resolve()
    _prepare_checkout(
        checkout,
        repository=args.repository,
        branch=args.branch,
        environment=environment,
    )
    experiment_path = checkout / EXPERIMENT_DIR
    try:
        experiment_path.relative_to(checkout)
    except ValueError as error:
        raise RuntimeError("experiment path escapes material checkout") from error
    copy_lightweight_evidence(staged, experiment_path)
    local_sha = _commit_and_push(
        checkout,
        experiment_path=experiment_path,
        branch=args.branch,
        environment=environment,
    )
    commit = checked(
        session.get(
            f"https://api.github.com/repos/{args.repository}/commits/{args.branch}",
            timeout=30,
        )
    ).json()
    if commit.get("sha") != local_sha:
        raise RuntimeError("remote material branch did not reach result commit")

    release = _get_or_create_release(
        session,
        repository=args.repository,
        tag=args.tag,
        branch=args.branch,
    )
    release_assets = {
        **large_assets,
        "clean-dcf-fdr-seed0-formal100-lightweight-evidence.tar.gz": staged.bundle_path,
        "clean-dcf-fdr-seed0-formal100-artifact-manifest.json": staged.manifest_path,
    }
    actions: dict[str, str] = {}
    for name, path in release_assets.items():
        actions[name] = upload_asset(session, release, path, name)
        release = checked(session.get(str(release["url"]), timeout=30)).json()
    expected = {name: path.stat().st_size for name, path in release_assets.items()}
    if not verify_assets(expected, list(release.get("assets", []))):
        raise RuntimeError("GitHub Release asset size verification failed")

    status = {
        "format_version": 1,
        "published": True,
        "repository": args.repository,
        "branch": args.branch,
        "result_commit": local_sha,
        "tag": args.tag,
        "release_url": release["html_url"],
        "decision": staged.comparison["decision"],
        "assets": {
            name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "action": actions[name],
            }
            for name, path in sorted(release_assets.items())
        },
    }
    write_json_atomic(staging_root / "publication-status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--dcf-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--material-checkout", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = ""
    try:
        publish(args)
    except Exception as error:
        if not args.check_only:
            try:
                token = read_private_token(Path(args.token_file))
            except Exception:
                token = ""
        print(f"publication failed: {sanitized_error(str(error), token)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

