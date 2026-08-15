"""Audit local IBER-BE deployment readiness without claiming remote state."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.iber_protocol import (  # noqa: E402
    DESIGN_VERSION,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    execution_environment,
)


COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
FOREIGN_PATH_PATTERN = re.compile(
    r"(?:^|[-_/])i[-_]?tber(?:$|[-_./])", re.IGNORECASE
)
SOURCE_TEMPLATE = "REPLACE_WITH_VERIFIED_40_CHARACTER_COMMIT_SHA"

REQUIRED_FILES = (
    "src/iber_protocol.py",
    "src/iber_sampling.py",
    "src/iber_head.py",
    "src/rtdetr_iber.py",
    "src/iber_cache.py",
    "src/iber_probe.py",
    "src/iber_evaluation.py",
    "src/iber_publication.py",
    "scripts/run_iber_canary.py",
    "scripts/evaluate_iber_stock.py",
    "scripts/cache_iber_evidence.py",
    "scripts/run_iber_probe.py",
    "scripts/train_iber.py",
    "scripts/evaluate_iber.py",
    "scripts/publish_iber_epoch.py",
    "scripts/restore_iber_checkpoint.py",
    "scripts/benchmark_iber.py",
    "scripts/run_iber_pipeline.py",
    "deploy/iber/__init__.py",
    "deploy/iber/verify_bundle.py",
    "deploy/iber/verify_host.sh",
    "deploy/iber/build_wheelhouse.sh",
    "deploy/iber/bootstrap_ubuntu.sh",
    "deploy/iber/artifact-manifest.template.json",
    "deploy/iber/publication-screen.template.json",
    "docs/IBER_BE_SERVER_GUIDE.md",
)

GUIDE_MARKERS = (
    "SSH host key",
    "mirror-first",
    "immutable source checkout",
    "chmod 600",
    "passed_with_runtime_amendment",
    "run_iber_canary.py",
    "run_iber_pipeline.py",
    "pipeline-state.json",
    "nvidia-smi",
    "publish_iber_epoch.py",
    "restore_iber_checkpoint.py",
    "engineering_invalid",
    "scientific_failed",
    "Gate-1",
    "禁止绕过 Gate-1",
    "http.version=HTTP/1.1",
    "git-http.env",
    "GIT_CONFIG_COUNT",
)


def _source_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip().lower()
    return commit if COMMIT_PATTERN.fullmatch(commit) else None


def _load_json(path: Path, invalid: dict[str, Any], key: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        invalid[key] = str(error)
        return None
    if not isinstance(payload, dict):
        invalid[key] = {"expected": "object", "actual": type(payload).__name__}
        return None
    return payload


def _manifest_invalid(payload: dict[str, Any]) -> dict[str, Any]:
    actual = {
        "format_version": payload.get("format_version"),
        "design_version": payload.get("design_version"),
        "source_commit": payload.get("source_commit"),
        "baseline_sha256": str(payload.get("baseline_sha256", "")).upper(),
        "dataset_sha256": str(payload.get("dataset", {}).get("sha256", "")).upper()
        if isinstance(payload.get("dataset"), dict)
        else None,
        "subset_sha256": str(payload.get("subset", {}).get("sha256", "")).upper()
        if isinstance(payload.get("subset"), dict)
        else None,
        "execution_environment": payload.get("execution_environment"),
    }
    expected = {
        "format_version": 1,
        "design_version": DESIGN_VERSION,
        "source_commit": SOURCE_TEMPLATE,
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "subset_sha256": EXPECTED_SUBSET_SHA256,
        "execution_environment": execution_environment(),
    }
    return {
        name: {"expected": expected_value, "actual": actual[name]}
        for name, expected_value in expected.items()
        if actual[name] != expected_value
    }


def _publication_invalid(payload: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "format_version": 1,
        "design_version": DESIGN_VERSION,
        "stage": "screen",
        "probe": "b3",
        "seed": 0,
        "expected_private_epochs": 30,
        "source_commit": SOURCE_TEMPLATE,
        "results_branch": "iber-be-v1-results",
        "tag": "iber-be-v1-rtdetr-l-live",
        "asset_prefix": "iber-be-v1.0-screen-seed0-b3",
        "token_file": "/data/uav/HANDOFFS/secrets/github_token",
        "retain": 3,
    }
    violations = {
        name: {"expected": expected_value, "actual": payload.get(name)}
        for name, expected_value in expected.items()
        if payload.get(name) != expected_value
    }
    identity_fields = (
        "design_version",
        "results_branch",
        "tag",
        "asset_prefix",
        "run_name",
        "results_repo",
        "gate1_decision",
    )
    foreign = {
        name: payload.get(name)
        for name in identity_fields
        if isinstance(payload.get(name), str)
        and FOREIGN_PATH_PATTERN.search(str(payload[name]))
    }
    if foreign:
        violations["foreign_identity"] = foreign
    results_repo = str(payload.get("results_repo", ""))
    gate1_decision = str(payload.get("gate1_decision", ""))
    if not results_repo.startswith("/data/uav/results/iber-be-v1-"):
        violations["results_repo"] = {"actual": results_repo}
    if not gate1_decision.startswith("/data/uav/runs/iber-be-v1/"):
        violations["gate1_decision"] = {"actual": gate1_decision}
    return violations


def audit_local_readiness(
    root: str | Path,
    *,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Return local evidence while every remote field stays unresolved."""
    repository = Path(root).resolve()
    missing = [relative for relative in REQUIRED_FILES if not (repository / relative).is_file()]
    invalid: dict[str, Any] = {}

    commit = (source_commit or _source_commit(repository) or "").lower()
    if COMMIT_PATTERN.fullmatch(commit) is None:
        invalid["source_commit"] = {"expected": "40 lowercase hex", "actual": commit or None}

    manifest_path = repository / "deploy/iber/artifact-manifest.template.json"
    if manifest_path.is_file():
        manifest = _load_json(manifest_path, invalid, "artifact_manifest")
        if manifest is not None:
            violations = _manifest_invalid(manifest)
            if violations:
                invalid["artifact_manifest_authority"] = violations

    publication_path = repository / "deploy/iber/publication-screen.template.json"
    if publication_path.is_file():
        publication = _load_json(publication_path, invalid, "publication_template")
        if publication is not None:
            violations = _publication_invalid(publication)
            if violations:
                invalid["publication_identity"] = violations

    guide_path = repository / "docs/IBER_BE_SERVER_GUIDE.md"
    if guide_path.is_file():
        guide = guide_path.read_text(encoding="utf-8")
        absent = [marker for marker in GUIDE_MARKERS if marker not in guide]
        if absent:
            invalid["guide_markers"] = absent

    for relative in (
        "deploy/iber/bootstrap_ubuntu.sh",
        "deploy/iber/build_wheelhouse.sh",
        "deploy/iber/verify_host.sh",
        "deploy/iber/artifact-manifest.template.json",
        "deploy/iber/publication-screen.template.json",
        "docs/IBER_BE_SERVER_GUIDE.md",
    ):
        path = repository / relative
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        exposed = [marker for marker in ("github_pat_", "a1314520") if marker in content]
        if exposed:
            invalid[f"embedded_credentials.{relative}"] = exposed

    ready = not missing and not invalid
    return {
        "status": "ready_waiting_for_server" if ready else "not_ready",
        "design_version": DESIGN_VERSION,
        "local": {
            "ready": ready,
            "root": str(repository),
            "source_commit": commit or None,
            "baseline_sha256": EXPECTED_BASELINE_SHA256,
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "subset_sha256": EXPECTED_SUBSET_SHA256,
            "execution_environment": execution_environment(),
            "required_files": list(REQUIRED_FILES),
            "missing": missing,
            "invalid": invalid,
            "minimum_disk_gib": 80,
            "secret_policy": "mode-600-token-file-never-config-log-or-git-remote",
        },
        "remote": {
            "endpoint": "unresolved",
            "host_key": "unresolved",
            "os": "unresolved",
            "gpu": "unresolved",
            "driver": "unresolved",
            "reported_memory_mib": "unresolved",
            "disk": "unresolved",
            "network": "unresolved",
        },
        "remote_verified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--source-commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_local_readiness(args.root, source_commit=args.source_commit)
    serialized = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if report["local"]["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
