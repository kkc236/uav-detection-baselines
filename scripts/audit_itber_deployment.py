"""Audit local I-TBER deployment readiness without claiming remote state."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.itber_protocol import (
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
)


REQUIRED_FILES = (
    "requirements-itber.lock",
    "src/itber_geometry.py",
    "src/itber_sampling.py",
    "src/itber_head.py",
    "src/itber_loss.py",
    "src/rtdetr_itber.py",
    "src/itber_protocol.py",
    "scripts/run_itber_canary.py",
    "deploy/itber/verify_bundle.py",
    "deploy/itber/verify_host.sh",
    "deploy/itber/build_wheelhouse.sh",
    "deploy/itber/bootstrap_ubuntu.sh",
    "deploy/itber/artifact-manifest.template.json",
    "deploy/itber/publication.env.template",
    "docs/ITBER_BARE_SERVER_GUIDE.md",
)

GUIDE_MARKERS = (
    "SSH host key",
    "至少 80 GiB",
    "run_itber_canary.py",
    "restore_itber_checkpoint.py",
    "每个 epoch",
    "chmod 600",
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
    commit = result.stdout.strip()
    return commit if len(commit) == 40 else None


def audit_local_readiness(root: str | Path) -> dict[str, Any]:
    """Return only local evidence; all remote fields stay explicitly unresolved."""
    root = Path(root).resolve()
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    invalid: dict[str, Any] = {}
    commit = _source_commit(root)
    if commit is None:
        invalid["source_commit"] = "missing"

    manifest_path = root / "deploy/itber/artifact-manifest.template.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            invalid["artifact_manifest"] = str(error)
        else:
            expected = {
                "baseline_sha256": EXPECTED_BASELINE_SHA256,
                "dataset_sha256": EXPECTED_DATASET_SHA256,
                "subset_sha256": EXPECTED_SUBSET_SHA256,
            }
            actual = {
                "baseline_sha256": str(manifest.get("baseline_sha256", "")).upper(),
                "dataset_sha256": str(manifest.get("dataset", {}).get("sha256", "")).upper(),
                "subset_sha256": str(manifest.get("subset", {}).get("sha256", "")).upper(),
            }
            if actual != expected:
                invalid["artifact_manifest_authority"] = {
                    "expected": expected,
                    "actual": actual,
                }

    publication_path = root / "deploy/itber/publication.env.template"
    if publication_path.is_file():
        publication = publication_path.read_text(encoding="utf-8")
        for setting in (
            "ITBER_GITHUB_REPOSITORY=",
            "ITBER_GITHUB_TOKEN_FILE=/data/uav/HANDOFFS/secrets/github_token",
            "ITBER_REQUIRE_EVERY_EPOCH_PUBLICATION=1",
        ):
            if setting not in publication:
                invalid.setdefault("publication_template", []).append(setting)

    guide_path = root / "docs/ITBER_BARE_SERVER_GUIDE.md"
    if guide_path.is_file():
        guide = guide_path.read_text(encoding="utf-8")
        absent_markers = [marker for marker in GUIDE_MARKERS if marker not in guide]
        if absent_markers:
            invalid["guide_markers"] = absent_markers

    ready = not missing and not invalid
    return {
        "status": "ready_waiting_for_server" if ready else "not_ready",
        "local": {
            "ready": ready,
            "repository_root": str(root),
            "source_commit": commit,
            "required_files": list(REQUIRED_FILES),
            "missing": missing,
            "invalid": invalid,
            "disk_budget_gib": 80,
            "secret_policy": "token-file-mode-600-no-secret-in-repository-or-log",
        },
        "remote": {
            "endpoint": "unresolved",
            "host_key": "unresolved",
            "os": "unresolved",
            "gpu": "unresolved",
            "driver": "unresolved",
            "disk": "unresolved",
            "network": "unresolved",
        },
        "remote_verified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_local_readiness(args.root)
    serialized = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if report["local"]["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
