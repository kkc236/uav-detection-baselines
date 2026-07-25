"""Independent source closure for SADED endpoint caching and evaluation."""

from __future__ import annotations

from pathlib import Path
import subprocess

from src.sbr_artifacts import git_provenance
from src.tascv_protocol import (
    require_clean_repo,
    sha256_file,
    source_bundle_sha256,
)


SADED_STAGE_SOURCE_FILES = (
    "scripts/adjudicate_saded_stage.py",
    "scripts/cache_saded_endpoint.py",
    "scripts/evaluate_saded_confirmation_once.py",
    "scripts/evaluate_saded_stage.py",
    "scripts/route_saded_pair.py",
    "scripts/seal_saded_confirmation_predictions.py",
    "src/saded.py",
    "src/saded_adjudicator.py",
    "src/saded_confirmation.py",
    "src/saded_stage.py",
    "src/saded_stage_protocol.py",
    "src/ascv_loc.py",
    "src/ascv_loc_protocol.py",
    "src/rtdetr_tascv.py",
    "src/sbr_artifacts.py",
    "src/sbr_fusion.py",
    "src/sbr_g0.py",
    "src/sbr_geometry.py",
    "src/sbr_metrics.py",
    "src/sbr_ppaf.py",
    "src/sbr_v2_audit.py",
    "src/tascv.py",
    "src/tascv_adjudicator.py",
    "src/tascv_cli.py",
    "src/tascv_diagnostics.py",
    "src/tascv_protocol.py",
    "src/tascv_stage.py",
)


def stage_source_hashes(repo_root: Path) -> dict[str, str]:
    root = Path(repo_root).resolve()
    return {
        relative: sha256_file(root / relative)
        for relative in SADED_STAGE_SOURCE_FILES
    }


def stage_source_state(repo_root: Path) -> dict:
    root = Path(repo_root).resolve()
    require_clean_repo(root)
    provenance = git_provenance(root)
    if (
        provenance.get("clean_tracked") is not True
        or provenance.get("untracked") is not False
    ):
        raise ValueError("SADED stage source is not clean")
    files = stage_source_hashes(root)
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "commit": provenance["commit"],
        "tree": tree,
        "clean_tracked": True,
        "untracked": False,
        "files": files,
        "bundle_sha256": source_bundle_sha256(files),
    }


__all__ = [
    "SADED_STAGE_SOURCE_FILES",
    "stage_source_hashes",
    "stage_source_state",
]
