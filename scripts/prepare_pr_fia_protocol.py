"""Create the immutable manifest for paired PR-FIA screens and Formal100."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fdr_protocol import validate_fdr_initial_state  # noqa: E402
from src.pr_fia_protocol import (  # noqa: E402
    FDR_INITIAL_STATE_SHA256,
    PR_FIA_PROTOCOL,
    PR_FIA_PROTOCOL_SHA256,
    build_run_identity,
    public_state_sha256,
    write_create_only_manifest,
)


SCREEN_VARIANTS = (
    "fdr_bpdd",
    "fdr_bpdd_pr_fia",
    "fdr",
    "fdr_pr_fia",
)
FORMAL_VARIANT = "fdr_bpdd_pr_fia"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _validate_hex(value: str, *, length: int, name: str) -> str:
    normalized = value.lower()
    if len(normalized) != length or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be exactly {length} hexadecimal characters")
    return normalized


def _validate_initial_state(path: Path) -> dict:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    validate_fdr_initial_state(artifact)
    return artifact


def prepare_manifest(
    *,
    source_commit: str,
    source_tree_sha256: str,
    initial_state: Path,
    output: Path,
) -> dict:
    """Bind source, state, protocol, four screens, and one formal run."""

    state_input = Path(initial_state)
    if not state_input.is_file() or state_input.is_symlink():
        raise FileNotFoundError(f"FDR initial state not found: {state_input}")
    initial_state = state_input.resolve()
    state_sha256 = _file_sha256(initial_state)
    if state_sha256 != FDR_INITIAL_STATE_SHA256:
        raise ValueError(
            "PR-FIA initial-state SHA256 mismatch: "
            f"expected={FDR_INITIAL_STATE_SHA256}, actual={state_sha256}"
        )
    artifact = _validate_initial_state(initial_state)
    fingerprints = artifact.get("fingerprints")
    if not isinstance(fingerprints, dict):
        raise ValueError("FDR initial-state fingerprints are missing")

    source = {
        "git_commit": _validate_hex(
            source_commit, length=40, name="source_commit"
        ),
        "tree_sha256": _validate_hex(
            source_tree_sha256, length=64, name="source_tree_sha256"
        ).upper(),
    }
    identities = {
        f"{variant}_screen": build_run_identity(
            source, stage="screen", variant=variant, seed=0
        )
        for variant in SCREEN_VARIANTS
    }
    identities[f"{FORMAL_VARIANT}_formal"] = build_run_identity(
        source, stage="formal", variant=FORMAL_VARIANT, seed=0
    )
    manifest = {
        "format_version": 1,
        "source": source,
        "source_sha256": public_state_sha256(source),
        "protocol": PR_FIA_PROTOCOL,
        "protocol_sha256": PR_FIA_PROTOCOL_SHA256,
        "initial_state": {
            "path": str(initial_state),
            "sha256": state_sha256,
            "fingerprints": dict(fingerprints),
        },
        "run_identities": identities,
    }
    manifest["manifest_sha256"] = public_state_sha256(manifest)
    write_create_only_manifest(Path(output).resolve(), manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_manifest(
        source_commit=args.source_commit,
        source_tree_sha256=args.source_tree_sha256,
        initial_state=args.initial_state,
        output=args.output,
    )


if __name__ == "__main__":
    main()
