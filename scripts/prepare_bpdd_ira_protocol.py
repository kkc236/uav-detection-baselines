"""Create one immutable FDR + BPDD + IRA formal protocol manifest."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bpdd_ira_protocol import (  # noqa: E402
    BPDD_IRA_PROTOCOL,
    BPDD_IRA_PROTOCOL_SHA256,
    FDR_INITIAL_STATE_SHA256,
    build_run_identity,
    public_state_sha256,
    write_create_only_manifest,
)
from src.fdr_protocol import validate_fdr_initial_state  # noqa: E402


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
    """Bind source, frozen state, protocol, and the sole formal run identity."""

    initial_state = Path(initial_state).resolve()
    if not initial_state.is_file() or initial_state.is_symlink():
        raise FileNotFoundError(f"FDR initial state not found: {initial_state}")
    state_sha256 = _file_sha256(initial_state)
    if state_sha256 != FDR_INITIAL_STATE_SHA256:
        raise ValueError(
            "BPDD+IRA initial-state SHA256 mismatch: "
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
    identity = build_run_identity(
        source, stage="formal", variant="fdr_bpdd_ira", seed=0
    )
    manifest = {
        "format_version": 1,
        "source": source,
        "source_sha256": public_state_sha256(source),
        "protocol": BPDD_IRA_PROTOCOL,
        "protocol_sha256": BPDD_IRA_PROTOCOL_SHA256,
        "initial_state": {
            "path": str(initial_state),
            "sha256": state_sha256,
            "fingerprints": dict(fingerprints),
        },
        "run_identities": {"fdr_bpdd_ira_formal": identity},
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
