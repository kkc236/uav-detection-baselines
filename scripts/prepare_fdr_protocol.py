"""Create the immutable FDR-only protocol manifest from committed authorities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fdr_protocol import (  # noqa: E402
    FDR_PROTOCOL,
    FDR_PROTOCOL_SHA256,
    build_run_identity,
    canonical_json_bytes,
    public_state_sha256,
    validate_fdr_initial_state,
    write_create_only_manifest,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _validate_hex(value: str, *, length: int, name: str) -> str:
    normalized = value.lower()
    if len(normalized) != length or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{name} must be exactly {length} hexadecimal characters")
    return normalized


def prepare_manifest(
    *,
    source_commit: str,
    source_tree_sha256: str,
    initial_state: Path,
    output: Path,
) -> dict:
    """Bind source, protocol, run IDs, and a prebuilt paired initialization artifact."""
    source = {
        "git_commit": _validate_hex(source_commit, length=40, name="source_commit"),
        "tree_sha256": _validate_hex(source_tree_sha256, length=64, name="source_tree_sha256").upper(),
    }
    state_path = Path(initial_state).resolve()
    if state_path.is_symlink() or not state_path.is_file():
        raise ValueError("initial_state must be a regular existing file")
    artifact = torch.load(state_path, map_location="cpu", weights_only=False)
    validate_fdr_initial_state(artifact)
    fingerprints = artifact.get("fingerprints", {})

    run_identities = {
        f"{variant}_{stage}": build_run_identity(source, stage=stage, variant=variant, seed=0)
        for stage in ("screen", "formal")
        for variant in ("control", "fdr")
    }
    manifest = {
        "format_version": 1,
        "source": source,
        "source_sha256": public_state_sha256(source),
        "protocol": FDR_PROTOCOL,
        "protocol_sha256": FDR_PROTOCOL_SHA256,
        "migration": artifact["migration"],
        "initial_state": {
            "path": str(state_path),
            "sha256": _file_sha256(state_path),
            "fingerprints": fingerprints,
        },
        "run_identities": run_identities,
    }
    manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest().upper()
    write_create_only_manifest(output, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare immutable FDR-only paired protocol authority.")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_manifest(
        source_commit=args.source_commit,
        source_tree_sha256=args.source_tree_sha256,
        initial_state=args.initial_state,
        output=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
