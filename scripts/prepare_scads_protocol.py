"""Bind SCADS source, protocol, and paired initial state into one manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fdr_protocol import write_create_only_manifest  # noqa: E402
from src.scads_protocol import (  # noqa: E402
    SCADS_PROTOCOL,
    SCADS_PROTOCOL_SHA256,
    build_run_identity,
    canonical_json_bytes,
    public_state_sha256,
    validate_scads_initial_state,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _validate_hex(value: str, length: int, name: str) -> str:
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
    source = {
        "git_commit": _validate_hex(source_commit, 40, "source_commit"),
        "tree_sha256": _validate_hex(source_tree_sha256, 64, "source_tree_sha256").upper(),
    }
    state_path = Path(initial_state).resolve()
    if state_path.is_symlink() or not state_path.is_file():
        raise ValueError("initial_state must be a regular existing file")
    artifact = torch.load(state_path, map_location="cpu", weights_only=False)
    validate_scads_initial_state(artifact)
    if artifact.get("metadata", {}).get("source_commit") != source["git_commit"]:
        raise ValueError("initial-state source commit does not match manifest source")
    identities = {
        f"{variant}_{stage}": build_run_identity(
            source,
            stage=stage,
            variant=variant,
            seed=0,
        )
        for stage in ("screen", "formal")
        for variant in ("fdr", "scads")
    }
    manifest = {
        "format_version": 1,
        "source": source,
        "source_sha256": public_state_sha256(source),
        "protocol": SCADS_PROTOCOL,
        "protocol_sha256": SCADS_PROTOCOL_SHA256,
        "migration": artifact["migration"],
        "initial_state": {
            "path": str(state_path),
            "sha256": _file_sha256(state_path),
            "fingerprints": artifact["fingerprints"],
        },
        "run_identities": identities,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest().upper()
    write_create_only_manifest(output, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare immutable SCADS protocol authority.")
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
