"""Create the immutable single-arm FDR+BPDD+RA-GLGM authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fdr_bpdd_ra_glgm_protocol import (  # noqa: E402
    BPDD_SOURCE_BLOB,
    COMBO_PROTOCOL,
    COMBO_PROTOCOL_SHA256,
    COMBO_STAGES,
    build_combo_run_identity,
)
from src.fdr_protocol import canonical_json_bytes, public_state_sha256  # noqa: E402
from src.lpr_protocol import dataset_signature  # noqa: E402
from src.ra_experiment_protocol import (  # noqa: E402
    current_source_identity,
    file_sha256,
    ignore_sidecar_signature,
)
from src.ra_glgm_protocol import validate_ra_glgm_initial_state  # noqa: E402


def _write_create_only(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_authority(
    *,
    source_commit: str,
    source_tree_sha256: str,
    gpu_uuid: str,
    initial_state: str | Path,
    dataset_root: str | Path,
    output: str | Path,
) -> dict:
    source = {
        "git_commit": source_commit.lower(),
        "tree_sha256": source_tree_sha256.upper(),
    }
    if source != current_source_identity(ROOT, require_clean=True):
        raise ValueError("supplied source differs from the clean combo checkout")
    if not gpu_uuid.startswith("GPU-") or any(character.isspace() for character in gpu_uuid):
        raise ValueError("gpu_uuid must be one NVIDIA GPU UUID token")
    bpdd_blob = (
        __import__("subprocess")
        .run(
            ["git", "-C", str(ROOT), "hash-object", "src/bpdd_loss.py"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )
    if bpdd_blob != BPDD_SOURCE_BLOB:
        raise ValueError("checked-out BPDD loss differs from locked 848f00cb algorithm")

    state_path = Path(initial_state).resolve()
    if state_path.is_symlink() or not state_path.is_file():
        raise FileNotFoundError("combo initial state is missing")
    artifact = torch.load(state_path, map_location="cpu", weights_only=False)
    validate_ra_glgm_initial_state(artifact)
    metadata = artifact.get("metadata", {})
    if metadata.get("seed") != 0 or metadata.get("initialization") != "fresh_scratch":
        raise ValueError("combo initial state was not built from fresh seed0 scratch")

    data_root = Path(dataset_root).resolve()
    positive = dataset_signature(data_root)
    if positive.get("sha256") != COMBO_PROTOCOL["dataset"]["sha256"]:
        raise ValueError("VisDrone positive dataset differs from combo protocol")
    ignore = ignore_sidecar_signature(data_root)
    expected_ignore = COMBO_PROTOCOL["dataset"]["ignore_sidecar"]
    for split in ("train", "val"):
        actual = ignore["splits"][split]
        if actual["files"] != expected_ignore["files"][split]:
            raise ValueError(f"VisDrone {split} ignore file count differs")
        if actual["boxes"] != expected_ignore["boxes"][split]:
            raise ValueError(f"VisDrone {split} ignore box count differs")

    identities = {
        stage: build_combo_run_identity(source, stage=stage, gpu_uuid=gpu_uuid)
        for stage in COMBO_STAGES
    }
    manifest = {
        "format_version": 1,
        "source": source,
        "source_sha256": public_state_sha256(source),
        "protocol": COMBO_PROTOCOL,
        "protocol_sha256": COMBO_PROTOCOL_SHA256,
        "gpu_uuid": gpu_uuid,
        "initial_state": {
            "path": str(state_path),
            "sha256": file_sha256(state_path),
            "fingerprints": artifact["fingerprints"],
        },
        "dataset_authority": {
            "root": str(data_root),
            "positive": positive,
            "ignore": ignore,
        },
        "run_identities": identities,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest().upper()
    _write_create_only(Path(output), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_authority(
        source_commit=args.source_commit,
        source_tree_sha256=args.source_tree_sha256,
        gpu_uuid=args.gpu_uuid,
        initial_state=args.initial_state,
        dataset_root=args.dataset_root,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
