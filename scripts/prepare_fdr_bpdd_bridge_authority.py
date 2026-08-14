"""Create B authority only after validating immutable A/C/D run manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fdr_bpdd_bridge_protocol import (  # noqa: E402
    BRIDGE_INITIAL_STATE_SHA256,
    BRIDGE_PROTOCOL,
    BRIDGE_PROTOCOL_SHA256,
    BRIDGE_STAGES,
    EXPECTED_GPU_UUID,
    SHARED_TRAINING,
    build_bridge_run_identity,
    file_sha256,
    validate_reference_snapshots,
)
from src.fdr_protocol import canonical_json_bytes, public_state_sha256  # noqa: E402
from src.lpr_protocol import dataset_signature  # noqa: E402
from src.ra_experiment_protocol import (  # noqa: E402
    current_source_identity,
    ignore_sidecar_signature,
)
from src.ra_glgm_protocol import validate_ra_glgm_initial_state  # noqa: E402


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_create_only(path: Path, payload: dict) -> None:
    data = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _validate_args(path: Path, *, label: str) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    mismatched = {
        key: (payload.get(key), expected)
        for key, expected in SHARED_TRAINING.items()
        if payload.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"{label} shared training arguments differ: {mismatched}")
    return {"path": str(path), "sha256": file_sha256(path)}


def _validate_source_extension() -> list[str]:
    subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", "5926ac7", "HEAD"],
        check=True,
    )
    result = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--name-status",
            "5926ac7",
            "HEAD",
            "--",
            "src",
            "scripts",
            "configs",
            "tests",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line]
    allowed_tokens = ("bpdd-bridge", "bpdd_bridge", "FDR_BPDD_BRIDGE")
    unexpected = [
        line for line in lines if not any(token in line for token in allowed_tokens)
    ]
    if unexpected:
        raise ValueError(f"bridge source changes non-bridge files: {unexpected}")
    if not lines or any(not line.startswith("A\t") for line in lines):
        raise ValueError("bridge source must add isolated files only")
    return lines


def prepare(args: argparse.Namespace) -> dict:
    source = current_source_identity(ROOT, require_clean=True)
    if source != {
        "git_commit": args.source_commit.lower(),
        "tree_sha256": args.source_tree_sha256.upper(),
    }:
        raise ValueError("supplied source differs from clean bridge checkout")
    source_diff = _validate_source_extension()
    if args.gpu_uuid != EXPECTED_GPU_UUID:
        raise ValueError("bridge GPU differs from A/C/D")

    initial = args.initial_state.resolve()
    if initial.is_symlink() or not initial.is_file():
        raise FileNotFoundError(initial)
    if file_sha256(initial) != BRIDGE_INITIAL_STATE_SHA256:
        raise ValueError("bridge initial-state SHA256 differs")
    artifact = torch.load(initial, map_location="cpu", weights_only=False)
    validate_ra_glgm_initial_state(artifact)
    if artifact.get("metadata", {}).get("initialization") != "fresh_scratch":
        raise ValueError("bridge initial state is not fresh scratch")

    a = _read_json(args.a_run.resolve())
    c = _read_json(args.c_run.resolve())
    d = _read_json(args.d_run.resolve())
    references = validate_reference_snapshots(a, c, d)
    reference_files = {
        "A": {"path": str(args.a_run.resolve()), "sha256": file_sha256(args.a_run)},
        "C": {"path": str(args.c_run.resolve()), "sha256": file_sha256(args.c_run)},
        "D": {"path": str(args.d_run.resolve()), "sha256": file_sha256(args.d_run)},
    }
    argument_files = {
        "A": _validate_args(args.a_args.resolve(), label="A"),
        "C": _validate_args(args.c_args.resolve(), label="C"),
        "D": _validate_args(args.d_args.resolve(), label="D"),
    }

    root = args.dataset_root.resolve()
    positive = dataset_signature(root)
    ignore = ignore_sidecar_signature(root)
    if positive["sha256"] != BRIDGE_PROTOCOL["dataset"]["sha256"]:
        raise ValueError("bridge positive dataset differs")
    if ignore["sha256"] != BRIDGE_PROTOCOL["dataset"]["ignore_sha256"]:
        raise ValueError("bridge ignore dataset differs")

    identities = {
        stage: build_bridge_run_identity(source, stage=stage, gpu_uuid=args.gpu_uuid)
        for stage in BRIDGE_STAGES
    }
    payload = {
        "format_version": 1,
        "source": source,
        "source_sha256": public_state_sha256(source),
        "source_extension_from_combo": source_diff,
        "protocol": BRIDGE_PROTOCOL,
        "protocol_sha256": BRIDGE_PROTOCOL_SHA256,
        "gpu_uuid": args.gpu_uuid,
        "initial_state": {
            "path": str(initial),
            "sha256": file_sha256(initial),
            "fingerprints": artifact["fingerprints"],
        },
        "dataset_authority": {
            "root": str(root),
            "positive": positive,
            "ignore": ignore,
        },
        "reference_snapshots": references,
        "reference_files": reference_files,
        "reference_argument_files": argument_files,
        "run_identities": identities,
    }
    payload["manifest_sha256"] = (
        hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()
    )
    _write_create_only(args.output.resolve(), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--a-run", type=Path, required=True)
    parser.add_argument("--c-run", type=Path, required=True)
    parser.add_argument("--d-run", type=Path, required=True)
    parser.add_argument("--a-args", type=Path, required=True)
    parser.add_argument("--c-args", type=Path, required=True)
    parser.add_argument("--d-args", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
