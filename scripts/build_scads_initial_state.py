"""Build one immutable byte-paired FDR/SCADS seed0 initial state."""

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

from src.rtdetr_fdr import FDRRTDETRDetectionModel  # noqa: E402
from src.rtdetr_scads import SCADSFDRRTDETRDetectionModel  # noqa: E402
from src.scads_protocol import build_scads_initial_state  # noqa: E402


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_artifact(*, source_commit: str) -> dict:
    if len(source_commit) != 40 or any(character not in "0123456789abcdef" for character in source_commit.lower()):
        raise ValueError("source_commit must be a 40-character hexadecimal commit")
    public_rng = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        fdr = FDRRTDETRDetectionModel(
            ROOT / "configs" / "rtdetr-l-fdr.yaml",
            nc=10,
            verbose=False,
            private_seed=10_000,
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        scads = SCADSFDRRTDETRDetectionModel(
            ROOT / "configs" / "rtdetr-l-fdr-scads.yaml",
            nc=10,
            verbose=False,
            private_seed=10_000,
            support_private_seed=20_000,
        )
    if not torch.equal(public_rng, torch.random.get_rng_state()):
        raise RuntimeError("paired model construction advanced public CPU RNG")
    return build_scads_initial_state(
        fdr.state_dict(),
        scads.state_dict(),
        metadata={
            "source_commit": source_commit.lower(),
            "seed": 0,
            "fdr_private_seed": 10_000,
            "scads_private_seed": 20_000,
            "nc": 10,
        },
    )


def write_artifact(output: Path, artifact: dict) -> dict:
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            torch.save(artifact, stream)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    summary = {
        "path": str(destination),
        "size": destination.stat().st_size,
        "sha256": _file_sha256(destination),
        "fingerprints": artifact["fingerprints"],
        "common_tensor_count": len(artifact["common_state"]),
        "scads_private_tensor_count": len(artifact["scads_private_state"]),
    }
    summary_path = destination.with_suffix(destination.suffix + ".json")
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the paired SCADS initial state.")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = write_artifact(
        args.output,
        build_artifact(source_commit=args.source_commit),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
