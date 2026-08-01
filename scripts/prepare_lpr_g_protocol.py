"""Create immutable seed0 data and initialization artifacts for LPR-G v2."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_lpr_protocol import prepare_data_files, validate_data_authority
from src.lpr_g_protocol import build_lpr_g_initial_state
from src.lpr_protocol import (
    CATEGORY_NAMES,
    EXPECTED_COMMON_FINGERPRINTS,
    EXPECTED_SOURCE_SHA256,
    category_mapping_sha256,
    current_environment,
    dataset_signature,
    environment_violations,
    file_sha256,
    source_violations,
)
from src.rtdetr_lpr_g import LPRGRTDETRDetectionModel
from ultralytics.nn.tasks import RTDETRDetectionModel


def _write_locked(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to replace changed protocol artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _save_locked_state(path: Path, artifact: dict) -> None:
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=False)
        if existing.get("fingerprints") != artifact.get("fingerprints") or existing.get(
            "metadata"
        ) != artifact.get("metadata"):
            raise FileExistsError(f"refusing to replace changed initial-state artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(artifact, temporary)
    os.replace(temporary, path)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _json_sha256(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return sha256(data).hexdigest().upper()


def create_initial_state_artifact(
    *,
    seed: int,
    nc: int = 10,
    channels: int = 3,
) -> dict:
    """Create exact common stock state plus seed-isolated LPR-G private state."""
    if seed != 0:
        raise ValueError("LPR-G v2 initial state is frozen to seed0")
    torch.manual_seed(seed)
    control = RTDETRDetectionModel("rtdetr-l.yaml", nc=nc, ch=channels, verbose=False)
    torch.manual_seed(seed)
    method = LPRGRTDETRDetectionModel(
        "rtdetr-l.yaml",
        nc=nc,
        ch=channels,
        verbose=False,
        private_seed=10_000,
    )
    return build_lpr_g_initial_state(
        control.state_dict(),
        method.state_dict(),
        seed=seed,
        metadata={
            "seed": seed,
            "innovation_seed": 10_000,
            "control_parameters": sum(parameter.numel() for parameter in control.parameters()),
            "lpr_g_parameters": sum(parameter.numel() for parameter in method.parameters()),
        },
    )


def prepare_protocol(dataset_root: Path, output_dir: Path, *, seed: int) -> dict:
    """Validate every frozen authority and write an immutable format-v2 manifest."""
    if seed != 0:
        raise ValueError("LPR-G v2 protocol is frozen to seed0")
    environment = current_environment()
    violations = environment_violations(environment)
    if violations:
        raise ValueError(f"environment does not match frozen authority: {violations}")
    source_drift = source_violations()
    if source_drift:
        raise ValueError(f"Ultralytics source does not match frozen authority: {source_drift}")

    dataset = dataset_signature(dataset_root)
    data_files = prepare_data_files(dataset_root, output_dir)
    validate_data_authority(dataset, data_files["subset"])

    output_dir = Path(output_dir).resolve()
    state_path = output_dir / "initial-state-seed0.pt"
    artifact = create_initial_state_artifact(seed=seed)
    expected_common = EXPECTED_COMMON_FINGERPRINTS[0]
    if artifact["fingerprints"]["common"] != expected_common:
        raise ValueError(
            "common initial state is not the Linux seed0 authority: "
            f"expected={expected_common}, actual={artifact['fingerprints']['common']}"
        )
    artifact["metadata"].update(
        {
            "dataset": dataset,
            "subset": data_files["subset"],
            "environment": environment,
            "category_mapping_sha256": category_mapping_sha256(CATEGORY_NAMES),
        }
    )
    _save_locked_state(state_path, artifact)

    manifest = {
        "format_version": 2,
        "seed": 0,
        "git_commit": _git_commit(),
        "dataset_root": str(Path(dataset_root).resolve()),
        "dataset": dataset,
        "subset": {
            **data_files["subset"],
            "path": str(data_files["subset_path"]),
            "file_sha256": file_sha256(data_files["subset_path"]),
        },
        "category_mapping_sha256": category_mapping_sha256(CATEGORY_NAMES),
        "environment": environment,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "data": {
            "screen": {
                "path": str(data_files["screen_data_path"]),
                "sha256": file_sha256(data_files["screen_data_path"]),
            },
            "formal": {
                "path": str(data_files["formal_data_path"]),
                "sha256": file_sha256(data_files["formal_data_path"]),
            },
        },
        "initial_state": {
            "path": str(state_path),
            "sha256": file_sha256(state_path),
            "fingerprints": artifact["fingerprints"],
        },
    }
    manifest["signature"] = _json_sha256(manifest)
    manifest_path = output_dir / "protocol-seed0.json"
    _write_locked(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare immutable strict seed0 LPR-G v2 paired-protocol artifacts."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0,), required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(
        json.dumps(
            prepare_protocol(args.dataset_root, args.output_dir, seed=args.seed),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
