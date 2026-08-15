"""Publish one fixed IBER-BE v1.0 seed0 B3 screen epoch transaction."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.iber_protocol import (  # noqa: E402
    DESIGN_VERSION,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    PROTOCOL_SHA256,
    RUNTIME_AMENDMENT_SHA256,
    SCREEN_EPOCHS,
    file_sha256,
)
from src.iber_publication import (  # noqa: E402
    ASSET_PREFIX,
    DEFAULT_TAG,
    RESULTS_BRANCH,
    RETAINED_PAIRS,
    PublicationConfig,
    PublicationIdentity,
    publish_with_retry,
)


_CONFIG_FIELDS = frozenset(
    {
        "format_version",
        "design_version",
        "stage",
        "probe",
        "seed",
        "expected_private_epochs",
        "repo",
        "repo_url",
        "source_branch",
        "source_commit",
        "results_branch",
        "tag",
        "asset_prefix",
        "run_name",
        "token_file",
        "results_repo",
        "gate1_decision",
        "retain",
    }
)


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError(f"IBER-BE publication config must have mode 600: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("IBER-BE publication config must be a JSON object")
    unknown = set(payload) - _CONFIG_FIELDS
    missing = _CONFIG_FIELDS - set(payload)
    if unknown:
        raise ValueError(
            "IBER-BE publication config contains forbidden fields: "
            + ", ".join(sorted(unknown))
        )
    if missing:
        raise ValueError(
            "IBER-BE publication config is missing fields: "
            + ", ".join(sorted(missing))
        )
    return payload


def _checkpoint_identity(path: Path) -> PublicationIdentity:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, Mapping):
        raise ValueError("IBER-BE private checkpoint is not a mapping")
    return PublicationIdentity(
        design_version=str(artifact.get("design_version", DESIGN_VERSION)),
        stage=str(artifact.get("stage", "")),
        probe=str(artifact.get("probe", "")),
        seed=artifact.get("seed"),
        baseline_sha256=str(artifact.get("baseline_sha256", EXPECTED_BASELINE_SHA256)),
        dataset_sha256=str(artifact.get("dataset_sha256", EXPECTED_DATASET_SHA256)),
        subset_sha256=str(artifact.get("subset_sha256", EXPECTED_SUBSET_SHA256)),
        category_sha256=str(artifact.get("category_sha256", "")),
        protocol_sha256=str(artifact.get("protocol_sha256", PROTOCOL_SHA256)),
        runtime_amendment_sha256=str(
            artifact.get("runtime_amendment_sha256", RUNTIME_AMENDMENT_SHA256)
        ),
        gate1_decision_sha256=str(artifact.get("gate1_decision_sha256", "")),
        source_commit=str(artifact.get("source_commit", "")),
    )


def load_publication_config(path: Path, checkpoint: Path) -> PublicationConfig:
    payload = _load_mapping(path)
    root = path.parent.resolve()

    def resolved(name: str) -> Path:
        value = Path(str(payload[name]))
        return (value if value.is_absolute() else root / value).resolve()

    identity = _checkpoint_identity(checkpoint)
    frozen = {
        "format_version": 1,
        "design_version": DESIGN_VERSION,
        "stage": "screen",
        "probe": "b3",
        "seed": 0,
        "expected_private_epochs": SCREEN_EPOCHS,
        "results_branch": RESULTS_BRANCH,
        "tag": DEFAULT_TAG,
        "asset_prefix": ASSET_PREFIX,
        "retain": RETAINED_PAIRS,
    }
    for name, expected in frozen.items():
        actual = payload.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"IBER-BE publication config {name} must be exactly {expected!r}"
            )
    source_commit = str(payload["source_commit"]).lower()
    if source_commit != identity.source_commit.lower():
        raise ValueError("IBER-BE publication config source_commit mismatch")
    gate1_decision = resolved("gate1_decision")
    if not gate1_decision.is_file():
        raise FileNotFoundError(gate1_decision)
    if file_sha256(gate1_decision) != identity.gate1_decision_sha256.upper():
        raise ValueError("IBER-BE publication config Gate-1 decision SHA-256 mismatch")

    return PublicationConfig(
        repo=str(payload["repo"]),
        repo_url=str(payload["repo_url"]),
        source_branch=str(payload["source_branch"]),
        run_name=str(payload["run_name"]),
        token_file=resolved("token_file"),
        results_repo=resolved("results_repo"),
        identity=identity,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    config_path = args.config.resolve()
    config = load_publication_config(config_path, checkpoint)
    record = publish_with_retry(run_dir, checkpoint, config)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
