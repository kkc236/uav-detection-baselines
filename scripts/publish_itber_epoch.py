"""Publish one complete I-TBER epoch from a credential-free JSON config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.train_itber import validate_gate1_cache_manifest  # noqa: E402
from src.itber_protocol import EXPECTED_BASELINE_SHA256, EXPECTED_DATASET_SHA256  # noqa: E402
from src.itber_publication import (  # noqa: E402
    PublicationConfig,
    PublicationIdentity,
    publish_with_retry,
)


CONFIG_KEYS = {
    "format_version",
    "design_version",
    "stage",
    "probe",
    "seed",
    "expected_private_epochs",
    "repo",
    "repo_url",
    "source_branch",
    "results_branch",
    "tag",
    "asset_prefix",
    "run_name",
    "token_file",
    "results_repo",
    "gate1_cache_manifest",
    "retain",
}


def load_publication_config(path: str | Path) -> PublicationConfig:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if set(payload) != CONFIG_KEYS:
        raise ValueError(
            f"I-TBER publication config keys mismatch: "
            f"missing={sorted(CONFIG_KEYS - set(payload))}, "
            f"unexpected={sorted(set(payload) - CONFIG_KEYS)}"
        )
    if payload["format_version"] != 1 or payload["design_version"] != "itber-v1.1":
        raise ValueError("I-TBER publication config format/design mismatch")
    expected_epochs = {"screen": 12, "formal": 30}
    if payload["stage"] not in expected_epochs or payload["expected_private_epochs"] != expected_epochs[payload["stage"]]:
        raise ValueError("I-TBER publication stage/epoch contract mismatch")
    cache_manifest = Path(payload["gate1_cache_manifest"]).resolve()
    cache_sha = validate_gate1_cache_manifest(cache_manifest)
    identity = PublicationIdentity(
        design_version="itber-v1.1",
        stage=str(payload["stage"]),
        probe=str(payload["probe"]),
        seed=int(payload["seed"]),
        baseline_sha256=EXPECTED_BASELINE_SHA256,
        dataset_sha256=EXPECTED_DATASET_SHA256,
        cache_manifest_sha256=cache_sha,
    )
    return PublicationConfig(
        repo=str(payload["repo"]),
        repo_url=str(payload["repo_url"]),
        source_branch=str(payload["source_branch"]),
        results_branch=str(payload["results_branch"]),
        tag=str(payload["tag"]),
        asset_prefix=str(payload["asset_prefix"]),
        run_name=str(payload["run_name"]),
        token_file=Path(payload["token_file"]).resolve(),
        results_repo=Path(payload["results_repo"]).resolve(),
        identity=identity,
        retain=int(payload["retain"]),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_publication_config(args.config)
    record = publish_with_retry(
        args.run_dir.resolve(),
        args.checkpoint.resolve(),
        config,
    )
    print(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
