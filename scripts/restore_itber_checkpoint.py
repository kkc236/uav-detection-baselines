"""Restore the newest complete and verified I-TBER private checkpoint pair."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.github_checkpoint_sync import github_session  # noqa: E402
from src.itber_protocol import EXPECTED_BASELINE_SHA256, EXPECTED_DATASET_SHA256  # noqa: E402
from src.itber_publication import (  # noqa: E402
    DEFAULT_TAG,
    PublicationIdentity,
    private_checkpoint_metadata,
    read_token_file,
)
from src.lpr_protocol import file_sha256  # noqa: E402


def select_latest_pair(
    assets: Iterable[dict[str, Any]],
    *,
    prefix: str,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Select the highest complete pair under one exact asset prefix."""
    pattern = re.compile(rf"^{re.escape(prefix)}-epoch-(\d+)\.(pt|json)$")
    pairs: dict[int, dict[str, dict[str, Any]]] = {}
    for asset in assets:
        match = pattern.match(str(asset.get("name", "")))
        if match:
            pairs.setdefault(int(match.group(1)), {})[match.group(2)] = asset
    complete = [(epoch, pair) for epoch, pair in pairs.items() if {"pt", "json"} <= pair.keys()]
    if not complete:
        raise FileNotFoundError(f"no complete I-TBER pair with prefix {prefix!r}")
    epoch, pair = max(complete, key=lambda item: item[0])
    return pair["pt"], pair["json"], epoch


def _download_asset(session: Any, asset: Mapping[str, Any], destination: Path) -> None:
    with session.get(
        str(asset["url"]),
        headers={"Accept": "application/octet-stream"},
        stream=True,
        timeout=(30, 3600),
    ) as response:
        response.raise_for_status()
        with destination.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    stream.write(chunk)
    if destination.stat().st_size != int(asset["size"]):
        raise RuntimeError("downloaded I-TBER asset size does not match GitHub metadata")


def verify_downloaded_checkpoint(
    path: str | Path,
    manifest: dict[str, Any],
    *,
    expected_epoch: int,
    expected_stage: str,
    expected_probe: str,
    expected_prefix: str,
    expected_cache_sha256: str,
):
    """Validate bytes, hash, resumability, and every scientific identity field."""
    identity = PublicationIdentity(
        design_version="itber-v1.1",
        stage=expected_stage,
        probe=expected_probe,
        seed=0,
        baseline_sha256=EXPECTED_BASELINE_SHA256,
        dataset_sha256=EXPECTED_DATASET_SHA256,
        cache_manifest_sha256=expected_cache_sha256,
    )
    for name, value in {"format_version": 1, **identity.as_dict()}.items():
        actual = manifest.get(name)
        if isinstance(actual, str) and name.endswith("sha256"):
            actual = actual.upper()
        if actual != value:
            label = name.replace("_sha256", "")
            raise RuntimeError(f"downloaded I-TBER {label} identity mismatch")
    metadata = private_checkpoint_metadata(path, identity=identity)
    checkpoint = manifest.get("checkpoint", {})
    if metadata.completed_epoch != expected_epoch or int(manifest.get("completed_epoch", -1)) != expected_epoch:
        raise RuntimeError("downloaded I-TBER checkpoint epoch mismatch")
    if metadata.bytes != int(checkpoint.get("bytes", -1)):
        raise RuntimeError("downloaded I-TBER checkpoint bytes mismatch")
    if metadata.sha256 != str(checkpoint.get("sha256", "")):
        raise RuntimeError("downloaded I-TBER checkpoint SHA-256 mismatch")
    if checkpoint.get("asset_name") != f"{expected_prefix}-epoch-{expected_epoch:04d}.pt":
        raise RuntimeError("downloaded I-TBER checkpoint asset prefix mismatch")
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "formal"), required=True)
    parser.add_argument("--probe", choices=("p3",), default="p3")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--repo", default="kkc236/uav-detection-baselines")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--asset-prefix", required=True)
    return parser


def restore_latest_checkpoint(args: argparse.Namespace) -> Path:
    expected_prefix = f"itber-v1.1-{args.stage}-seed0-{args.probe}"
    if args.asset_prefix != expected_prefix:
        raise ValueError(f"I-TBER asset prefix must be exactly {expected_prefix!r}")
    cache_sha = file_sha256(args.cache_manifest)
    session = github_session(read_token_file(args.token_file))
    response = session.get(
        f"https://api.github.com/repos/{args.repo}/releases/tags/{args.tag}", timeout=30
    )
    response.raise_for_status()
    checkpoint_asset, manifest_asset, epoch = select_latest_pair(
        response.json().get("assets", []), prefix=args.asset_prefix
    )
    checkpoint_root = args.run_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    temporary_manifest = checkpoint_root / ".restore-itber-manifest.json.tmp"
    temporary_checkpoint = checkpoint_root / ".restore-itber-checkpoint.pt.tmp"
    try:
        _download_asset(session, manifest_asset, temporary_manifest)
        manifest = json.loads(temporary_manifest.read_text(encoding="utf-8"))
        _download_asset(session, checkpoint_asset, temporary_checkpoint)
        metadata = verify_downloaded_checkpoint(
            temporary_checkpoint,
            manifest,
            expected_epoch=epoch,
            expected_stage=args.stage,
            expected_probe=args.probe,
            expected_prefix=args.asset_prefix,
            expected_cache_sha256=cache_sha,
        )
        destination = checkpoint_root / f"epoch-{epoch:04d}.pt"
        if destination.exists():
            existing = private_checkpoint_metadata(
                destination,
                identity=PublicationIdentity(
                    design_version="itber-v1.1",
                    stage=args.stage,
                    probe=args.probe,
                    seed=0,
                    baseline_sha256=EXPECTED_BASELINE_SHA256,
                    dataset_sha256=EXPECTED_DATASET_SHA256,
                    cache_manifest_sha256=cache_sha,
                ),
            )
            if existing.sha256 != metadata.sha256:
                raise FileExistsError(f"refusing to replace changed I-TBER checkpoint: {destination}")
            temporary_checkpoint.unlink()
        else:
            os.replace(temporary_checkpoint, destination)
        print(
            json.dumps(
                {
                    "checkpoint": str(destination.resolve()),
                    "completed_epoch": epoch,
                    "sha256": metadata.sha256,
                    "stage": args.stage,
                    "probe": args.probe,
                    "seed": 0,
                },
                sort_keys=True,
            )
        )
        return destination.resolve()
    finally:
        temporary_manifest.unlink(missing_ok=True)
        temporary_checkpoint.unlink(missing_ok=True)


def main() -> int:
    args = _build_parser().parse_args()
    args.run_dir = args.run_dir.resolve()
    args.cache_manifest = args.cache_manifest.resolve()
    args.token_file = args.token_file.resolve()
    restore_latest_checkpoint(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
