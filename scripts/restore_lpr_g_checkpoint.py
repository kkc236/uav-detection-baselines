"""Restore the newest exact, validated LPR-G v2 checkpoint/manifest pair."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sync_experiment_checkpoint import validate_token_file
from src.github_checkpoint_sync import checkpoint_metadata, github_session


def select_latest_pair(
    assets: Iterable[dict[str, Any]],
    *,
    prefix: str,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Select only a complete .pt/.json pair with the exact arm prefix."""
    pattern = re.compile(rf"^{re.escape(prefix)}-epoch-(\d+)\.(pt|json)$")
    pairs: dict[int, dict[str, dict[str, Any]]] = {}
    for asset in assets:
        match = pattern.match(str(asset.get("name", "")))
        if match:
            pairs.setdefault(int(match.group(1)), {})[match.group(2)] = asset
    complete = [(epoch, pair) for epoch, pair in pairs.items() if {"pt", "json"} <= pair.keys()]
    if not complete:
        raise FileNotFoundError(
            f"No matching checkpoint and manifest pair with prefix {prefix!r} was found"
        )
    epoch, pair = max(complete, key=lambda item: item[0])
    return pair["pt"], pair["json"], epoch


def _download_asset(session, asset: dict[str, Any], destination: Path) -> None:
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
        destination.unlink(missing_ok=True)
        raise RuntimeError("Downloaded asset size does not match GitHub metadata")


def verify_downloaded_checkpoint(
    path: Path,
    manifest: dict[str, Any],
    *,
    expected_epoch: int,
    expected_variant: str,
    expected_stage: str,
    expected_prefix: str,
):
    """Validate resumability, integrity, and exact LPR-G scientific identity."""
    metadata = checkpoint_metadata(path)
    checkpoint = manifest.get("checkpoint", {})
    if metadata.completed_epoch != expected_epoch or int(
        manifest.get("completed_epoch", -1)
    ) != expected_epoch:
        raise RuntimeError("Downloaded checkpoint epoch does not match its manifest")
    if metadata.bytes != int(checkpoint.get("bytes", -1)):
        raise RuntimeError("Downloaded checkpoint size does not match its manifest")
    if metadata.sha256 != str(checkpoint.get("sha256", "")):
        raise RuntimeError("Downloaded checkpoint SHA-256 does not match its manifest")
    expected_asset_name = f"{expected_prefix}-epoch-{expected_epoch:04d}.pt"
    if checkpoint.get("asset_name") != expected_asset_name:
        raise RuntimeError("Downloaded checkpoint asset name does not match its arm prefix")
    if manifest.get("design_version") != "lpr-g-v2":
        raise RuntimeError("Downloaded checkpoint design version is not lpr-g-v2")
    if manifest.get("variant") != expected_variant:
        raise RuntimeError("Downloaded checkpoint variant does not match the requested arm")
    if manifest.get("stage") != expected_stage:
        raise RuntimeError("Downloaded checkpoint stage does not match the requested stage")
    if manifest.get("seed") != 0:
        raise RuntimeError("Downloaded checkpoint is not the frozen seed0 arm")
    return metadata


def restore_latest_checkpoint(args: argparse.Namespace) -> Path:
    expected_identity = f"{args.stage}-seed0-{args.variant}"
    if expected_identity not in args.asset_prefix:
        raise ValueError(
            f"asset prefix must isolate the requested arm as {expected_identity!r}"
        )
    token = validate_token_file(args.token_file)
    session = github_session(token)
    response = session.get(
        f"https://api.github.com/repos/{args.repo}/releases/tags/{args.tag}", timeout=30
    )
    response.raise_for_status()
    checkpoint_asset, manifest_asset, expected_epoch = select_latest_pair(
        response.json().get("assets", []), prefix=args.asset_prefix
    )

    weights = args.run_dir / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    temporary_manifest = weights / ".restore-lpr-g-manifest.json.tmp"
    temporary_checkpoint = weights / ".restore-lpr-g-checkpoint.pt.tmp"
    try:
        _download_asset(session, manifest_asset, temporary_manifest)
        manifest = json.loads(temporary_manifest.read_text(encoding="utf-8"))
        _download_asset(session, checkpoint_asset, temporary_checkpoint)
        metadata = verify_downloaded_checkpoint(
            temporary_checkpoint,
            manifest,
            expected_epoch=expected_epoch,
            expected_variant=args.variant,
            expected_stage=args.stage,
            expected_prefix=args.asset_prefix,
        )
        destination = weights / f"epoch{expected_epoch - 1}.pt"
        temporary_checkpoint.replace(destination)
        print(
            json.dumps(
                {
                    "checkpoint": str(destination.resolve()),
                    "completed_epoch": metadata.completed_epoch,
                    "sha256": metadata.sha256,
                    "variant": args.variant,
                    "stage": args.stage,
                    "seed": 0,
                },
                sort_keys=True,
            )
        )
        return destination.resolve()
    finally:
        temporary_manifest.unlink(missing_ok=True)
        temporary_checkpoint.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restore a validated LPR-G v2 checkpoint from GitHub."
    )
    parser.add_argument("--variant", choices=("control", "lprg"), required=True)
    parser.add_argument("--stage", choices=("screen", "formal"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--repo", default="kkc236/uav-detection-baselines")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--asset-prefix", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.run_dir = args.run_dir.resolve()
    args.token_file = args.token_file.resolve()
    restore_latest_checkpoint(args)


if __name__ == "__main__":
    main()
