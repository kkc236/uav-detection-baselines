from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sync_btdse_checkpoint import validate_token_file
from src.github_checkpoint_sync import checkpoint_metadata, github_session


def select_latest_asset(assets: Iterable[dict[str, Any]], *, prefix: str) -> tuple[dict[str, Any], int]:
    pattern = re.compile(rf"^{re.escape(prefix)}-epoch-(\d+)\.pt$")
    candidates: list[tuple[int, dict[str, Any]]] = []
    for asset in assets:
        match = pattern.match(str(asset.get("name", "")))
        if match:
            candidates.append((int(match.group(1)), asset))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint asset with prefix {prefix!r} was found")
    epoch, asset = max(candidates, key=lambda item: item[0])
    return asset, epoch


def restore_latest_checkpoint(args: argparse.Namespace) -> Path:
    token = validate_token_file(args.token_file)
    session = github_session(token)
    release_response = session.get(
        f"https://api.github.com/repos/{args.repo}/releases/tags/{args.tag}", timeout=30
    )
    release_response.raise_for_status()
    asset, expected_epoch = select_latest_asset(release_response.json().get("assets", []), prefix=args.asset_prefix)

    weights = args.run_dir / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    destination = weights / f"epoch{expected_epoch}.pt"
    temporary = destination.with_suffix(".pt.tmp")
    with session.get(
        str(asset["url"]),
        headers={"Accept": "application/octet-stream"},
        stream=True,
        timeout=(30, 3600),
    ) as response:
        response.raise_for_status()
        with temporary.open("wb") as file:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    file.write(chunk)
    if temporary.stat().st_size != int(asset["size"]):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Downloaded checkpoint size does not match the GitHub asset")

    metadata = checkpoint_metadata(temporary)
    if metadata.completed_epoch != expected_epoch:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checkpoint epoch mismatch: asset says {expected_epoch}, payload says {metadata.completed_epoch}"
        )
    temporary.replace(destination)
    print(
        json.dumps(
            {
                "checkpoint": str(destination.resolve()),
                "completed_epoch": metadata.completed_epoch,
                "sha256": metadata.sha256,
            },
            sort_keys=True,
        )
    )
    return destination.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Restore the newest validated IOQC-SA checkpoint from GitHub.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--repo", default="kkc236/uav-detection-baselines")
    parser.add_argument("--tag", default="ioqc-sa-rtdetr-l-live")
    parser.add_argument("--asset-prefix", default="ioqc-sa-last")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.run_dir = args.run_dir.resolve()
    args.token_file = args.token_file.resolve()
    restore_latest_checkpoint(args)


if __name__ == "__main__":
    main()
