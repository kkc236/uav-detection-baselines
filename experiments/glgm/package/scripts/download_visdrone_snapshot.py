#!/usr/bin/env python3
"""Download a pinned VisDrone train/val snapshot from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO = "banu4prasad/VisDrone-Dataset"
DEFAULT_REVISION = "62fc5e7387775e8f1f4fa5e027555024124763fa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--workers", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.revision or len(args.revision) != 40:
        raise ValueError("--revision must be a full 40-character commit hash")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

    snapshot_path = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        revision=args.revision,
        local_dir=output,
        allow_patterns=[
            "README.md",
            "visdrone.yaml",
            "VisDrone2019-DET-train/**",
            "VisDrone2019-DET-val/**",
        ],
        max_workers=args.workers,
    )
    receipt = {
        "schema": "glgm-dataset-source-v1",
        "repo": args.repo,
        "revision": args.revision,
        "endpoint": args.endpoint,
        "snapshot_path": str(Path(snapshot_path).resolve()),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path = output / "SOURCE_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
