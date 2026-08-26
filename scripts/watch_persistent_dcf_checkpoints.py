"""Preserve selected full Persistent DCF checkpoints without changing training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch


MILESTONES = (25, 50, 66, 75, 90, 100)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_summary(path: Path) -> dict[str, int | bool]:
    """Require one full, resumable Ultralytics checkpoint."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("epoch"), int):
        raise RuntimeError("checkpoint epoch is missing")
    for key in ("optimizer", "scaler", "ema"):
        if payload.get(key) is None:
            raise RuntimeError(f"checkpoint {key} is missing")
    return {
        "paper_epoch": int(payload["epoch"]) + 1,
        "optimizer": True,
        "scaler": True,
        "ema": True,
    }


def preserve_milestone(
    source: Path, target_root: Path, *, expected_paper_epoch: int
) -> Path:
    """Copy and atomically publish one verified milestone checkpoint."""

    source = source.resolve()
    summary = checkpoint_summary(source)
    if summary["paper_epoch"] != expected_paper_epoch:
        raise RuntimeError(
            f"expected paper epoch {expected_paper_epoch}, "
            f"found {summary['paper_epoch']}"
        )
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / f"epoch{expected_paper_epoch:04d}.pt"
    temporary = target.with_suffix(".pt.tmp")
    shutil.copy2(source, temporary)
    copied = checkpoint_summary(temporary)
    if copied != summary:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("copied checkpoint summary changed")
    os.replace(temporary, target)
    manifest = {
        **summary,
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "path": target.name,
    }
    manifest_tmp = target.with_suffix(".json.tmp")
    manifest_path = target.with_suffix(".json")
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(manifest_tmp, manifest_path)
    return target


def completed_epochs(results_csv: Path) -> int:
    """Count canonical completed result rows."""

    if not results_csv.exists():
        return 0
    with results_csv.open(newline="", encoding="utf-8-sig") as stream:
        return len(list(csv.DictReader(stream)))


def watch(run_dir: Path, poll_seconds: float) -> int:
    """Poll `last.pt` and preserve each milestone before it is overwritten."""

    run_dir = run_dir.resolve()
    source = run_dir / "weights" / "last.pt"
    target_root = run_dir / "milestone-checkpoints"
    results = run_dir / "results.csv"
    while True:
        completed = completed_epochs(results)
        missing = [
            epoch
            for epoch in MILESTONES
            if not (target_root / f"epoch{epoch:04d}.pt").exists()
        ]
        if missing and source.exists():
            current = checkpoint_summary(source)["paper_epoch"]
            expected = missing[0]
            if current == expected:
                preserve_milestone(
                    source, target_root, expected_paper_epoch=expected
                )
            elif current > expected:
                raise RuntimeError(
                    f"missed milestone {expected}; current checkpoint is {current}"
                )
        if completed >= 100 and not missing:
            return 0
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    args = parser.parse_args()
    return watch(args.run_dir, args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
