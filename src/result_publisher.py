from __future__ import annotations

import csv
import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


RESULT_PATTERNS = (
    "results.csv",
    "args.yaml",
    "results.png",
    "confusion_matrix*.png",
    "PR_curve.png",
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png",
    "labels*.jpg",
)


@dataclass
class PublicationStatus:
    completed_epoch: int = 0
    git_push_verified: bool = False
    release_verified: bool = False

    @property
    def shutdown_allowed(self) -> bool:
        return self.completed_epoch >= 100 and self.git_push_verified and self.release_verified


def latest_completed_epoch(results_csv: Path) -> int:
    if not results_csv.exists():
        return 0
    with results_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return 0
    try:
        return int(float(rows[-1].get("epoch", "0")))
    except (TypeError, ValueError):
        return 0


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verified_release_assets(
    expected: Mapping[str, int],
    actual: Iterable[Mapping[str, object]],
) -> bool:
    actual_sizes = {str(item.get("name")): int(item.get("size", -1)) for item in actual}
    return all(actual_sizes.get(name) == size for name, size in expected.items())


def collect_result_artifacts(run_dir: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    seen: set[Path] = set()
    for pattern in RESULT_PATTERNS:
        for source in run_dir.glob(pattern):
            if not source.is_file() or source in seen:
                continue
            target = destination / source.name
            shutil.copy2(source, target)
            copied.append(target)
            seen.add(source)
    return copied


def repo_slug_from_remote(remote: str) -> str:
    match = re.search(r"github\.com[/:]([^/]+/[^/]+?)(?:\.git)?$", remote.strip())
    if not match:
        raise ValueError(f"Unsupported GitHub remote: {remote}")
    return match.group(1)
