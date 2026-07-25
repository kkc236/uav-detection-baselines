from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tascv_protocol import (
    resolve_control_allowlist,
    sha256_file,
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve stock controls using provenance only; performance "
            "artifacts are forbidden."
        )
    )
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _reject_candidate_path(path: Path) -> None:
    raw = str(path).replace("\\", "/").lower()
    resolved = path.resolve().as_posix().lower()
    forbidden = (
        "test-dev",
        "test_dev",
        "/metrics/",
        "/results/",
        "/deltas/",
        "/gates/",
        "val_annotation",
    )
    if any(token in raw or token in resolved for token in forbidden):
        raise ValueError(f"forbidden control candidate path: {path}")


def main() -> None:
    args = build_parser().parse_args()
    requirements_path = args.requirements.resolve()
    requirements = json.loads(
        requirements_path.read_text(encoding="utf-8")
    )
    candidates = []
    for path in args.candidate:
        _reject_candidate_path(path)
        candidates.append(
            json.loads(path.resolve().read_text(encoding="utf-8"))
        )
    allowlist = resolve_control_allowlist(requirements, candidates)
    allowlist["requirements"] = {
        "path": requirements_path.as_posix(),
        "sha256": sha256_file(requirements_path),
    }
    _atomic_json(args.output.resolve(), allowlist)


if __name__ == "__main__":
    main()
