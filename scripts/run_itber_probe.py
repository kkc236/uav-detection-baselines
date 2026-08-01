"""Run fresh P0-P3 I-TBER Probe arms and write immutable Gate 1 decision."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.itber_cache import load_evidence_cache  # noqa: E402
from src.itber_probe import evaluate_gate1, train_probe_arm  # noqa: E402
from src.itber_protocol import (  # noqa: E402
    EXPECTED_BASELINE_SHA256,
    EXPECTED_CATEGORY_SHA256,
    EXPECTED_DATASET_SHA256,
    write_immutable_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    cache = load_evidence_cache(
        args.cache_root,
        expected_authority={
            "baseline_sha256": EXPECTED_BASELINE_SHA256,
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "category_sha256": EXPECTED_CATEGORY_SHA256,
            "source_commit": commit,
        },
    )
    device = torch.device(f"cuda:{args.device}")
    reports = {
        probe: train_probe_arm(cache, probe=probe, output_root=args.output_root, device=device)
        for probe in ("p0", "p1", "p2", "p3")
    }
    decision = evaluate_gate1(reports)
    write_immutable_report(args.output_root / "gate1-decision.json", decision)
    print(json.dumps(decision, sort_keys=True, separators=(",", ":")))
    return 0 if decision["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
