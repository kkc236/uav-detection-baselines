"""Run fresh B0-B3 IBER-BE Probe arms and freeze the Gate-1 decision."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.iber_cache import load_evidence_cache  # noqa: E402
from src.iber_probe import ARM_ORDER, evaluate_gate1, train_probe_arm  # noqa: E402
from src.iber_protocol import (  # noqa: E402
    DESIGN_VERSION,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    RUNTIME_AMENDMENT_SHA256,
    write_immutable_report,
)


EXPECTED_CATEGORY_SHA256 = (
    "1455A1F5D9FA9988799815B36CF2CE6D5044B5BD7CAD7CC614D6B5E5059EF2A6"
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser.parse_args(argv)


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip().lower()


def _engineering_failure(output_root: Path, error: BaseException) -> int:
    payload = {
        "design_version": DESIGN_VERSION,
        "stage": "gate1_decision",
        "status": "engineering_invalid",
        "error_type": type(error).__name__,
        "error": str(error),
    }
    try:
        write_immutable_report(output_root / "gate1-engineering-invalid.json", payload)
    except (OSError, ValueError):
        pass
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        cache = load_evidence_cache(
            args.cache_root,
            expected_authority={
                "baseline_sha256": EXPECTED_BASELINE_SHA256,
                "dataset_sha256": EXPECTED_DATASET_SHA256,
                "category_sha256": EXPECTED_CATEGORY_SHA256,
                "subset_sha256": EXPECTED_SUBSET_SHA256,
                "source_commit": _source_commit(),
                "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
            },
        )
        device = torch.device(
            f"cuda:{args.device}" if str(args.device).isdigit() else str(args.device)
        )
        reports = {
            arm: train_probe_arm(
                cache,
                arm=arm,
                output_root=args.output_root,
                device=device,
            )
            for arm in ARM_ORDER
        }
        decision = evaluate_gate1(reports)
        write_immutable_report(args.output_root / "gate1-decision.json", decision)
        print(json.dumps(decision, sort_keys=True, separators=(",", ":")))
        if decision["status"] == "passed":
            return 0
        if decision["status"] == "scientific_failed":
            return 2
        return 1
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return _engineering_failure(args.output_root, error)


if __name__ == "__main__":
    raise SystemExit(main())
