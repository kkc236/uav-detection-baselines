"""Run paired Ultralytics-native validation for the final ACR-EG model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.acr_eg_ultralytics_native import (
    EXPECTED_BASELINE_SHA256,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_DATASET_SIGNATURE,
    run_native_evaluation,
    validate_native_protocol,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and five-view ACR-EG with Ultralytics metrics."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--expected-records", type=int, default=548)
    parser.add_argument("--expected-epoch", type=int, default=99)
    parser.add_argument(
        "--expected-baseline-sha256",
        default=EXPECTED_BASELINE_SHA256,
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=EXPECTED_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--dataset-signature",
        default=EXPECTED_DATASET_SIGNATURE,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser


def evaluate(args: argparse.Namespace) -> Path:
    validate_native_protocol(args)
    return run_native_evaluation(args)


def main() -> None:
    print(evaluate(build_parser().parse_args()))


if __name__ == "__main__":
    main()

