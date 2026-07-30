"""Run the sealed live ACR-EG versus mature RT-DETR-L evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.acr_eg_live_evaluation import run_live_evaluation


EXPECTED_BASELINE_SHA256 = (
    "54CE60289DD34C6750B8BA5F7516EEF"
    "CF3AFEF6C174C6E4F3B1EF810C883099B"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "66E0B8D27706CDA594BE657B20BFD01C"
    "AA536D90B7EA0A05EDC2FEEC11C6E2B4"
)
EXPECTED_DATASET_SIGNATURE = (
    "A9A0C00DC640BCAAEFE9360F5E3B553"
    "82E74E169B5AEEF15EB1F0AE2A571228A"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the final live five-view ACR-EG checkpoint."
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


def validate_protocol(args: argparse.Namespace) -> None:
    if (
        str(args.device) != "0"
        or args.batch != 1
        or args.workers != 0
        or args.imgsz != 640
        or args.conf != 0.001
        or args.max_det != 300
        or args.expected_records != 548
        or args.expected_epoch != 99
        or not args.amp
    ):
        raise ValueError("ACR_EG_LIVE_PROTOCOL_DRIFT")
    for value in (
        args.expected_baseline_sha256,
        args.expected_checkpoint_sha256,
        args.dataset_signature,
    ):
        normalized = str(value).upper()
        if len(normalized) != 64 or any(
            character not in "0123456789ABCDEF" for character in normalized
        ):
            raise ValueError("ACR_EG_LIVE_DIGEST_INVALID")
    if args.limit is not None:
        if not args.smoke:
            raise ValueError("ACR_EG_LIVE_LIMIT_REQUIRES_SMOKE")
        if args.limit != 1:
            raise ValueError("ACR_EG_LIVE_SMOKE_LIMIT_DRIFT")
    elif args.smoke:
        raise ValueError("ACR_EG_LIVE_SMOKE_REQUIRES_LIMIT")


def evaluate(args: argparse.Namespace) -> Path:
    validate_protocol(args)
    return run_live_evaluation(args)


def main() -> None:
    print(evaluate(build_parser().parse_args()))


if __name__ == "__main__":
    main()
