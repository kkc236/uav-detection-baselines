#!/usr/bin/env python3
"""Run a standalone sealed SADED stage adjudication."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.saded_adjudicator import (  # noqa: E402
    adjudicate_confirmation_result,
    adjudicate_formal_seed0,
    adjudicate_formal_three_seed,
    adjudicate_screen_seed0,
    adjudicate_screen_three_seed,
)
from src.sbr_artifacts import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    write_checksums,
)
from src.tascv_protocol import reject_forbidden_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently adjudicate a sealed SADED stage."
    )
    parser.add_argument(
        "--stage",
        choices=(
            "SCREEN_SEED0",
            "SCREEN_THREE_SEED",
            "FORMAL_SEED0",
            "FORMAL_THREE_SEED",
            "CONFIRMATION",
        ),
        required=True,
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        action="append",
    )
    parser.add_argument(
        "--evaluation-anchor-sha256",
        action="append",
    )
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--result-anchor-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluation_roots = args.evaluation_root or []
    for value in (*evaluation_roots, args.output):
        reject_forbidden_path(value, context="SADED adjudication")
    if args.result_root is not None:
        reject_forbidden_path(
            args.result_root,
            context="SADED confirmation adjudication result",
        )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("SADED adjudication output already exists")
    expected = (
        0
        if args.stage == "CONFIRMATION"
        else (
            1
            if args.stage in {"SCREEN_SEED0", "FORMAL_SEED0"}
            else 3
        )
    )
    evaluation_anchors = args.evaluation_anchor_sha256 or []
    if (
        len(evaluation_roots) != expected
        or len(evaluation_anchors) != expected
    ):
        raise ValueError(
            f"{args.stage} requires exactly {expected} evaluations"
        )
    paired = list(
        zip(evaluation_roots, evaluation_anchors)
    )
    if args.stage == "CONFIRMATION":
        if args.result_root is None or not args.result_anchor_sha256:
            raise ValueError(
                "CONFIRMATION requires result root and anchor"
            )
        decision = adjudicate_confirmation_result(
            args.result_root,
            result_anchor_sha256=args.result_anchor_sha256,
        )
    elif args.result_root is not None or args.result_anchor_sha256:
        raise ValueError("non-confirmation stage forbids result inputs")
    elif args.stage == "SCREEN_SEED0":
        decision = adjudicate_screen_seed0(
            paired[0][0].resolve(),
            evaluation_anchor_sha256=paired[0][1],
        )
    elif args.stage == "FORMAL_SEED0":
        decision = adjudicate_formal_seed0(
            paired[0][0].resolve(),
            evaluation_anchor_sha256=paired[0][1],
        )
    else:
        evaluations = {
            seed: (root.resolve(), anchor)
            for seed, (root, anchor) in enumerate(paired)
        }
        decision = (
            adjudicate_screen_three_seed(evaluations)
            if args.stage == "SCREEN_THREE_SEED"
            else adjudicate_formal_three_seed(evaluations)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.adjudication-staging-",
            dir=output.parent,
        )
    )
    try:
        adjudication_dir = staging / "adjudication"
        adjudication_dir.mkdir()
        gate_path = atomic_write_json(
            adjudication_dir / "gate.json",
            decision,
        )
        checksums_path = write_checksums(
            adjudication_dir / "checksums.sha256",
            [gate_path],
            root=adjudication_dir,
        )
        atomic_write_json(
            staging / "adjudication_anchor.json",
            {
                "schema_version": "saded-adjudication-anchor/v1",
                "stage": args.stage,
                "gate_sha256": sha256_file(gate_path),
                "checksums_sha256": sha256_file(checksums_path),
                "evaluation_anchor_sha256": [
                    str(value).lower()
                    for value in evaluation_anchors
                ],
                "result_anchor_sha256": (
                    str(args.result_anchor_sha256).lower()
                    if args.result_anchor_sha256
                    else None
                ),
                "decision": decision["decision"],
            },
        )
        shutil.move(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(decision["decision"])
    if decision["decision"] == "INVALID":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
