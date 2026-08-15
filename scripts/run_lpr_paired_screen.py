"""Prepare and run the frozen three-seed paired LPR screen in order."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCREEN_ORDER = (
    (0, "control"),
    (0, "lpr"),
    (1, "lpr"),
    (1, "control"),
    (2, "control"),
    (2, "lpr"),
)


def _run_name(seed: int, variant: str) -> str:
    return f"screen-seed{seed}-{variant}-lpr-v1"


def build_arm_command(
    *,
    python: Path,
    protocol_dir: Path,
    project: Path,
    seed: int,
    variant: str,
) -> list[str]:
    return [
        str(python),
        "scripts/train_rtdetr_lpr.py",
        "--variant",
        variant,
        "--stage",
        "screen",
        "--seed",
        str(seed),
        "--protocol-manifest",
        str((protocol_dir / f"protocol-seed{seed}.json").resolve()),
        "--initial-state",
        str((protocol_dir / f"initial-state-seed{seed}.pt").resolve()),
        "--project",
        str(project.resolve()),
    ]


def build_pairs_manifest(project: Path) -> dict:
    root = Path(project).resolve()
    return {
        "format_version": 1,
        "order": [{"seed": seed, "variant": variant} for seed, variant in SCREEN_ORDER],
        "pairs": {
            str(seed): {
                "control": str(root / _run_name(seed, "control")),
                "lpr": str(root / _run_name(seed, "lpr")),
            }
            for seed in (0, 1, 2)
        },
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen three-seed paired LPR screen.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol_dir = args.protocol_dir.resolve()
    project = args.project.resolve()
    for seed in (0, 1, 2):
        subprocess.run(
            [
                str(args.python),
                "scripts/prepare_lpr_protocol.py",
                "--dataset-root",
                str(args.dataset_root.resolve()),
                "--output-dir",
                str(protocol_dir),
                "--seed",
                str(seed),
            ],
            cwd=ROOT,
            check=True,
        )
    for seed, variant in SCREEN_ORDER:
        subprocess.run(
            build_arm_command(
                python=args.python,
                protocol_dir=protocol_dir,
                project=project,
                seed=seed,
                variant=variant,
            ),
            cwd=ROOT,
            check=True,
        )
    pairs_path = project / "paired-screen-runs.json"
    _atomic_json(pairs_path, build_pairs_manifest(project))
    output = project / "paired-screen-gate.json"
    subprocess.run(
        [
            str(args.python),
            "scripts/evaluate_lpr_gate.py",
            "--pairs-manifest",
            str(pairs_path),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    print(output)


if __name__ == "__main__":
    main()
