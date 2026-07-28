"""Evaluate the YAML-configured ACR-EG wrapper against the mature baseline."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_gcqf_g0


def build_parser() -> argparse.ArgumentParser:
    parser = evaluate_gcqf_g0.build_parser()
    parser.description = (
        "Evaluate integrated ACR-EG against the sealed mature RT-DETR baseline."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    return parser


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "integration": {
            "module": "ACR-EG",
            "forward_integration": True,
            "config": str(args.config.resolve()),
            "config_sha256": _sha256_file(args.config),
        },
        "baseline": {
            "checkpoint": str(args.baseline_checkpoint.resolve()),
            "sha256": _sha256_file(args.baseline_checkpoint),
            "source": "mature RT-DETR-L 100-epoch baseline",
        },
        "module_checkpoint": str(args.module.resolve()),
        "module_sha256": _sha256_file(args.module),
    }


def evaluate(args: argparse.Namespace) -> Path:
    output = evaluate_gcqf_g0.evaluate(args)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["integration_protocol"] = build_protocol(args)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    print(evaluate(build_parser().parse_args()))


if __name__ == "__main__":
    main()
