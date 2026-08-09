#!/usr/bin/env python3
"""Select passing GLGM-v2 candidates using only preregistered gate receipts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, action="append", required=True)
    parser.add_argument("--maximum", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.maximum < 1:
        raise ValueError("maximum must be positive")

    gates = [json.loads(path.read_text(encoding="utf-8")) for path in args.gate]
    passing = [gate for gate in gates if gate.get("pass") is True]
    passing.sort(
        key=lambda gate: (
            float(gate["ranking"]["last_map_delta_pp"]),
            float(gate["ranking"]["last_recall_delta_pp"]),
            float(gate["ranking"]["mean_early_map_delta_pp"]),
        ),
        reverse=True,
    )
    selected = passing[: args.maximum]
    payload = {
        "schema": "glgm-v2-candidate-selection-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "maximum": args.maximum,
        "passing_count": len(passing),
        "selected": [gate["variant"] for gate in selected],
        "ranked_passing": [
            {"variant": gate["variant"], "ranking": gate["ranking"]} for gate in passing
        ],
        "failed": [gate["variant"] for gate in gates if gate.get("pass") is not True],
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    for gate in selected:
        print(gate["variant"])


if __name__ == "__main__":
    main()
