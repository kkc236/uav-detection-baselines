from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.ascv_loc_adjudicator import adjudicate_formal, adjudicate_screen
from src.ascv_loc_protocol import sha256_file


def _reject_forbidden(path: Path) -> None:
    normalized = path.resolve().as_posix().lower()
    if "test-dev" in normalized or "test_dev" in normalized:
        raise ValueError(f"test-dev is forbidden in ASCV adjudication: {path}")


def load_records(paths: dict[tuple[int, str], Path]) -> tuple[dict, list[dict]]:
    records: dict[str, dict] = {}
    inputs = []
    for (seed, arm), raw_path in sorted(paths.items()):
        if arm not in {"control", "ascv"}:
            raise ValueError(f"unknown paired arm: {arm}")
        path = Path(raw_path).resolve()
        _reject_forbidden(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if set(payload) != {"A", "C"}:
            raise ValueError(f"metrics payload must contain exact A/C arms: {path}")
        records.setdefault(str(seed), {})[arm] = {"A": payload["A"], "C": payload["C"]}
        inputs.append(
            {
                "seed": seed,
                "arm": arm,
                "path": path.as_posix(),
                "sha256": sha256_file(path),
            }
        )
    return records, inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone frozen ASCV-Loc paired adjudicator.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=("screen", "formal-seed0", "formal-paper"),
    )
    for seed in (0, 1, 2):
        parser.add_argument(f"--seed{seed}-control", type=Path)
        parser.add_argument(f"--seed{seed}-ascv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = (0,) if args.mode == "formal-seed0" else (0, 1, 2)
    paths = {}
    for seed in seeds:
        for arm in ("control", "ascv"):
            value = getattr(args, f"seed{seed}_{arm}")
            if value is None:
                raise ValueError(f"{args.mode} requires --seed{seed}-{arm}")
            paths[(seed, arm)] = value
    records, inputs = load_records(paths)
    if args.mode == "screen":
        adjudication = adjudicate_screen(records)
    else:
        adjudication = adjudicate_formal(
            records,
            require_three_seeds=args.mode == "formal-paper",
        )
    output = {
        **adjudication,
        "input_metrics": inputs,
    }
    path = args.output.resolve()
    _reject_forbidden(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
