from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tascv_adjudicator import (
    adjudicate_mechanism,
    adjudicate_preflight,
    expected_artifact_paths,
    replay_preflight_gate,
)
from src.tascv_protocol import reject_forbidden_path, sha256_file


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
        description="Independently adjudicate a frozen T-ASCV endpoint."
    )
    parser.add_argument(
        "--stage",
        choices=("PREFLIGHT_1", "TINY_MECHANISM_500"),
        required=True,
    )
    parser.add_argument("--records", type=Path)
    parser.add_argument(
        "--summary",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = reject_forbidden_path(
        args.output,
        context="T-ASCV adjudication output",
    )
    if output_path.exists():
        raise ValueError("T-ASCV adjudication output already exists")
    expected_summaries = 2 if args.stage == "PREFLIGHT_1" else 1
    if len(args.summary) != expected_summaries:
        raise ValueError(
            f"{args.stage} requires exactly "
            f"{expected_summaries} summaries"
        )
    summaries: dict[str, dict] = {}
    bindings: dict[str, dict] = {}
    for path in args.summary:
        resolved = reject_forbidden_path(
            path,
            context="T-ASCV adjudication summary",
        )
        if resolved == output_path:
            raise ValueError("T-ASCV output aliases a summary input")
        summary = json.loads(resolved.read_text(encoding="utf-8"))
        if resolved != expected_artifact_paths(summary)["summary"]:
            raise ValueError("T-ASCV summary endpoint drift")
        arm = summary.get("arm")
        if arm not in {"control", "tascv"} or arm in summaries:
            raise ValueError("duplicate or invalid T-ASCV summary arm")
        summaries[arm] = summary
        bindings[arm] = {
            "path": resolved.as_posix(),
            "sha256": sha256_file(resolved),
        }
    if args.stage == "PREFLIGHT_1":
        if args.records is not None:
            raise ValueError("PREFLIGHT_1 forbids mechanism records")
        decision = adjudicate_preflight(summaries)
    else:
        if set(summaries) != {"tascv"} or args.records is None:
            raise ValueError(
                "TINY_MECHANISM_500 requires T-ASCV summary and records"
            )
        summary = summaries["tascv"]
        predecessor = summary.get("predecessor_evidence", {})
        predecessor_path = reject_forbidden_path(
            predecessor.get("path", ""),
            context="T-ASCV mechanism predecessor",
        )
        if (
            not predecessor_path.is_file()
            or sha256_file(predecessor_path)
            != predecessor.get("sha256")
        ):
            raise ValueError("mechanism predecessor binding drift")
        replayed_preflight = replay_preflight_gate(
            json.loads(
                predecessor_path.read_text(encoding="utf-8")
            )
        )
        if (
            replayed_preflight.get("protocol_manifest_sha256")
            != summary.get("protocol_manifest_sha256")
            or replayed_preflight.get("protocol_source_commit")
            != summary.get("protocol_source_commit")
        ):
            raise ValueError("mechanism predecessor source drift")
        records_path = reject_forbidden_path(
            args.records,
            context="T-ASCV adjudication records",
        )
        if records_path == output_path:
            raise ValueError("T-ASCV output aliases mechanism records")
        record_binding = summary.get("mechanism_records", {})
        if (
            records_path.as_posix() != record_binding.get("path")
            or sha256_file(records_path)
            != record_binding.get("sha256")
            or records_path
            != expected_artifact_paths(summary)["records"]
        ):
            raise ValueError("mechanism record binding drift")
        records = [
            json.loads(line)
            for line in records_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line
        ]
        decision = adjudicate_mechanism(summary, records)
        decision["mechanism_records_binding"] = record_binding
    decision["summary_bindings"] = bindings
    _atomic_json(output_path, decision)
    if decision["decision"] == "INVALID":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
