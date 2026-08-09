"""Run the frozen FDR/FrequencyCM candidate-complementarity upper-bound oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


IMAGE_SIZE = 640
BATCH_SIZE = 8
WORKERS = 8
CONFIDENCE = 0.001
MAX_DET = 300
NMS = False
VAL_COUNT = 548
NUM_CLASSES = 10
FDR_CHECKPOINT_SHA256 = (
    "C2F638744508ADFE7B6C4A1EF3E08C503273F628062E4650AD59FFFF4C6588C2"
)
FREQUENCYCM_CHECKPOINT_SHA256 = (
    "2BBCD6057FEFED5792F786A18E603F8FECA3EC426A6F68938F5F8ADA1603A141"
)
FREQUENCYCM_SOURCE_COMMIT = "d3655b14c17a3c8ca14e1888517b6fde4e059766"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdr-checkpoint", type=Path, required=True)
    parser.add_argument("--frequencycm-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser.parse_args(argv)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable report output already exists: {path}")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_summary(report_root: Path, payload: Mapping[str, Any]) -> None:
    root = Path(report_root)
    summary_path = root / "oracle-summary.json"
    markdown_path = root / "frequencycm-complementarity-report.md"
    sums_path = root / "SHA256SUMS.txt"
    for path in (summary_path, markdown_path, sums_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"immutable report output already exists: {path}")

    normalized = dict(payload)
    normalized["interpretation"] = "non_deployable_design_selection_evidence"
    summary = _canonical_json(normalized)
    decision = str(normalized.get("decision", {}).get("decision", "unknown"))
    candidate_delta = normalized.get("oracle", {}).get("candidate_map_delta", "unknown")
    coverage_delta = normalized.get("coverage", {}).get(
        "tiny_small_recall50_delta", "unknown"
    )
    markdown = (
        "# FrequencyCM Complementarity Oracle\n\n"
        "> This report uses ground truth and is non-deployable design-selection evidence.\n\n"
        f"- Decision: `{decision}`\n"
        f"- Candidate-oracle mAP delta: `{candidate_delta}`\n"
        f"- Tiny/small recall@0.50 delta: `{coverage_delta}`\n"
    ).encode("utf-8")
    _write_create_only(summary_path, summary)
    _write_create_only(markdown_path, markdown)
    sums = "".join(
        f"{hashlib.sha256(content).hexdigest().upper()}  {name}\n"
        for name, content in (
            (summary_path.name, summary),
            (markdown_path.name, markdown),
        )
    ).encode("ascii")
    _write_create_only(sums_path, sums)


def main(argv: Sequence[str] | None = None) -> int:
    _parse_args(argv)
    raise RuntimeError("full complementarity execution is not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())

