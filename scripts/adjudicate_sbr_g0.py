#!/usr/bin/env python3
"""Independent, fail-closed adjudication of an SBR-RTDETR G0-A run.

The adjudicator deliberately does not trust ``g0_gate.json`` for the scientific
decision.  It verifies the immutable provenance and artifact checksums, then
recomputes the frozen C-versus-A gate from ``g0_metrics.json``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

from src.sbr_artifacts import atomic_write_json, sha256_file, write_checksums


REQUIRED = (
    "g0_manifest.json",
    "g0_metrics.json",
    "g0_deltas.json",
    "g0_gate.json",
    "raw_views.jsonl.gz",
    "arm_predictions.jsonl.gz",
    "checksums.sha256",
)
HEX64 = set("0123456789abcdefABCDEF")


def _json(path: Path) -> Any:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_no} is not an object")
            rows.append(row)
    return rows


def _finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is non-finite")
    return result


def _hash(value: Any, name: str) -> str:
    result = str(value or "")
    if len(result) != 64 or any(ch not in HEX64 for ch in result):
        raise ValueError(f"{name} is not a SHA-256 digest")
    return result.lower()


def _verify_checksums(root: Path) -> list[str]:
    checksum_file = root / "checksums.sha256"
    lines = checksum_file.read_text(encoding="utf-8").splitlines()
    seen: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            digest, rel = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError("invalid checksums.sha256 line") from exc
        digest = _hash(digest, f"checksum {rel}")
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"unsafe checksum path: {rel}")
        target = (root / rel_path).resolve()
        if root.resolve() not in target.parents or not target.is_file():
            raise ValueError(f"missing checksum target: {rel}")
        if sha256_file(target).lower() != digest:
            raise ValueError(f"artifact checksum mismatch: {rel}")
        seen.append(rel_path.as_posix())
    return seen


def _recompute_g0a(metrics: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, float], str]:
    if not isinstance(metrics.get("A"), Mapping) or not isinstance(metrics.get("C"), Mapping):
        raise ValueError("g0_metrics.json must contain A and C metrics")
    a, c = metrics["A"], metrics["C"]
    keys = ("AP-tiny-SBR", "mAP50-95", "AP75", "AP-large-SBR")
    deltas = {
        key: _finite_number(c.get(key), f"C.{key}") - _finite_number(a.get(key), f"A.{key}")
        for key in keys
    }
    deltas["tiny_recall"] = _finite_number(c.get("tiny_recall"), "C.tiny_recall") - _finite_number(
        a.get("tiny_recall"), "A.tiny_recall"
    )
    passed = (
        deltas["AP-tiny-SBR"] >= 0.01
        and deltas["mAP50-95"] >= 0.003
        and deltas["tiny_recall"] >= 0.02
        and deltas["AP75"] >= -0.002
        and deltas["AP-large-SBR"] >= -0.005
    )
    return deltas, "SBR_G0A_PASS" if passed else "SBR_G0A_FAIL"


def _close(a: Any, b: Any) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)


def adjudicate_evidence(evidence: Path | str) -> dict[str, Any]:
    """Adjudicate evidence and write ``independent_adjudication.json``.

    A malformed/tampered evidence directory still receives a FAIL report, but
    its pre-existing checksum manifest is left untouched for forensic review.
    """
    root = Path(evidence).resolve()
    report: dict[str, Any]
    checksum_ok = False
    try:
        if not root.is_dir():
            raise ValueError(f"evidence directory does not exist: {root}")
        missing = [name for name in REQUIRED if not (root / name).is_file()]
        if missing:
            raise ValueError(f"missing evidence files: {', '.join(missing)}")
        listed = _verify_checksums(root)
        checksum_ok = True
        for name in REQUIRED[:-1]:
            if name not in listed:
                raise ValueError(f"{name} is not covered by checksums.sha256")

        manifest = _json(root / "g0_manifest.json")
        metrics = _json(root / "g0_metrics.json")
        deltas = _json(root / "g0_deltas.json")
        runner_gate = _json(root / "g0_gate.json")
        if not isinstance(manifest, Mapping) or manifest.get("mode") != "g0-a":
            raise ValueError("evidence is not a G0-A manifest")
        expected = {
            "source_hash": _hash(manifest.get("source_hash"), "manifest.source_hash"),
            "checkpoint_hash": _hash(manifest.get("checkpoint_hash"), "manifest.checkpoint_hash"),
            "dataset_signature": _hash(manifest.get("dataset_signature"), "manifest.dataset_signature"),
            "protocol_hash": _hash(manifest.get("protocol_hash"), "manifest.protocol_hash"),
        }
        source = manifest.get("source")
        if isinstance(source, Mapping) and str(source.get("commit", "")) != expected["source_hash"]:
            raise ValueError("manifest source commit/hash mismatch")
        for name, value in expected.items():
            if str(runner_gate.get(name, "")).lower() != value:
                raise ValueError(f"g0_gate provenance mismatch: {name}")
        if int(manifest.get("image_count", 0)) != 548:
            raise ValueError("G0-A manifest must contain exactly 548 images")

        raw_rows = _jsonl_gz(root / "raw_views.jsonl.gz")
        arm_rows = _jsonl_gz(root / "arm_predictions.jsonl.gz")
        image_ids = set(manifest.get("image_list", []))
        if not image_ids or len(image_ids) != 548:
            raise ValueError("manifest image_list is missing or incomplete")
        valid_arms = set("ABCDEF")
        for row in raw_rows:
            if row.get("arm") not in valid_arms or row.get("image_id") not in image_ids:
                raise ValueError("raw-view provenance is invalid")
        for row in arm_rows:
            if row.get("arm") not in valid_arms or row.get("image_id") not in image_ids:
                raise ValueError("arm-prediction provenance is invalid")

        recomputed, independent_gate = _recompute_g0a(metrics)
        if not isinstance(deltas, Mapping):
            raise ValueError("g0_deltas.json must be an object")
        delta_mismatch = [key for key, value in recomputed.items() if key not in deltas or not _close(deltas[key], value)]
        if delta_mismatch:
            raise ValueError(f"g0_deltas mismatch: {', '.join(delta_mismatch)}")
        report = {
            "status": "SBR_G0A_INDEPENDENT_PASS" if independent_gate == "SBR_G0A_PASS" else "SBR_G0A_INDEPENDENT_FAIL",
            "decision": "PASS" if independent_gate == "SBR_G0A_PASS" else "FAIL",
            "independent_gate": independent_gate,
            "runner_status": str(runner_gate.get("status", "")),
            "source_hash": expected["source_hash"],
            "checkpoint_hash": expected["checkpoint_hash"],
            "dataset_signature": expected["dataset_signature"],
            "protocol_hash": expected["protocol_hash"],
            "recomputed_deltas": recomputed,
            "raw_view_rows": len(raw_rows),
            "arm_prediction_rows": len(arm_rows),
            "checksums_verified": True,
        }
    except Exception as exc:
        report = {
            "status": "SBR_G0A_INDEPENDENT_FAIL",
            "decision": "FAIL",
            "independent_gate": "SBR_G0A_FAIL",
            "checksums_verified": checksum_ok,
            "error": str(exc),
        }

    atomic_write_json(root / "independent_adjudication.json", report)
    # A valid report becomes part of the checksummed evidence set.  On a
    # failed integrity check we retain the old manifest for forensic review.
    if checksum_ok:
        files = [p for p in root.iterdir() if p.is_file() and p.name != "checksums.sha256"]
        write_checksums(root / "checksums.sha256", files, root=root)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independently adjudicate SBR G0-A evidence")
    parser.add_argument("--evidence", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        report = adjudicate_evidence(build_parser().parse_args(argv).evidence)
    except Exception as exc:
        print(f"SBR_ADJUDICATOR_FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("decision") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
