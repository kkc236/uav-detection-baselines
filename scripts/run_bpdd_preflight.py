"""Run immutable fail-closed B0-B4 gates before the paired BPDD screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GATE_ORDER = ("B0", "B1", "B2", "B3", "B4")
FIXED_RUNTIME = {"device": "cuda:0", "batch": 8, "imgsz": 640, "amp_scale": 128.0}
SCREEN_AUTHORITY = {"schedule_epochs": 50, "cutoff_epoch": 30, "seed": 0}
GateRunner = Callable[["PreflightContext"], Mapping[str, Any]]


@dataclass(frozen=True)
class PreflightContext:
    protocol_manifest: Path
    initial_state: Path
    dataset_root: Path
    report_root: Path
    repository_root: Path = ROOT

    def __post_init__(self) -> None:
        for field in (
            "protocol_manifest",
            "initial_state",
            "dataset_root",
            "report_root",
            "repository_root",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _record(gate: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    return {
        "format_version": 1,
        "gate": gate,
        "payload": body,
        "payload_sha256": hashlib.sha256(_canonical(body)).hexdigest().upper(),
    }


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical(payload) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_inputs(context: PreflightContext) -> None:
    if not context.protocol_manifest.is_file():
        raise FileNotFoundError(f"BPDD protocol manifest not found: {context.protocol_manifest}")
    if not context.initial_state.is_file():
        raise FileNotFoundError(f"FDR initial state not found: {context.initial_state}")
    if not context.dataset_root.is_dir():
        raise FileNotFoundError(f"VisDrone root not found: {context.dataset_root}")
    if not context.repository_root.is_dir():
        raise FileNotFoundError(f"repository root not found: {context.repository_root}")
    if context.report_root.exists():
        raise FileExistsError(f"BPDD preflight report root already exists: {context.report_root}")


def _default_runner(gate: str) -> GateRunner:
    module = importlib.import_module("src.bpdd_runtime_preflight")
    runner = getattr(module, f"run_{gate.lower()}", None)
    if not callable(runner):
        raise RuntimeError(f"BPDD runtime preflight does not expose run_{gate.lower()}")
    return runner


def run_preflight(
    context: PreflightContext,
    *,
    gate_runners: Mapping[str, GateRunner] | None = None,
) -> dict[str, Any]:
    _validate_inputs(context)
    context.report_root.mkdir(parents=True, exist_ok=False)
    supplied = dict(gate_runners or {})
    unknown = set(supplied) - set(GATE_ORDER)
    if unknown:
        raise ValueError(f"unknown BPDD preflight gates: {sorted(unknown)}")
    states: dict[str, str] = {}
    blocked_by: str | None = None
    hashes: dict[str, str] = {}
    for gate in GATE_ORDER:
        if blocked_by is not None:
            evidence: dict[str, Any] = {
                "status": "blocked",
                "gate": gate,
                "blocked_by": blocked_by,
            }
        else:
            try:
                evidence = dict((supplied.get(gate) or _default_runner(gate))(context))
            except Exception as error:
                evidence = {
                    "status": "engineering_failed",
                    "gate": gate,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
        status = str(evidence.get("status", "engineering_failed"))
        if status not in {"passed", "engineering_failed", "scientific_failed", "blocked"}:
            status = "engineering_failed"
            evidence = {
                "status": status,
                "gate": gate,
                "reason": "invalid gate status",
            }
        if status != "passed" and blocked_by is None:
            blocked_by = gate
        states[gate] = status
        record = _record(gate, evidence)
        _write_create_only(context.report_root / f"{gate}.json", record)
        hashes[gate] = hashlib.sha256(_canonical(record)).hexdigest().upper()
    eligible = set(states) == set(GATE_ORDER) and all(
        states[gate] == "passed" for gate in GATE_ORDER
    )
    failed_status = (
        "scientific_failed"
        if any(value == "scientific_failed" for value in states.values())
        else "engineering_failed"
    )
    decision = {
        "status": "passed" if eligible else failed_status,
        "screen_eligible": eligible,
        "gate_states": states,
        "gate_report_sha256": hashes,
        "fixed_runtime": dict(FIXED_RUNTIME),
        "screen_authority": dict(SCREEN_AUTHORITY),
    }
    _write_create_only(context.report_root / "decision.json", _record("decision", decision))
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    decision = run_preflight(
        PreflightContext(
            protocol_manifest=args.protocol_manifest,
            initial_state=args.initial_state,
            dataset_root=args.dataset_root,
            report_root=args.report_root,
        )
    )
    print(json.dumps(decision, sort_keys=True, allow_nan=False))
    return 0 if decision["screen_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
