"""Supervise Gate 0 -> Probe -> screen -> formal I-TBER execution."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.train_itber import (  # noqa: E402
    stage_protocol,
    validate_gate1_cache_manifest,
    validate_resume_checkpoint,
)
from src.itber_evaluation import evaluate_formal_gate, write_immutable_report  # noqa: E402
from src.itber_protocol import (  # noqa: E402
    ACCEPTED_GATE_STATUSES,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_CATEGORY_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    EXPECTED_ENVIRONMENT,
    ProtocolViolation,
    current_execution_environment,
    validate_authorities,
)
from src.lpr_protocol import (  # noqa: E402
    CATEGORY_NAMES,
    category_mapping_sha256,
    dataset_signature,
    file_sha256,
    select_hashed_subset,
    subset_signature,
    ultralytics_source_paths,
)


TERMINAL_PHASES = {"engineering_invalid", "scientific_failed", "formal_complete"}


@dataclass(frozen=True)
class PipelineEvidence:
    authority: str | None
    gate0: str | None
    stock_authority: str | None
    cache_complete: bool
    gate1: str | None
    screen: str | None
    formal: str | None


def next_pipeline_phase(evidence: PipelineEvidence) -> str:
    """Return the only permitted next phase under immutable gate evidence."""
    for status in (evidence.authority, evidence.gate0, evidence.stock_authority):
        if status == "engineering_invalid":
            return "engineering_invalid"
    if evidence.authority is None:
        return "authority"
    if evidence.authority not in ACCEPTED_GATE_STATUSES:
        return "engineering_invalid"
    if evidence.gate0 is None:
        return "gate0"
    if evidence.gate0 not in ACCEPTED_GATE_STATUSES:
        return "engineering_invalid"
    if evidence.stock_authority is None:
        return "stock_authority"
    if evidence.stock_authority not in ACCEPTED_GATE_STATUSES:
        return "engineering_invalid"
    if not evidence.cache_complete:
        return "cache"
    if evidence.gate1 is None:
        return "probe"
    if evidence.gate1 == "engineering_invalid":
        return "engineering_invalid"
    if evidence.gate1 != "passed":
        return "scientific_failed"
    if evidence.screen is None:
        return "screen"
    if evidence.screen == "engineering_invalid":
        return "engineering_invalid"
    if evidence.screen != "passed":
        return "scientific_failed"
    if evidence.formal is None:
        return "formal"
    if evidence.formal == "engineering_invalid":
        return "engineering_invalid"
    if evidence.formal != "passed":
        return "scientific_failed"
    return "formal_complete"


def atomic_write_state(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically update state while requiring append-only history."""
    destination = Path(path)
    history = payload.get("history")
    if not isinstance(history, list):
        raise ValueError("I-TBER pipeline state history must be a list")
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        previous_history = previous.get("history", [])
        if history[: len(previous_history)] != previous_history:
            raise ValueError("I-TBER pipeline history is not append-only")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, destination)
    return destination


def build_train_command(
    *,
    stage: str,
    baseline_checkpoint: Path,
    dataset_root: Path,
    cache_manifest: Path,
    output_root: Path,
    publication_config: Path,
    resume_checkpoint: Path | None,
    cache_manifest_sha256: str,
) -> list[str]:
    """Build one frozen stage command and reject cross-stage resume."""
    stage_protocol(stage)
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "train_itber.py"),
        "--stage",
        stage,
        "--baseline-checkpoint",
        str(baseline_checkpoint),
        "--dataset-root",
        str(dataset_root),
        "--gate1-cache-manifest",
        str(cache_manifest),
        "--publication-config",
        str(publication_config),
        "--output-root",
        str(output_root),
    ]
    if resume_checkpoint is not None:
        artifact = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        validate_resume_checkpoint(
            artifact,
            stage=stage,
            cache_manifest_sha256=cache_manifest_sha256,
        )
        command.extend(("--resume-checkpoint", str(resume_checkpoint)))
    return command


def _json_status(path: Path, *, nested_decision: bool = False) -> str | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = payload.get("decision", {}) if nested_decision else payload
    status = source.get("status")
    return str(status) if status is not None else None


def _latest_checkpoint(run_root: Path, *, stage: str, cache_sha: str) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in (run_root / "checkpoints").glob("epoch-*.pt"):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        validate_resume_checkpoint(artifact, stage=stage, cache_manifest_sha256=cache_sha)
        candidates.append((int(artifact["epoch"]), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _formal_decision(run_root: Path) -> dict[str, Any]:
    reports = []
    for epoch in range(26, 31):
        path = run_root / "evaluations" / f"epoch-{epoch:04d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing formal evaluation epoch {epoch}")
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    deltas = [float(report["refined"]["map"]) - float(report["stock"]["map"]) for report in reports]
    last = reports[-1]
    diagnostics = dict(last["diagnostics"])
    all_detector_unchanged = all(
        report["diagnostics"].get("detector_sha_before")
        == report["diagnostics"].get("detector_sha_after")
        for report in reports
    )
    if not all_detector_unchanged:
        diagnostics["detector_sha_after"] = "FORMAL_DETECTOR_DRIFT"
    decision = evaluate_formal_gate(
        last["stock"],
        last["refined"],
        diagnostics,
        tail5_map_delta=sum(deltas) / len(deltas),
    )
    return {
        **decision,
        "epochs": [26, 27, 28, 29, 30],
        "tail5_deltas": deltas,
        "all_detector_unchanged": all_detector_unchanged,
    }


def _authority_report(baseline: Path, dataset_root: Path) -> dict[str, Any]:
    images = sorted((dataset_root / "images" / "train").glob("*.jpg"))
    subset = select_hashed_subset(images, root=dataset_root, fraction=0.10)
    source_sha256 = {
        name: file_sha256(path) for name, path in ultralytics_source_paths().items()
    }
    actual = {
        "baseline_sha256": file_sha256(baseline),
        "dataset_sha256": str(dataset_signature(dataset_root)["sha256"]),
        "subset_sha256": subset_signature(subset, root=dataset_root),
        "category_sha256": category_mapping_sha256(CATEGORY_NAMES),
        "environment": current_execution_environment(),
        "source_sha256": source_sha256,
    }
    try:
        authority = validate_authorities(
            baseline_sha256=str(actual["baseline_sha256"]),
            dataset_sha256=str(actual["dataset_sha256"]),
            subset_sha256=str(actual["subset_sha256"]),
            category_sha256=str(actual["category_sha256"]),
            source_sha256=source_sha256,
            environment=actual["environment"],
        )
    except ProtocolViolation as error:
        return {
            "status": "engineering_invalid",
            "expected_environment": EXPECTED_ENVIRONMENT,
            "actual": actual,
            "violations": error.violations,
        }
    return {**authority, "actual": actual, "violations": {}}


def _pipeline_evidence(run_root: Path, cache_root: Path) -> PipelineEvidence:
    gate0_reports = sorted((run_root / "gate0").glob("attempt-*.json"))
    gate0 = _json_status(gate0_reports[-1]) if gate0_reports else None
    return PipelineEvidence(
        authority=_json_status(run_root / "authority.json"),
        gate0=gate0,
        stock_authority=_json_status(run_root / "stock-authority.json"),
        cache_complete=(cache_root / "manifest.json").is_file(),
        gate1=_json_status(run_root / "probe" / "gate1-decision.json"),
        screen=_json_status(
            run_root / "screen" / "evaluations" / "epoch-0012.json",
            nested_decision=True,
        ),
        formal=_json_status(run_root / "formal" / "formal-decision.json"),
    )


def _run_command(
    command: list[str],
    *,
    phase: str,
    run_root: Path,
    state_path: Path,
    state: dict[str, Any],
) -> int:
    log_path = run_root / "logs" / f"{phase}-{time.time_ns()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=os.name != "nt",
        )
        state["active_process"] = {
            "phase": phase,
            "pid": process.pid,
            "command": command,
            "log": str(log_path.resolve()),
        }
        atomic_write_state(state_path, state)
        return_code = process.wait()
    state["last_process"] = {**state.pop("active_process"), "return_code": return_code}
    atomic_write_state(state_path, state)
    return return_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--screen-publication-config", type=Path, required=True)
    parser.add_argument("--formal-publication-config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    baseline = args.baseline_checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()
    run_root = args.run_root.resolve()
    cache_root = args.cache_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "pipeline-state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {"format_version": 1, "design_version": "itber-v1.1", "history": []}
    )
    while True:
        evidence = _pipeline_evidence(run_root, cache_root)
        phase = next_pipeline_phase(evidence)
        if state.get("phase") != phase:
            state["phase"] = phase
            state["history"].append({"phase": phase, "time_ns": time.time_ns()})
            atomic_write_state(state_path, state)
        if phase in TERMINAL_PHASES:
            return 0 if phase == "formal_complete" else (2 if phase == "scientific_failed" else 1)
        if phase == "authority":
            write_immutable_report(run_root / "authority.json", _authority_report(baseline, dataset_root))
            continue
        if phase == "gate0":
            gate0_root = run_root / "gate0"
            gate0_root.mkdir(parents=True, exist_ok=True)
            output = gate0_root / f"attempt-{len(list(gate0_root.glob('attempt-*.json'))) + 1:03d}.json"
            command = [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "run_itber_canary.py"),
                "--baseline-checkpoint",
                str(baseline),
                "--dataset-root",
                str(dataset_root),
                "--output",
                str(output),
            ]
        elif phase == "cache":
            command = [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "cache_itber_evidence.py"),
                "--baseline-checkpoint",
                str(baseline),
                "--dataset-root",
                str(dataset_root),
                "--output-root",
                str(cache_root),
            ]
        elif phase == "probe":
            command = [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "run_itber_probe.py"),
                "--cache-root",
                str(cache_root),
                "--output-root",
                str(run_root / "probe"),
            ]
        elif phase in {"screen", "formal"}:
            cache_manifest = cache_root / "manifest.json"
            cache_sha = validate_gate1_cache_manifest(cache_manifest)
            stage_root = run_root / phase
            resume = _latest_checkpoint(stage_root, stage=phase, cache_sha=cache_sha)
            publication_config = (
                args.screen_publication_config if phase == "screen" else args.formal_publication_config
            ).resolve()
            command = build_train_command(
                stage=phase,
                baseline_checkpoint=baseline,
                dataset_root=dataset_root,
                cache_manifest=cache_manifest,
                output_root=stage_root,
                publication_config=publication_config,
                resume_checkpoint=resume,
                cache_manifest_sha256=cache_sha,
            )
        else:
            raise RuntimeError(f"unhandled I-TBER pipeline phase: {phase}")
        return_code = _run_command(
            command,
            phase=phase,
            run_root=run_root,
            state_path=state_path,
            state=state,
        )
        if return_code != 0:
            refreshed = next_pipeline_phase(_pipeline_evidence(run_root, cache_root))
            if refreshed not in TERMINAL_PHASES:
                state["phase"] = "engineering_invalid"
                state["history"].append(
                    {"phase": "engineering_invalid", "failed_action": phase, "return_code": return_code, "time_ns": time.time_ns()}
                )
                atomic_write_state(state_path, state)
                return 1
            continue
        if phase == "formal":
            formal_root = run_root / "formal"
            final_checkpoint = formal_root / "checkpoints" / "epoch-0030.pt"
            if final_checkpoint.is_file():
                write_immutable_report(formal_root / "formal-decision.json", _formal_decision(formal_root))


if __name__ == "__main__":
    raise SystemExit(main())
