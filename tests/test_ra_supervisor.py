from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.supervise_ra_glgm import (
    TRAIN_STEPS,
    append_audit_event,
    build_train_command,
    run_name,
    validate_audit_chain,
    validate_smoke_advancement,
)


def test_supervisor_order_is_sequential_and_formal_follows_screen() -> None:
    assert TRAIN_STEPS == (
        ("smoke", "baseline"),
        ("smoke", "ra_glgm"),
        ("screen", "baseline"),
        ("screen", "ra_glgm"),
        ("formal", "baseline"),
        ("formal", "ra_glgm"),
    )


def test_train_command_exposes_no_scientific_overrides_and_formal_requires_gate(tmp_path: Path) -> None:
    common = {
        "python": Path("/venv/bin/python"),
        "protocol_manifest": tmp_path / "protocol.json",
        "initial_state": tmp_path / "initial.pt",
        "learnability_report": tmp_path / "learnability.json",
        "dataset_root": tmp_path / "VisDrone",
        "output_root": tmp_path / "runs",
        "stage": "screen",
        "variant": "ra_glgm",
    }
    command = build_train_command(**common)
    assert command[:2] == [str(Path("/venv/bin/python")), "scripts/train_rtdetr_ra_glgm.py"]
    assert command[command.index("--name") + 1] == run_name("screen", "ra_glgm")
    assert command[command.index("--learnability-report") + 1] == str(
        (tmp_path / "learnability.json").resolve()
    )
    assert not {
        "--epochs",
        "--batch",
        "--workers",
        "--device",
        "--optimizer",
        "--lr0",
        "--imgsz",
    }.intersection(command)
    with pytest.raises(ValueError, match="Gate"):
        build_train_command(**{**common, "stage": "formal"})


def test_audit_events_are_hash_chained_and_tampering_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    first = append_audit_event(path, {"event": "start"})
    second = append_audit_event(path, {"event": "exit", "returncode": 0})
    assert second["previous_sha256"] == first["event_sha256"]
    validate_audit_chain(path)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["event"] = "changed"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        validate_audit_chain(path)


def _smoke_decision(variant: str) -> dict:
    return {
        "decision": "complete",
        "stage": "smoke",
        "completed_epoch": 2,
        "amp_scale": 128.0,
        "amp_skipped_steps": 0,
        "public_gradient_finite": True,
        "fdr_gradient_finite": True,
        "ra_private_gradient_nonzero": True if variant == "ra_glgm" else None,
    }


@pytest.mark.parametrize("variant", ("baseline", "ra_glgm"))
def test_smoke_advancement_accepts_only_complete_optimizer_evidence(variant: str) -> None:
    validate_smoke_advancement(_smoke_decision(variant), variant=variant)


@pytest.mark.parametrize(
    ("variant", "field", "value"),
    (
        ("baseline", "amp_skipped_steps", 1),
        ("baseline", "public_gradient_finite", False),
        ("baseline", "fdr_gradient_finite", False),
        ("ra_glgm", "ra_private_gradient_nonzero", False),
        ("ra_glgm", "completed_epoch", 1),
    ),
)
def test_smoke_advancement_rejects_missing_engineering_evidence(
    variant: str, field: str, value: object
) -> None:
    decision = _smoke_decision(variant)
    decision[field] = value
    with pytest.raises(RuntimeError, match="Smoke2 optimizer/gradient gate"):
        validate_smoke_advancement(decision, variant=variant)
