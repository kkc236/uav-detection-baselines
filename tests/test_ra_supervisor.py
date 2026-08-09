from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

import scripts.supervise_ra_glgm as supervisor_module
from scripts.supervise_ra_glgm import (
    TRAIN_STEPS,
    append_audit_event,
    build_train_command,
    ensure_locked_evaluation,
    revalidate_supervisor_authority,
    run_name,
    validate_audit_chain,
    validate_smoke_advancement,
    validate_supervisor_evaluator,
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
        "public_gradient_nonzero": True,
        "fdr_gradient_nonzero": True,
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
        ("baseline", "public_gradient_nonzero", False),
        ("baseline", "fdr_gradient_nonzero", False),
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


@pytest.mark.parametrize("preexisting", (False, True))
def test_fresh_and_reused_locked_evaluations_share_the_same_post_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preexisting: bool
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    output = run / "locked-evaluation.jsonl"
    if preexisting:
        output.write_text("{}\n", encoding="utf-8")
    boundaries: list[str] = []
    verified: list[tuple[Path, str, str]] = []
    subprocess_calls: list[list[str]] = []

    monkeypatch.setattr(
        supervisor_module,
        "revalidate_supervisor_authority",
        lambda _protocol, _authority: boundaries.append("authority") or {},
    )
    monkeypatch.setattr(
        supervisor_module,
        "validate_evaluated_arm",
        lambda checked_run, *, variant, stage: verified.append(
            (checked_run, variant, stage)
        )
        or {},
    )

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        subprocess_calls.append(command)
        output.write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(supervisor_module.subprocess, "run", fake_run)
    origin = ensure_locked_evaluation(
        python=Path("python"),
        evaluator_script=tmp_path / "evaluator.py",
        protocol=tmp_path / "protocol.json",
        authority={"fixed": True},
        run=run,
        stage="formal",
        variant="baseline",
        audit=tmp_path / "audit.jsonl",
        environment={},
    )

    assert origin == ("reused" if preexisting else "fresh")
    assert boundaries == ["authority", "authority"]
    assert verified == [(run, "baseline", "formal")]
    assert bool(subprocess_calls) is not preexisting
    events = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "locked_evaluation_verified"
    assert events[-1]["origin"] == origin


def test_supervisor_boundary_rejects_a_replaced_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {"manifest_sha256": "A" * 64}
    monkeypatch.setattr(supervisor_module, "load_authority", lambda _path: dict(expected))
    assert revalidate_supervisor_authority(tmp_path / "protocol.json", expected) == expected

    monkeypatch.setattr(
        supervisor_module,
        "load_authority",
        lambda _path: {"manifest_sha256": "B" * 64},
    )
    with pytest.raises(ValueError, match="changed after supervisor start"):
        revalidate_supervisor_authority(tmp_path / "protocol.json", expected)


def test_supervisor_rejects_an_evaluator_cli_override(tmp_path: Path) -> None:
    locked = tmp_path / "locked.py"
    locked.write_text("# locked\n", encoding="utf-8")
    foreign = tmp_path / "foreign.py"
    foreign.write_text("# foreign\n", encoding="utf-8")
    authority = {
        "locked_evaluator": {
            "path": str(locked.resolve()),
            "sha256": supervisor_module.file_sha256(locked),
        }
    }

    assert validate_supervisor_evaluator(locked, authority) == locked.resolve()
    with pytest.raises(ValueError, match="path differs"):
        validate_supervisor_evaluator(foreign, authority)


def test_supervisor_rejects_locked_evaluator_byte_drift(tmp_path: Path) -> None:
    locked = tmp_path / "locked.py"
    locked.write_text("# original\n", encoding="utf-8")
    authority = {
        "locked_evaluator": {
            "path": str(locked.resolve()),
            "sha256": supervisor_module.file_sha256(locked),
        }
    }
    locked.write_text("# changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bytes differ"):
        validate_supervisor_evaluator(locked, authority)
