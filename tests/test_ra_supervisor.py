from __future__ import annotations

import json
import os
from types import SimpleNamespace
from pathlib import Path

import pytest

import scripts.supervise_ra_glgm as supervisor_module
from scripts.supervise_ra_glgm import (
    EXPLORE50_TRAIN_STEPS,
    TRAIN_STEPS,
    append_audit_event,
    acquire_lock,
    build_evaluator_command,
    build_gate_command,
    build_train_command,
    ensure_locked_evaluation,
    ensure_fresh_run_slot,
    release_lock,
    revalidate_supervisor_authority,
    run_name,
    validate_audit_chain,
    validate_smoke_advancement,
    validate_supervisor_evaluator,
)


def test_supervisor_order_is_sequential_smoke_then_screen_then_formal() -> None:
    assert TRAIN_STEPS == (
        ("smoke", "baseline"),
        ("smoke", "ra_glgm"),
        ("screen", "baseline"),
        ("screen", "ra_glgm"),
        ("formal", "baseline"),
        ("formal", "ra_glgm"),
    )
    assert EXPLORE50_TRAIN_STEPS == (
        ("explore50", "baseline"),
        ("explore50", "ra_glgm"),
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


def test_screen30_evaluator_and_gate_commands_use_frozen_tail5(tmp_path: Path) -> None:
    evaluator = build_evaluator_command(
        python=Path("/venv/bin/python"),
        evaluator_script=tmp_path / "evaluator.py",
        run=tmp_path / "screen-run",
        protocol_manifest=tmp_path / "protocol.json",
        stage="screen",
    )
    assert evaluator[evaluator.index("--epochs") + 1] == "26,27,28,29,30"

    gate = build_gate_command(
        python=Path("/venv/bin/python"),
        output_root=tmp_path / "runs",
        gate_output=tmp_path / "screen-gate.json",
        stage="screen",
    )
    assert gate[gate.index("--stage") + 1] == "screen"
    assert run_name("screen", "baseline") in gate[gate.index("--baseline-run") + 1]
    assert run_name("screen", "ra_glgm") in gate[gate.index("--ra-run") + 1]

    explore = build_evaluator_command(
        python=Path("/venv/bin/python"),
        evaluator_script=tmp_path / "evaluator.py",
        run=tmp_path / "explore50-run",
        protocol_manifest=tmp_path / "protocol.json",
        stage="explore50",
    )
    assert explore[explore.index("--epochs") + 1] == "5,10,15,20,25,30,35,40,45,50"
    assert run_name("explore50", "baseline").endswith("ra-glgm-v1.2-long50")


def test_supervisor_accepts_only_v12_formal_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "protocol_sha256": supervisor_module.RA_EXPERIMENT_PROTOCOL_SHA256,
        "report_name": "RA-GLGM-Formal100-v1.2",
        "primary_evidence": ["epoch100", "tail3_mean"],
        "engineering": {"complete": True},
        "formal_success": False,
    }
    monkeypatch.setattr(
        supervisor_module,
        "validate_formal_report",
        lambda *_args, **_kwargs: dict(expected),
    )
    assert supervisor_module._validate_formal_report(tmp_path / "report.json") == expected


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


def test_supervisor_reclaims_only_a_proven_stale_pid_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "supervisor.lock"
    lock.write_text(
        json.dumps({"pid": 999_999, "process_start_identity": "dead-start"}),
        encoding="ascii",
    )
    current = os.getpid()
    monkeypatch.setattr(
        supervisor_module,
        "_process_start_identity",
        lambda pid: "current-start" if pid == current else None,
    )
    acquire_lock(lock)
    owner = json.loads(lock.read_text(encoding="ascii"))
    assert owner == {"pid": current, "process_start_identity": "current-start"}
    release_lock(lock)
    assert not lock.exists()


def test_supervisor_rejects_a_live_pid_start_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "supervisor.lock"
    lock.write_text(
        json.dumps({"pid": 123, "process_start_identity": "live-start"}),
        encoding="ascii",
    )
    monkeypatch.setattr(
        supervisor_module,
        "_process_start_identity",
        lambda pid: "live-start" if pid == 123 else "current-start",
    )
    with pytest.raises(RuntimeError, match="already exists"):
        acquire_lock(lock)


def test_fresh_orphan_run_is_atomically_quarantined_before_exact_fresh_launch(
    tmp_path: Path,
) -> None:
    run = tmp_path / "screen-seed0-baseline-ra-glgm-v1.2"
    run.mkdir()
    (run / "partial.txt").write_text("preserve me", encoding="utf-8")
    quarantine = ensure_fresh_run_slot(run)
    assert quarantine is not None
    assert not run.exists()
    assert quarantine.parent == tmp_path.resolve()
    assert (quarantine / "partial.txt").read_text(encoding="utf-8") == "preserve me"


def test_fresh_slot_never_quarantines_a_runtime_manifest(tmp_path: Path) -> None:
    run = tmp_path / "screen-seed0-baseline-ra-glgm-v1.2"
    run.mkdir()
    (run / "ra-run.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must use recovery"):
        ensure_fresh_run_slot(run)
    assert run.is_dir()


def _terminal_status_args(tmp_path: Path) -> SimpleNamespace:
    initial = tmp_path / "initial.pt"
    initial.write_bytes(b"initial")
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    learnability = tmp_path / "learnability.json"
    learnability.write_text("{}", encoding="utf-8")
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("# evaluator", encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    return SimpleNamespace(
        protocol_manifest=protocol,
        initial_state=initial,
        learnability_report=learnability,
        dataset_root=dataset,
        output_root=tmp_path / "runs",
        evaluator_script=evaluator,
        python=Path("python"),
        audit=tmp_path / "audit.jsonl",
        status=tmp_path / "status.json",
        log=tmp_path / "supervisor.log",
        lock=tmp_path / "supervisor.lock",
        poll_seconds=1,
        max_attempts=1,
    )


def _stub_terminal_status_boundaries(
    monkeypatch: pytest.MonkeyPatch, args: SimpleNamespace
) -> None:
    authority = {
        "initial_state": {"sha256": supervisor_module.file_sha256(args.initial_state)}
    }
    monkeypatch.setattr(supervisor_module, "TRAIN_STEPS", (("smoke", "baseline"),))
    monkeypatch.setattr(supervisor_module, "load_authority", lambda _path: authority)
    monkeypatch.setattr(supervisor_module, "validate_learnability_report", lambda *_a, **_k: {})
    monkeypatch.setattr(supervisor_module, "validate_supervisor_evaluator", lambda *_a, **_k: args.evaluator_script)
    monkeypatch.setattr(supervisor_module, "revalidate_supervisor_authority", lambda *_a, **_k: authority)
    monkeypatch.setattr(supervisor_module, "acquire_lock", lambda _path: None)
    monkeypatch.setattr(supervisor_module, "release_lock", lambda _path: None)


def test_supervisor_atomically_records_failed_terminal_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _terminal_status_args(tmp_path)
    _stub_terminal_status_boundaries(monkeypatch, args)
    monkeypatch.setattr(
        supervisor_module,
        "_run_child",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("injected failure")),
    )

    with pytest.raises(RuntimeError, match="injected failure"):
        supervisor_module.run_supervisor(args)
    status = json.loads(args.status.read_text(encoding="utf-8"))
    assert status["process_state"] == "failed"
    assert status["error_type"] == "RuntimeError"
    assert status["stage"] == "smoke"


def test_supervisor_atomically_records_interrupted_terminal_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _terminal_status_args(tmp_path)
    _stub_terminal_status_boundaries(monkeypatch, args)
    orphan = args.output_root / run_name("smoke", "baseline")
    orphan.mkdir(parents=True)
    (orphan / "partial.txt").write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(supervisor_module, "_run_child", lambda *_a, **_k: (143, 15))

    assert supervisor_module.run_supervisor(args) == 143
    status = json.loads(args.status.read_text(encoding="utf-8"))
    assert status["process_state"] == "interrupted"
    assert status["signal"] == 15
    events = [json.loads(line) for line in args.audit.read_text(encoding="utf-8").splitlines()]
    quarantine = next(
        event["quarantine"]
        for event in events
        if event["event"] == "manifest_free_run_quarantined"
    )
    assert (Path(quarantine) / "partial.txt").read_text(encoding="utf-8") == "preserve"
