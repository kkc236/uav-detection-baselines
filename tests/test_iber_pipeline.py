from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts.run_iber_pipeline import (
    PHASE_ORDER,
    PipelineEvidence,
    PipelinePaths,
    atomic_write_state,
    build_phase_command,
    finalize_screen_decision,
    next_pipeline_phase,
    run_tracked_subprocess,
    select_verified_resume,
    validate_prerequisite_sources,
)


SOURCE_COMMIT = "a" * 40


def _evidence(**updates: object) -> PipelineEvidence:
    values: dict[str, object] = {
        "authority": None,
        "gate0": None,
        "stock_authority": None,
        "cache_complete": False,
        "gate1": None,
        "screen_completed_epoch": 0,
        "screen_verified_epoch": 0,
        "screen_decision": None,
        "source_consistent": True,
    }
    values.update(updates)
    return PipelineEvidence(**values)


def _passed_through_gate1(**updates: object) -> PipelineEvidence:
    return _evidence(
        authority="passed_with_runtime_amendment",
        gate0="passed_with_runtime_amendment",
        stock_authority="passed_with_runtime_amendment",
        cache_complete=True,
        gate1="passed",
        **updates,
    )


def test_state_machine_has_exact_order_and_terminal_decisions() -> None:
    assert PHASE_ORDER == (
        "authority",
        "gate0",
        "stock_authority",
        "cache",
        "probe",
        "screen30",
        "screen_decision",
    )
    assert next_pipeline_phase(_evidence()) == "authority"
    assert next_pipeline_phase(_evidence(authority="passed")) == "gate0"
    assert next_pipeline_phase(
        _evidence(authority="passed", gate0="passed")
    ) == "stock_authority"
    assert next_pipeline_phase(
        _evidence(authority="passed", gate0="passed", stock_authority="passed")
    ) == "cache"
    assert next_pipeline_phase(
        _evidence(
            authority="passed",
            gate0="passed",
            stock_authority="passed",
            cache_complete=True,
        )
    ) == "probe"
    assert next_pipeline_phase(_passed_through_gate1()) == "screen30"
    assert next_pipeline_phase(
        _passed_through_gate1(
            screen_completed_epoch=30,
            screen_verified_epoch=30,
        )
    ) == "screen_decision"
    assert next_pipeline_phase(
        _passed_through_gate1(
            screen_completed_epoch=30,
            screen_verified_epoch=30,
            screen_decision="passed",
        )
    ) == "screen_complete"
    assert next_pipeline_phase(
        _passed_through_gate1(
            screen_completed_epoch=30,
            screen_verified_epoch=30,
            screen_decision="scientific_failed",
        )
    ) == "scientific_failed"


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"authority": "engineering_invalid"}, "engineering_invalid"),
        (
            {"authority": "passed", "gate0": "engineering_invalid"},
            "engineering_invalid",
        ),
        (
            {
                "authority": "passed",
                "gate0": "passed",
                "stock_authority": "engineering_invalid",
            },
            "engineering_invalid",
        ),
        (
            {
                "authority": "passed",
                "gate0": "passed",
                "stock_authority": "passed",
                "cache_complete": True,
                "gate1": "scientific_failed",
            },
            "scientific_failed",
        ),
        (
            {
                "authority": "passed",
                "gate0": "passed",
                "stock_authority": "passed",
                "cache_complete": True,
                "gate1": "engineering_invalid",
            },
            "engineering_invalid",
        ),
        (
            {
                "authority": "passed",
                "source_consistent": False,
            },
            "engineering_invalid",
        ),
    ],
)
def test_engineering_and_gate1_failures_stop_before_screen(
    updates: dict[str, object], expected: str
) -> None:
    assert next_pipeline_phase(_evidence(**updates)) == expected


def test_incomplete_screen_can_only_continue_from_verified_tip() -> None:
    assert next_pipeline_phase(
        _passed_through_gate1(
            screen_completed_epoch=18,
            screen_verified_epoch=17,
        )
    ) == "screen30"
    assert next_pipeline_phase(
        _passed_through_gate1(
            screen_completed_epoch=17,
            screen_verified_epoch=18,
        )
    ) == "engineering_invalid"
    assert next_pipeline_phase(
        _passed_through_gate1(
            screen_completed_epoch=31,
            screen_verified_epoch=30,
        )
    ) == "engineering_invalid"


def test_prerequisite_reports_must_share_the_exact_source_commit() -> None:
    reports = {
        "authority": {"source_commit": SOURCE_COMMIT},
        "gate0": {"authority": {"source_commit": SOURCE_COMMIT}},
        "stock_authority": {"source_commit": SOURCE_COMMIT},
        "cache": {"authority": {"source_commit": SOURCE_COMMIT}},
        "gate1": {
            "reports": {
                arm: {"cache_authority": {"source_commit": SOURCE_COMMIT}}
                for arm in ("b0", "b1", "b2", "b3")
            }
        },
    }
    assert validate_prerequisite_sources(reports, SOURCE_COMMIT) == SOURCE_COMMIT
    reports["gate1"]["reports"]["b2"]["cache_authority"][
        "source_commit"
    ] = "b" * 40
    with pytest.raises(ValueError, match="source commit"):
        validate_prerequisite_sources(reports, SOURCE_COMMIT)


def _write_verified_epoch(run_root: Path, epoch: int) -> None:
    checkpoint = run_root / "checkpoints" / f"epoch-{epoch:04d}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"checkpoint-{epoch}".encode())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ledger = run_root / "publication-ledger.jsonl"
    with ledger.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                {
                    "design_version": "iber-be-v1.0",
                    "stage": "screen",
                    "probe": "b3",
                    "seed": 0,
                    "completed_epoch": epoch,
                    "verified": True,
                    "source_commit": SOURCE_COMMIT,
                    "checkpoint": {"sha256": digest},
                },
                sort_keys=True,
            )
            + "\n"
        )


def test_resume_selects_only_highest_contiguous_verified_checkpoint(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "screen"
    _write_verified_epoch(run_root, 1)
    _write_verified_epoch(run_root, 2)
    resume = select_verified_resume(run_root, source_commit=SOURCE_COMMIT)
    assert resume is not None
    assert resume.name == "epoch-0002.pt"

    rows = (run_root / "publication-ledger.jsonl").read_text(encoding="utf-8")
    (run_root / "publication-ledger.jsonl").write_text(
        rows.replace('"completed_epoch": 2', '"completed_epoch": 3'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contiguous"):
        select_verified_resume(run_root, source_commit=SOURCE_COMMIT)


def test_resume_rejects_unverified_hash_or_source_drift(tmp_path: Path) -> None:
    run_root = tmp_path / "screen"
    _write_verified_epoch(run_root, 1)
    ledger = run_root / "publication-ledger.jsonl"
    row = json.loads(ledger.read_text(encoding="utf-8"))

    for field, value, message in (
        ("verified", False, "verified"),
        ("source_commit", "b" * 40, "source commit"),
    ):
        changed = dict(row)
        changed[field] = value
        ledger.write_text(json.dumps(changed) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            select_verified_resume(run_root, source_commit=SOURCE_COMMIT)

    ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")
    row["checkpoint"]["sha256"] = "0" * 64
    ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        select_verified_resume(run_root, source_commit=SOURCE_COMMIT)


def test_atomic_state_preserves_append_only_history(tmp_path: Path) -> None:
    path = tmp_path / "pipeline-state.json"
    first = {
        "phase": "gate0",
        "history": [{"phase": "authority", "sequence": 1}],
    }
    atomic_write_state(path, first)
    second = {
        "phase": "stock_authority",
        "history": [
            {"phase": "authority", "sequence": 1},
            {"phase": "gate0", "sequence": 2},
        ],
    }
    atomic_write_state(path, second)
    assert json.loads(path.read_text(encoding="utf-8")) == second
    assert not path.with_suffix(".json.tmp").exists()
    with pytest.raises(ValueError, match="append-only"):
        atomic_write_state(
            path,
            {"phase": "cache", "history": [{"phase": "tampered"}]},
        )


def test_subprocess_records_pid_process_group_exact_command_and_safe_log(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "pipeline-state.json"
    state = {"phase": "gate0", "history": []}
    command = [sys.executable, "-c", "print('child-ok')"]
    result = run_tracked_subprocess(
        command,
        phase="gate0",
        run_root=tmp_path,
        state_path=state_path,
        state=state,
    )
    assert result == 0
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    process = persisted["last_process"]
    assert process["pid"] > 0
    assert process["process_group_id"] == process["pid"]
    assert process["command"] == command
    assert process["return_code"] == 0
    log = Path(process["log"]).read_text(encoding="utf-8")
    assert json.dumps(command, separators=(",", ":")) in log
    assert "child-ok" in log

    with pytest.raises(ValueError, match="credential"):
        run_tracked_subprocess(
            [sys.executable, "-c", "pass", "--token", "github_pat_secret"],
            phase="gate0",
            run_root=tmp_path,
            state_path=state_path,
            state=persisted,
        )
    assert "github_pat_secret" not in "".join(
        path.read_text(encoding="utf-8")
        for path in tmp_path.rglob("*.log")
    )


def test_subprocess_redacts_credentials_emitted_by_child(tmp_path: Path) -> None:
    state_path = tmp_path / "pipeline-state.json"
    state = {"phase": "probe", "history": []}
    emitted = "github_pat_DO_NOT_PERSIST"
    result = run_tracked_subprocess(
        [
            sys.executable,
            "-c",
            "print('github_' + 'pat_' + 'DO_NOT_PERSIST')",
        ],
        phase="probe",
        run_root=tmp_path,
        state_path=state_path,
        state=state,
    )
    assert result == 0
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    log = Path(persisted["last_process"]["log"]).read_text(encoding="utf-8")
    assert emitted not in log
    assert "[REDACTED]" in log

    with pytest.raises(ValueError, match="credential"):
        run_tracked_subprocess(
            [sys.executable, "-c", "pass", "--password=do-not-persist"],
            phase="probe",
            run_root=tmp_path,
            state_path=state_path,
            state=persisted,
        )


def test_phase_commands_use_independent_subprocess_boundaries(tmp_path: Path) -> None:
    paths = PipelinePaths(
        baseline_checkpoint=tmp_path / "baseline.pt",
        dataset_root=tmp_path / "dataset",
        run_root=tmp_path / "run",
        cache_root=tmp_path / "cache",
        publication_config=tmp_path / "publication.json",
        device="0",
    )
    assert "run_iber_canary.py" in " ".join(build_phase_command("gate0", paths))
    stock = build_phase_command("stock_authority", paths)
    assert "evaluate_iber_stock.py" in " ".join(stock)
    assert "--device" not in stock
    assert "cache_iber_evidence.py" in " ".join(
        build_phase_command("cache", paths)
    )
    assert "run_iber_probe.py" in " ".join(
        build_phase_command("probe", paths)
    )
    screen = build_phase_command(
        "screen30", paths, resume_checkpoint=tmp_path / "epoch-0017.pt"
    )
    assert "train_iber.py" in " ".join(screen)
    assert "--resume-checkpoint" in screen
    with pytest.raises(ValueError, match="no subprocess command"):
        build_phase_command("screen_decision", paths)
    for command in (screen,):
        for forbidden in (
            "--phase",
            "--skip",
            "--epochs",
            "--seed",
            "--batch",
            "--workers",
            "--imgsz",
            "--token",
        ):
            assert forbidden not in command


@pytest.mark.parametrize("status", ["passed", "scientific_failed"])
def test_screen_decision_finalizes_existing_epoch30_evaluation(
    tmp_path: Path, status: str
) -> None:
    evaluation = tmp_path / "screen" / "evaluations" / "epoch-0030.json"
    evaluation.parent.mkdir(parents=True)
    evaluation.write_text(
        json.dumps(
            {
                "format_version": 1,
                "design_version": "iber-be-v1.0",
                "stage": "screen",
                "probe": "b3",
                "seed": 0,
                "epoch": 30,
                "source_commit": SOURCE_COMMIT,
                "decision": {"status": status, "conditions": {"map": status == "passed"}},
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "screen" / "screen-decision.json"
    payload = finalize_screen_decision(
        evaluation,
        destination,
        source_commit=SOURCE_COMMIT,
    )
    assert payload["status"] == status
    assert payload["evaluation_epoch"] == 30
    assert payload["source_commit"] == SOURCE_COMMIT
    assert json.loads(destination.read_text(encoding="utf-8"))["status"] == status


def test_screen_decision_rejects_best_epoch_or_source_substitution(
    tmp_path: Path,
) -> None:
    evaluation = tmp_path / "epoch-0029.json"
    payload = {
        "design_version": "iber-be-v1.0",
        "stage": "screen",
        "probe": "b3",
        "seed": 0,
        "epoch": 29,
        "source_commit": SOURCE_COMMIT,
        "decision": {"status": "passed"},
    }
    evaluation.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="epoch30"):
        finalize_screen_decision(
            evaluation,
            tmp_path / "decision.json",
            source_commit=SOURCE_COMMIT,
        )
    payload["epoch"] = 30
    payload["source_commit"] = "b" * 40
    evaluation.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source commit"):
        finalize_screen_decision(
            evaluation,
            tmp_path / "decision.json",
            source_commit=SOURCE_COMMIT,
        )


def test_pipeline_source_has_no_old_identity_shared_task_imports_or_skip_cli() -> None:
    source = Path("scripts/run_iber_pipeline.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "itber" not in lowered.replace("iber", "")
    assert "src.iber_evaluation" not in source
    assert "src.iber_publication" not in source
    assert "from scripts.train_iber" not in source
    assert 'add_argument("--phase"' not in source
    assert 'add_argument("--skip"' not in source
