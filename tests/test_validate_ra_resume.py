from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import scripts.validate_ra_resume as resume_module
from scripts.validate_ra_resume import validate_optimizer_evidence, validate_resume
from src.ra_experiment_protocol import RA_EXPERIMENT_PROTOCOL_SHA256, file_sha256


@pytest.fixture(autouse=True)
def _stub_learnability_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resume_module,
        "validate_learnability_report",
        lambda _path, **_kwargs: {},
    )


def _write_partial_run(tmp_path: Path, completed: int = 3) -> tuple[Path, Path, Path]:
    initial = tmp_path / "initial.pt"
    initial.write_bytes(b"paired")
    source = {"git_commit": "a" * 40, "tree_sha256": "B" * 64}
    identity = {
        "source_sha256": "S" * 64,
        "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
        "stage": "screen",
        "variant": "baseline",
        "seed": 0,
        "pair_id": "paired-screen-seed0",
        "run_id": "baseline-screen-seed0",
    }
    authority = {
        "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
        "source": source,
        "gpu_uuid": "GPU-fixed",
        "initial_state": {"path": str(initial.resolve()), "sha256": file_sha256(initial)},
        "run_identities": {"baseline_screen": identity},
    }
    protocol = tmp_path / "protocol.json"
    dataset_authority = {
        "root": str((tmp_path / "VisDrone").resolve()),
        "positive": {"sha256": "D" * 64},
        "ignore": {"sha256": "E" * 64},
    }
    authority["dataset_authority"] = dataset_authority
    protocol.write_text(json.dumps(authority), encoding="utf-8")
    learnability = tmp_path / "learnability.json"
    learnability.write_text("{}\n", encoding="utf-8")
    run = tmp_path / "run"
    weights = run / "weights"
    weights.mkdir(parents=True)
    runtime = {
        "format_version": 1,
        "protocol_sha256": RA_EXPERIMENT_PROTOCOL_SHA256,
        "source": source,
        "run_identity": identity,
        "initial_state": authority["initial_state"],
        "dataset_authority": dataset_authority,
        "learnability_report_sha256": file_sha256(learnability),
        "gpu_uuid": "GPU-fixed",
        "schedule_epochs": 50,
        "cutoff_epoch": 30,
    }
    (run / "ra-run.json").write_text(json.dumps(runtime), encoding="utf-8")
    evidence, optimizer, queue = [], [], []
    for epoch in range(1, completed + 1):
        checkpoint = weights / f"epoch{epoch - 1}.pt"
        torch.save(
            {"epoch": epoch - 1, "optimizer": {"state": {}}, "ema": {"state": {}}},
            checkpoint,
        )
        evidence.append(
            {
                "completed_epoch": epoch,
                "variant": "baseline",
                "stage": "screen",
                "run_id": identity["run_id"],
                "map": 0.1,
                "map50": 0.2,
                "map75": 0.08,
                "precision": 0.3,
                "recall": 0.4,
                "cuda_peak_mib": 15000.0,
            }
        )
        optimizer.append(
            {
                "optimizer_attempt": epoch,
                "completed_epoch": epoch,
                "run_id": identity["run_id"],
                "variant": "baseline",
                "stage": "screen",
                "amp_scale_before": 128.0,
                "amp_scale_after": 128.0,
                "amp_step_skipped": False,
                "gradient_norm_finite": True,
                "gradient_norm": 4.0,
                "fdr_gradient_norm": 2.0,
            }
        )
        queue.append(
            {
                "run_id": identity["run_id"],
                "completed_epoch": epoch,
                "status": "pending",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint),
            }
        )
    (run / "ra-epochs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in evidence), encoding="utf-8"
    )
    (run / "optimizer-evidence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in optimizer), encoding="utf-8"
    )
    (run / "publication-queue.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in queue), encoding="utf-8"
    )
    return run, protocol, learnability


def test_resume_selects_only_latest_exact_epoch_checkpoint(tmp_path: Path) -> None:
    run, protocol, learnability = _write_partial_run(tmp_path)

    decision = validate_resume(
        run,
        variant="baseline",
        stage="screen",
        protocol_manifest=protocol,
        learnability_report=learnability,
    )

    assert decision["decision"] == "resume"
    assert decision["completed_epoch"] == 3
    assert decision["checkpoint"].endswith("epoch2.pt")
    assert decision["authority"] == "same-run exact-epoch only"
    assert decision["optimizer_attempts"] == 3
    assert decision["amp_skipped_steps"] == 0
    assert decision["public_gradient_finite"] is True
    assert decision["fdr_gradient_finite"] is True
    assert decision["public_gradient_nonzero"] is True
    assert decision["fdr_gradient_nonzero"] is True


def test_resume_rejects_queue_hash_drift(tmp_path: Path) -> None:
    run, protocol, learnability = _write_partial_run(tmp_path)
    queue = [json.loads(line) for line in (run / "publication-queue.jsonl").read_text().splitlines()]
    queue[-1]["checkpoint_sha256"] = "0" * 64
    (run / "publication-queue.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in queue), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="checkpoint/queue"):
        validate_resume(
            run,
            variant="baseline",
            stage="screen",
            protocol_manifest=protocol,
            learnability_report=learnability,
        )


def test_resume_rejects_cross_arm_authority(tmp_path: Path) -> None:
    run, protocol, learnability = _write_partial_run(tmp_path)

    with pytest.raises(ValueError, match="identity"):
        validate_resume(
            run,
            variant="ra_glgm",
            stage="screen",
            protocol_manifest=protocol,
            learnability_report=learnability,
        )


def test_resume_rejects_dataset_authority_drift(tmp_path: Path) -> None:
    run, protocol, learnability = _write_partial_run(tmp_path)
    runtime_path = run / "ra-run.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["dataset_authority"]["ignore"]["sha256"] = "0" * 64
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset authority"):
        validate_resume(
            run,
            variant="baseline",
            stage="screen",
            protocol_manifest=protocol,
            learnability_report=learnability,
        )


def test_resume_rejects_changed_learnability_report(tmp_path: Path) -> None:
    run, protocol, learnability = _write_partial_run(tmp_path)
    learnability.write_text('{"changed":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="learnability report"):
        validate_resume(
            run,
            variant="baseline",
            stage="screen",
            protocol_manifest=protocol,
            learnability_report=learnability,
        )


def _read_optimizer(run: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run / "optimizer-evidence.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _write_optimizer(run: Path, rows: list[dict]) -> None:
    (run / "optimizer-evidence.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("optimizer_attempt", 9, "sequence"),
        ("completed_epoch", 9, "sequence"),
        ("run_id", "foreign", "authority"),
        ("variant", "ra_glgm", "authority"),
        ("stage", "formal", "authority"),
        ("amp_scale_before", 64.0, "AMP/gradient"),
        ("amp_scale_after", 64.0, "AMP/gradient"),
        ("amp_step_skipped", True, "AMP/gradient"),
        ("gradient_norm_finite", False, "AMP/gradient"),
        ("gradient_norm", float("nan"), "gradient_norm"),
        ("fdr_gradient_norm", None, "fdr_gradient_norm"),
    ),
)
def test_resume_rejects_adversarial_optimizer_evidence(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    run, protocol, learnability = _write_partial_run(tmp_path)
    rows = _read_optimizer(run)
    rows[0][field] = value
    _write_optimizer(run, rows)

    with pytest.raises(ValueError, match=message):
        validate_resume(
            run,
            variant="baseline",
            stage="screen",
            protocol_manifest=protocol,
            learnability_report=learnability,
        )


def test_resume_rejects_missing_optimizer_evidence(tmp_path: Path) -> None:
    run, protocol, learnability = _write_partial_run(tmp_path)
    (run / "optimizer-evidence.jsonl").unlink()

    with pytest.raises(ValueError, match="missing or unreadable"):
        validate_resume(
            run,
            variant="baseline",
            stage="screen",
            protocol_manifest=protocol,
            learnability_report=learnability,
        )


def test_optimizer_evidence_requires_every_completed_epoch(tmp_path: Path) -> None:
    run, protocol, learnability = _write_partial_run(tmp_path)
    rows = _read_optimizer(run)
    rows[1]["completed_epoch"] = 3
    _write_optimizer(run, rows)

    with pytest.raises(ValueError, match="cover every completed epoch"):
        validate_resume(
            run,
            variant="baseline",
            stage="screen",
            protocol_manifest=protocol,
            learnability_report=learnability,
        )


def test_method_optimizer_evidence_requires_nonzero_private_gradient(tmp_path: Path) -> None:
    path = tmp_path / "optimizer-evidence.jsonl"
    rows = [
        {
            "optimizer_attempt": attempt,
            "completed_epoch": attempt,
            "run_id": "method-smoke",
            "variant": "ra_glgm",
            "stage": "smoke",
            "amp_scale_before": 128.0,
            "amp_scale_after": 128.0,
            "amp_step_skipped": False,
            "gradient_norm_finite": True,
            "gradient_norm": 3.0,
            "fdr_gradient_norm": 1.0,
            "ra_glgm_gradient_norm": 0.0,
        }
        for attempt in (1, 2)
    ]
    _write_optimizer(tmp_path, rows)

    with pytest.raises(ValueError, match="no nonzero RA private gradient in every epoch"):
        validate_optimizer_evidence(
            path,
            run_id="method-smoke",
            variant="ra_glgm",
            stage="smoke",
            completed_epochs=2,
        )

    rows[-1]["ra_glgm_gradient_norm"] = 0.25
    _write_optimizer(tmp_path, rows)
    with pytest.raises(ValueError, match="no nonzero RA private gradient in every epoch"):
        validate_optimizer_evidence(
            path,
            run_id="method-smoke",
            variant="ra_glgm",
            stage="smoke",
            completed_epochs=2,
        )

    for row in rows:
        row["ra_glgm_gradient_norm"] = 0.25
    _write_optimizer(tmp_path, rows)
    report = validate_optimizer_evidence(
        path,
        run_id="method-smoke",
        variant="ra_glgm",
        stage="smoke",
        completed_epochs=2,
    )
    assert report["ra_private_gradient_nonzero"] is True


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("gradient_norm", "nonzero public gradient"),
        ("fdr_gradient_norm", "nonzero FDR gradient"),
    ),
)
def test_optimizer_evidence_requires_nonzero_public_and_fdr_gradient_each_epoch(
    tmp_path: Path, field: str, message: str
) -> None:
    run, protocol, learnability = _write_partial_run(tmp_path)
    rows = _read_optimizer(run)
    rows[1][field] = 0.0
    _write_optimizer(run, rows)

    with pytest.raises(ValueError, match=message):
        validate_resume(
            run,
            variant="baseline",
            stage="screen",
            protocol_manifest=protocol,
            learnability_report=learnability,
        )


def test_resume_accepts_audited_attempts_from_only_the_interrupted_next_epoch(
    tmp_path: Path,
) -> None:
    run, protocol, learnability = _write_partial_run(tmp_path)
    rows = _read_optimizer(run)
    trailing = dict(rows[-1])
    trailing.update(
        {
            "optimizer_attempt": len(rows) + 1,
            "completed_epoch": 4,
        }
    )
    rows.append(trailing)
    _write_optimizer(run, rows)

    decision = validate_resume(
        run,
        variant="baseline",
        stage="screen",
        protocol_manifest=protocol,
        learnability_report=learnability,
    )

    assert decision["decision"] == "resume"
    assert decision["trailing_uncommitted_optimizer_attempts"] == 1

    rows[-1]["completed_epoch"] = 5
    _write_optimizer(run, rows)
    with pytest.raises(ValueError, match="sequence"):
        validate_resume(
            run,
            variant="baseline",
            stage="screen",
            protocol_manifest=protocol,
            learnability_report=learnability,
        )
