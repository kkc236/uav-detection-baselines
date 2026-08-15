from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.compare_lpr_g import compare_runs, load_arm_evidence


def _write_arm(path: Path, *, method: bool, epochs: int = 30) -> None:
    path.mkdir(parents=True)
    with (path / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("epoch", "metrics/mAP50-95(B)", "metrics/mAP50(B)"),
        )
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            writer.writerow(
                {
                    "epoch": epoch,
                    "metrics/mAP50-95(B)": 0.091 if method else 0.09,
                    "metrics/mAP50(B)": 0.191 if method else 0.19,
                }
            )
    diagnostics = []
    for epoch in range(1, epochs + 1):
        diagnostics.append(
            {
                "epoch": epoch,
                "map75": 0.041 if method else 0.04,
                "loss_bbox_refine": 0.3 if method else None,
                "loss_giou_refine": 0.4 if method else None,
                "gate_p95": 0.01 if method else None,
                "residual_rms": 0.001 if method else None,
                "lpr_g_gradient_norm": 1.0 if method else None,
            }
        )
    (path / "lpr_g_diagnostics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in diagnostics), encoding="utf-8"
    )
    audits = [
        {
            "epoch": epoch,
            "common_model_sha256": f"{epoch:064x}",
            "common_optimizer_sha256": f"{epoch + 100:064x}",
        }
        for epoch in range(1, epochs + 1)
    ]
    (path / "common_state_audit.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in audits), encoding="utf-8"
    )
    ledger = [
        {"completed_epoch": epoch, "verified": True, "checkpoint": {"sha256": f"{epoch:064x}"}}
        for epoch in range(1, epochs + 1)
    ]
    (path / "publication-ledger.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ledger), encoding="utf-8"
    )
    (path / "optimizer-evidence.jsonl").write_text(
        json.dumps(
            {
                "optimizer_attempt": 1,
                "amp_scale_before": 128.0,
                "amp_scale_after": 128.0,
                "amp_step_skipped": False,
                "gradient_norm_finite": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_comparator_requires_exact_epochs_pairing_and_publication(tmp_path: Path) -> None:
    control = tmp_path / "control"
    method = tmp_path / "method"
    _write_arm(control, method=False)
    _write_arm(method, method=True)
    ablation = {
        "stock": {"map": 0.09, "ap75": 0.04},
        "refined": {"map": 0.091, "ap75": 0.041},
    }
    benchmark = {"parameters": {}, "gflops": {}, "latency": {}}

    report = compare_runs(control, method, ablation=ablation, benchmark=benchmark)

    assert report["status"] == "passed"
    assert report["engineering"]["publication_records"] == 60
    assert report["engineering"]["common_state_equal"] is True


def test_cuda_nondeterministic_post_epoch_hashes_are_recorded_not_rejected(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control"
    method = tmp_path / "method"
    _write_arm(control, method=False)
    _write_arm(method, method=True)
    audit_path = method / "common_state_audit.jsonl"
    audits = [json.loads(line) for line in audit_path.read_text().splitlines()]
    audits[0]["common_model_sha256"] = "F" * 64
    audit_path.write_text(
        "".join(json.dumps(row) + "\n" for row in audits), encoding="utf-8"
    )
    ablation = {
        "stock": {"map": 0.09, "ap75": 0.04},
        "refined": {"map": 0.091, "ap75": 0.041},
    }

    report = compare_runs(control, method, ablation=ablation, benchmark={})

    assert report["status"] == "passed"
    assert report["engineering"]["common_state_equal"] is False
    assert report["engineering"]["common_state_exact_epochs"] == 29
    assert report["engineering"]["common_state_fingerprints_complete"] is True


def test_arm_parser_rejects_postfit_rows_and_gaps(tmp_path: Path) -> None:
    run = tmp_path / "method"
    _write_arm(run, method=True)
    diagnostics = run / "lpr_g_diagnostics.jsonl"
    with diagnostics.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"epoch": 31, "map75": 0.1}) + "\n")

    with pytest.raises(ValueError, match="epochs 1-30"):
        load_arm_evidence(run, method=True)


def test_arm_parser_rejects_missing_and_duplicate_epochs(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    duplicate = tmp_path / "duplicate"
    _write_arm(missing, method=True, epochs=29)
    _write_arm(duplicate, method=True)
    diagnostics = duplicate / "lpr_g_diagnostics.jsonl"
    rows = [json.loads(line) for line in diagnostics.read_text().splitlines()]
    rows[0]["epoch"] = 2
    diagnostics.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="epochs 1-30"):
        load_arm_evidence(missing, method=True)
    with pytest.raises(ValueError, match="epochs 1-30"):
        load_arm_evidence(duplicate, method=True)


def test_tail10_uses_exact_epochs_21_to_30(tmp_path: Path) -> None:
    run = tmp_path / "control"
    _write_arm(run, method=False)
    results = run / "results.csv"
    with results.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("epoch", "metrics/mAP50-95(B)", "metrics/mAP50(B)"),
        )
        writer.writeheader()
        for epoch in range(1, 31):
            writer.writerow(
                {
                    "epoch": epoch,
                    "metrics/mAP50-95(B)": epoch,
                    "metrics/mAP50(B)": epoch * 2,
                }
            )

    evidence = load_arm_evidence(run, method=False)

    assert evidence["metrics"]["final"]["map"] == 30
    assert evidence["metrics"]["tail10"]["map"] == pytest.approx(25.5)
