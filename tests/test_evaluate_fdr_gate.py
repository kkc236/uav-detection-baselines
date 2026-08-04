from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path("scripts/evaluate_fdr_gate.py")


def _load_module():
    assert SCRIPT.is_file(), "FDR Gate2 evaluator has not been implemented"
    spec = importlib.util.spec_from_file_location("evaluate_fdr_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest(variant: str, *, source_sha: str = "S" * 64) -> dict:
    return {
        "format_version": 1,
        "protocol_sha256": "P" * 64,
        "source": {"git_commit": "a" * 40, "tree_sha256": "T" * 64},
        "run_identity": {
            "source_sha256": source_sha,
            "protocol_sha256": "P" * 64,
            "run_id": f"{variant}-screen-seed0-authority",
            "stage": "screen",
            "variant": variant,
            "seed": 0,
        },
        "initial_state": {"path": "/authority/initial-state.pt", "sha256": "I" * 64},
        "data": "/authority/screen-data.yaml",
        "screen_cutoff_epoch": 30,
        "publication_queue": f"/runs/{variant}/publication-queue.jsonl",
    }


def _write_arm(
    root: Path,
    variant: str,
    *,
    map_delta: float = 0.0,
    map75_delta: float = 0.0,
    manifest: dict | None = None,
) -> Path:
    run = root / variant
    run.mkdir(parents=True)
    rows = []
    csv_rows = []
    for epoch in range(1, 31):
        map_value = 0.10 + epoch * 0.001 + map_delta
        map75 = 0.06 + epoch * 0.0008 + map75_delta
        precision = 0.30 + epoch * 0.001 + map_delta / 2
        recall = 0.40 + epoch * 0.0015 + map_delta / 3
        row = {
            "completed_epoch": epoch,
            "variant": variant,
            "stage": "screen",
            "run_id": f"{variant}-screen-seed0-authority",
            "precision": precision,
            "recall": recall,
            "map50": map_value + 0.10,
            "map": map_value,
            "map75": map75,
            "loss_giou": 2.0 - epoch * 0.02,
            "loss_class": 1.5 - epoch * 0.01,
            "loss_bbox": 1.0 - epoch * 0.005,
            "loss_fgl": 0.5 - epoch * 0.003 if variant == "fdr" else None,
            "loss_fgl_aux": 0.3 - epoch * 0.002 if variant == "fdr" else None,
            "loss_bbox_pre": 1.1 - epoch * 0.004 if variant == "fdr" else None,
            "loss_giou_pre": 1.7 - epoch * 0.008 if variant == "fdr" else None,
            "gradient_norm": 3.0,
            "fdr_gradient_norm": 2.0 if variant == "fdr" else None,
            "cuda_peak_mib": 1000.0,
        }
        rows.append(row)
        csv_rows.append(
            {
                "epoch": epoch,
                "metrics/precision(B)": precision,
                "metrics/recall(B)": recall,
                "metrics/mAP50(B)": map_value + 0.10,
                "metrics/mAP50-95(B)": map_value,
                "val/giou_loss": 2.0 - epoch * 0.02,
                "val/cls_loss": 1.5 - epoch * 0.01,
                "val/l1_loss": 1.0 - epoch * 0.005,
            }
        )
    (run / "fdr-epochs.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    with (run / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (run / "fdr-run.json").write_text(
        json.dumps(manifest or _manifest(variant), allow_nan=False), encoding="utf-8"
    )
    return run


def test_cli_exposes_only_paired_run_dirs_and_create_only_output() -> None:
    assert SCRIPT.is_file(), "FDR Gate2 evaluator has not been implemented"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--control-run" in result.stdout
    assert "--fdr-run" in result.stdout
    assert "--output" in result.stdout
    for forbidden in ("--threshold", "--epochs", "--seed", "--tail"):
        assert forbidden not in result.stdout


def test_valid_pair_reports_final_tail3_best_deltas_and_passes_frozen_gate(
    tmp_path: Path,
) -> None:
    module = _load_module()
    control = _write_arm(tmp_path, "control")
    fdr = _write_arm(tmp_path, "fdr", map_delta=0.006, map75_delta=0.004)

    report = module.evaluate_gate(control, fdr)

    assert report["engineering"]["complete"] is True
    assert report["engineering"]["checks"]["continuous_30_epochs"] is True
    assert report["engineering"]["checks"]["paired_authority"] is True
    assert report["metrics"]["final"]["delta"]["map"] == pytest.approx(0.006)
    assert report["metrics"]["final"]["delta"]["map75"] == pytest.approx(0.004)
    assert report["metrics"]["tail3"]["delta"]["map"] == pytest.approx(0.006)
    assert report["metrics"]["best"]["control"]["epoch"] == 30
    assert report["metrics"]["best"]["fdr"]["epoch"] == 30
    assert set(report["metrics"]["final"]["delta"]) == {
        "map",
        "map75",
        "precision",
        "recall",
    }
    assert report["loss_trends"]["control"]["loss_giou"]["tail3_minus_first3"] < 0
    assert report["loss_trends"]["fdr"]["loss_fgl"]["tail3_minus_first3"] < 0
    assert report["gate"]["checks"] == {
        "final_map_strictly_positive": True,
        "tail3_mean_map_strictly_positive": True,
        "final_ap75_strictly_positive": True,
    }
    assert report["formal_eligible"] is True


def test_equal_final_map_fails_strict_scientific_gate(tmp_path: Path) -> None:
    module = _load_module()
    control = _write_arm(tmp_path, "control")
    fdr = _write_arm(tmp_path, "fdr", map_delta=0.0, map75_delta=0.001)

    report = module.evaluate_gate(control, fdr)

    assert report["engineering"]["complete"] is True
    assert report["gate"]["checks"]["final_map_strictly_positive"] is False
    assert report["gate"]["checks"]["tail3_mean_map_strictly_positive"] is False
    assert report["formal_eligible"] is False


def test_epoch_gap_is_engineering_incomplete_and_cannot_be_formal_eligible(
    tmp_path: Path,
) -> None:
    module = _load_module()
    control = _write_arm(tmp_path, "control")
    fdr = _write_arm(tmp_path, "fdr", map_delta=0.006, map75_delta=0.004)
    jsonl = fdr / "fdr-epochs.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    rows[14]["completed_epoch"] = 99
    jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = module.evaluate_gate(control, fdr)

    assert report["engineering"]["complete"] is False
    assert report["engineering"]["checks"]["continuous_30_epochs"] is False
    assert report["formal_eligible"] is False
    assert report["engineering"]["errors"]


def test_authority_drift_is_reported_without_relaxing_gate(tmp_path: Path) -> None:
    module = _load_module()
    control = _write_arm(tmp_path, "control")
    changed = _manifest("fdr", source_sha="X" * 64)
    fdr = _write_arm(
        tmp_path,
        "fdr",
        map_delta=0.006,
        map75_delta=0.004,
        manifest=changed,
    )

    report = module.evaluate_gate(control, fdr)

    assert report["engineering"]["checks"]["paired_authority"] is False
    assert report["engineering"]["complete"] is False
    assert report["formal_eligible"] is False


def test_jsonl_and_results_csv_metric_drift_is_engineering_incomplete(
    tmp_path: Path,
) -> None:
    module = _load_module()
    control = _write_arm(tmp_path, "control")
    fdr = _write_arm(tmp_path, "fdr", map_delta=0.006, map75_delta=0.004)
    rows = list(csv.DictReader((fdr / "results.csv").open(encoding="utf-8")))
    rows[-1]["metrics/mAP50-95(B)"] = "0.999"
    with (fdr / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = module.evaluate_gate(control, fdr)

    assert report["engineering"]["checks"]["jsonl_results_consistent"] is False
    assert report["formal_eligible"] is False


def test_report_writer_is_create_only(tmp_path: Path) -> None:
    module = _load_module()
    output = tmp_path / "gate2.json"
    report = {"formal_eligible": False}

    assert module.write_create_only_report(output, report) == output.resolve()
    with pytest.raises(FileExistsError):
        module.write_create_only_report(output, report)
    assert json.loads(output.read_text(encoding="utf-8")) == report
