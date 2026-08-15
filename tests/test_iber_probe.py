"""Frozen engineering and scientific Gate-1 tests for IBER-BE."""

from __future__ import annotations

import copy
import math
import stat
from pathlib import Path

import pytest
import torch

import scripts.run_iber_probe as run_iber_probe
from scripts.run_iber_probe import _parse_args
from src.iber_probe import (
    AMP_AUTHORITY,
    ARM_ORDER,
    _move_batch,
    _save_checkpoint_immutable,
    evaluate_gate1,
)
from src.iber_protocol import (
    BOUNDARY_LOSS_CONTRACT,
    DESIGN_VERSION,
    PRIVATE_OPTIMIZER,
    PRIVATE_SEED,
    PROBE_EPOCHS,
)


CACHE_AUTHORITY = {
    "baseline_sha256": "A" * 64,
    "dataset_sha256": "B" * 64,
    "category_sha256": "C" * 64,
    "subset_sha256": "D" * 64,
    "source_commit": "e" * 40,
    "runtime_amendment_sha256": "F" * 64,
}


def _metrics(
    *,
    edge_mae: float,
    matched_iou_delta: float,
    tiny_direction_accuracy: float,
    small_direction_accuracy: float,
) -> dict[str, float]:
    stock_iou = 0.50
    return {
        "edge_mae": edge_mae,
        "stock_matched_iou": stock_iou,
        "refined_matched_iou": stock_iou + matched_iou_delta,
        "matched_iou_delta": matched_iou_delta,
        "tiny_direction_accuracy": tiny_direction_accuracy,
        "small_direction_accuracy": small_direction_accuracy,
        "gate_mean": 0.20,
        "gate_p95": 0.80,
        "residual_rms": 0.10,
        "gradient_rms": 0.02,
        "total_loss": 1.0,
        "f3_boundary_rms": 0.15,
        "rgb_boundary_rms": 0.16,
    }


def _report(arm: str, metrics: dict[str, float]) -> dict:
    return {
        "design_version": DESIGN_VERSION,
        "stage": "gate1_probe",
        "arm": arm,
        "epochs": PROBE_EPOCHS,
        "evaluated_epoch": PROBE_EPOCHS,
        "checkpoint_epoch": PROBE_EPOCHS,
        "selection": "epoch12_only",
        "private_seed": PRIVATE_SEED,
        "batch_size": 8,
        "optimizer": dict(PRIVATE_OPTIMIZER),
        "amp": dict(AMP_AUTHORITY),
        "parameter_count": 24_836,
        "initialization_sha256": "1" * 64,
        "cache_authority": dict(CACHE_AUTHORITY),
        "history": [
            {"epoch": epoch, "total_loss": 2.0 - epoch / 20}
            for epoch in range(1, PROBE_EPOCHS + 1)
        ],
        "boundary_loss": {
            "contract": dict(BOUNDARY_LOSS_CONTRACT),
            "bucket_counts": {
                "direction": [100, 200, 300],
                "margin": [80, 160, 240],
            },
            "batches_per_epoch": 81,
        },
        "metrics": metrics,
    }


def passing_reports() -> dict[str, dict]:
    return {
        "b0": _report(
            "b0",
            _metrics(
                edge_mae=0.100,
                matched_iou_delta=0.002,
                tiny_direction_accuracy=0.50,
                small_direction_accuracy=0.55,
            ),
        ),
        "b1": _report(
            "b1",
            _metrics(
                edge_mae=0.097,
                matched_iou_delta=0.003,
                tiny_direction_accuracy=0.51,
                small_direction_accuracy=0.56,
            ),
        ),
        "b2": _report(
            "b2",
            _metrics(
                edge_mae=0.096,
                matched_iou_delta=0.004,
                tiny_direction_accuracy=0.52,
                small_direction_accuracy=0.57,
            ),
        ),
        "b3": _report(
            "b3",
            _metrics(
                edge_mae=0.094,
                matched_iou_delta=0.006,
                tiny_direction_accuracy=0.54,
                small_direction_accuracy=0.59,
            ),
        ),
    }


def test_gate1_accepts_only_the_exact_b0_b1_b2_b3_arm_set() -> None:
    assert ARM_ORDER == ("b0", "b1", "b2", "b3")
    reports = passing_reports()
    decision = evaluate_gate1(reports)
    assert decision["status"] == "passed"
    assert decision["arm_order"] == list(ARM_ORDER)
    assert all(decision["engineering"].values())
    assert all(decision["conditions"].values())


@pytest.mark.parametrize("missing", ARM_ORDER)
def test_gate1_rejects_each_missing_arm(missing: str) -> None:
    reports = passing_reports()
    del reports[missing]
    assert evaluate_gate1(reports)["status"] == "engineering_invalid"


def test_gate1_rejects_an_extra_or_misidentified_arm() -> None:
    reports = passing_reports()
    reports["b4"] = copy.deepcopy(reports["b3"])
    assert evaluate_gate1(reports)["status"] == "engineering_invalid"

    reports = passing_reports()
    reports["b2"]["arm"] = "b1"
    assert evaluate_gate1(reports)["status"] == "engineering_invalid"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("design_version", "iber-be-v1.1"),
        ("stage", "probe"),
        ("epochs", 11),
        ("evaluated_epoch", 11),
        ("checkpoint_epoch", 11),
        ("selection", "best"),
        ("private_seed", 0),
        ("batch_size", 7),
    ],
)
def test_gate1_rejects_wrong_frozen_report_authority(field: str, bad_value: object) -> None:
    reports = passing_reports()
    reports["b3"][field] = bad_value
    decision = evaluate_gate1(reports)
    assert decision["status"] == "engineering_invalid"
    assert not all(decision["engineering"].values())


def test_gate1_forbids_best_epoch_substitution_even_with_twelve_history_rows() -> None:
    reports = passing_reports()
    reports["b3"]["evaluated_epoch"] = 7
    reports["b3"]["checkpoint_epoch"] = 7
    reports["b3"]["selection"] = "best"
    assert len(reports["b3"]["history"]) == 12
    assert evaluate_gate1(reports)["status"] == "engineering_invalid"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report.update(optimizer={**report["optimizer"], "lr": 2e-3}),
        lambda report: report.update(amp={**report["amp"], "init_scale": 64.0}),
        lambda report: report.update(parameter_count=report["parameter_count"] + 1),
        lambda report: report.update(initialization_sha256="2" * 64),
        lambda report: report.update(cache_authority={**report["cache_authority"], "source_commit": "f" * 40}),
        lambda report: report["boundary_loss"]["contract"].update(
            direction_margin=0.50
        ),
        lambda report: report.update(history=report["history"][:-1]),
    ],
)
def test_gate1_rejects_optimizer_amp_capacity_initialization_cache_loss_or_history_drift(mutate) -> None:
    reports = passing_reports()
    mutate(reports["b2"])
    assert evaluate_gate1(reports)["status"] == "engineering_invalid"


@pytest.mark.parametrize("metric", tuple(_metrics(edge_mae=1, matched_iou_delta=1, tiny_direction_accuracy=1, small_direction_accuracy=1)))
@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_gate1_rejects_every_nonfinite_metric(metric: str, invalid: float) -> None:
    reports = passing_reports()
    reports["b1"]["metrics"][metric] = invalid
    assert evaluate_gate1(reports)["status"] == "engineering_invalid"


def _reports_failing_only(condition: str) -> dict[str, dict]:
    reports = passing_reports()
    metrics = {arm: report["metrics"] for arm, report in reports.items()}
    if condition == "edge_over_b0":
        metrics["b1"]["edge_mae"] = 0.099
        metrics["b2"]["edge_mae"] = 0.098
        metrics["b3"]["edge_mae"] = 0.096
    elif condition == "edge_over_b1":
        metrics["b1"]["edge_mae"] = 0.090
        metrics["b2"]["edge_mae"] = 0.092
        metrics["b3"]["edge_mae"] = 0.089
    elif condition == "matched_iou":
        metrics["b0"]["matched_iou_delta"] = 0.001
        metrics["b1"]["matched_iou_delta"] = 0.002
        metrics["b2"]["matched_iou_delta"] = 0.003
        metrics["b3"]["matched_iou_delta"] = 0.004999
    elif condition == "tiny_direction":
        metrics["b3"]["tiny_direction_accuracy"] = 0.529999
    elif condition == "small_direction":
        metrics["b3"]["small_direction_accuracy"] = 0.579999
    elif condition == "b3_best_primary":
        metrics["b2"]["edge_mae"] = metrics["b3"]["edge_mae"] - 1e-6
    elif condition == "finite_activity":
        metrics["b3"]["gradient_rms"] = 0.0
    else:
        raise AssertionError(condition)
    return reports


@pytest.mark.parametrize(
    "condition",
    (
        "edge_over_b0",
        "edge_over_b1",
        "matched_iou",
        "tiny_direction",
        "small_direction",
        "b3_best_primary",
        "finite_activity",
    ),
)
def test_each_scientific_condition_is_individually_mandatory(condition: str) -> None:
    decision = evaluate_gate1(_reports_failing_only(condition))
    assert decision["status"] == "scientific_failed"
    assert decision["conditions"][condition] is False
    assert sum(not value for value in decision["conditions"].values()) == 1


def test_gate1_threshold_boundaries_are_exact_and_unrounded() -> None:
    reports = passing_reports()
    reports["b3"]["metrics"].update(
        edge_mae=reports["b0"]["metrics"]["edge_mae"] * 0.95,
        matched_iou_delta=0.005,
        tiny_direction_accuracy=reports["b0"]["metrics"]["tiny_direction_accuracy"] + 0.03,
        small_direction_accuracy=reports["b0"]["metrics"]["small_direction_accuracy"] + 0.03,
    )
    reports["b1"]["metrics"]["edge_mae"] = reports["b3"]["metrics"]["edge_mae"] / 0.985
    assert evaluate_gate1(reports)["status"] == "passed"

    reports["b3"]["metrics"]["matched_iou_delta"] = math.nextafter(0.005, 0.0)
    assert evaluate_gate1(reports)["conditions"]["matched_iou"] is False


@pytest.mark.parametrize(
    ("metric", "bad_value"),
    [
        ("gradient_rms", 0.0),
        ("gate_mean", 1e-4),
        ("gate_p95", 1e-3),
        ("gate_p95", 0.999),
        ("residual_rms", 1e-4),
    ],
)
def test_b3_activity_bounds_are_strict(metric: str, bad_value: float) -> None:
    reports = passing_reports()
    reports["b3"]["metrics"][metric] = bad_value
    decision = evaluate_gate1(reports)
    assert decision["status"] == "scientific_failed"
    assert decision["conditions"]["finite_activity"] is False


def _cache_record(index: int, *, targets: int, pixel: int) -> dict[str, object]:
    return {
        "index": index,
        "image_id": f"image-{index}",
        "hidden": torch.full((300, 8), float(index)),
        "stock_boxes": torch.full((300, 4), 0.5),
        "stock_scores": torch.zeros(300, 10),
        "f3": torch.full((4, 2, 2), float(index), dtype=torch.float16),
        "image_rgb": torch.full((3, 640, 640), pixel, dtype=torch.uint8),
        "target_edges": torch.full((targets, 4), 0.5),
        "match_source": torch.arange(targets),
        "match_target": torch.arange(targets),
    }


def test_probe_batch_uses_only_trajectory_free_evidence_and_offsets_matches() -> None:
    records = (
        _cache_record(0, targets=2, pixel=17),
        _cache_record(1, targets=1, pixel=255),
    )

    evidence, targets, matches = _move_batch(records, torch.device("cpu"))

    assert set(evidence) == {
        "hidden",
        "stock_boxes",
        "stock_scores",
        "f3",
        "image_rgb",
    }
    assert evidence["image_rgb"].dtype is torch.float32
    torch.testing.assert_close(
        evidence["image_rgb"][:, 0, 0, 0],
        torch.tensor([17.0 / 255.0, 1.0]),
    )
    assert targets.shape == (3, 4)
    assert matches[0][1].tolist() == [0, 1]
    assert matches[1][1].tolist() == [2]


def test_probe_checkpoint_is_tensor_only_and_refuses_overwrite(tmp_path: Path) -> None:
    checkpoint = tmp_path / "b3-epoch-0012.pt"
    try:
        _save_checkpoint_immutable(
            checkpoint,
            {"epoch": 12, "refiner": {"weight": torch.ones(2)}},
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert payload["epoch"] == 12
        torch.testing.assert_close(payload["refiner"]["weight"], torch.ones(2))
        with pytest.raises(FileExistsError, match="overwrite"):
            _save_checkpoint_immutable(checkpoint, {"epoch": 12})
    finally:
        if checkpoint.exists():
            checkpoint.chmod(stat.S_IREAD | stat.S_IWRITE)


def test_probe_cli_has_no_authority_overrides() -> None:
    args = _parse_args(
        [
            "--cache-root",
            "cache",
            "--output-root",
            "output",
        ]
    )
    assert args.cache_root == Path("cache")
    assert args.output_root == Path("output")
    assert args.device == "0"


def test_probe_cli_records_floating_point_failures_as_engineering_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"

    def fail_load(*_args, **_kwargs):
        raise FloatingPointError("NONFINITE_IBER_PROBE_GRADIENT")

    monkeypatch.setattr(run_iber_probe, "_source_commit", lambda: "e" * 40)
    monkeypatch.setattr(run_iber_probe, "load_evidence_cache", fail_load)
    result = run_iber_probe.main(
        [
            "--cache-root",
            str(tmp_path / "cache"),
            "--output-root",
            str(output_root),
        ]
    )
    report = output_root / "gate1-engineering-invalid.json"
    try:
        assert result == 1
        assert report.is_file()
        assert "FloatingPointError" in report.read_text(encoding="utf-8")
    finally:
        if report.exists():
            report.chmod(stat.S_IREAD | stat.S_IWRITE)
