from __future__ import annotations

import copy

import torch

from src.itber_metrics import aligned_iou, area_bucket, direction_accuracy
from src.itber_probe import evaluate_gate1


def _report(
    probe: str,
    *,
    edge_mae: float,
    iou_delta: float,
    tiny_direction: float,
    small_direction: float,
) -> dict:
    return {
        "probe": probe,
        "epochs": 12,
        "parameter_count": 12345,
        "initialization_sha256": "A" * 64,
        "metrics": {
            "edge_mae": edge_mae,
            "matched_iou_delta": iou_delta,
            "tiny_direction_accuracy": tiny_direction,
            "small_direction_accuracy": small_direction,
            "gate_mean": 0.2,
            "gate_p95": 0.7,
            "residual_rms": 0.3,
            "finite": True,
        },
    }


def _passing_reports() -> dict[str, dict]:
    return {
        "p0": _report("p0", edge_mae=0.1000, iou_delta=0.0000, tiny_direction=0.50, small_direction=0.55),
        "p1": _report("p1", edge_mae=0.0980, iou_delta=0.0020, tiny_direction=0.51, small_direction=0.56),
        "p2": _report("p2", edge_mae=0.0970, iou_delta=0.0040, tiny_direction=0.52, small_direction=0.57),
        "p3": _report("p3", edge_mae=0.0940, iou_delta=0.0060, tiny_direction=0.54, small_direction=0.59),
    }


def test_gate1_requires_every_preregistered_p3_gain() -> None:
    decision = evaluate_gate1(_passing_reports())

    assert decision["status"] == "passed"
    assert all(decision["conditions"].values())


def test_gate1_fails_each_scientific_boundary_without_rounding() -> None:
    mutations = {
        "edge_over_p0": ("edge_mae", 0.0950000001),
        "edge_over_p2": ("edge_mae", 0.0955450001),
        "matched_iou": ("matched_iou_delta", 0.0049999999),
        "tiny_direction": ("tiny_direction_accuracy", 0.5299999999),
        "small_direction": ("small_direction_accuracy", 0.5799999999),
    }
    for expected_condition, (metric, value) in mutations.items():
        reports = _passing_reports()
        reports["p3"]["metrics"][metric] = value
        decision = evaluate_gate1(reports)
        assert decision["status"] == "scientific_failed"
        assert decision["conditions"][expected_condition] is False


def test_gate1_rejects_nonfinite_wrong_epoch_or_unequal_capacity_as_engineering() -> None:
    for mutate in ("finite", "epoch", "capacity", "initialization"):
        reports = copy.deepcopy(_passing_reports())
        if mutate == "finite":
            reports["p2"]["metrics"]["finite"] = False
        elif mutate == "epoch":
            reports["p1"]["epochs"] = 11
        elif mutate == "capacity":
            reports["p2"]["parameter_count"] += 1
        else:
            reports["p3"]["initialization_sha256"] = "B" * 64
        assert evaluate_gate1(reports)["status"] == "engineering_invalid"


def test_p3_must_be_best_on_both_primary_metrics() -> None:
    reports = _passing_reports()
    reports["p1"]["metrics"]["matched_iou_delta"] = 0.007

    decision = evaluate_gate1(reports)

    assert decision["status"] == "scientific_failed"
    assert decision["conditions"]["p3_best_primary"] is False


def test_metric_helpers_compute_iou_area_and_direction_exactly() -> None:
    first = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 1.0, 1.0]])
    second = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]])
    torch.testing.assert_close(aligned_iou(first, second), torch.tensor([1.0, 0.25]))

    buckets = area_bucket(torch.tensor([[0.5, 0.5, 0.01, 0.01], [0.5, 0.5, 0.04, 0.04]]), image_size=640)
    assert buckets.tolist() == [0, 1]
    accuracy = direction_accuracy(
        torch.tensor([[1.0, -1.0, -1.0, 1.0]]),
        torch.tensor([[1.0, -0.5, 0.0, -1.0]]),
    )
    torch.testing.assert_close(accuracy, torch.tensor(2 / 3))
