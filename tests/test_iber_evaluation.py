from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.iber_evaluation import (
    EVALUATION_CONSTANTS,
    assert_repeated_evaluations,
    compute_detection_metrics,
    compute_refinement_diagnostics,
    evaluate_gate2,
    finite_noncollapsed_activity,
)
from scripts.evaluate_iber import (
    _assert_shared_prediction_scores,
    _globalize_match_indices,
    _last5_history,
)
from src.iber_protocol import (
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    PROTOCOL_SHA256,
    RUNTIME_AMENDMENT_SHA256,
)


def _predictions() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "boxes": torch.tensor(
                [
                    [0.20, 0.20, 8 / 640, 8 / 640],
                    [0.60, 0.60, 24 / 640, 24 / 640],
                ],
                dtype=torch.float32,
            ),
            "scores": torch.tensor([0.95, 0.90]),
            "classes": torch.tensor([0, 1]),
        }
    ]


def _targets() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "boxes": _predictions()[0]["boxes"].clone(),
            "classes": torch.tensor([0, 1]),
        }
    ]


def _metrics(**updates: float) -> dict[str, float]:
    values = {
        "map": 0.2400,
        "ap50": 0.4200,
        "ap75": 0.2400,
        "ap_tiny": 0.1000,
        "ap_small": 0.2000,
        "precision": 0.5000,
        "recall": 0.4300,
    }
    values.update(updates)
    return values


def _diagnostics(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "finite": True,
        "matched_count": 201,
        "matched_improved": 101,
        "matched_degraded": 100,
        "matched_equal": 0,
        "stock_iou_mean": 0.60,
        "refined_iou_mean": 0.61,
        "stock_edge_mae": 0.08,
        "refined_edge_mae": 0.07,
        "edge_mae_delta": -0.01,
        "matched_iou_delta_mean": 0.01,
        "matched_correction_rms": 0.020,
        "unmatched_correction_rms": 0.005,
        "unmatched_to_matched_rms_ratio": 0.25,
        "f3_embedding_rms": 0.2,
        "rgb_embedding_rms": 0.1,
        "gate_mean": 0.30,
        "gate_std": 0.10,
        "gate_p05": 0.05,
        "gate_p95": 0.80,
        "residual_rms": 0.20,
        "residual_abs_p95": 0.75,
        "detector_sha_before": "A" * 64,
        "detector_sha_after": "A" * 64,
    }
    values.update(updates)
    return values


def _passing_inputs():
    stock = _metrics()
    refined = _metrics(
        map=stock["map"] + 0.002,
        ap75=stock["ap75"] + 0.003,
        ap50=stock["ap50"] - 0.0005,
        ap_tiny=stock["ap_tiny"] + 1e-12,
    )
    repeat = {"stock": stock, "refined": refined, "diagnostics": _diagnostics()}
    repeats = [copy.deepcopy(repeat) for _ in range(3)]
    return stock, refined, _diagnostics(), repeats


def test_evaluation_constants_match_frozen_no_nms_contract() -> None:
    assert EVALUATION_CONSTANTS == {
        "seed": 0,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "device": "0",
        "max_det": 300,
        "nms": False,
        "cache": False,
        "conf": 0.001,
        "half": False,
        "repeats": 3,
    }


def test_detection_metrics_include_full_tiny_and_small_ap() -> None:
    metrics = compute_detection_metrics(_predictions(), _targets(), image_size=640)
    assert set(metrics) == {
        "map",
        "ap50",
        "ap75",
        "ap_tiny",
        "ap_small",
        "precision",
        "recall",
    }
    assert all(metrics[name] > 0.99 for name in ("map", "ap50", "ap75", "ap_tiny", "ap_small"))


def test_refinement_diagnostics_include_edge_iou_leakage_and_embeddings() -> None:
    stock = torch.tensor([[[0.50, 0.50, 0.40, 0.40], [0.20, 0.20, 0.10, 0.10]]])
    refined = torch.tensor([[[0.50, 0.50, 0.50, 0.50], [0.20, 0.20, 0.10, 0.10]]])
    targets = torch.tensor([[0.50, 0.50, 0.50, 0.50]])
    matches = [(torch.tensor([0]), torch.tensor([0]))]
    correction = torch.tensor([[[0.05] * 4, [0.01] * 4]])
    gates = torch.tensor([[[0.2] * 4, [0.1] * 4]])
    residuals = torch.tensor([[[0.5] * 4, [0.1] * 4]])
    f3 = torch.full((1, 2, 4, 32), 0.2)
    rgb = torch.full((1, 2, 4, 16), 0.1)

    report = compute_refinement_diagnostics(
        stock,
        refined,
        targets,
        matches,
        correction,
        gates,
        residuals,
        f3,
        rgb,
        target_group_sizes=[1],
    )
    assert report["matched_improved"] == 1
    assert report["matched_degraded"] == 0
    assert report["matched_iou_delta_mean"] > 0
    assert report["refined_edge_mae"] < report["stock_edge_mae"]
    assert report["matched_correction_rms"] == pytest.approx(0.05)
    assert report["unmatched_correction_rms"] == pytest.approx(0.01)
    assert report["f3_embedding_rms"] == pytest.approx(0.2)
    assert report["rgb_embedding_rms"] == pytest.approx(0.1)
    assert report["finite"] is True


def test_three_repeats_require_exact_values() -> None:
    _, _, _, repeats = _passing_inputs()
    assert assert_repeated_evaluations(repeats) == repeats[0]
    changed = copy.deepcopy(repeats)
    changed[2]["refined"]["map"] += 1e-12
    with pytest.raises(ValueError, match="repeat 3"):
        assert_repeated_evaluations(changed)


def test_gate2_passes_only_exact_epoch30_boundaries_and_last5() -> None:
    stock, refined, diagnostics, repeats = _passing_inputs()
    decision = evaluate_gate2(
        stock,
        refined,
        diagnostics,
        repeats=repeats,
        last5_stock_map=[stock["map"]] * 5,
        last5_refined_map=[refined["map"]] * 5,
        checkpoint_epoch=30,
    )
    assert decision["status"] == "passed"
    assert all(decision["conditions"].values())


@pytest.mark.parametrize(
    ("condition", "change"),
    [
        ("map", "map"),
        ("ap75", "ap75"),
        ("ap50", "ap50"),
        ("tiny_or_small", "tiny_or_small"),
        ("matched_counts", "matched_counts"),
        ("unmatched_rms", "unmatched_rms"),
        ("activity", "activity"),
        ("repeatability", "repeatability"),
        ("last5", "last5"),
    ],
)
def test_each_gate2_condition_is_mandatory(condition: str, change: str) -> None:
    stock, refined, diagnostics, repeats = _passing_inputs()
    last5_stock = [stock["map"]] * 5
    last5_refined = [refined["map"]] * 5
    if change == "map":
        refined["map"] = stock["map"] + 0.002 - 1e-12
    elif change == "ap75":
        refined["ap75"] = stock["ap75"] + 0.003 - 1e-12
    elif change == "ap50":
        refined["ap50"] = stock["ap50"] - 0.0005 - 1e-12
    elif change == "tiny_or_small":
        refined["ap_tiny"] = stock["ap_tiny"]
        refined["ap_small"] = stock["ap_small"]
    elif change == "matched_counts":
        diagnostics["matched_improved"] = diagnostics["matched_degraded"]
        diagnostics["matched_equal"] += 1
    elif change == "unmatched_rms":
        diagnostics["unmatched_correction_rms"] = 0.005000000001
    elif change == "activity":
        diagnostics["gate_std"] = 0.0
    elif change == "repeatability":
        repeats[2]["refined"]["map"] += 1e-12
    elif change == "last5":
        last5_refined = list(last5_stock)
    decision = evaluate_gate2(
        stock,
        refined,
        diagnostics,
        repeats=repeats,
        last5_stock_map=last5_stock,
        last5_refined_map=last5_refined,
        checkpoint_epoch=30,
    )
    assert decision["conditions"][condition] is False
    expected_status = "engineering_invalid" if change == "repeatability" else "scientific_failed"
    assert decision["status"] == expected_status


def test_gate2_rejects_non_epoch30_and_changed_detector_as_engineering_invalid() -> None:
    stock, refined, diagnostics, repeats = _passing_inputs()
    for epoch, changed in (
        (29, diagnostics),
        (30, {**diagnostics, "detector_sha_after": "B" * 64}),
    ):
        decision = evaluate_gate2(
            stock,
            refined,
            changed,
            repeats=repeats,
            last5_stock_map=[stock["map"]] * 5,
            last5_refined_map=[refined["map"]] * 5,
            checkpoint_epoch=epoch,
        )
        assert decision["status"] == "engineering_invalid"


@pytest.mark.parametrize(
    "diagnostic_update",
    [
        {"matched_count": 999},
        {"detector_sha_before": "not-a-sha", "detector_sha_after": "not-a-sha"},
    ],
)
def test_gate2_treats_malformed_diagnostics_as_engineering_invalid(
    diagnostic_update: dict[str, object],
) -> None:
    stock, refined, diagnostics, repeats = _passing_inputs()
    diagnostics.update(diagnostic_update)
    decision = evaluate_gate2(
        stock,
        refined,
        diagnostics,
        repeats=repeats,
        last5_stock_map=[stock["map"]] * 5,
        last5_refined_map=[refined["map"]] * 5,
        checkpoint_epoch=30,
    )
    assert decision["status"] == "engineering_invalid"


def test_gate2_treats_missing_diagnostic_field_as_engineering_invalid() -> None:
    stock, refined, diagnostics, repeats = _passing_inputs()
    diagnostics.pop("matched_improved")
    decision = evaluate_gate2(
        stock,
        refined,
        diagnostics,
        repeats=repeats,
        last5_stock_map=[stock["map"]] * 5,
        last5_refined_map=[refined["map"]] * 5,
        checkpoint_epoch=30,
    )
    assert decision["status"] == "engineering_invalid"


def test_matcher_indices_reject_cross_image_targets() -> None:
    stock = torch.full((2, 1, 4), 0.5)
    refined = stock.clone()
    features = torch.ones(2, 1, 4, 2)
    with pytest.raises(ValueError, match="crosses image boundary"):
        compute_refinement_diagnostics(
            stock,
            refined,
            torch.tensor(
                [[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]
            ),
            [
                (torch.tensor([0]), torch.tensor([1])),
                (torch.tensor([0]), torch.tensor([1])),
            ],
            torch.zeros_like(stock),
            torch.full_like(stock, 0.5),
            torch.zeros_like(stock),
            features,
            features,
            target_group_sizes=[1, 1],
        )


def test_batch_global_match_indices_are_offset_once_across_batches() -> None:
    matches = [
        (torch.tensor([1, 3]), torch.tensor([0, 1])),
        (torch.tensor([2]), torch.tensor([2])),
    ]
    converted = _globalize_match_indices(
        matches,
        groups=[2, 1],
        prior_target_count=5,
        query_count=4,
    )
    assert converted[0][1].tolist() == [5, 6]
    assert converted[1][1].tolist() == [7]


def test_stock_and_refined_postprocess_must_keep_identical_scores_and_classes() -> None:
    stock = torch.zeros(1, 2, 6)
    stock[..., 4] = torch.tensor([0.9, 0.8])
    stock[..., 5] = torch.tensor([1.0, 2.0])
    refined = stock.clone()
    _assert_shared_prediction_scores(stock, refined)
    refined[0, 1, 4] += 1e-6
    with pytest.raises(RuntimeError, match="shared scores/classes"):
        _assert_shared_prediction_scores(stock, refined)


def test_activity_requires_both_boundary_embeddings() -> None:
    assert finite_noncollapsed_activity(_diagnostics()) is True
    assert finite_noncollapsed_activity(_diagnostics(rgb_embedding_rms=0.0)) is False
    assert finite_noncollapsed_activity(_diagnostics(f3_embedding_rms=0.0)) is False


def test_epoch30_history_uses_exactly_published_epochs_26_to_29(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    output = run_root / "evaluations" / "epoch-0030.json"
    output.parent.mkdir(parents=True)
    rows = []
    ledger = []
    for epoch in range(1, 30):
        checkpoint_sha = f"{epoch:064x}"
        rows.append(
            {
                "design_version": "iber-be-v1.0",
                "stage": "screen",
                "probe": "b3",
                "seed": 0,
                "epoch": epoch,
                "evaluation": {
                    "design_version": "iber-be-v1.0",
                    "epoch": epoch,
                    "private_checkpoint": {"sha256": checkpoint_sha},
                    "stock": {"map": epoch / 1000},
                    "refined": {"map": epoch / 1000 + 0.001},
                },
            }
        )
        ledger.append(
            {
                "design_version": "iber-be-v1.0",
                "stage": "screen",
                "probe": "b3",
                "seed": 0,
                "baseline_sha256": EXPECTED_BASELINE_SHA256,
                "dataset_sha256": EXPECTED_DATASET_SHA256,
                "subset_sha256": EXPECTED_SUBSET_SHA256,
                "category_sha256": "a" * 64,
                "protocol_sha256": PROTOCOL_SHA256,
                "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
                "gate1_decision_sha256": "b" * 64,
                "source_commit": "c" * 40,
                "completed_epoch": epoch,
                "checkpoint": {"bytes": 10, "sha256": checkpoint_sha},
                "remote_verification": {
                    "checkpoint": {"bytes": 10, "sha256": checkpoint_sha},
                    "manifest": {"bytes": 20, "sha256": "d" * 64},
                },
                "result_commit_sha": "e" * 40,
                "result_commit_verified": True,
                "verified": True,
            }
        )
    (run_root / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (run_root / "publication-ledger.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ledger), encoding="utf-8"
    )
    current = {
        "design_version": "iber-be-v1.0",
        "epoch": 30,
        "private_checkpoint": {"sha256": "f" * 64},
        "stock": {"map": 0.030},
        "refined": {"map": 0.031},
    }
    stock, refined = _last5_history(output, current)
    assert stock == pytest.approx([0.026, 0.027, 0.028, 0.029, 0.030])
    assert refined == pytest.approx([0.027, 0.028, 0.029, 0.030, 0.031])

    rows[10]["epoch"] = 99
    (run_root / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="contiguous"):
        _last5_history(output, current)

    rows[10]["epoch"] = 11
    (run_root / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    ledger[-1]["result_commit_verified"] = False
    (run_root / "publication-ledger.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ledger), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="result commit"):
        _last5_history(output, current)


def test_evaluation_cli_has_only_authority_paths() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_iber.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for allowed in (
        "--baseline-checkpoint",
        "--private-checkpoint",
        "--dataset-root",
        "--gate1-decision",
        "--output",
    ):
        assert allowed in result.stdout
    for forbidden in (
        "--stage",
        "--imgsz",
        "--batch",
        "--workers",
        "--max-det",
        "--nms",
        "--repeats",
        "--seed",
        "--epoch",
    ):
        assert forbidden not in result.stdout


def test_evaluator_source_is_independent_safe_and_same_checkpoint() -> None:
    source = Path("scripts/evaluate_iber.py").read_text(encoding="utf-8")
    for marker in (
        "FrozenIBERAdapter",
        "output.stock_scores",
        "output.stock_boxes",
        "output.refined_boxes",
        "weights_only=True",
        "RUNTIME_AMENDMENT_SHA256",
        "PROTOCOL_SHA256",
        "last5_stock_map",
        "last5_refined_map",
    ):
        assert marker in source
    assert "FrozenITBERAdapter" not in source
    assert "rtdetr_itber" not in source
    assert "itber-v1.1" not in source
