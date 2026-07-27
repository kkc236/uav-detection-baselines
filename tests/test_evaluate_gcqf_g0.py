import copy

import pytest
import torch

from scripts.evaluate_gcqf_g0 import (
    STATES,
    _stringify_mapping_keys,
    per_seed_gate,
    build_parser,
    load_gcqf_module,
    metric_deltas,
)
from scripts.train_gcqf_g0 import build_module_artifact
from src.gcqf import GCQF


def _metrics(**overrides):
    values = {
        "mAP50-95": 0.20,
        "AP75": 0.18,
        "AP-tiny-SBR": 0.10,
        "AP-medium-SBR": 0.25,
        "AP-large-SBR": 0.30,
        "tiny_recall": 0.60,
    }
    values.update(overrides)
    return values


def test_evaluator_cli_reports_five_required_states():
    args = build_parser().parse_args(
        [
            "--cache",
            "val/manifest.json",
            "--module",
            "best-module.pt",
            "--data",
            "visdrone.yaml",
            "--output",
            "evaluation.json",
            "--calibration",
            "calibration.json",
        ]
    )

    assert args.states == [
        "Global",
        "Raw-Union",
        "Fixed-SADED",
        "Residual-Off",
        "Full-GCQF",
    ]
    assert STATES == args.states
    assert args.batch == 8


def test_metric_deltas_are_method_minus_reference():
    delta = metric_deltas(
        _metrics(),
        _metrics(**{"mAP50-95": 0.205, "AP-tiny-SBR": 0.11}),
    )

    assert delta["mAP50-95"] == pytest.approx(0.005)
    assert delta["AP-tiny-SBR"] == pytest.approx(0.01)


def test_evaluation_stringifies_nested_metric_threshold_keys():
    converted = _stringify_mapping_keys(
        {
            "per_threshold": {
                "tiny": {
                    0.5: {"tp": 3},
                    0.75: {"tp": 2},
                }
            }
        }
    )

    assert converted["per_threshold"]["tiny"] == {
        "0.5": {"tp": 3},
        "0.75": {"tp": 2},
    }


def test_per_seed_gate_enforces_global_and_fixed_saded_success_budgets():
    metrics = {
        "Global": _metrics(),
        "Raw-Union": _metrics(),
        "Fixed-SADED": _metrics(
            **{
                "mAP50-95": 0.224,
                "AP-tiny-SBR": 0.12,
                "AP-medium-SBR": 0.24,
            }
        ),
        "Full-GCQF": _metrics(
            **{
                "mAP50-95": 0.225,
                "AP-tiny-SBR": 0.125,
                "tiny_recall": 0.63,
                "AP-medium-SBR": 0.248,
                "AP-large-SBR": 0.296,
            }
        ),
        "Residual-Off": _metrics(**{"mAP50-95": 0.223}),
    }

    gate = per_seed_gate(
        metrics,
        anchor_exact=True,
        protected_exact=True,
        max_det_exact=True,
        residual_statistics={
            "mean_abs": 0.1,
            "saturation_fraction": 0.0,
        },
    )

    assert gate["map_gain_vs_global"] is True
    assert gate["tiny_gain_vs_global"] is True
    assert gate["tiny_recall_gain_vs_global"] is True
    assert gate["medium_budget_vs_global"] is True
    assert gate["large_budget_vs_global"] is True
    assert gate["medium_recovery_vs_fixed"] is True
    assert gate["map_nonnegative_vs_fixed"] is True
    assert gate["passed"] is True


def test_per_seed_gate_rejects_insufficient_medium_recovery():
    metrics = {
        "Global": _metrics(),
        "Raw-Union": _metrics(),
        "Fixed-SADED": _metrics(**{"AP-medium-SBR": 0.24}),
        "Residual-Off": _metrics(),
        "Full-GCQF": _metrics(
            **{
                "mAP50-95": 0.21,
                "AP-tiny-SBR": 0.12,
                "tiny_recall": 0.63,
                "AP-medium-SBR": 0.247,
                "AP-large-SBR": 0.30,
            }
        ),
    }

    gate = per_seed_gate(
        metrics,
        anchor_exact=True,
        protected_exact=True,
        max_det_exact=True,
        residual_statistics={"mean_abs": 0.1, "saturation_fraction": 0.0},
    )

    assert gate["medium_recovery_vs_fixed"] is False
    assert gate["passed"] is False


def test_module_loader_rejects_state_or_schema_drift(tmp_path):
    module = GCQF(
        query_dim=8,
        num_classes=3,
        num_heads=2,
        num_views=4,
    )
    artifact = build_module_artifact(
        module,
        seed=0,
        epoch=1,
        train_cache_sha256="A" * 64,
        source_commit="B" * 40,
        train_image_ids=("a.jpg",),
        calibration_image_ids=("b.jpg",),
        positive_weights={"tiny": 2.0, "risk": 3.0, "retain": 4.0},
    )
    path = tmp_path / "module.pt"
    torch.save(artifact, path)

    loaded = load_gcqf_module(path, device=torch.device("cpu"))

    assert isinstance(loaded, GCQF)
    broken = copy.deepcopy(artifact)
    broken["schema_version"] = "wrong"
    torch.save(broken, path)
    with pytest.raises(ValueError, match="schema"):
        load_gcqf_module(path, device=torch.device("cpu"))
