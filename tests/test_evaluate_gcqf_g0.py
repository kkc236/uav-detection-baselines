import copy

import pytest
import torch

from scripts.evaluate_gcqf_g0 import (
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


def test_evaluator_cli_reports_four_required_states():
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
        ]
    )

    assert args.states == [
        "Global",
        "Raw-Union",
        "Fixed-SADED",
        "Full-GCQF",
        "Residual-Off",
    ]
    assert args.batch == 8


def test_metric_deltas_are_method_minus_reference():
    delta = metric_deltas(
        _metrics(),
        _metrics(**{"mAP50-95": 0.205, "AP-tiny-SBR": 0.11}),
    )

    assert delta["mAP50-95"] == pytest.approx(0.005)
    assert delta["AP-tiny-SBR"] == pytest.approx(0.01)


def test_per_seed_gate_is_relative_to_fixed_saded_not_only_global():
    metrics = {
        "Global": _metrics(**{"mAP50-95": 0.18}),
        "Raw-Union": _metrics(**{"mAP50-95": 0.19}),
        "Fixed-SADED": _metrics(),
        "Full-GCQF": _metrics(
            **{
                "mAP50-95": 0.204,
                "AP-tiny-SBR": 0.106,
                "AP-medium-SBR": 0.249,
                "AP-large-SBR": 0.299,
            }
        ),
        "Residual-Off": _metrics(),
    }

    gate = per_seed_gate(
        metrics,
        anchor_exact=True,
        protected_exact=True,
        residual_statistics={
            "mean_abs": 0.1,
            "saturation_fraction": 0.0,
        },
    )

    assert gate["map_beats_fixed_saded"] is True
    assert gate["large_within_fixed_budget"] is True
    assert gate["advance_seed"] is True


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
        val_cache_sha256="B" * 64,
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
