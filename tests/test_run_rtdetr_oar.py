from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from decimal import Decimal
from pathlib import Path

import pytest
import torch

from src.rtdetr_quality_probe import c1_features


SCRIPT = Path("scripts/run_rtdetr_oar.py")


def _load_module():
    assert SCRIPT.is_file(), "OAR runner has not been implemented"
    spec = importlib.util.spec_from_file_location("run_rtdetr_oar_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _observed_d0() -> tuple[dict[str, dict[str, float]], dict[int, float]]:
    return (
        {
            "stock": {
                "map": 0.28628865801344866,
                "ap75": 0.292364074762,
                "ap50": 0.476388925325,
            },
            "presence": {"map": 0.294608200682, "ap75": 0.300115294639},
            "query_iou": {"map": 0.324185693386, "ap75": 0.344708985960},
            "same_class": {"map": 0.409733588907, "ap75": 0.413238330995},
        },
        {
            20: 0.304549967436,
            40: 0.335016751522,
            60: 0.358000690130,
            100: 0.385568106152,
        },
    )


def test_sparse_d0_fails_without_extending_the_grid() -> None:
    module = _load_module()
    decomposition, restricted = _observed_d0()

    reports = module._sparse_d0_reports(
        decomposition=decomposition,
        restricted_map=restricted,
    )

    decision = reports["sparse-d0-decision.json"]
    assert decision["status"] == "scientific_failed"
    assert decision["selected_k"] is None
    assert decision["frozen_k_grid"] == [20, 40, 60, 100]
    assert decision["next_authority"] == "oar-all-pair-amendment"
    assert (
        Decimal(reports["d0-k-coverage.json"]["coverage"]["100"]["recovered"])
        < module.OAR_GAIN_RECOVERY
    )
    assert reports["d0-k-coverage.json"]["recovery_threshold"] == "0.90"
    assert decision["grid_extended"] is False
    assert reports["d0-oracle-decomposition.json"]["metrics"]["stock"]["map"] == (
        "0.28628865801344866"
    )


def test_sparse_d0_reports_are_canonical_create_only_and_drift_safe(
    tmp_path: Path,
) -> None:
    module = _load_module()
    decomposition, restricted = _observed_d0()
    reports = module._sparse_d0_reports(
        decomposition=decomposition,
        restricted_map=restricted,
    )

    first = module._write_sparse_d0_reports(tmp_path, reports)
    second = module._write_sparse_d0_reports(tmp_path, reports)

    assert first == second
    assert set(first) == {
        "d0-oracle-decomposition.json",
        "d0-k-coverage.json",
        "sparse-d0-decision.json",
    }
    for name, digest in first.items():
        path = tmp_path / name
        assert len(digest) == 64
        raw = path.read_bytes()
        assert raw.endswith(b"\n")
        assert json.loads(raw) == reports[name]

    changed = json.loads(json.dumps(reports))
    changed["sparse-d0-decision.json"]["status"] = "passed"
    with pytest.raises(RuntimeError, match="immutable|drift"):
        module._write_sparse_d0_reports(tmp_path, changed)


def test_sparse_d0_rejects_authority_drift_and_ignores_mapping_order() -> None:
    module = _load_module()
    decomposition, restricted = _observed_d0()
    reversed_decomposition = dict(reversed(tuple(decomposition.items())))
    assert module._sparse_d0_reports(
        decomposition=reversed_decomposition,
        restricted_map=dict(reversed(tuple(restricted.items()))),
    )["sparse-d0-decision.json"]["status"] == "scientific_failed"

    changed = {name: dict(metrics) for name, metrics in decomposition.items()}
    changed["same_class"]["map"] += 2e-12
    with pytest.raises(RuntimeError, match="metric drift"):
        module._sparse_d0_reports(
            decomposition=changed,
            restricted_map=restricted,
        )

    extra = {name: dict(metrics) for name, metrics in decomposition.items()}
    extra["presence"]["ap50"] = 0.1
    with pytest.raises(ValueError, match="exactly"):
        module._sparse_d0_reports(
            decomposition=extra,
            restricted_map=restricted,
        )


def test_runner_cli_and_scientific_controls_are_frozen() -> None:
    module = _load_module()
    argv = [
        "--baseline-checkpoint",
        "baseline.pt",
        "--dataset-root",
        "dataset",
        "--cache-root",
        "cache",
        "--historical-report-root",
        "historical",
        "--report-root",
        "report",
    ]

    assert module._parse_args(argv) == Namespace(
        baseline_checkpoint=Path("baseline.pt"),
        dataset_root=Path("dataset"),
        cache_root=Path("cache"),
        historical_report_root=Path("historical"),
        report_root=Path("report"),
        device="0",
    )
    assert module.EPOCHS == 20
    assert module.BATCH_SIZE == 8
    for forbidden in ("--epochs", "--lr", "--seed", "--batch"):
        with pytest.raises(SystemExit):
            module._parse_args([*argv, forbidden, "1"])


def test_r2_features_reuse_exact_detached_276_value_representation() -> None:
    module = _load_module()
    boxes = torch.rand(2, 300, 4, requires_grad=True)
    logits = torch.randn(2, 300, 10, requires_grad=True)
    hidden = torch.randn(2, 300, 256, requires_grad=True)

    features = module._oar_r2_features(boxes, logits, hidden)
    control = c1_features(boxes, logits, num_classes=10)

    assert features.shape == (2, 300, 10, 276)
    assert not features.requires_grad
    assert torch.equal(features[..., :20], control)
    expected_hidden = hidden.detach().float().unsqueeze(2).expand(-1, -1, 10, -1)
    assert torch.equal(features[..., 20:], expected_hidden)


def test_checkpoint_selection_uses_map_ap75_ap50_then_earliest_epoch() -> None:
    module = _load_module()
    history = [
        {"epoch": 1, "metrics": {"map": 0.2, "ap75": 0.11, "ap50": 0.30}},
        {"epoch": 2, "metrics": {"map": 0.2, "ap75": 0.11, "ap50": 0.31}},
        {"epoch": 3, "metrics": {"map": 0.2, "ap75": 0.11, "ap50": 0.31}},
    ]

    assert module._select_checkpoint(history)["epoch"] == 2
