from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest


SCRIPT = Path("scripts/run_bpdd_mature_signal_probe.py")
FDR_EPOCH100_SHA256 = (
    "C2F638744508ADFE7B6C4A1EF3E08C503273F628062E4650AD59FFFF4C6588C2"
)


def _load_module():
    assert SCRIPT.is_file(), "mature BPDD signal probe has not been implemented"
    spec = importlib.util.spec_from_file_location(
        "run_bpdd_mature_signal_probe_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_and_authority_are_frozen() -> None:
    module = _load_module()
    argv = [
        "--checkpoint",
        "epoch100.pt",
        "--dataset-root",
        "VisDrone",
        "--report-root",
        "report",
    ]

    assert module._parse_args(argv) == Namespace(
        checkpoint=Path("epoch100.pt"),
        dataset_root=Path("VisDrone"),
        report_root=Path("report"),
        device="0",
    )
    assert module.FDR_EPOCH100_SHA256 == FDR_EPOCH100_SHA256
    assert module.BATCH_LIMIT == 16
    assert module.OFFICIAL_VAL_OPENED is False
    for forbidden in ("--margin", "--temperature", "--weight", "--batch-limit"):
        with pytest.raises(SystemExit):
            module._parse_args([*argv, forbidden, "1"])


def test_gate_requires_usable_teacher_and_mixture_ablation_support() -> None:
    module = _load_module()
    passing = {
        "batches": 16,
        "final_matched_queries": 100,
        "statistics_finite": True,
        "active_edge_ratio_max": 0.1,
        "mean_teacher_improvement_max": 0.2,
        "mixture_beats_final_ratio_mean": 0.36,
        "mean_mixture_advantage_over_final": 0.01,
    }

    decision = module._decide(passing)

    assert decision["status"] == "passed"
    assert decision["screen30_eligible"] is True
    assert all(decision["checks"].values())

    for key, value in (
        ("active_edge_ratio_max", 0.0),
        ("mean_teacher_improvement_max", 0.0),
        ("mixture_beats_final_ratio_mean", 0.0),
        ("mean_mixture_advantage_over_final", 0.0),
    ):
        failed = dict(passing)
        failed[key] = value
        result = module._decide(failed)
        assert result["status"] == "scientific_failed"
        assert result["screen30_eligible"] is False


def test_report_is_create_only_and_excludes_official_val(tmp_path: Path) -> None:
    module = _load_module()
    payload = {
        "status": "passed",
        "screen30_eligible": True,
        "data_scope": "fixed_train10_only",
        "official_val_opened": False,
    }

    module._write_report(tmp_path, payload)

    report = json.loads((tmp_path / "mature-signal.json").read_text("utf-8"))
    assert report == payload
    assert (tmp_path / "SHA256SUMS.txt").is_file()
    with pytest.raises(FileExistsError):
        module._write_report(tmp_path, payload)


def test_checkpoint_hash_is_verified_before_model_loading(tmp_path: Path) -> None:
    module = _load_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"authority")

    with pytest.raises(RuntimeError, match="checkpoint SHA256 mismatch"):
        module._verify_checkpoint(checkpoint)
