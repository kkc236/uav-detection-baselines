from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest


SCRIPT = Path("scripts/run_rtdetr_complementarity_oracle.py")


def _load_module():
    assert SCRIPT.is_file(), "complementarity-oracle runner has not been implemented"
    spec = importlib.util.spec_from_file_location(
        "run_rtdetr_complementarity_oracle_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_exists_and_cli_is_frozen() -> None:
    module = _load_module()
    argv = [
        "--fdr-checkpoint",
        "fdr.pt",
        "--frequencycm-checkpoint",
        "frequencycm.pt",
        "--dataset-root",
        "dataset",
        "--cache-root",
        "cache",
        "--report-root",
        "report",
    ]

    assert module._parse_args(argv) == Namespace(
        fdr_checkpoint=Path("fdr.pt"),
        frequencycm_checkpoint=Path("frequencycm.pt"),
        dataset_root=Path("dataset"),
        cache_root=Path("cache"),
        report_root=Path("report"),
        device="0",
    )
    assert module.IMAGE_SIZE == 640
    assert module.BATCH_SIZE == 8
    assert module.WORKERS == 8
    assert module.CONFIDENCE == 0.001
    assert module.MAX_DET == 300
    assert module.NMS is False
    assert module.VAL_COUNT == 548
    assert module.NUM_CLASSES == 10
    for forbidden in (
        "--threshold",
        "--alpha",
        "--max-det",
        "--conf",
        "--batch",
        "--workers",
        "--nms",
    ):
        with pytest.raises(SystemExit):
            module._parse_args([*argv, forbidden, "1"])


def test_runner_is_bound_to_exact_checkpoint_authorities() -> None:
    module = _load_module()
    assert module.FDR_CHECKPOINT_SHA256 == (
        "C2F638744508ADFE7B6C4A1EF3E08C503273F628062E4650AD59FFFF4C6588C2"
    )
    assert module.FREQUENCYCM_CHECKPOINT_SHA256 == (
        "2BBCD6057FEFED5792F786A18E603F8FECA3EC426A6F68938F5F8ADA1603A141"
    )
    assert module.FREQUENCYCM_SOURCE_COMMIT == (
        "d3655b14c17a3c8ca14e1888517b6fde4e059766"
    )


def test_report_writer_is_create_only_and_labels_oracle_non_deployable(
    tmp_path: Path,
) -> None:
    module = _load_module()
    payload = {
        "decision": {"decision": "yellow"},
        "stock": {
            "fdr": {"map": 0.28966},
            "frequencycm": {"map": 0.28609},
        },
        "oracle": {"candidate_map_delta": 0.004},
        "coverage": {"tiny_small_recall50_delta": 0.012},
    }

    module._write_summary(tmp_path, payload)

    summary = json.loads((tmp_path / "oracle-summary.json").read_text("utf-8"))
    assert summary["interpretation"] == "non_deployable_design_selection_evidence"
    assert (tmp_path / "frequencycm-complementarity-report.md").is_file()
    assert (tmp_path / "SHA256SUMS.txt").is_file()
    with pytest.raises(FileExistsError):
        module._write_summary(tmp_path, payload)

