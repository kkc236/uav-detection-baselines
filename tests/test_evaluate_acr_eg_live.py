from __future__ import annotations

from pathlib import Path

import pytest


def _required(tmp_path: Path) -> list[str]:
    return [
        "--checkpoint",
        str(tmp_path / "epoch99.pt"),
        "--baseline-checkpoint",
        str(tmp_path / "baseline.pt"),
        "--data",
        str(tmp_path / "data.yaml"),
        "--output",
        str(tmp_path / "evaluation"),
    ]


def test_live_evaluator_defaults_are_frozen(tmp_path: Path) -> None:
    from scripts.evaluate_acr_eg_live import (
        EXPECTED_BASELINE_SHA256,
        EXPECTED_CHECKPOINT_SHA256,
        EXPECTED_DATASET_SIGNATURE,
        build_parser,
        validate_protocol,
    )

    args = build_parser().parse_args(_required(tmp_path))
    validate_protocol(args)

    assert args.device == "0"
    assert args.batch == 1
    assert args.workers == 0
    assert args.imgsz == 640
    assert args.conf == 0.001
    assert args.max_det == 300
    assert args.expected_records == 548
    assert args.expected_epoch == 99
    assert args.expected_baseline_sha256 == EXPECTED_BASELINE_SHA256
    assert args.expected_checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
    assert args.dataset_signature == EXPECTED_DATASET_SIGNATURE


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--device", "1"),
        ("--batch", "2"),
        ("--imgsz", "1088"),
        ("--conf", "0.01"),
        ("--max-det", "100"),
        ("--expected-records", "547"),
    ],
)
def test_live_evaluator_rejects_protocol_drift(
    tmp_path: Path,
    flag: str,
    value: str,
) -> None:
    from scripts.evaluate_acr_eg_live import build_parser, validate_protocol

    args = build_parser().parse_args(_required(tmp_path) + [flag, value])
    with pytest.raises(ValueError, match="ACR_EG_LIVE_PROTOCOL_DRIFT"):
        validate_protocol(args)


def test_smoke_limit_is_explicit_and_final_run_has_no_limit(tmp_path: Path) -> None:
    from scripts.evaluate_acr_eg_live import build_parser, validate_protocol

    parser = build_parser()
    final_args = parser.parse_args(_required(tmp_path))
    validate_protocol(final_args)
    assert final_args.limit is None

    smoke_args = parser.parse_args(_required(tmp_path) + ["--smoke", "--limit", "1"])
    validate_protocol(smoke_args)
    assert smoke_args.limit == 1

    invalid = parser.parse_args(_required(tmp_path) + ["--limit", "1"])
    with pytest.raises(ValueError, match="ACR_EG_LIVE_LIMIT_REQUIRES_SMOKE"):
        validate_protocol(invalid)
