from pathlib import Path

import pytest


def required_args(tmp_path: Path) -> list[str]:
    return [
        "--checkpoint",
        str(tmp_path / "epoch99.pt"),
        "--baseline-checkpoint",
        str(tmp_path / "baseline.pt"),
        "--data",
        str(tmp_path / "VisDrone.yaml"),
        "--output",
        str(tmp_path / "native-evaluation"),
    ]


def test_native_cli_defaults_are_frozen(tmp_path: Path) -> None:
    from scripts.evaluate_acr_eg_ultralytics_native import (
        build_parser,
        validate_native_protocol,
    )

    args = build_parser().parse_args(required_args(tmp_path))
    validate_native_protocol(args)
    assert args.device == "0"
    assert args.batch == 1
    assert args.workers == 0
    assert args.imgsz == 640
    assert args.conf == 0.001
    assert args.max_det == 300
    assert args.expected_records == 548
    assert args.expected_epoch == 99
    assert args.amp is True


def test_native_cli_requires_explicit_one_image_smoke(tmp_path: Path) -> None:
    from scripts.evaluate_acr_eg_ultralytics_native import (
        build_parser,
        validate_native_protocol,
    )

    parser = build_parser()
    smoke = parser.parse_args(
        required_args(tmp_path) + ["--smoke", "--limit", "1"]
    )
    validate_native_protocol(smoke)

    invalid = parser.parse_args(required_args(tmp_path) + ["--limit", "1"])
    with pytest.raises(ValueError, match="ACR_EG_NATIVE_PROTOCOL_DRIFT"):
        validate_native_protocol(invalid)

