from __future__ import annotations

from pathlib import Path

from scripts.evaluate_acr_eg_integrated import build_parser, build_protocol, evaluate


def test_integrated_evaluator_records_mature_baseline_and_yaml(tmp_path) -> None:
    config = tmp_path / "rtdetr-l-gcte.yaml"
    baseline = tmp_path / "matched-baseline-best-epoch-0100.pt"
    module = tmp_path / "best-module.pt"
    args = build_parser().parse_args(
        [
            "--cache",
            str(tmp_path / "val-cache"),
            "--module",
            str(module),
            "--data",
            str(tmp_path / "data.yaml"),
            "--output",
            str(tmp_path / "evaluation.json"),
            "--calibration",
            str(tmp_path / "calibration.json"),
            "--config",
            str(config),
            "--baseline-checkpoint",
            str(baseline),
        ]
    )
    protocol = build_protocol(args)

    assert protocol["integration"]["forward_integration"] is True
    assert protocol["integration"]["config"] == str(config.resolve())
    assert protocol["baseline"]["checkpoint"] == str(baseline.resolve())
    assert protocol["module_checkpoint"] == str(module.resolve())


def test_integrated_evaluator_normalizes_cache_directory(
    tmp_path,
    monkeypatch,
) -> None:
    cache = tmp_path / "val-cache"
    cache.mkdir()
    output = tmp_path / "evaluation.json"
    args = build_parser().parse_args(
        [
            "--cache",
            str(cache),
            "--module",
            str(tmp_path / "module.pt"),
            "--data",
            str(tmp_path / "data.yaml"),
            "--output",
            str(output),
            "--calibration",
            str(tmp_path / "calibration.json"),
            "--config",
            str(tmp_path / "config.yaml"),
            "--baseline-checkpoint",
            str(tmp_path / "baseline.pt"),
        ]
    )
    observed = {}

    def fake_evaluate(value):
        observed["cache"] = value.cache
        output.write_text("{}\n", encoding="utf-8")
        return output

    monkeypatch.setattr(
        "scripts.evaluate_acr_eg_integrated.evaluate_gcqf_g0.evaluate",
        fake_evaluate,
    )
    evaluate(args)

    assert observed["cache"] == cache / "manifest.json"
