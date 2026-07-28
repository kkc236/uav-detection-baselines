from __future__ import annotations

from scripts.package_acr_eg_integrated import build_parser


def test_packager_requires_yaml_baseline_module_and_exact_commit(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--config",
            str(tmp_path / "rtdetr-l-gcte.yaml"),
            "--baseline-checkpoint",
            str(tmp_path / "baseline.pt"),
            "--module-checkpoint",
            str(tmp_path / "module.pt"),
            "--source-commit",
            "a" * 40,
            "--output",
            str(tmp_path / "integrated.pt"),
        ]
    )

    assert args.config.name == "rtdetr-l-gcte.yaml"
    assert args.baseline_checkpoint.name == "baseline.pt"
    assert args.module_checkpoint.name == "module.pt"
    assert args.source_commit == "a" * 40
    assert args.output.name == "integrated.pt"
