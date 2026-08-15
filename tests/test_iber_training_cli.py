from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_training_cli_exposes_only_operational_paths_device_and_resume() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/train_iber.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for allowed in (
        "--baseline-checkpoint",
        "--dataset-root",
        "--gate1-decision",
        "--publication-config",
        "--output-root",
        "--device",
        "--resume-checkpoint",
    ):
        assert allowed in result.stdout
    for forbidden in (
        "--stage",
        "--epochs",
        "--seed",
        "--batch",
        "--workers",
        "--imgsz",
        "--lr",
        "--mosaic",
        "--close-mosaic",
    ):
        assert forbidden not in result.stdout


def test_training_source_locks_frozen_on_the_fly_screen_and_publication() -> None:
    source = Path("scripts/train_iber.py").read_text(encoding="utf-8")
    for marker in (
        "FrozenIBERAdapter",
        "detector.eval()",
        "requires_grad_(False)",
        'probe="b3"',
        "training_step",
        "select_hashed_subset",
        "subset_signature",
        "close_mosaic",
        "amp_scale",
        "epoch-{epoch:04d}.pt",
        "detector_sha_before",
        "detector_sha_after",
        "optimizer_evidence",
        "results.jsonl",
        "diagnostics.jsonl",
        "evaluate_iber.py",
        "publish_iber_epoch.py",
        "publication-ledger.jsonl",
        "highest_contiguous_verified_epoch",
        "generator.manual_seed",
        "loader.reset()",
    ):
        assert marker in source
    assert "FrozenITBERAdapter" not in source
    assert "rtdetr_itber" not in source
    assert "itber-v1.1" not in source
    assert "weights_only=False" not in source
    assert "weights_only=True" in source
