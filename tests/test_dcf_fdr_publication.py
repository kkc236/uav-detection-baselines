from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from src.dcf_fdr_publication import (
    EXPECTED_INITIAL_STATE_SHA256,
    EXPECTED_SOURCE_COMMIT,
    ArmSpec,
    PublicationGateError,
    build_comparison,
    stage_evidence,
    validate_arm,
)


METRICS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


def make_arm(
    base: Path,
    arm: str,
    *,
    epochs=range(1, 101),
    best_map: str = "0.29696",
    source_commit: str = EXPECTED_SOURCE_COMMIT,
    initial_sha: str = EXPECTED_INITIAL_STATE_SHA256,
    log_text: str = "training complete\n",
) -> ArmSpec:
    output_root = base / arm
    run_name = f"formal-seed0-{arm}-fdr-v1"
    run_dir = output_root / run_name
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    rows = []
    epoch_values = list(epochs)
    for epoch in epoch_values:
        value = 0.20 + int(epoch) / 2000
        rows.append(
            {
                "epoch": str(epoch),
                METRICS[0]: f"{value + 0.20:.5f}",
                METRICS[1]: f"{value + 0.10:.5f}",
                METRICS[2]: f"{value + 0.05:.5f}",
                METRICS[3]: f"{value:.5f}",
            }
        )
    if rows:
        rows[-1][METRICS[3]] = best_map
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", *METRICS))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "args.yaml").write_text("epochs: 100\nseed: 0\n", encoding="utf-8")
    torch.save({"model": {"weight": torch.tensor([1.0])}}, weights / "best.pt")
    torch.save({"ema": {"weight": torch.tensor([2.0])}}, weights / "last.pt")
    authority = {
        "format_version": 1,
        "method": "clean_fdr" if arm == "clean" else "dcf_fdr",
        "source": {"git_commit": source_commit},
        "initial_state": {"sha256": initial_sha},
        "settings": {"epochs": 100, "seed": 0, "name": run_name},
    }
    authority_dir = output_root / "authority"
    authority_dir.mkdir(parents=True)
    (authority_dir / f"{run_name}.json").write_text(
        json.dumps(authority), encoding="utf-8"
    )
    (output_root / f"train-{arm}.log").write_text(log_text, encoding="utf-8")
    return ArmSpec(arm=arm, output_root=output_root, run_name=run_name)


def test_validate_arm_requires_exact_continuous_formal100(tmp_path: Path) -> None:
    incomplete = make_arm(tmp_path / "incomplete", "clean", epochs=range(1, 100))
    with pytest.raises(PublicationGateError, match="exactly 100"):
        validate_arm(incomplete)

    discontinuous = make_arm(
        tmp_path / "gap", "clean", epochs=[*range(1, 50), *range(51, 102)]
    )
    with pytest.raises(PublicationGateError, match="continuous"):
        validate_arm(discontinuous)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source", "bad-commit", "source commit"),
        ("initial", "bad-state", "initial-state"),
    ],
)
def test_validate_arm_rejects_wrong_authority(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    kwargs = {"source_commit": value} if field == "source" else {"initial_sha": value}
    spec = make_arm(tmp_path, "clean", **kwargs)
    with pytest.raises(PublicationGateError, match=message):
        validate_arm(spec)


def test_validate_arm_rejects_terminal_error_and_unreadable_checkpoint(
    tmp_path: Path,
) -> None:
    failed = make_arm(tmp_path / "failed", "clean", log_text="CUDA out of memory\n")
    with pytest.raises(PublicationGateError, match="terminal failure"):
        validate_arm(failed)

    corrupt = make_arm(tmp_path / "corrupt", "clean")
    (corrupt.run_dir / "weights" / "best.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(PublicationGateError, match="unreadable checkpoint"):
        validate_arm(corrupt)


def test_comparison_uses_unrounded_best_map_and_aligns_all_epochs(tmp_path: Path) -> None:
    clean = validate_arm(make_arm(tmp_path, "clean", best_map="0.29696"))
    dcf = validate_arm(make_arm(tmp_path, "dcf", best_map="0.29697"))
    report, rows = build_comparison(clean, dcf)

    assert report["decision"] == "passed_nonnegative"
    assert report["best_delta"][METRICS[-1]] == pytest.approx(0.00001)
    assert len(rows) == 100
    assert rows[-1]["epoch"] == 100
    assert rows[-1][f"delta/{METRICS[-1]}"] == pytest.approx(0.00001)


def test_stage_evidence_is_complete_and_contains_no_absolute_run_paths(
    tmp_path: Path,
) -> None:
    clean = validate_arm(make_arm(tmp_path, "clean"))
    dcf = validate_arm(make_arm(tmp_path, "dcf", best_map="0.29700"))
    staged = stage_evidence(clean, dcf, tmp_path / "stage")

    expected = {
        "clean/results.csv",
        "clean/args.yaml",
        "clean/authority.json",
        "clean/train.log.gz",
        "clean/summary.json",
        "dcf/results.csv",
        "dcf/args.yaml",
        "dcf/authority.json",
        "dcf/train.log.gz",
        "dcf/summary.json",
        "aligned-epochs.csv",
        "comparison.json",
        "RESULTS.md",
        "artifact-manifest.json",
        "lightweight-evidence.tar.gz",
    }
    assert expected.issubset(
        {path.relative_to(staged.root).as_posix() for path in staged.root.rglob("*") if path.is_file()}
    )
    manifest_text = staged.manifest_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["source_commit"] == EXPECTED_SOURCE_COMMIT
    assert manifest["decision"] == "passed_nonnegative"
    assert all("sha256" in item for item in manifest["artifacts"].values())

