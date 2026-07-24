from __future__ import annotations

from pathlib import Path
import gzip
import json
import subprocess
import sys

import pytest

from src.sbr_artifacts import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_route_cli_help_does_not_import_evaluator():
    completed = subprocess.run(
        [sys.executable, "scripts/route_saded.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--input-manifest" in completed.stdout
    assert "--output" in completed.stdout
    assert "src.sbr_metrics" not in completed.stderr


def test_route_replay_seals_exact_gt_free_contract(tmp_path: Path):
    from scripts.route_saded import route_replay
    from tests.test_sbr_ppaf_cli import _floor_sealed_fixture

    manifest = _floor_sealed_fixture(tmp_path)
    output = tmp_path / "saded-route"

    result = route_replay(manifest, output, require_clean=False)

    assert result == output.resolve()
    assert {item.name for item in output.iterdir()} == {
        "route",
        "route_anchor.json",
    }
    assert {item.name for item in (output / "route").iterdir()} == {
        "route_manifest.json",
        "predictions.jsonl.gz",
        "capacity.json",
        "route_invariants.json",
        "checksums.sha256",
    }
    with gzip.open(
        output / "route" / "predictions.jsonl.gz",
        "rt",
        encoding="utf-8",
    ) as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == 1
    assert set(rows[0]["arms"]) == {"A", "route_control"}
    assert not {
        "gt_boxes",
        "gt_classes",
        "ignore_boxes",
        "matches",
    }.intersection(rows[0])
    invariants = json.loads(
        (output / "route" / "route_invariants.json").read_text(
            encoding="utf-8"
        )
    )
    assert invariants["passed"] is True


def test_route_replay_rejects_existing_output_without_modification(
    tmp_path: Path,
):
    from scripts.route_saded import route_replay
    from tests.test_sbr_ppaf_cli import _floor_sealed_fixture

    manifest = _floor_sealed_fixture(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        route_replay(manifest, output, require_clean=False)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_r0_gate_uses_only_frozen_safety_thresholds():
    from scripts.evaluate_saded import adjudicate_r0

    passing = adjudicate_r0(
        deltas={"AP75": -0.002, "AP-large-SBR": -0.005},
        aggregate_remaining_slots=1,
        invariants_passed=True,
    )
    failing = adjudicate_r0(
        deltas={"AP75": -0.002001, "AP-large-SBR": -0.005},
        aggregate_remaining_slots=1,
        invariants_passed=True,
    )

    assert passing["decision"] == "R0_GO"
    assert passing["failures"] == []
    assert failing["decision"] == "R0_STOP"
    assert failing["failures"] == ["AP75_delta<-0.002"]


def test_evaluate_replay_seals_one_isolated_r0_decision(tmp_path: Path):
    from scripts.evaluate_saded import evaluate_replay
    from scripts.route_saded import route_replay
    from tests.test_sbr_ppaf_cli import _floor_sealed_fixture

    manifest = _floor_sealed_fixture(tmp_path)
    route_root = route_replay(
        manifest,
        tmp_path / "route",
        require_clean=False,
    )
    output = tmp_path / "evaluation"

    result = evaluate_replay(
        manifest,
        route_root / "route",
        sha256_file(route_root / "route_anchor.json"),
        output,
        require_clean=False,
    )

    assert result == output.resolve()
    assert {item.name for item in output.iterdir()} == {
        "evaluation_manifest.json",
        "metrics.json",
        "deltas.json",
        "capacity.json",
        "evaluation_invariants.json",
        "r0_gate.json",
        "checksums.sha256",
    }
    gate = json.loads(
        (output / "r0_gate.json").read_text(encoding="utf-8")
    )
    assert gate["decision"] in {"R0_GO", "R0_STOP"}
    assert gate["decision"] != "INVALID"


def test_evaluate_replay_rejects_route_mutation(tmp_path: Path):
    from scripts.evaluate_saded import evaluate_replay
    from scripts.route_saded import route_replay
    from tests.test_sbr_ppaf_cli import _floor_sealed_fixture

    manifest = _floor_sealed_fixture(tmp_path)
    route_root = route_replay(
        manifest,
        tmp_path / "route",
        require_clean=False,
    )
    manifest_path = route_root / "route" / "route_manifest.json"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum"):
        evaluate_replay(
            manifest,
            route_root / "route",
            sha256_file(route_root / "route_anchor.json"),
            tmp_path / "evaluation",
            require_clean=False,
        )

    assert not (tmp_path / "evaluation").exists()
