from __future__ import annotations

from pathlib import Path
import gzip
import json
import subprocess
import sys

import pytest


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
