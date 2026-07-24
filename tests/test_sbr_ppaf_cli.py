from __future__ import annotations

from pathlib import Path
import gzip
import json
import subprocess
import sys

import pytest

from src.sbr_artifacts import (
    atomic_write_json,
    atomic_write_jsonl_gz,
    sha256_file,
    write_checksums,
)
from src.sbr_ppaf import A_FLOOR


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_importing_router_core_does_not_import_evaluator():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import src.sbr_ppaf; "
                "assert 'src.sbr_metrics' not in sys.modules"
            ),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_route_cli_help_does_not_import_evaluator():
    completed = subprocess.run(
        [sys.executable, "scripts/route_sbr_ppaf.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--input-manifest" in completed.stdout
    assert "--output" in completed.stdout


def _floor_sealed_fixture(tmp_path: Path) -> Path:
    from tests.test_sbr_v2_audit_cli import (
        _gzip_rows,
        _make_input,
        _reseal_changed_input,
    )

    manifest, evidence, _ = _make_input(tmp_path)
    raw_path = evidence / "raw_views.jsonl.gz"
    raw_rows = _gzip_rows(raw_path)
    for row in raw_rows:
        if row["source_order"] == 0:
            row["score"] = A_FLOOR
    atomic_write_jsonl_gz(raw_path, raw_rows)
    _reseal_changed_input(manifest, evidence, "raw_views", raw_path)

    arm_path = evidence / "arm_predictions.jsonl.gz"
    arm_rows = _gzip_rows(arm_path)
    arm_rows[0]["records"][0]["score"] = A_FLOOR
    arm_rows[0]["predictions"][0]["score"] = A_FLOOR
    arm_rows[2]["records"][0]["score"] = A_FLOOR
    total = A_FLOOR + 0.95
    arm_rows[2]["predictions"][0]["box"] = [
        80.0 * 0.95 / total,
        0.0,
        (200.0 * A_FLOOR + 280.0 * 0.95) / total,
        200.0,
    ]
    atomic_write_jsonl_gz(arm_path, arm_rows)
    _reseal_changed_input(manifest, evidence, "arm_predictions", arm_path)
    return manifest


def test_route_replay_seals_exact_prediction_only_contract(tmp_path: Path):
    from scripts.route_sbr_ppaf import route_replay

    manifest = _floor_sealed_fixture(tmp_path)
    output = tmp_path / "route-output"

    result = route_replay(manifest, output, require_clean=False)

    assert result == output.resolve()
    assert {item.name for item in output.iterdir()} == {
        "route",
        "route_anchor.json",
    }
    assert {item.name for item in (output / "route").iterdir()} == {
        "route_manifest.json",
        "predictions.jsonl.gz",
        "coverage.json",
        "route_invariants.json",
        "checksums.sha256",
    }
    invariants = json.loads(
        (output / "route" / "route_invariants.json").read_text(
            encoding="utf-8"
        )
    )
    assert invariants["passed"] is True
    assert invariants["a_floor"]["actual_a_min"] == A_FLOOR
    with gzip.open(
        output / "route" / "predictions.jsonl.gz",
        "rt",
        encoding="utf-8",
    ) as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == 1
    assert set(rows[0]["arms"]) == {"A", "C", "All-A", "P1", "P2", "P3"}
    assert not {
        "gt_boxes",
        "gt_classes",
        "ignore_boxes",
        "matches",
    }.intersection(rows[0])


def test_route_replay_rejects_existing_output_without_modification(
    tmp_path: Path,
):
    from scripts.route_sbr_ppaf import route_replay

    manifest = _floor_sealed_fixture(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        route_replay(manifest, output, require_clean=False)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_frozen_c_box_accepts_only_sub_picometer_accumulation_drift():
    from scripts.route_sbr_ppaf import FrozenArmImage, _assert_frozen
    from src.sbr_fusion import Detection

    prediction = Detection(
        box=(0.0, 0.0, 10.0, 10.0),
        global_xyxy=(0.0, 0.0, 10.0, 10.0),
        score=0.8,
        class_id=0,
        source_order=1,
        query_index=2,
    )

    def frozen(delta):
        return FrozenArmImage(
            records=(),
            predictions=(
                {
                    "box": [0.0, 0.0, 10.0 + delta, 10.0],
                    "global_xyxy": [0.0, 0.0, 10.0, 10.0],
                    "score": 0.8,
                    "class_id": 0,
                    "source_order": 1,
                    "query_index": 2,
                },
            ),
        )

    _assert_frozen("C", "i.jpg", (), (prediction,), frozen(5e-13))
    with pytest.raises(ValueError, match="predictions"):
        _assert_frozen("C", "i.jpg", (), (prediction,), frozen(2e-12))
    with pytest.raises(ValueError, match="predictions"):
        _assert_frozen("A", "i.jpg", (), (prediction,), frozen(5e-13))


def test_route_authenticates_but_does_not_parse_sealed_method_metrics(
    tmp_path: Path,
):
    from scripts.route_sbr_ppaf import route_replay
    from tests.test_sbr_v2_audit_cli import _reseal_changed_input

    manifest = _floor_sealed_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    metrics_path = (
        manifest.parent / payload["files"]["g0_metrics"]["uri"]
    ).resolve()
    metrics_path.write_bytes(b"method-metrics-must-remain-opaque-to-route")
    evidence = (
        manifest.parent / payload["original_evidence_root"]["uri"]
    ).resolve()
    _reseal_changed_input(
        manifest,
        evidence,
        "g0_metrics",
        metrics_path,
    )

    result = route_replay(
        manifest,
        tmp_path / "opaque-metrics-route",
        require_clean=False,
    )

    assert result.is_dir()


def test_evaluate_cli_help_exposes_only_closure_inputs():
    completed = subprocess.run(
        [sys.executable, "scripts/evaluate_sbr_ppaf.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--input-manifest" in completed.stdout
    assert "--route" in completed.stdout
    assert "--route-anchor-sha256" in completed.stdout
    assert "--output" in completed.stdout


def test_evaluation_verifies_route_then_writes_stop_closure(tmp_path: Path):
    from scripts.evaluate_sbr_ppaf import evaluate_replay
    from scripts.route_sbr_ppaf import route_replay

    manifest = _floor_sealed_fixture(tmp_path)
    root = tmp_path / "result"
    route_replay(manifest, root, require_clean=False)
    evaluation = root / "evaluation"

    result = evaluate_replay(
        manifest,
        root / "route",
        sha256_file(root / "route_anchor.json"),
        evaluation,
        require_clean=False,
    )

    assert result == evaluation.resolve()
    assert {item.name for item in evaluation.iterdir()} == {
        "evaluation_manifest.json",
        "metrics.json",
        "deltas.json",
        "evaluation_invariants.json",
        "primary_gate.json",
        "checksums.sha256",
    }
    gate = json.loads(
        (evaluation / "primary_gate.json").read_text(encoding="utf-8")
    )
    assert gate["status"] == "SP_PPAF_STOP"
    assert gate["selected_arm"] == "none"
    invariants = json.loads(
        (evaluation / "evaluation_invariants.json").read_text(
            encoding="utf-8"
        )
    )
    assert invariants["passed"] is True
    assert invariants["route_snapshot_unchanged"] is True


def test_evaluation_rejects_tampered_route_before_loading_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from scripts.evaluate_sbr_ppaf import evaluate_replay
    from scripts.route_sbr_ppaf import route_replay
    import src.sbr_artifacts as artifacts

    manifest = _floor_sealed_fixture(tmp_path)
    root = tmp_path / "tampered"
    route_replay(manifest, root, require_clean=False)
    predictions = root / "route" / "predictions.jsonl.gz"
    predictions.write_bytes(predictions.read_bytes() + b"tamper")
    called = False

    def forbidden_loader(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("dataset loader must not run")

    monkeypatch.setattr(artifacts, "load_dataset", forbidden_loader)

    with pytest.raises(ValueError, match="checksum|anchor|snapshot"):
        evaluate_replay(
            manifest,
            root / "route",
            sha256_file(root / "route_anchor.json"),
            root / "evaluation",
            require_clean=False,
        )

    assert called is False
    assert not (root / "evaluation").exists()


def test_external_anchor_rejects_coherently_resealed_route(tmp_path: Path):
    from scripts.evaluate_sbr_ppaf import evaluate_replay
    from scripts.route_sbr_ppaf import route_replay

    manifest = _floor_sealed_fixture(tmp_path)
    root = tmp_path / "resealed"
    route_replay(manifest, root, require_clean=False)
    route = root / "route"
    external_anchor_sha256 = sha256_file(root / "route_anchor.json")
    predictions = route / "predictions.jsonl.gz"
    with gzip.open(predictions, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    rows[0]["arms"]["P3"][0]["box"][0] += 1.0
    atomic_write_jsonl_gz(predictions, rows)
    route_manifest_path = route / "route_manifest.json"
    route_manifest = json.loads(
        route_manifest_path.read_text(encoding="utf-8")
    )
    route_manifest["predictions_sha256"] = sha256_file(predictions)
    atomic_write_json(route_manifest_path, route_manifest)
    write_checksums(
        route / "checksums.sha256",
        [
            route / "route_manifest.json",
            route / "predictions.jsonl.gz",
            route / "coverage.json",
            route / "route_invariants.json",
        ],
        root=route,
    )
    anchor_path = root / "route_anchor.json"
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchor["route_checksums_sha256"] = sha256_file(
        route / "checksums.sha256"
    )
    anchor["route_manifest_sha256"] = sha256_file(
        route / "route_manifest.json"
    )
    anchor["predictions_sha256"] = sha256_file(predictions)
    atomic_write_json(anchor_path, anchor)

    with pytest.raises(ValueError, match="anchor"):
        evaluate_replay(
            manifest,
            route,
            external_anchor_sha256,
            root / "evaluation",
            require_clean=False,
        )


def test_evaluation_rejects_source_mismatch_before_loading_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import scripts.evaluate_sbr_ppaf as evaluation_module
    from scripts.route_sbr_ppaf import route_replay
    import src.sbr_artifacts as artifacts

    manifest = _floor_sealed_fixture(tmp_path)
    root = tmp_path / "source-mismatch"
    route_replay(manifest, root, require_clean=False)
    real_state = evaluation_module._source_state(require_clean=False)
    changed_state = {
        **real_state,
        "commit": (
            "0" * 40
            if real_state.get("commit") != "0" * 40
            else "1" * 40
        ),
    }
    monkeypatch.setattr(
        evaluation_module,
        "_source_state",
        lambda **kwargs: changed_state,
    )
    called = False

    def forbidden_loader(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("dataset loader must not run")

    monkeypatch.setattr(artifacts, "load_dataset", forbidden_loader)

    with pytest.raises(ValueError, match="source"):
        evaluation_module.evaluate_replay(
            manifest,
            root / "route",
            sha256_file(root / "route_anchor.json"),
            root / "evaluation",
            require_clean=False,
        )

    assert called is False
