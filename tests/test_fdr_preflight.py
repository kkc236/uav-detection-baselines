from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_fdr_preflight import (
    FIXED_RUNTIME,
    GATE_ORDER,
    SCREEN_AUTHORITY,
    PreflightContext,
    build_parser,
    canonical_sha256,
    decide_preflight,
    make_evidence_record,
    run_f0,
    run_preflight,
    summarize_representation,
    validate_4090_evidence,
    validate_f4_evidence,
    write_create_only_report,
)
from src.fdr_protocol import DFINE_COMMIT, canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]


def _context(tmp_path: Path) -> PreflightContext:
    tmp_path.mkdir(parents=True, exist_ok=True)
    protocol = tmp_path / "protocol.json"
    protocol.write_bytes(
        canonical_json_bytes(
            {
                "protocol": {"dfine_commit": DFINE_COMMIT},
                "source": {"commit": "a" * 40, "tree_sha256": "B" * 64},
            }
        )
        + b"\n"
    )
    checkpoint = tmp_path / "baseline.pt"
    checkpoint.write_bytes(b"baseline-authority")
    dataset = tmp_path / "VisDrone"
    dataset.mkdir()
    return PreflightContext(
        protocol_manifest=protocol,
        baseline_checkpoint=checkpoint,
        dataset_root=dataset,
        report_root=tmp_path / "report",
        repository_root=ROOT,
    )


def _f1() -> dict:
    return {
        "status": "passed",
        "device": "cpu",
        "checks": {
            "neutral_encode_decode": True,
            "cumulative_residual": True,
            "fgl_zero_stock_exact": True,
            "classification_stock_exact": True,
            "matcher_stock_exact": True,
            "top300_stock_exact": True,
            "nms_stock_exact": True,
        },
    }


def _f2() -> dict:
    return {
        "status": "passed",
        "device": "cpu",
        "shapes": {
            "corner_logits": [6, 8, 300, 132],
            "boxes": [6, 8, 300, 4],
            "scores": [6, 8, 300, 10],
        },
        "cases": {
            "normal_queries": True,
            "dn_queries": True,
            "empty_gt": True,
            "mixed_empty_gt": True,
            "boundary_clipping": True,
            "auxiliary_layers": True,
            "finite_forward": True,
            "finite_backward": True,
        },
        "amp": {"enabled": True, "scale": 128.0, "skipped_steps": 0},
    }


def _f3() -> dict:
    return {
        "status": "passed",
        "runtime": {"device": "cuda:0", "batch": 8, "imgsz": 640},
        "hardware": {
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "device_index": 0,
            "total_memory_bytes": 25_769_803_776,
            "compute_capability": [8, 9],
            "driver": "550.142",
            "cuda": "12.1",
            "torch": "2.5.1+cu121",
            "torchvision": "0.20.1+cu121",
        },
        "single_step": {
            "real_visdrone_batch": True,
            "forward": True,
            "backward": True,
            "optimizer": "MuSGD",
            "optimizer_steps": 1,
            "loss_finite": True,
            "gradients_finite": True,
            "expected_gradient_coverage": True,
            "unexpected_trainable_parameters": 0,
            "excluded_components": [],
            "amp_scale_before": 128.0,
            "amp_scale_after": 128.0,
            "amp_skipped_steps": 0,
            "validation_postprocess": True,
            "checkpoint_roundtrip": True,
        },
        "immutability": {
            "source_before_sha256": "A" * 64,
            "source_after_sha256": "A" * 64,
            "ultralytics_before_sha256": "B" * 64,
            "ultralytics_after_sha256": "B" * 64,
            "baseline_public_state_sha256": "C" * 64,
            "fdr_public_state_sha256": "C" * 64,
            "baseline_data_order_sha256": "D" * 64,
            "fdr_data_order_sha256": "D" * 64,
        },
    }


def _representation() -> dict:
    return summarize_representation(
        reference_boxes=[
            [0.50, 0.50, 0.02, 0.02],
            [0.40, 0.40, 0.04, 0.04],
            [0.60, 0.60, 0.10, 0.10],
        ],
        reconstructed_boxes=[
            [0.50, 0.50, 0.02, 0.02],
            [0.41, 0.40, 0.04, 0.04],
            [0.60, 0.60, 0.10, 0.10],
        ],
        target_indices=[
            [0, 1, 2, 31],
            [1, 1, 1, 1],
            [31, 31, 5, 5],
        ],
        object_widths=[10.0, 20.0, 50.0],
        object_heights=[10.0, 20.0, 50.0],
    )


def _f4() -> dict:
    return {
        "status": "passed",
        "official_reference_match": True,
        "reconstruction_tolerance": 0.02,
        "representation": _representation(),
    }


def _runners(order: list[str]) -> dict[str, object]:
    reports = {"F1": _f1(), "F2": _f2(), "F3": _f3(), "F4": _f4()}

    def runner(gate: str):
        def execute(context: PreflightContext) -> dict:
            assert context.runtime == FIXED_RUNTIME
            order.append(gate)
            return reports[gate]

        return execute

    return {gate: runner(gate) for gate in reports}


def test_gate_order_and_all_pass_authorize_screen(tmp_path: Path) -> None:
    context = _context(tmp_path)
    order: list[str] = []

    decision = run_preflight(context, gate_runners=_runners(order))

    assert order == ["F1", "F2", "F3", "F4"]
    assert decision["gate_states"] == {gate: "passed" for gate in GATE_ORDER}
    assert decision["status"] == "passed"
    assert decision["screen_eligible"] is True
    assert decision["screen_authority"] == {
        "schedule_epochs": 50,
        "cutoff_epoch": 30,
    }
    assert sorted(path.name for path in context.report_root.iterdir()) == [
        "F0.json",
        "F1.json",
        "F2.json",
        "F3.json",
        "F4.json",
        "decision.json",
    ]


@pytest.mark.parametrize("failed_gate", ["F1", "F2", "F3", "F4"])
def test_failure_blocks_every_later_gate(tmp_path: Path, failed_gate: str) -> None:
    context = _context(tmp_path)
    called: list[str] = []
    reports = _runners(called)

    def fail(_: PreflightContext) -> dict:
        called.append(failed_gate)
        return {"status": "engineering_failed", "reason": "injected failure"}

    reports[failed_gate] = fail
    decision = run_preflight(context, gate_runners=reports)

    failed_index = GATE_ORDER.index(failed_gate)
    assert decision["gate_states"][failed_gate] == "engineering_failed"
    assert called == list(GATE_ORDER[1 : failed_index + 1])
    for gate in GATE_ORDER[failed_index + 1 :]:
        assert decision["gate_states"][gate] == "blocked"
    assert decision["screen_eligible"] is False


def test_callback_exception_is_recorded_and_blocks(tmp_path: Path) -> None:
    context = _context(tmp_path)
    called: list[str] = []
    reports = _runners(called)

    def explode(_: PreflightContext) -> dict:
        called.append("F2")
        raise RuntimeError("shape contract failed")

    reports["F2"] = explode
    decision = run_preflight(context, gate_runners=reports)

    assert decision["gate_states"] == {
        "F0": "passed",
        "F1": "passed",
        "F2": "engineering_failed",
        "F3": "blocked",
        "F4": "blocked",
    }
    report = json.loads((context.report_root / "F2.json").read_text("utf-8"))
    assert report["payload"]["error_type"] == "RuntimeError"
    assert "shape contract failed" in report["payload"]["reason"]


def test_explicit_callback_failure_reason_is_preserved(tmp_path: Path) -> None:
    context = _context(tmp_path)
    reports = _runners([])
    reports["F1"] = lambda _: {
        "status": "engineering_failed",
        "reason": "neutral identity mismatch",
        "diagnostic": {"max_error": 0.25},
    }

    run_preflight(context, gate_runners=reports)

    record = json.loads((context.report_root / "F1.json").read_text("utf-8"))
    assert record["payload"]["reason"] == "neutral identity mismatch"
    assert record["payload"]["diagnostic"] == {"max_error": 0.25}


def test_decision_is_eligible_only_when_every_gate_passed() -> None:
    states = {gate: "passed" for gate in GATE_ORDER}
    assert decide_preflight(states) == {"status": "passed", "screen_eligible": True}
    for failed in GATE_ORDER:
        candidate = dict(states)
        candidate[failed] = "engineering_failed"
        assert decide_preflight(candidate) == {
            "status": "engineering_failed",
            "screen_eligible": False,
        }
    incomplete = {gate: "passed" for gate in GATE_ORDER[:-1]}
    assert decide_preflight(incomplete)["screen_eligible"] is False


def test_f0_binds_third_party_authority_and_math_golden(tmp_path: Path) -> None:
    evidence = run_f0(_context(tmp_path))

    assert evidence["status"] == "passed"
    assert evidence["authority"]["commit"] == DFINE_COMMIT
    assert evidence["authority"]["path"].endswith(
        "third_party/dfine_7fe2f888/AUTHORITY.json"
    )
    assert len(evidence["authority"]["sha256"]) == 64
    assert evidence["authority"]["vendored_reference_sha256"] == (
        json.loads(
            (ROOT / "third_party/dfine_7fe2f888/AUTHORITY.json").read_text("utf-8")
        )["vendored_reference_sha256"]
    )
    assert evidence["math_golden"] == {
        "weighting_float32": True,
        "weighting_float64": True,
        "integral": True,
        "distance2bbox": True,
        "bbox2distance": True,
        "fgl": True,
    }


def test_reports_are_canonical_hashed_and_create_only(tmp_path: Path) -> None:
    payload = {"status": "passed", "z": 2, "a": [1]}
    record = make_evidence_record("F1", payload)
    path = tmp_path / "F1.json"

    write_create_only_report(path, record)

    assert path.read_bytes() == canonical_json_bytes(record) + b"\n"
    assert record["payload_sha256"] == canonical_sha256(payload)
    assert record["payload_sha256"] == hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest().upper()
    with pytest.raises(FileExistsError):
        write_create_only_report(path, record)


def test_report_writer_rejects_symlink_destination(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("untouched", encoding="utf-8")
    link = tmp_path / "F1.json"
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="symlink|reparse"):
        write_create_only_report(link, make_evidence_record("F1", _f1()))
    assert target.read_text("utf-8") == "untouched"


def test_preflight_rejects_existing_or_symlink_report_root(tmp_path: Path) -> None:
    existing = _context(tmp_path / "existing")
    existing.report_root.mkdir()
    with pytest.raises(FileExistsError):
        run_preflight(existing, gate_runners=_runners([]))

    symlink_context = _context(tmp_path / "symlink")
    target = tmp_path / "real-report"
    target.mkdir()
    try:
        symlink_context.report_root.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    with pytest.raises(ValueError, match="symlink|reparse"):
        run_preflight(symlink_context, gate_runners=_runners([]))


def test_4090_schema_is_strict_and_source_model_authority_is_immutable() -> None:
    assert validate_4090_evidence(_f3())["status"] == "passed"

    wrong_gpu = _f3()
    wrong_gpu["hardware"]["gpu_name"] = "NVIDIA A100-SXM4-80GB"
    with pytest.raises(ValueError, match="RTX 4090"):
        validate_4090_evidence(wrong_gpu)

    wrong_runtime = _f3()
    wrong_runtime["runtime"]["batch"] = 4
    with pytest.raises(ValueError, match="batch"):
        validate_4090_evidence(wrong_runtime)

    changed_source = _f3()
    changed_source["immutability"]["source_after_sha256"] = "E" * 64
    with pytest.raises(ValueError, match="source"):
        validate_4090_evidence(changed_source)

    changed_public_model = _f3()
    changed_public_model["immutability"]["fdr_public_state_sha256"] = "E" * 64
    with pytest.raises(ValueError, match="public"):
        validate_4090_evidence(changed_public_model)


def test_representation_statistics_cover_edges_sizes_and_errors() -> None:
    report = _representation()

    assert report["count"] == 3
    assert report["reconstruction"]["l1"] == pytest.approx(0.01 / 12.0)
    assert report["reconstruction"]["max"] == pytest.approx(0.01)
    assert report["saturation"]["per_edge"] == {
        "left": {"count": 2, "rate": pytest.approx(2 / 3)},
        "top": {"count": 1, "rate": pytest.approx(1 / 3)},
        "right": {"count": 0, "rate": 0.0},
        "bottom": {"count": 1, "rate": pytest.approx(1 / 3)},
    }
    assert report["saturation"]["total"] == {
        "count": 4,
        "rate": pytest.approx(4 / 12),
    }
    assert report["invalid_boxes"] == 0
    assert report["nonfinite_rows"] == 0
    assert report["nonfinite_values"] == 0
    assert report["stratification"]["tiny"]["count"] == 1
    assert report["stratification"]["small"]["count"] == 1
    assert report["stratification"]["other"]["count"] == 1
    assert report["object_size"]["width_mean"] == pytest.approx(80 / 3)
    assert report["object_size"]["height_mean"] == pytest.approx(80 / 3)


def test_representation_nonfinite_and_invalid_are_counted_and_fail_f4() -> None:
    report = summarize_representation(
        reference_boxes=[[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
        reconstructed_boxes=[[0.5, 0.5, -0.1, 0.2], [float("nan"), 0.5, 0.2, 0.2]],
        target_indices=[[1, 1, 1, 1], [1, 1, 1, 1]],
        object_widths=[20.0, 20.0],
        object_heights=[20.0, 20.0],
    )
    assert report["invalid_boxes"] == 1
    assert report["nonfinite_rows"] == 1
    assert report["nonfinite_values"] == 1

    evidence = {
        "status": "passed",
        "official_reference_match": True,
        "reconstruction_tolerance": 1.0,
        "representation": report,
    }
    with pytest.raises(ValueError, match="non-finite"):
        validate_f4_evidence(evidence)


def test_failed_f4_report_preserves_representation_diagnostics(tmp_path: Path) -> None:
    context = _context(tmp_path)
    runners = _runners([])
    representation = summarize_representation(
        reference_boxes=[[0.5, 0.5, 0.2, 0.2]],
        reconstructed_boxes=[[float("nan"), 0.5, 0.2, 0.2]],
        target_indices=[[0, 1, 31, 1]],
        object_widths=[float("nan")],
        object_heights=[20.0],
    )
    runners["F4"] = lambda _: {
        "status": "passed",
        "official_reference_match": True,
        "reconstruction_tolerance": 1.0,
        "representation": representation,
    }

    decision = run_preflight(context, gate_runners=runners)

    assert decision["gate_states"]["F4"] == "engineering_failed"
    record = json.loads((context.report_root / "F4.json").read_text("utf-8"))
    assert record["payload"]["representation"]["nonfinite_rows"] == 1
    assert record["payload"]["representation"]["nonfinite_values"] == 2
    assert record["payload"]["representation"]["saturation"]["total"]["count"] == 2


def test_cli_exposes_only_authority_paths_and_freezes_runtime() -> None:
    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }
    assert option_strings == {
        "--protocol-manifest",
        "--baseline-checkpoint",
        "--dataset-root",
        "--report-root",
    }
    for forbidden in (
        "--reg-max",
        "--reg-scale",
        "--fgl-weight",
        "--threshold",
        "--seed",
        "--device",
        "--batch",
        "--imgsz",
        "--subset",
    ):
        assert forbidden not in option_strings
    assert FIXED_RUNTIME == {"device": "cuda:0", "batch": 8, "imgsz": 640}
    assert SCREEN_AUTHORITY == {"schedule_epochs": 50, "cutoff_epoch": 30}
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--protocol-manifest",
                "protocol.json",
                "--baseline-checkpoint",
                "baseline.pt",
                "--dataset-root",
                "VisDrone",
                "--report-root",
                "report",
                "--batch",
                "4",
            ]
        )


def test_preflight_context_rejects_missing_authority_inputs(tmp_path: Path) -> None:
    context = PreflightContext(
        protocol_manifest=tmp_path / "missing.json",
        baseline_checkpoint=tmp_path / "missing.pt",
        dataset_root=tmp_path / "missing-dataset",
        report_root=tmp_path / "report",
        repository_root=ROOT,
    )
    with pytest.raises(FileNotFoundError, match="protocol"):
        run_preflight(context, gate_runners=_runners([]))


def test_parser_type_is_standard_argparse() -> None:
    assert isinstance(build_parser(), argparse.ArgumentParser)


def test_direct_script_help_works_from_repository_root() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_fdr_preflight.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--protocol-manifest" in result.stdout
    assert "--baseline-checkpoint" in result.stdout
