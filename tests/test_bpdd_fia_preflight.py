from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest


def _context(tmp_path: Path):
    from scripts.run_bpdd_fia_preflight import PreflightContext

    manifest = tmp_path / "protocol.json"
    manifest.write_text("{}", encoding="utf-8")
    initial_state = tmp_path / "initial.pt"
    initial_state.write_bytes(b"fdr-authority")
    dataset_root = tmp_path / "VisDrone"
    dataset_root.mkdir()
    return PreflightContext(
        protocol_manifest=manifest,
        initial_state=initial_state,
        dataset_root=dataset_root,
        report_root=tmp_path / "reports",
        repository_root=Path(__file__).resolve().parents[1],
    )


def _passing_runners(called: list[str] | None = None):
    def runner(gate: str):
        def execute(_context):
            if called is not None:
                called.append(gate)
            return {
                "status": "passed",
                "gate": gate,
                "checks": {"ok": True},
            }

        return execute

    return {f"I{index}": runner(f"I{index}") for index in range(5)}


def test_preflight_cli_bootstraps_repository_before_production_imports() -> None:
    import scripts.run_bpdd_fia_preflight as cli

    source = inspect.getsource(cli)
    bootstrap = source.index("sys.path.insert(0, str(ROOT))")
    production_import = source.index("def run_i0")
    assert bootstrap < production_import


def test_default_runtime_exposes_all_five_combined_gates() -> None:
    import scripts.run_bpdd_fia_preflight as cli

    assert cli.GATE_ORDER == ("I0", "I1", "I2", "I3", "I4")
    for name in ("run_i0", "run_i1", "run_i2", "run_i3", "run_i4"):
        assert callable(getattr(cli, name))


def test_preflight_requires_all_gates_and_writes_create_only_evidence(
    tmp_path: Path,
) -> None:
    from scripts.run_bpdd_fia_preflight import run_preflight

    context = _context(tmp_path)
    decision = run_preflight(context, gate_runners=_passing_runners())

    assert decision["status"] == "passed"
    assert decision["formal_eligible"] is True
    assert decision["gate_states"] == {f"I{index}": "passed" for index in range(5)}
    assert decision["fixed_runtime"] == {
        "device": "cuda:0",
        "batch": 8,
        "imgsz": 640,
        "amp_scale": 128.0,
        "queries": 300,
    }
    for name in (*decision["gate_states"], "decision"):
        path = context.report_root / f"{name}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["gate"] == name
        assert len(record["payload_sha256"]) == 64
    with pytest.raises(FileExistsError, match="already exists"):
        run_preflight(context, gate_runners=_passing_runners())


def test_preflight_stops_after_shared_state_failure(tmp_path: Path) -> None:
    from scripts.run_bpdd_fia_preflight import run_preflight

    context = _context(tmp_path)
    called: list[str] = []
    runners = _passing_runners(called)
    runners["I1"] = lambda _context: {
        "status": "engineering_failed",
        "gate": "I1",
        "reason": "shared state mismatch",
    }

    decision = run_preflight(context, gate_runners=runners)

    assert called == ["I0"]
    assert decision["gate_states"] == {
        "I0": "passed",
        "I1": "engineering_failed",
        "I2": "blocked",
        "I3": "blocked",
        "I4": "blocked",
    }
    assert decision["status"] == "engineering_failed"
    assert decision["formal_eligible"] is False


def test_preflight_converts_runner_exception_to_fail_closed_evidence(
    tmp_path: Path,
) -> None:
    from scripts.run_bpdd_fia_preflight import run_preflight

    context = _context(tmp_path)
    runners = _passing_runners()

    def explode(_context):
        raise RuntimeError("CUDA loss became non-finite")

    runners["I2"] = explode
    decision = run_preflight(context, gate_runners=runners)
    record = json.loads((context.report_root / "I2.json").read_text("utf-8"))

    assert decision["formal_eligible"] is False
    assert record["payload"]["status"] == "engineering_failed"
    assert record["payload"]["error_type"] == "RuntimeError"
    assert "non-finite" in record["payload"]["reason"]


def test_preflight_rejects_unknown_gate_injection(tmp_path: Path) -> None:
    from scripts.run_bpdd_fia_preflight import run_preflight

    with pytest.raises(ValueError, match="unknown BPDD FIA preflight gates"):
        run_preflight(
            _context(tmp_path),
            gate_runners={**_passing_runners(), "I5": lambda _context: {}},
        )


def test_invalid_pass_payload_is_failed_closed(tmp_path: Path) -> None:
    from scripts.run_bpdd_fia_preflight import run_preflight

    context = _context(tmp_path)
    runners = _passing_runners()
    runners["I0"] = lambda _context: {
        "status": "passed",
        "gate": "wrong-gate",
        "checks": {"ok": True},
    }

    decision = run_preflight(context, gate_runners=runners)

    assert decision["gate_states"]["I0"] == "engineering_failed"
    assert decision["formal_eligible"] is False


def test_gradient_summary_distinguishes_zero_and_live_groups() -> None:
    import torch

    from scripts.run_bpdd_fia_preflight import summarize_gradient_group

    live = torch.nn.Parameter(torch.tensor([1.0]))
    zero = torch.nn.Parameter(torch.tensor([2.0]))
    absent = torch.nn.Parameter(torch.tensor([3.0]))
    live.grad = torch.tensor([0.5])
    zero.grad = torch.tensor([0.0])

    report = summarize_gradient_group([live, zero, absent])

    assert report == {
        "parameter_tensors": 3,
        "gradient_tensors": 2,
        "finite": True,
        "nonzero_tensors": 1,
        "l2_norm": 0.5,
    }


def test_prediction_contract_requires_finite_batch_by_300_queries() -> None:
    import torch

    from scripts.run_bpdd_fia_preflight import validate_prediction_contract

    report = validate_prediction_contract(torch.zeros(8, 300, 14), batch_size=8)
    assert report == {
        "batch": 8,
        "queries": 300,
        "finite": True,
    }

    with pytest.raises(RuntimeError, match="300-query"):
        validate_prediction_contract(torch.zeros(8, 299, 14), batch_size=8)
    invalid = torch.zeros(8, 300, 14)
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        validate_prediction_contract(invalid, batch_size=8)


def test_environment_authority_rejects_missing_or_changed_fields() -> None:
    from scripts.run_bpdd_fia_preflight import validate_environment_authority

    expected = {
        "model": "Ultralytics RT-DETR-L",
        "torch": "2.5.1+cu121",
        "gpu": "NVIDIA GeForce RTX 4090",
    }
    assert validate_environment_authority(dict(expected), expected) == dict(expected)
    with pytest.raises(ValueError, match="missing model"):
        validate_environment_authority(
            {"torch": expected["torch"], "gpu": expected["gpu"]},
            expected,
        )
    with pytest.raises(ValueError, match="mismatch for torch"):
        validate_environment_authority(
            {**expected, "torch": "2.6.0+cu124"},
            expected,
        )
