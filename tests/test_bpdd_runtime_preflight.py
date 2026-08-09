from __future__ import annotations

import torch
from pathlib import Path

import src.bpdd_runtime_preflight as runtime
from src.bpdd_runtime_preflight import summarize_assignment_continuity
from scripts.run_bpdd_preflight import PreflightContext, run_preflight


def _layer(*pairs: tuple[int, int, int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    by_image: list[tuple[torch.Tensor, torch.Tensor]] = []
    for image in range(2):
        selected = [(query, target) for batch, query, target in pairs if batch == image]
        by_image.append(
            (
                torch.tensor([query for query, _target in selected], dtype=torch.long),
                torch.tensor([target for _query, target in selected], dtype=torch.long),
            )
        )
    return by_image


def test_assignment_continuity_is_measured_against_final_stock_targets() -> None:
    assignments = [
        _layer((0, 1, 4), (1, 2, 7)),
        _layer((0, 1, 4), (1, 2, 8)),
        _layer((0, 1, 4), (1, 2, 7)),
    ]

    report = summarize_assignment_continuity(assignments)

    assert report["final_matched_queries"] == 2
    assert report["layers"][0]["query_support_rate"] == 1.0
    assert report["layers"][0]["same_target_rate"] == 1.0
    assert report["layers"][1]["query_support_rate"] == 1.0
    assert report["layers"][1]["same_target_rate"] == 0.5
    assert report["overall_same_target_rate"] == 0.75


def test_assignment_continuity_handles_empty_final_assignment() -> None:
    empty = _layer()
    report = summarize_assignment_continuity([empty, empty])

    assert report["final_matched_queries"] == 0
    assert report["overall_query_support_rate"] == 0.0
    assert report["overall_same_target_rate"] == 0.0


def test_default_runtime_exposes_all_five_bpdd_gates() -> None:
    for name in ("run_b0", "run_b1", "run_b2", "run_b3", "run_b4"):
        assert callable(getattr(runtime, name))


def test_ordered_preflight_fails_closed_and_blocks_later_gates(tmp_path: Path) -> None:
    manifest = tmp_path / "protocol.json"
    manifest.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "initial.pt"
    checkpoint.write_bytes(b"authority")
    dataset = tmp_path / "VisDrone"
    dataset.mkdir()
    context = PreflightContext(
        protocol_manifest=manifest,
        initial_state=checkpoint,
        dataset_root=dataset,
        report_root=tmp_path / "reports",
        repository_root=Path(__file__).resolve().parents[1],
    )
    called: list[str] = []

    def passed(name: str):
        def execute(_context: PreflightContext) -> dict:
            called.append(name)
            return {"status": "passed", "gate": name, "checks": {"ok": True}}
        return execute

    runners = {name: passed(name) for name in ("B0", "B1", "B2", "B3", "B4")}
    runners["B2"] = lambda _context: {
        "status": "engineering_failed",
        "gate": "B2",
        "reason": "checkpoint mismatch",
    }

    decision = run_preflight(context, gate_runners=runners)

    assert called == ["B0", "B1"]
    assert decision["gate_states"] == {
        "B0": "passed",
        "B1": "passed",
        "B2": "engineering_failed",
        "B3": "blocked",
        "B4": "blocked",
    }
    assert decision["screen_eligible"] is False
    assert (context.report_root / "decision.json").is_file()


def test_all_five_preflight_gates_are_required_for_screen(tmp_path: Path) -> None:
    manifest = tmp_path / "protocol.json"
    manifest.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "initial.pt"
    checkpoint.write_bytes(b"authority")
    dataset = tmp_path / "VisDrone"
    dataset.mkdir()
    context = PreflightContext(
        protocol_manifest=manifest,
        initial_state=checkpoint,
        dataset_root=dataset,
        report_root=tmp_path / "reports",
        repository_root=Path(__file__).resolve().parents[1],
    )
    runners = {
        name: (lambda gate: lambda _context: {"status": "passed", "gate": gate})(name)
        for name in ("B0", "B1", "B2", "B3", "B4")
    }

    decision = run_preflight(context, gate_runners=runners)

    assert decision["status"] == "passed"
    assert decision["screen_eligible"] is True
