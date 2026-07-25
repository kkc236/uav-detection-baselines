from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import train_rtdetr_tascv as runner
from src.tascv_stage import TASCVStage


def test_training_runner_never_imports_stopped_runtime() -> None:
    source = inspect.getsource(runner)
    assert "rtdetr_ascv_loc" not in source
    assert "ASCVLocTrainer" not in source
    assert "ascv_loc_adjudicator" not in source


def test_mechanism_jsonl_is_ordered_and_atomic(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    runner._atomic_jsonl(
        path,
        [{"batch": 1}, {"batch": 2}],
    )
    assert [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ] == [{"batch": 1}, {"batch": 2}]
    assert not path.with_suffix(".jsonl.tmp").exists()


def test_uncaught_training_failure_seals_runtime_invalid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = SimpleNamespace(
        project=tmp_path,
        name="preflight-control",
        stage=TASCVStage.PREFLIGHT_1,
        arm="control",
        seed=0,
    )
    monkeypatch.setattr(
        runner,
        "build_parser",
        lambda: SimpleNamespace(parse_args=lambda: args),
    )
    monkeypatch.setattr(
        runner,
        "_run",
        lambda _args, _protocol: (_ for _ in ()).throw(
            RuntimeError("simulated OOM")
        ),
    )
    monkeypatch.setattr(
        runner,
        "validate_protocol_inputs",
        lambda _args: {},
    )

    with pytest.raises(RuntimeError, match="simulated OOM"):
        runner.main()

    invalid = json.loads(
        (
            tmp_path
            / "preflight-control/runtime_invalid.json"
        ).read_text(encoding="utf-8")
    )
    assert invalid["decision"] == "INVALID"
    assert invalid["error_type"] == "RuntimeError"
