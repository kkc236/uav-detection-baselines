from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path("scripts/run_rtdetr_oar.py")


def _load_module():
    assert SCRIPT.is_file(), "OAR runner has not been implemented"
    spec = importlib.util.spec_from_file_location("run_rtdetr_oar_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _observed_d0() -> tuple[dict[str, dict[str, float]], dict[int, float]]:
    return (
        {
            "stock": {
                "map": 0.28628865801344866,
                "ap75": 0.292364074762,
                "ap50": 0.476388925325,
            },
            "presence": {"map": 0.294608200682, "ap75": 0.300115294639},
            "query_iou": {"map": 0.324185693386, "ap75": 0.344708985960},
            "same_class": {"map": 0.409733588907, "ap75": 0.413238330995},
        },
        {
            20: 0.304549967436,
            40: 0.335016751522,
            60: 0.358000690130,
            100: 0.385568106152,
        },
    )


def test_sparse_d0_fails_without_extending_the_grid() -> None:
    module = _load_module()
    decomposition, restricted = _observed_d0()

    reports = module._sparse_d0_reports(
        decomposition=decomposition,
        restricted_map=restricted,
    )

    decision = reports["sparse-d0-decision.json"]
    assert decision["status"] == "scientific_failed"
    assert decision["selected_k"] is None
    assert decision["frozen_k_grid"] == [20, 40, 60, 100]
    assert decision["next_authority"] == "oar-all-pair-amendment"
    assert reports["d0-k-coverage.json"]["coverage"]["100"]["recovered"] < "0.90"


def test_sparse_d0_reports_are_canonical_create_only_and_drift_safe(
    tmp_path: Path,
) -> None:
    module = _load_module()
    decomposition, restricted = _observed_d0()
    reports = module._sparse_d0_reports(
        decomposition=decomposition,
        restricted_map=restricted,
    )

    first = module._write_sparse_d0_reports(tmp_path, reports)
    second = module._write_sparse_d0_reports(tmp_path, reports)

    assert first == second
    assert set(first) == {
        "d0-oracle-decomposition.json",
        "d0-k-coverage.json",
        "sparse-d0-decision.json",
    }
    for name, digest in first.items():
        path = tmp_path / name
        assert len(digest) == 64
        raw = path.read_bytes()
        assert raw.endswith(b"\n")
        assert json.loads(raw) == reports[name]

    changed = json.loads(json.dumps(reports))
    changed["sparse-d0-decision.json"]["status"] = "passed"
    with pytest.raises(RuntimeError, match="immutable|drift"):
        module._write_sparse_d0_reports(tmp_path, changed)
