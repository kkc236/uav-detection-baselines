from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.adjudicate_ascv_loc import load_records


def _metrics(path: Path, value: float = 0.1) -> None:
    view = {
        "mAP50-95": value,
        "AP-tiny-SBR": value,
        "tiny_recall": value,
        "AP75": value,
        "AP-large-SBR": value,
    }
    path.write_text(json.dumps({"A": view, "C": view}))


def test_load_records_requires_exact_seed_arm_files_and_preserves_ac(tmp_path: Path) -> None:
    paths = {}
    for seed in (0, 1, 2):
        for arm in ("control", "ascv"):
            path = tmp_path / f"s{seed}-{arm}.json"
            _metrics(path, 0.1 + seed / 100)
            paths[(seed, arm)] = path

    records, inputs = load_records(paths)

    assert set(records) == {"0", "1", "2"}
    assert set(records["0"]) == {"control", "ascv"}
    assert set(records["0"]["control"]) == {"A", "C"}
    assert len(inputs) == 6
    assert all(record["sha256"] for record in inputs)


def test_load_records_rejects_test_dev_and_non_ac_payloads(tmp_path: Path) -> None:
    forbidden = tmp_path / "test-dev" / "metrics.json"
    forbidden.parent.mkdir()
    _metrics(forbidden)
    with pytest.raises(ValueError, match="test-dev"):
        load_records({(0, "control"): forbidden})

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"A": {}, "B": {}}))
    with pytest.raises(ValueError, match="exact A/C"):
        load_records({(0, "control"): invalid})
