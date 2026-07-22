from pathlib import Path

import pytest

from src.ebc_qp_config import EBCQPConfig, assert_ultralytics_source_lock


def test_v1_defaults_are_the_frozen_values():
    cfg = EBCQPConfig()

    assert cfg.query_budget == 300
    assert cfg.p2_candidates == 50
    assert cfg.warmup_epochs == 3
    assert cfg.tiny_radius == 16.0
    assert cfg.p2_anchor_size == 0.025
    assert cfg.lambda_p2 == 0.25
    assert cfg.lambda_ebc == 0.05
    assert cfg.local_radius == 1


def test_source_lock_rejects_a_changed_file(tmp_path: Path):
    changed = tmp_path / "head.py"
    changed.write_text("changed", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source lock mismatch"):
        assert_ultralytics_source_lock({"head.py": changed})
