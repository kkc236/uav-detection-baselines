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
    assert cfg.quality_weighted_ebc is False
    assert cfg.learnable_fusion_gamma is False
    assert cfg.p2_c2_grad_scale == 0.0
    assert cfg.contribution_separated_aux_gradients is False
    assert cfg.local_radius == 1


def test_tsg_routing_scale_is_bounded():
    assert EBCQPConfig(p2_c2_grad_scale=0.1).p2_c2_grad_scale == 0.1
    with pytest.raises(ValueError, match="p2_c2_grad_scale"):
        EBCQPConfig(p2_c2_grad_scale=-0.1)
    with pytest.raises(ValueError, match="p2_c2_grad_scale"):
        EBCQPConfig(p2_c2_grad_scale=1.1)


def test_contribution_separated_mode_rejects_stock_coupling_features():
    cfg = EBCQPConfig(
        lambda_p2=0.1,
        lambda_quality=0.0,
        lambda_ebc=0.0,
        query_injection_enabled=False,
        p2_c2_grad_scale=0.1,
        contribution_separated_aux_gradients=True,
    )
    assert cfg.contribution_separated_aux_gradients is True
    with pytest.raises(ValueError, match="query injection"):
        EBCQPConfig(contribution_separated_aux_gradients=True)


def test_source_lock_rejects_a_changed_file(tmp_path: Path):
    changed = tmp_path / "head.py"
    changed.write_text("changed", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source lock mismatch"):
        assert_ultralytics_source_lock({"head.py": changed})
