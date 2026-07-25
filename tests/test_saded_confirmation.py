from __future__ import annotations

import pytest

from scripts.evaluate_saded_confirmation_once import _create_claim
from scripts.seal_saded_confirmation_predictions import (
    build_parser as build_sealer_parser,
)
from src.saded_confirmation import adjudicate_confirmation_metrics


def _metrics(
    *,
    map_value: float,
    tiny: float,
    recall: float,
    ap75: float,
    large: float,
):
    return {
        "mAP50-95": map_value,
        "AP-tiny-SBR": tiny,
        "tiny_recall": recall,
        "AP75": ap75,
        "AP-large-SBR": large,
    }


def test_confirmation_reapplies_primary_and_attribution_rules():
    by_seed = {}
    for seed in range(3):
        arm_a = _metrics(
            map_value=0.10,
            tiny=0.10,
            recall=0.20,
            ap75=0.10,
            large=0.20,
        )
        route_control = _metrics(
            map_value=0.101,
            tiny=0.108,
            recall=0.215,
            ap75=0.099,
            large=0.198,
        )
        route_treatment = _metrics(
            map_value=0.104,
            tiny=0.112,
            recall=0.225,
            ap75=0.099,
            large=0.198,
        )
        by_seed[str(seed)] = {
            "A": arm_a,
            "route_control": route_control,
            "route_treatment": route_treatment,
        }

    result = adjudicate_confirmation_metrics(by_seed)

    assert result["decision"] == "TASCV_CONFIRMATION_GO"
    assert result["primary"]["mean"]["mAP50-95"] == pytest.approx(
        0.004
    )
    assert result["attribution"]["counts"]["mAP_positive"] == 3


def test_confirmation_stops_when_primary_mean_misses_a_gate():
    arm = _metrics(
        map_value=0.10,
        tiny=0.10,
        recall=0.20,
        ap75=0.10,
        large=0.20,
    )
    by_seed = {
        str(seed): {
            "A": arm,
            "route_control": arm,
            "route_treatment": {
                **arm,
                "mAP50-95": 0.102,
            },
        }
        for seed in range(3)
    }

    result = adjudicate_confirmation_metrics(by_seed)

    assert result["decision"] == "TASCV_STOP"
    assert "primary:mAP50-95<0.003" in result["failures"]


def test_confirmation_sealer_has_no_free_split_or_threshold_options():
    options = {
        option
        for action in build_sealer_parser()._actions
        for option in action.option_strings
    }
    assert "--split" not in options
    assert "--data" not in options
    assert "--checkpoint" not in options
    assert "--seed" not in options
    assert "--arm" not in options
    assert "--threshold" not in options


def test_confirmation_gt_claim_is_exclusive_and_not_retryable(tmp_path):
    claim = tmp_path / "claim.json"
    _create_claim(claim, {"state": "CONSUMED"})

    with pytest.raises(FileExistsError):
        _create_claim(claim, {"state": "CONSUMED"})

    assert claim.exists()
