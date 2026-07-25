from __future__ import annotations

import pytest

from src.saded_single_model_adjudicator import (
    FORMAL_THRESHOLDS,
    adjudicate_single_model,
)


PRIMARY_KEYS = {
    "AP-tiny-SBR",
    "mAP50-95",
    "tiny_recall",
    "AP75",
    "AP-large-SBR",
}


def _metrics(**values: float) -> dict[str, float]:
    defaults = {
        "AP-tiny-SBR": 0.10,
        "mAP50-95": 0.20,
        "tiny_recall": 0.50,
        "AP75": 0.18,
        "AP-large-SBR": 0.14,
    }
    defaults.update(values)
    return defaults


def test_exact_sealed_r0_values_pass_all_five_formal_gates():
    arm_a = _metrics(
        **{
            "AP-tiny-SBR": 0.07105714429171933,
            "mAP50-95": 0.18062139655466955,
            "tiny_recall": 0.5537479710786484,
            "AP75": 0.16666558492813208,
            "AP-large-SBR": 0.14584679380950474,
        }
    )
    route = _metrics(
        **{
            "AP-tiny-SBR": 0.11025116665398038,
            "mAP50-95": 0.20647030730840651,
            "tiny_recall": 0.6555260439722591,
            "AP75": 0.18690233019506805,
            "AP-large-SBR": 0.1439375720841185,
        }
    )

    result = adjudicate_single_model(
        arm_a=arm_a,
        route_control=route,
        invariants_passed=True,
    )

    assert result["decision"] == "SADED_SINGLE_SEED_GO"
    assert result["failures"] == []
    assert set(result["deltas"]) == PRIMARY_KEYS
    assert all(result["gates"].values())


def test_all_thresholds_are_inclusive():
    arm_a = {key: 0.0 for key in FORMAL_THRESHOLDS}
    route = dict(FORMAL_THRESHOLDS)

    result = adjudicate_single_model(
        arm_a=arm_a,
        route_control=route,
        invariants_passed=True,
    )

    assert result["decision"] == "SADED_SINGLE_SEED_GO"


def test_one_below_threshold_is_scientific_stop():
    arm_a = {key: 0.0 for key in FORMAL_THRESHOLDS}
    route = dict(FORMAL_THRESHOLDS)
    route["mAP50-95"] -= 1e-9

    result = adjudicate_single_model(
        arm_a=arm_a,
        route_control=route,
        invariants_passed=True,
    )

    assert result["decision"] == "SADED_SINGLE_SEED_STOP"
    assert result["failures"] == ["mAP50-95_delta<0.003"]


def test_failed_invariants_are_invalid_without_metric_use():
    result = adjudicate_single_model(
        arm_a={},
        route_control={},
        invariants_passed=False,
    )

    assert result == {
        "schema_version": "saded-single-model-formal-adjudication/v1",
        "decision": "INVALID",
        "failures": ["evidence_invariants_failed"],
    }


@pytest.mark.parametrize(
    ("arm_a", "route_control"),
    [
        (
            {
                key: value
                for key, value in _metrics().items()
                if key != "AP75"
            },
            _metrics(),
        ),
        (
            _metrics(),
            {
                key: value
                for key, value in _metrics().items()
                if key != "AP75"
            },
        ),
        (_metrics(**{"AP75": float("nan")}), _metrics()),
        (_metrics(), _metrics(**{"AP75": True})),
    ],
)
def test_metric_schema_or_nonfinite_value_is_invalid(
    arm_a,
    route_control,
):
    result = adjudicate_single_model(
        arm_a=arm_a,
        route_control=route_control,
        invariants_passed=True,
    )

    assert result["decision"] == "INVALID"
    assert result["failures"] == ["metric_schema_or_value_invalid"]
