from scripts.calibrate_sr_peg_g0 import (
    select_calibration,
    threshold_grid,
)


def _candidate(
    thresholds,
    *,
    medium: float,
    large: float,
    map: float,
    tiny: float,
    recall: float = 0.0,
):
    return {
        "thresholds": {
            "tiny_utility": thresholds[0],
            "non_tiny_risk": thresholds[1],
            "global_retain": thresholds[2],
        },
        "deltas": {
            "mAP50-95": map,
            "AP-tiny-SBR": tiny,
            "tiny_recall": recall,
            "AP-medium-SBR": medium,
            "AP-large-SBR": large,
        },
    }


def test_calibration_prefers_budget_safe_map_then_tiny_then_thresholds():
    rows = [
        _candidate(
            (0.5, 0.5, 0.5),
            medium=-0.001,
            large=-0.004,
            map=0.010,
            tiny=0.02,
        ),
        _candidate(
            (0.4, 0.4, 0.4),
            medium=-0.003,
            large=-0.001,
            map=0.020,
            tiny=0.03,
        ),
    ]

    selected = select_calibration(rows)

    assert selected["thresholds"] == {
        "tiny_utility": 0.5,
        "non_tiny_risk": 0.5,
        "global_retain": 0.5,
    }


def test_threshold_grid_contains_only_effective_global_retain_settings():
    values = threshold_grid()

    assert len(values) == 3
    assert len(set(values)) == 3
    assert all(row[:2] == (0.5, 0.5) for row in values)
    assert set(value for row in values for value in row) == {0.4, 0.5, 0.6}


def test_calibration_tie_break_is_deterministic_and_risk_conservative():
    rows = [
        _candidate(
            (0.4, 0.4, 0.4),
            medium=0,
            large=0,
            map=0.01,
            tiny=0.02,
        ),
        _candidate(
            (0.4, 0.6, 0.4),
            medium=0,
            large=0,
            map=0.01,
            tiny=0.02,
        ),
    ]

    assert select_calibration(rows)["thresholds"]["non_tiny_risk"] == 0.6


def test_calibration_rejects_when_no_setting_meets_scale_budgets():
    rows = [
        _candidate(
            (0.5, 0.5, 0.5),
            medium=-0.003,
            large=-0.006,
            map=0.1,
            tiny=0.1,
        )
    ]

    try:
        select_calibration(rows)
    except ValueError as error:
        assert "budget" in str(error)
    else:
        raise AssertionError("unsafe calibration must fail closed")
