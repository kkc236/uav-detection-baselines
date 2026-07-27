from scripts.adjudicate_gcqf_g0 import two_seed_gate


def _seed(map_delta, tiny_delta, ap75_delta, medium_delta, large_delta):
    return {
        "deltas": {
            "full_minus_anchor": {
                "mAP50-95": map_delta,
                "AP-tiny-SBR": tiny_delta,
                "AP75": ap75_delta,
                "AP-medium-SBR": medium_delta,
                "AP-large-SBR": large_delta,
            },
            "full_minus_global": {
                "AP-large-SBR": -0.003,
            },
        },
        "anchor_reference": {"exact": True},
        "protected_global_exact": True,
        "per_seed_gate": {
            "residual_is_active": True,
            "residual_not_saturated": True,
        },
    }


def test_two_seed_gate_uses_preregistered_average_and_direction_rules():
    gate = two_seed_gate(
        [
            _seed(0.004, 0.006, 0.002, -0.001, -0.001),
            _seed(0.003, 0.005, 0.004, 0.000, -0.002),
        ]
    )

    assert gate["both_seed_map_positive"] is True
    assert gate["mean_map_at_least_0_003"] is True
    assert gate["mean_tiny_or_ap75_material"] is True
    assert gate["advance_accuracy"] is True


def test_two_seed_gate_fails_if_one_seed_is_negative():
    gate = two_seed_gate(
        [
            _seed(0.007, 0.010, 0.004, 0.000, -0.001),
            _seed(-0.001, 0.006, 0.004, 0.000, -0.001),
        ]
    )

    assert gate["both_seed_map_positive"] is False
    assert gate["advance_accuracy"] is False
