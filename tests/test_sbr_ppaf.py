from __future__ import annotations

from dataclasses import replace
import inspect
import math

import pytest

from src.sbr_v2_audit import AuditRawDetection, reconstruct_c_clusters


def raw(
    arm: str,
    *,
    source: int,
    query: int,
    score: float,
    box: tuple[float, float, float, float],
    index: int,
    cls: int = 0,
) -> AuditRawDetection:
    return AuditRawDetection.synthetic(
        "i.jpg",
        arm,
        source=source,
        query=query,
        score=score,
        box=box,
        width=640,
        height=640,
        original_index=index,
        cls=cls,
    )


def test_tail_score_map_is_strictly_inside_frozen_band_and_monotone():
    from src.sbr_ppaf import (
        A_FLOOR,
        C_CEILING,
        CONF_THRESHOLD,
        map_tail_score,
    )

    values = [CONF_THRESHOLD, 0.1, 0.5, 1.0]
    mapped = [map_tail_score(value) for value in values]

    assert all(CONF_THRESHOLD < value < C_CEILING < A_FLOOR for value in mapped)
    assert mapped == sorted(mapped)
    assert len(set(mapped)) == len(mapped)


@pytest.mark.parametrize("value", [True, -1.0, math.nan, math.inf, 1.1])
def test_tail_score_map_rejects_invalid_scores(value):
    from src.sbr_ppaf import map_tail_score

    with pytest.raises(ValueError):
        map_tail_score(value)


def test_router_public_api_contains_no_ground_truth_inputs():
    from src.sbr_ppaf import build_ppaf_arms

    names = set(inspect.signature(build_ppaf_arms).parameters)
    assert not names & {
        "gt",
        "gt_boxes",
        "gt_classes",
        "ignore_boxes",
        "matches",
    }


def test_primary_arms_protect_only_a_large_and_fill_from_tile_nonlarge():
    from src.sbr_ppaf import build_ppaf_arms

    a_large = raw(
        "A", source=0, query=1, score=0.8, box=(0, 0, 120, 120), index=1
    ).to_detection()
    a_small = raw(
        "A", source=0, query=2, score=0.7, box=(200, 0, 220, 20), index=2
    ).to_detection()
    c_full_only = raw(
        "C", source=0, query=9, score=0.6, box=(400, 0, 420, 20), index=11
    )
    c_tile = raw(
        "C", source=1, query=3, score=0.5, box=(300, 0, 320, 20), index=12
    )
    reconstruction = reconstruct_c_clusters((c_full_only, c_tile))

    result = build_ppaf_arms(
        image_id="i.jpg",
        width=640,
        height=640,
        a_final=(a_large, a_small),
        c_reconstruction=reconstruction,
        c_raw=(c_full_only, c_tile),
    )

    assert result.arms["P1"][0] == a_large
    assert a_small not in result.arms["P1"]
    assert result.arms["P3"][0] == a_large
    assert result.arms["All-A"][:2] == (a_large, a_small)
    assert len(result.arms["P1"]) == 2
    assert all(len(predictions) <= 300 for predictions in result.arms.values())


def test_tail_preserves_fused_c_box_while_mapping_only_score():
    from src.sbr_ppaf import build_ppaf_arms

    tile_seed = raw(
        "C", source=1, query=1, score=0.9, box=(0, 0, 40, 40), index=10
    )
    tile_member = raw(
        "C", source=2, query=2, score=0.8, box=(10, 0, 50, 40), index=11
    )
    reconstruction = reconstruct_c_clusters((tile_seed, tile_member))
    assert reconstruction.pre_cap_predictions[0].box != tile_seed.global_xyxy

    result = build_ppaf_arms(
        image_id="i.jpg",
        width=640,
        height=640,
        a_final=(),
        c_reconstruction=reconstruction,
        c_raw=(tile_seed, tile_member),
    )

    routed = result.arms["P3"][0]
    assert routed.box == reconstruction.pre_cap_predictions[0].box
    assert routed.global_xyxy == reconstruction.pre_cap_predictions[0].global_xyxy
    assert routed.score != reconstruction.pre_cap_predictions[0].score


def test_exact_effective_size_96_remains_tail_eligible():
    from src.sbr_ppaf import build_ppaf_arms

    boundary = raw(
        "C", source=1, query=1, score=0.8, box=(0, 0, 96, 96), index=10
    )
    reconstruction = reconstruct_c_clusters((boundary,))

    result = build_ppaf_arms(
        image_id="i.jpg",
        width=640,
        height=640,
        a_final=(),
        c_reconstruction=reconstruction,
        c_raw=(boundary,),
    )

    assert len(result.arms["P3"]) == 1


def test_p2_removes_cluster_with_exact_selected_full_provenance():
    from src.sbr_ppaf import build_ppaf_arms

    a_large = raw(
        "A", source=0, query=4, score=0.8, box=(0, 0, 120, 120), index=1
    ).to_detection()
    c_full = raw(
        "C", source=0, query=4, score=0.8, box=(0, 0, 120, 120), index=10
    )
    c_local = raw(
        "C", source=1, query=4, score=0.9, box=(10, 10, 80, 80), index=11
    )
    reconstruction = reconstruct_c_clusters((c_full, c_local))

    result = build_ppaf_arms(
        image_id="i.jpg",
        width=640,
        height=640,
        a_final=(a_large,),
        c_reconstruction=reconstruction,
        c_raw=(c_full, c_local),
    )

    assert len(result.arms["P1"]) == 2
    assert result.arms["P2"] == (a_large,)
    assert result.coverage["P2"]["provenance_rejected"] == 1


def test_p3_removes_only_same_class_tile_only_fragment_at_exact_half_ios():
    from src.sbr_ppaf import build_ppaf_arms

    a_large = raw(
        "A",
        source=0,
        query=1,
        score=0.8,
        box=(0, 0, 120, 120),
        index=1,
        cls=2,
    ).to_detection()
    fragment = raw(
        "C",
        source=1,
        query=2,
        score=0.7,
        box=(-30, 0, 30, 60),
        index=10,
        cls=2,
    )
    other_class = raw(
        "C",
        source=2,
        query=3,
        score=0.6,
        box=(-30, 0, 30, 60),
        index=11,
        cls=3,
    )
    reconstruction = reconstruct_c_clusters((fragment, other_class))

    result = build_ppaf_arms(
        image_id="i.jpg",
        width=640,
        height=640,
        a_final=(a_large,),
        c_reconstruction=reconstruction,
        c_raw=(fragment, other_class),
    )

    assert len(result.arms["P2"]) == 3
    assert len(result.arms["P3"]) == 2
    assert result.arms["P3"][1].class_id == 3
    assert result.coverage["P3"]["fragment_rejected"] == 1


def test_all_a_with_no_capacity_is_byte_for_byte_a():
    from src.sbr_ppaf import MAX_DET, build_ppaf_arms

    a = tuple(
        raw(
            "A",
            source=0,
            query=index,
            score=0.9 - index / 1000,
            box=(index * 2, 0, index * 2 + 1, 1),
            index=index,
        ).to_detection()
        for index in range(MAX_DET)
    )
    tail = raw(
        "C", source=1, query=999, score=0.99, box=(0, 10, 2, 12), index=999
    )

    result = build_ppaf_arms(
        image_id="i.jpg",
        width=640,
        height=640,
        a_final=a,
        c_reconstruction=reconstruct_c_clusters((tail,)),
        c_raw=(tail,),
    )

    assert result.arms["All-A"] == a


def test_router_rejects_reconstruction_that_does_not_match_raw():
    from src.sbr_ppaf import build_ppaf_arms

    one = raw(
        "C", source=1, query=1, score=0.8, box=(0, 0, 20, 20), index=10
    )
    reconstruction = reconstruct_c_clusters((one,))
    tampered = replace(
        reconstruction,
        cluster_members=((999,),),
    )

    with pytest.raises(ValueError, match="reconstruction|provenance|cluster"):
        build_ppaf_arms(
            image_id="i.jpg",
            width=640,
            height=640,
            a_final=(),
            c_reconstruction=tampered,
            c_raw=(one,),
        )


def test_real_score_domain_detects_mapping_collisions(monkeypatch):
    import src.sbr_ppaf as ppaf

    monkeypatch.setattr(ppaf, "map_tail_score", lambda score: 0.002)
    result = ppaf.verify_tail_score_domain((0.1, 0.2, 0.2))

    assert result["distinct_input_count"] == 2
    assert result["collision_free"] is False
    assert result["passed"] is False


def test_real_score_domain_rejects_a_true_nextafter_collision():
    from src.sbr_ppaf import verify_tail_score_domain

    from src.sbr_ppaf import CONF_THRESHOLD

    left = CONF_THRESHOLD
    right = math.nextafter(left, math.inf)
    assert left < right

    result = verify_tail_score_domain((left, right))

    assert result["collision_free"] is False
    assert result["passed"] is False


def test_a_floor_requires_exact_real_cache_minimum():
    from src.sbr_ppaf import A_FLOOR, verify_a_floor

    assert verify_a_floor((0.8, A_FLOOR))["passed"] is True
    result = verify_a_floor((0.8, math.nextafter(A_FLOOR, math.inf)))
    assert result["passed"] is False
    assert result["actual_a_min"] != A_FLOOR


def test_decision_prefers_p3_then_fallback_and_otherwise_stops():
    from src.sbr_ppaf import decide_ppaf

    a = {
        "AP-tiny-SBR": 0.0,
        "mAP50-95": 0.0,
        "tiny_recall": 0.0,
        "AP75": 0.0,
        "AP-large-SBR": 0.0,
    }
    passing = {
        "AP-tiny-SBR": 0.010,
        "mAP50-95": 0.003,
        "tiny_recall": 0.020,
        "AP75": -0.002,
        "AP-large-SBR": -0.005,
    }

    assert (
        decide_ppaf(a, passing, a, invariants_passed=True)["status"]
        == "SP_PPAF_PASS"
    )
    assert (
        decide_ppaf(a, a, passing, invariants_passed=True)["status"]
        == "SP_PPAF_FALLBACK_PASS"
    )
    assert (
        decide_ppaf(a, a, a, invariants_passed=True)["status"]
        == "SP_PPAF_STOP"
    )
    assert (
        decide_ppaf(a, passing, passing, invariants_passed=False)["status"]
        == "SP_PPAF_INVALID"
    )


def test_decision_selects_p3_when_both_p3_and_fallback_pass():
    from src.sbr_ppaf import decide_ppaf

    baseline = {
        "AP-tiny-SBR": 0.0,
        "mAP50-95": 0.0,
        "tiny_recall": 0.0,
        "AP75": 0.0,
        "AP-large-SBR": 0.0,
    }
    passing = {
        "AP-tiny-SBR": 0.010,
        "mAP50-95": 0.003,
        "tiny_recall": 0.020,
        "AP75": -0.002,
        "AP-large-SBR": -0.005,
    }

    result = decide_ppaf(
        baseline,
        passing,
        passing,
        invariants_passed=True,
    )

    assert result["status"] == "SP_PPAF_PASS"
    assert result["selected_arm"] == "P3"


def test_decision_invalid_short_circuits_nonfinite_metrics():
    from src.sbr_ppaf import decide_ppaf

    broken = {
        "AP-tiny-SBR": math.nan,
        "mAP50-95": math.nan,
        "tiny_recall": math.nan,
        "AP75": math.nan,
        "AP-large-SBR": math.nan,
    }

    result = decide_ppaf(
        broken,
        broken,
        broken,
        invariants_passed=False,
    )

    assert result["status"] == "SP_PPAF_INVALID"
    assert result["selected_arm"] == "none"


def test_decision_never_accepts_p1_or_p2_inputs():
    from src.sbr_ppaf import decide_ppaf

    parameters = set(inspect.signature(decide_ppaf).parameters)
    assert "p1_metrics" not in parameters
    assert "p2_metrics" not in parameters


@pytest.mark.parametrize(
    "metric,threshold",
    [
        ("AP-tiny-SBR", 0.010),
        ("mAP50-95", 0.003),
        ("tiny_recall", 0.020),
        ("AP75", -0.002),
        ("AP-large-SBR", -0.005),
    ],
)
def test_each_gate_is_inclusive_and_fails_one_ulp_below(metric, threshold):
    from src.sbr_ppaf import decide_ppaf

    baseline = {
        "AP-tiny-SBR": 0.0,
        "mAP50-95": 0.0,
        "tiny_recall": 0.0,
        "AP75": 0.0,
        "AP-large-SBR": 0.0,
    }
    exact = {
        "AP-tiny-SBR": 0.010,
        "mAP50-95": 0.003,
        "tiny_recall": 0.020,
        "AP75": -0.002,
        "AP-large-SBR": -0.005,
    }
    assert (
        decide_ppaf(baseline, exact, baseline, invariants_passed=True)[
            "status"
        ]
        == "SP_PPAF_PASS"
    )
    below = dict(exact)
    below[metric] = math.nextafter(threshold, -math.inf)
    assert (
        decide_ppaf(baseline, below, baseline, invariants_passed=True)[
            "status"
        ]
        == "SP_PPAF_STOP"
    )
