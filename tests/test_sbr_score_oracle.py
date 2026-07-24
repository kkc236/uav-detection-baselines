import math
from dataclasses import replace

import pytest

from src.sbr_score_oracle import (
    GATES,
    OracleImage,
    apply_group_overlay,
    evaluate_oracle_image,
    find_aggressor_groups,
    gate_oracle_metrics,
    replay_overlay,
    verify_oracle_image_invariants,
)
from src.sbr_v2_audit import AuditRawDetection, reconstruct_c_clusters


def raw(
    source,
    score,
    box,
    *,
    query=0,
    original=0,
    cls=0,
    arm="C",
):
    return AuditRawDetection.synthetic(
        "images/i.jpg",
        arm,
        source=source,
        score=score,
        box=box,
        query=query,
        original_index=original,
        cls=cls,
        width=640,
        height=640,
    )


def test_group_requires_mixed_cluster_and_strict_tile_advantage():
    full = raw(0, 0.80, (0, 0, 100, 100), original=10)
    tied = raw(1, 0.80, (0, 0, 90, 90), original=11)
    low = raw(2, 0.79, (0, 0, 80, 80), original=12)

    assert find_aggressor_groups((full, tied, low)) == ()


def test_group_caps_every_tile_strictly_above_best_full():
    full_low = raw(
        0, 0.70, (0, 0, 100, 100), query=2, original=10
    )
    full = raw(0, 0.80, (0, 0, 100, 100), query=1, original=11)
    tile_a = raw(1, 0.95, (0, 0, 90, 90), original=20)
    tile_b = raw(2, 0.90, (0, 0, 80, 80), original=21)

    group = find_aggressor_groups(
        (full_low, full, tile_a, tile_b)
    )[0]

    assert group.full_anchor_index == 11
    assert group.aggressor_indices == (20, 21)
    assert group.anchor_score == 0.80
    assert group.unit_id.startswith("images/i.jpg:")


def test_group_is_gt_free_and_class_aware_by_stock_cluster():
    full = raw(0, 0.80, (0, 0, 100, 100), original=1, cls=0)
    other_class = raw(
        1, 0.99, (0, 0, 100, 100), original=2, cls=1
    )
    far_tile = raw(
        1, 0.99, (300, 300, 400, 400), original=3, cls=0
    )

    assert find_aggressor_groups((full, other_class, far_tile)) == ()


def test_stock_probe_is_strict_at_ios_half_and_nontransitive():
    records = (
        raw(1, 0.90, (0, 0, 100, 100), original=1),
        raw(1, 0.80, (40, 0, 140, 100), original=2),
        raw(1, 0.70, (80, 0, 180, 100), original=3),
    )

    reconstruction = reconstruct_c_clusters(records)

    assert reconstruction.cluster_members == ((1, 2), (3,))
    exact_half = (
        raw(1, 0.90, (0, 0, 100, 100), original=4),
        raw(1, 0.80, (50, 0, 150, 100), original=5),
    )
    assert reconstruct_c_clusters(exact_half).cluster_members == (
        (4,),
        (5,),
    )


def test_stock_order_uses_score_source_query_original_index():
    records = (
        raw(
            1,
            0.80,
            (0, 0, 10, 10),
            query=2,
            original=8,
        ),
        raw(
            1,
            0.80,
            (20, 0, 30, 10),
            query=2,
            original=7,
        ),
    )

    assert reconstruct_c_clusters(records).cluster_members[0][0] == 7


def test_math_float_predecessor_assumption_is_strict():
    predecessor = math.nextafter(0.8, -math.inf)

    assert predecessor < 0.8


def test_overlay_uses_exact_float64_predecessor_and_full_bypass():
    full = raw(0, 0.80, (0, 0, 100, 100), original=10)
    tile = raw(1, 0.90, (10, 0, 110, 100), original=20)
    group = find_aggressor_groups((full, tile))[0]

    overlaid, patches = apply_group_overlay((full, tile), (group,))
    mapped = {record.original_index: record for record in overlaid}

    assert mapped[10] == full
    assert mapped[20].score == math.nextafter(0.80, -math.inf)
    assert patches[0].old_score == 0.90
    assert patches[0].new_score == math.nextafter(
        0.80, -math.inf
    )
    assert replace(mapped[20], score=tile.score) == tile


def test_post_overlay_conf_filter_precedes_reclustering():
    full = raw(0, 0.001, (0, 0, 100, 100), original=10)
    tile = raw(1, 0.002, (10, 0, 110, 100), original=20)
    group = find_aggressor_groups((full, tile))[0]

    replay = replay_overlay((full, tile), (group,))

    assert tuple(
        record.original_index for record in replay.active_raw
    ) == (10,)
    assert (
        replay.reconstruction.standard_predictions[0].source_order
        == 0
    )


def test_noop_replay_matches_stock_reconstruction():
    records = (
        raw(0, 0.8, (0, 0, 100, 100), original=1),
        raw(1, 0.7, (0, 0, 90, 90), original=2),
    )

    replay = replay_overlay(records, ())

    assert replay.reconstruction == reconstruct_c_clusters(records)
    assert replay.patches == ()


def test_overlay_rejects_group_not_derived_from_stock_clusters():
    full = raw(0, 0.80, (0, 0, 100, 100), original=10)
    tile = raw(1, 0.90, (10, 0, 110, 100), original=20)
    group = find_aggressor_groups((full, tile))[0]
    forged = replace(group, unit_id="forged")

    with pytest.raises(ValueError, match="eligible"):
        apply_group_overlay((full, tile), (forged,))


def test_post_overlay_filter_is_inclusive_at_exact_conf():
    full = raw(0, 0.001, (0, 0, 100, 100), original=10)

    replay = replay_overlay((full,), ())

    assert replay.active_raw == (full,)


def test_replay_preserves_the_complete_post_overlay_population():
    full = raw(0, 0.001, (0, 0, 100, 100), original=10)
    tile = raw(1, 0.002, (10, 0, 110, 100), original=20)
    group = find_aggressor_groups((full, tile))[0]

    replay = replay_overlay((full, tile), (group,))

    assert tuple(
        record.original_index for record in replay.overlaid_raw
    ) == (10, 20)
    assert replay.overlaid_raw[1].score < 0.001
    assert replay.active_raw == (full,)


def oracle_image(c_raw, gt_boxes):
    a_raw = tuple(
        raw(
            0,
            0.99 - index * 0.01,
            box,
            original=1000 + index,
            arm="A",
        )
        for index, box in enumerate(gt_boxes)
    )
    return OracleImage(
        image_id="images/i.jpg",
        width=640,
        height=640,
        gt_boxes=tuple(gt_boxes),
        gt_classes=tuple(0 for _ in gt_boxes),
        ignore_boxes=(),
        a_raw=a_raw,
        c_raw=tuple(c_raw),
    )


def safe_large_recovery_image():
    full = raw(0, 0.80, (0, 0, 200, 200), original=10)
    local = raw(1, 0.90, (50, 0, 250, 200), original=11)
    return oracle_image((full, local), ((0, 0, 200, 200),))


def tiny_tradeoff_image():
    full = raw(0, 0.80, (0, 0, 200, 200), original=10)
    local = raw(1, 0.90, (10, 10, 20, 20), original=11)
    return oracle_image(
        (full, local),
        ((0, 0, 200, 200), (10, 10, 20, 20)),
    )


def interacting_groups_image():
    full_a = raw(0, 0.80, (0, 0, 100, 100), original=10)
    local_a = raw(1, 0.95, (0, 0, 50, 100), original=11)
    full_b = raw(0, 0.79, (40, 0, 140, 100), original=20)
    local_b = raw(2, 0.94, (90, 0, 140, 100), original=21)
    return oracle_image(
        (full_a, local_a, full_b, local_b),
        ((0, 0, 100, 100), (40, 0, 140, 100)),
    )


def test_group_selected_only_for_large_gain_without_protected_loss():
    result = evaluate_oracle_image(safe_large_recovery_image())

    assert len(result.groups) == 1
    assert result.events[0].selected is True
    assert (
        sum(
            row["large"]
            for row in result.events[0].tp_delta.values()
        )
        > 0
    )
    assert all(
        row["all"] >= 0
        and row["tiny"] >= 0
        and row["large"] >= 0
        for row in result.events[0].tp_delta.values()
    )


def test_group_event_records_absolute_single_before_after_profiles():
    result = evaluate_oracle_image(safe_large_recovery_image())
    event = result.events[0]

    assert event.before_profile == result.stock_profile
    assert set(event.after_profile) == {
        f"{threshold:.2f}"
        for threshold in (
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
        )
    }
    for threshold in event.after_profile.values():
        assert set(threshold) == {
            "all",
            "tiny",
            "small",
            "medium",
            "large",
        }
        for counts in threshold.values():
            assert {"tp", "fp", "gt"} <= set(counts)


def test_complete_invariant_verifier_accepts_exact_oracle_replay():
    image = safe_large_recovery_image()
    result = evaluate_oracle_image(image)

    report = verify_oracle_image_invariants(image, result)

    assert report["passed"] is True
    assert all(
        value is True
        for key, value in report.items()
        if key != "passed" and isinstance(value, bool)
    )


@pytest.mark.parametrize(
    ("mutation", "failed_key"),
    (
        ("selected_score", "modified_scores_exact_predecessor"),
        ("unselected_box", "non_score_fields_bit_identical"),
        ("active_population", "active_exclusions_exact"),
        ("patch_score", "patches_exact"),
    ),
)
def test_invariant_verifier_rejects_fault_injection(
    mutation, failed_key
):
    image = safe_large_recovery_image()
    result = evaluate_oracle_image(image)
    joint = result.joint
    original = joint.retained_raw
    overlaid = list(joint.overlaid_raw)
    active = list(joint.active_raw)
    patches = list(joint.patches)

    if mutation == "selected_score":
        selected = patches[0].original_index
        position = next(
            index
            for index, record in enumerate(overlaid)
            if record.original_index == selected
        )
        overlaid[position] = replace(
            overlaid[position],
            score=math.nextafter(
                overlaid[position].score, -math.inf
            ),
        )
        active = [
            record
            for record in overlaid
            if record.score >= 0.001
        ]
    elif mutation == "unselected_box":
        position = next(
            index
            for index, record in enumerate(overlaid)
            if record.source_order == 0
        )
        overlaid[position] = replace(
            overlaid[position],
            global_xyxy=(0.0, 0.0, 199.0, 200.0),
        )
        active = [
            record
            for record in overlaid
            if record.score >= 0.001
        ]
    elif mutation == "active_population":
        active = []
    else:
        patches[0] = replace(
            patches[0],
            new_score=math.nextafter(
                patches[0].new_score, -math.inf
            ),
        )

    forged_joint = replace(
        joint,
        retained_raw=original,
        overlaid_raw=tuple(overlaid),
        active_raw=tuple(active),
        patches=tuple(patches),
    )
    forged_result = replace(result, joint=forged_joint)

    report = verify_oracle_image_invariants(image, forged_result)

    assert report["passed"] is False
    assert report[failed_key] is False


def test_any_threshold_tiny_loss_rejects_group():
    event = evaluate_oracle_image(tiny_tradeoff_image()).events[0]

    assert event.selected is False
    assert event.reason == "TP_SAFETY_FAIL"


def test_joint_pass_applies_all_independently_selected_groups_once():
    result = evaluate_oracle_image(interacting_groups_image())

    assert [event.selected for event in result.events] == [True, True]
    assert {patch.unit_id for patch in result.joint.patches} == {
        event.unit_id for event in result.events
    }
    assert result.selection_rounds == 1
    independent_gain = sum(
        row["large"]
        for event in result.events
        for row in event.tp_delta.values()
    )
    joint_gain = sum(
        result.joint_profile[key]["large"]["tp"]
        - result.stock_profile[key]["large"]["tp"]
        for key in result.joint_profile
    )
    assert 0 < joint_gain < independent_gain


def test_gate_is_joint_minus_a_and_inclusive_without_tolerance():
    a = {
        "AP-tiny-SBR": 0.0,
        "mAP50-95": 0.0,
        "tiny_recall": 0.0,
        "AP75": 0.002,
        "AP-large-SBR": 0.005,
    }
    oracle = {
        "AP-tiny-SBR": 0.010,
        "mAP50-95": 0.003,
        "tiny_recall": 0.020,
        "AP75": 0.0,
        "AP-large-SBR": 0.0,
    }

    decision = gate_oracle_metrics(a, oracle, selected_count=1)

    assert decision.status == "SBR_SCORE_ORACLE_GO"
    assert all(decision.gates.values())
    for name, threshold in GATES.items():
        below_a = dict(a)
        below_oracle = dict(oracle)
        if threshold >= 0:
            below_oracle[name] = math.nextafter(
                below_oracle[name], -math.inf
            )
        else:
            below_a[name] = math.nextafter(
                below_a[name], math.inf
            )
        failed = gate_oracle_metrics(
            below_a, below_oracle, selected_count=1
        )
        assert failed.status == "SBR_SCORE_ORACLE_STOP"
        assert failed.gates[name] is False
