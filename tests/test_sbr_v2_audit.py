from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
from src.sbr_fusion import Detection
import src.sbr_v2_audit as audit_module

from src.sbr_v2_audit import (
    AttributionCategory,
    AuditRawDetection,
    AuditImage,
    audit_image_at_threshold,
    effective_size,
    group_relevant_raw_rows,
    map_full_a_to_c,
    match_large_targets,
    reconstruct_c_clusters,
)


def pred(box, score, *, cls=0, source=0, query=0, original_index=0):
    return AuditRawDetection.synthetic(
        "i.jpg", "C", source=source, query=query, score=score, cls=cls,
        box=box, width=640, height=640, original_index=original_index,
    ).to_detection()


def test_effective_size_uses_arm_a_640_gain():
    assert effective_size((0, 0, 200, 200), width=1000, height=500) == pytest.approx(128.0)


def test_large_matching_is_class_aware_and_ignore_neutral():
    result = match_large_targets(
        predictions=[
            pred((0, 0, 120, 120), .9, cls=0),
            pred((200, 200, 260, 260), .8, cls=1, original_index=1),
        ], gt_boxes=[(0, 0, 120, 120)], gt_classes=[0],
        ignore_boxes=[(200, 200, 260, 260)], width=640, height=640,
        iou_threshold=.75,
    )
    assert result.gt_to_prediction == {0: 0}
    assert result.neutral_prediction_indices == (1,)


def test_audit_propagates_full_to_c_mapping_integrity_failure():
    a = AuditRawDetection.synthetic("i.jpg", "A", query=0, box=(0, 0, 120, 120))
    fixture = AuditImage("i.jpg", 640, 640, ((0, 0, 120, 120),), (0,), (a,), ())
    with pytest.raises(ValueError, match="missing"):
        audit_image_at_threshold(fixture, .75)


def test_reconstructed_ties_use_original_raw_index_after_shuffling_input():
    high_index = AuditRawDetection.synthetic("i.jpg", "C", source=1, query=0, score=.8, box=(0, 0, 1, 1), original_index=100)
    low_index = AuditRawDetection.synthetic("i.jpg", "C", source=1, query=0, score=.8, box=(3, 0, 4, 1), original_index=1)
    result = reconstruct_c_clusters([high_index, low_index])
    assert result.cluster_members == ((1,), (100,))


def test_mixed_anchor_tie_uses_source_query_original_order():
    a_first = AuditRawDetection.synthetic("i.jpg", "A", query=1, score=.8, box=(0, 0, 100, 100), original_index=20)
    a_second = AuditRawDetection.synthetic("i.jpg", "A", query=2, score=.8, box=(20, 20, 120, 120), original_index=10)
    c_first = AuditRawDetection.synthetic("i.jpg", "C", query=1, score=.8, box=(0, 0, 100, 100), original_index=200)
    c_second = AuditRawDetection.synthetic("i.jpg", "C", query=2, score=.8, box=(20, 20, 120, 120), original_index=100)
    local = AuditRawDetection.synthetic("i.jpg", "C", source=1, query=0, score=.9, box=(15, 15, 115, 115), original_index=300)
    fixture = AuditImage("i.jpg", 640, 640, ((0, 0, 100, 100),), (0,), (a_first,), (c_first, c_second, local), ())
    event = audit_image_at_threshold(fixture, .75).events[0]
    assert event.category is AttributionCategory.MIXED_CLUSTER_LOCALIZATION
    assert event.counterfactual_recovers is True


def test_mixed_geometric_match_lost_to_assignment_is_competition_not_localization():
    competing_gt = (5, 0, 105, 100)
    target_gt = (0, 0, 100, 100)
    a = AuditRawDetection.synthetic(
        "mixed-competition.jpg",
        "A",
        score=.8,
        box=target_gt,
        original_index=1,
    )
    c_full = AuditRawDetection.synthetic(
        "mixed-competition.jpg",
        "C",
        score=.8,
        box=target_gt,
        original_index=1,
    )
    c_local = AuditRawDetection.synthetic(
        "mixed-competition.jpg",
        "C",
        source=1,
        score=.9,
        box=competing_gt,
        original_index=2,
    )
    fixture = AuditImage(
        "mixed-competition.jpg",
        640,
        640,
        (competing_gt, target_gt),
        (0, 0),
        (a,),
        (c_full, c_local),
    )

    event = audit_image_at_threshold(fixture, .75).events[0]

    assert event.gt_index == 1
    assert event.category is AttributionCategory.MATCHING_COMPETITION
    assert event.counterfactual_recovers is False


def test_attribution_reports_truncation_competition_and_class_candidate_precedence():
    target = (0, 0, 100, 100)
    a = AuditRawDetection.synthetic("i.jpg", "A", source=0, query=0, score=.8, box=target, original_index=1)
    distractors = [AuditRawDetection.synthetic("i.jpg", "C", source=1, query=i, score=.9, box=(300 + 2*i, 0, 301 + 2*i, 1), original_index=1000+i) for i in range(300)]
    c_target = AuditRawDetection.synthetic("i.jpg", "C", source=0, query=0, score=.8, box=target, original_index=1)
    trunc = AuditImage("i.jpg", 640, 640, (target,), (0,), (a,), tuple([c_target, *distractors]), ())
    assert audit_image_at_threshold(trunc, .75).events[0].category is AttributionCategory.FINAL_300_TRUNCATION

    g0, g1 = (0, 0, 110, 110), (5, 5, 105, 105)
    a_comp = AuditImage("comp", 640, 640, (g0, g1), (0, 0), (Detection(g0, .8, 0, 0, 0), Detection(target, .7, 0, 0, 1)), (Detection(target, .7, 0, 0, 1),), ())
    comp_events = audit_image_at_threshold(a_comp, .75).events
    assert comp_events[0].category is AttributionCategory.MATCHING_COMPETITION

    no_candidate = AuditImage("candidate", 640, 640, (target,), (0,), (Detection(target, .8, 0, 0, 0),), (Detection((200, 200, 300, 300), .8, 0, 0, 0),), ())
    assert audit_image_at_threshold(no_candidate, .75).events[0].category is AttributionCategory.CLASS_OR_CANDIDATE_LOSS


def test_out_of_cap_nonmatching_cluster_is_not_attributed_to_truncation():
    target = (0, 0, 100, 100)
    a = AuditRawDetection.synthetic(
        "nonmatching-cap.jpg", "A", score=.1, box=target, original_index=1
    )
    c_full = AuditRawDetection.synthetic(
        "nonmatching-cap.jpg", "C", score=.1, box=target, original_index=1
    )
    c_local = AuditRawDetection.synthetic(
        "nonmatching-cap.jpg",
        "C",
        source=1,
        score=.8,
        box=(40, 0, 140, 100),
        original_index=2,
    )
    distractors = tuple(
        AuditRawDetection.synthetic(
            "nonmatching-cap.jpg",
            "C",
            source=2,
            query=i,
            score=.9,
            box=(300 + 2 * i, 0, 301 + 2 * i, 1),
            original_index=1000 + i,
        )
        for i in range(300)
    )
    fixture = AuditImage(
        "nonmatching-cap.jpg",
        640,
        640,
        (target,),
        (0,),
        (a,),
        (c_full, c_local, *distractors),
    )

    event = audit_image_at_threshold(fixture, .75).events[0]

    assert event.category is AttributionCategory.CLASS_OR_CANDIDATE_LOSS


def test_matching_pre_cap_cluster_removed_by_cap_is_attributed_to_truncation():
    target = (0, 0, 100, 100)
    a = AuditRawDetection.synthetic(
        "matching-cap.jpg", "A", score=.8, box=target, original_index=1
    )
    c_target = AuditRawDetection.synthetic(
        "matching-cap.jpg", "C", score=.8, box=target, original_index=1
    )
    distractors = tuple(
        AuditRawDetection.synthetic(
            "matching-cap.jpg",
            "C",
            source=1,
            query=i,
            score=.9,
            box=(300 + 2 * i, 0, 301 + 2 * i, 1),
            original_index=1000 + i,
        )
        for i in range(300)
    )
    fixture = AuditImage(
        "matching-cap.jpg",
        640,
        640,
        (target,),
        (0,),
        (a,),
        (c_target, *distractors),
    )

    event = audit_image_at_threshold(fixture, .75).events[0]

    assert event.category is AttributionCategory.FINAL_300_TRUNCATION


def test_attribution_excludes_c_tp_and_neutral_events_and_keeps_exact_event_id():
    g0, g1 = (0, 0, 100, 100), (200, 200, 300, 300)
    fixture = AuditImage("nested/exact.jpg", 640, 640, (g0, g1), (0, 0), (Detection(g0, .8, 0, 0, 0), Detection(g1, .7, 0, 0, 1)), (Detection(g0, .8, 0, 0, 0), Detection((200, 200, 250, 250), .7, 0, 0, 1)), ())
    result = audit_image_at_threshold(fixture, .75)
    assert len(result.events) == 1
    event = result.events[0]
    assert (event.image_id, event.gt_index, event.iou_threshold) == ("nested/exact.jpg", 1, .75)


def test_other_category_is_used_for_neutralized_candidate_after_a_tp():
    gt = (0, 0, 100, 100)
    # A's larger box remains an A-TP while the C candidate is neutralized by
    # an ignore region covering >= half of its own area.
    a = Detection((0, 0, 110, 110), .8, 0, 0, 0)
    c = Detection(gt, .8, 0, 0, 0)
    fixture = AuditImage("other", 640, 640, (gt,), (0,), (a,), (c,), ((40, 0, 100, 100),))
    assert audit_image_at_threshold(fixture, .75).events[0].category is AttributionCategory.OTHER


def test_matcher_confidence_boundary_and_frozen_final_cap():
    gt = [(0, 0, 100, 100)]
    preds = [Detection(gt[0], .001, 0, 0, 0), Detection(gt[0], .0009, 0, 0, 1)]
    result = match_large_targets(preds, gt, [0], width=640, height=640, iou_threshold=.75)
    assert result.gt_to_prediction == {0: 0}
    many = [Detection((3*i, 0, 3*i+1, 1), .5, 0, 0, i) for i in range(301)]
    assert len(match_large_targets(many, [], [], width=640, height=640, iou_threshold=.75).ordered_predictions) == 300


def test_matcher_neutralizes_out_of_large_bin_targets():
    result = match_large_targets([Detection((0, 0, 50, 50), .9, 0, 0, 0)], [(0, 0, 50, 50)], [0], width=640, height=640, iou_threshold=.75)
    assert result.gt_to_prediction == {}
    assert result.neutral_prediction_indices == (0,)


def test_matcher_uses_highest_iou_then_lowest_gt_index_and_raw_original_order():
    pred_match = Detection((0, 0, 100, 100), .9, 0, 0, 0)
    result = match_large_targets([pred_match], [(0, 0, 100, 100), (0, 0, 100, 100)], [0, 0], width=640, height=640, iou_threshold=.75)
    assert result.pred_to_gt == {0: 0}
    p_high_index = AuditRawDetection.synthetic("i.jpg", "C", source=1, query=0, score=.8, box=(0, 0, 1, 1), original_index=100)
    p_low_index = AuditRawDetection.synthetic("i.jpg", "C", source=1, query=0, score=.8, box=(3, 0, 4, 1), original_index=1)
    ordered = match_large_targets([p_high_index, p_low_index], [], [], width=640, height=640, iou_threshold=.75).ordered_predictions
    assert [p.original_index for p in ordered] == [1, 100]


@pytest.mark.parametrize(
    "prediction",
    [
        Detection((float("nan"), 0, 1, 1), .8, 0, 0, 0),
        Detection((0, 0, 0, 1), .8, 0, 0, 0),
        Detection((0, 0, 1, 1), .8, True, 0, 0),
        Detection((0, 0, 1, 1), .8, 0, 1.0, 0),
        Detection((0, 0, 1, 1), -.1, 0, 0, 0),
        Detection((0, 0, 1, 1), 1.1, 0, 0, 0),
        {"box": (0, 0, 1, 1), "score": float("inf"), "class_id": 0},
        {"box": (0, 0, 1, 1), "score": .8, "class_id": -1},
        {"box": (0, 0, 1, 1), "score": .8, "class_id": 0, "source_order": -1},
        {"box": (0, 0, 1, 1), "score": .8, "class_id": 0, "query_index": 1.0},
        {"box": (0, 0, 1, 1), "score": .8, "class_id": 0, "original_index": True},
        SimpleNamespace(
            box=(0, 0, 1, 1),
            score=.8,
            class_id=0,
            source_order=0,
            query_index=0,
            original_index=2.0,
        ),
    ],
)
def test_matcher_rejects_malformed_prediction_records(prediction):
    with pytest.raises(ValueError):
        match_large_targets([prediction], [], [], width=640, height=640, iou_threshold=.75)


def test_matcher_rejects_detection_with_boolean_score():
    with pytest.raises(ValueError):
        match_large_targets(
            [Detection((0, 0, 1, 1), True, 0, 0, 0)],
            [],
            [],
            width=640,
            height=640,
            iou_threshold=.75,
        )


@pytest.mark.parametrize("score", [True, "0.8"])
def test_audit_raw_detection_rejects_non_real_score_before_conversion(score):
    with pytest.raises(ValueError):
        AuditRawDetection.synthetic("i.jpg", "C", score=score)


def test_matcher_validates_ignore_boxes_even_without_eligible_predictions():
    for malformed in (
        (0, 0, float("nan"), 1),
        (0, 0, 0, 1),
        (0, 0, 1),
    ):
        with pytest.raises(ValueError):
            match_large_targets(
                [],
                [],
                [],
                ignore_boxes=[malformed],
                width=640,
                height=640,
                iou_threshold=.75,
            )


@pytest.mark.parametrize(
    "width,height",
    [
        (0, 640),
        (640, 0),
        (1.0, 640),
        (640, True),
    ],
)
def test_matcher_validates_image_dimensions_even_without_ground_truth(
    width, height
):
    with pytest.raises(ValueError):
        match_large_targets(
            [],
            [],
            [],
            width=width,
            height=height,
            iou_threshold=.75,
        )


def test_matcher_rejects_noninteger_original_index_on_detection():
    prediction = Detection((0, 0, 1, 1), .8, 0, 0, 0)
    object.__setattr__(prediction, "original_index", 2.0)

    with pytest.raises(ValueError):
        match_large_targets(
            [prediction], [], [], width=640, height=640, iou_threshold=.75
        )


def raw(
    image_id: str,
    arm: str,
    *,
    source: int = 0,
    query: int = 0,
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "arm": arm,
        "source_order": source,
        "query_index": query,
    }


def test_group_relevant_raw_rows_keeps_only_a_and_c_in_exact_manifest_order():
    rows = [
        raw("a.jpg", "A"),
        raw("a.jpg", "B", source=1),
        raw("a.jpg", "C"),
        raw("c.jpg", "D"),
        raw("c.jpg", "C"),
    ]

    groups = list(group_relevant_raw_rows(rows, ["a.jpg", "b.jpg", "c.jpg"]))

    assert [group.image_id for group in groups] == ["a.jpg", "b.jpg", "c.jpg"]
    assert [[row["arm"] for row in group.rows] for group in groups] == [
        ["A", "C"],
        [],
        ["C"],
    ]


@pytest.mark.parametrize(
    "rows,manifest",
    [
        ([raw("unknown.jpg", "A")], ["a.jpg"]),
        (
            [raw("a.jpg", "A"), raw("b.jpg", "A"), raw("a.jpg", "C")],
            ["a.jpg", "b.jpg"],
        ),
        ([raw("b.jpg", "A"), raw("a.jpg", "A")], ["a.jpg", "b.jpg"]),
        ([raw(r"nested\a.jpg", "A")], ["nested/a.jpg"]),
    ],
    ids=["unknown", "repeated-group", "out-of-order", "no-path-normalization"],
)
def test_group_relevant_raw_rows_rejects_invalid_image_groups(rows, manifest):
    with pytest.raises(ValueError):
        list(group_relevant_raw_rows(rows, manifest))


def test_audit_raw_detection_is_immutable_and_preserves_detection_provenance():
    audit = AuditRawDetection.synthetic(
        "nested/i.jpg",
        "C",
        source=2,
        query=7,
        score=0.8,
        cls=3,
        box=(1, 2, 9, 10),
        width=20,
        height=20,
        tile_bounds=(0, 0, 20, 20),
        original_index=41,
    )

    assert audit.identity_key == ("nested/i.jpg", 3, 2, 7)
    with pytest.raises(FrozenInstanceError):
        audit.score = 0.1

    detection = audit.to_detection()
    assert detection.box == (1.0, 2.0, 9.0, 10.0)
    assert detection.score == 0.8
    assert detection.class_id == 3
    assert detection.source_order == 2
    assert detection.query_index == 7
    assert detection.network_xyxy == audit.network_xyxy
    assert detection.view_xyxy == audit.view_xyxy
    assert detection.global_xyxy == audit.global_xyxy
    assert detection.tile_local_box == audit.view_xyxy
    assert detection.global_box == audit.global_xyxy
    assert detection.tile_bounds == (0, 0, 20, 20)
    assert detection.tile_index == 1


def test_a_full_detection_maps_to_exact_c_key_and_raw_index():
    a = AuditRawDetection.synthetic(
        "i.jpg", "A", source=0, query=7, score=0.8, original_index=2
    )
    c = AuditRawDetection.synthetic(
        "i.jpg", "C", source=0, query=7, score=0.8, original_index=17
    )
    local_a = AuditRawDetection.synthetic(
        "i.jpg", "A", source=1, query=8, score=0.7, original_index=3
    )
    local_c = AuditRawDetection.synthetic(
        "i.jpg", "C", source=1, query=8, score=0.7, original_index=18
    )

    assert map_full_a_to_c([a, local_a], [c, local_c]) == {
        a.identity_key: 17
    }


def test_synthetic_default_indices_support_the_plan_full_plus_tile_fixture():
    full = AuditRawDetection.synthetic("i.jpg", "C", source=0, query=7)
    tile = AuditRawDetection.synthetic("i.jpg", "C", source=1, query=7)

    assert (full.original_index, tile.original_index) == (0, 1)
    assert reconstruct_c_clusters([full, tile]).cluster_members == ((0, 1),)


def test_mapping_rejects_missing_or_colliding_full_records():
    a = AuditRawDetection.synthetic("i.jpg", "A", query=7, original_index=2)
    c = AuditRawDetection.synthetic("i.jpg", "C", query=7, original_index=17)
    duplicate = replace(c, original_index=18)

    with pytest.raises(ValueError, match="missing"):
        map_full_a_to_c([a], [])
    with pytest.raises(ValueError, match="collision"):
        map_full_a_to_c([a], [c, duplicate])
    with pytest.raises(ValueError, match="collision"):
        map_full_a_to_c([a, replace(a, original_index=3)], [c])


@pytest.mark.parametrize(
    "changes",
    [
        {"score": 0.8000000000000002},
        {"network_xyxy": (1.0, 0.0, 2.0, 2.0)},
        {"view_xyxy": (1.0, 0.0, 2.0, 2.0)},
        {"global_xyxy": (1.0, 0.0, 2.0, 2.0)},
        # Canonical JSON distinguishes these numerically equal float spellings.
        {"network_xyxy": (-0.0, 0.0, 2.0, 2.0)},
    ],
    ids=["score", "network", "view", "global", "canonical-float-bytes"],
)
def test_mapping_rejects_byte_level_score_or_coordinate_disagreement(changes):
    a = AuditRawDetection.synthetic(
        "i.jpg", "A", query=7, score=0.8, box=(0, 0, 2, 2)
    )
    c = replace(
        AuditRawDetection.synthetic(
            "i.jpg", "C", query=7, score=0.8, box=(0, 0, 2, 2)
        ),
        **changes,
    )

    with pytest.raises(ValueError, match="disagreement"):
        map_full_a_to_c([a], [c])


def test_reconstructed_clusters_use_strict_ios_and_original_raw_indices():
    full = AuditRawDetection.synthetic(
        "i.jpg",
        "C",
        source=0,
        query=0,
        score=0.9,
        box=(0, 0, 10, 10),
        width=20,
        original_index=12,
    )
    nested_tile = AuditRawDetection.synthetic(
        "i.jpg",
        "C",
        source=1,
        query=1,
        score=0.8,
        box=(1, 1, 9, 9),
        width=20,
        original_index=3,
    )
    exact_half = AuditRawDetection.synthetic(
        "i.jpg",
        "C",
        source=2,
        query=2,
        score=0.7,
        box=(5, 0, 15, 10),
        original_index=99,
        width=20,
    )

    result = reconstruct_c_clusters([exact_half, nested_tile, full])

    assert result.cluster_members == ((12, 3), (99,))
    assert result.pre_cap_predictions[0].box == pytest.approx(
        (8 / 17, 8 / 17, 162 / 17, 162 / 17)
    )
    assert result.pre_cap_predictions[0].members == (
        full.to_detection(),
        nested_tile.to_detection(),
    )
    assert result.pre_cap_predictions[1] is result.standard_predictions[1]


def test_reconstructed_pre_cap_order_uses_frozen_deterministic_tie_breaks():
    detections = [
        AuditRawDetection.synthetic(
            "i.jpg", "C", source=2, query=5, score=0.8, box=(0, 0, 1, 1), original_index=10
        ),
        AuditRawDetection.synthetic(
            "i.jpg", "C", source=4, query=9, score=0.9, box=(3, 0, 4, 1), original_index=11
        ),
        AuditRawDetection.synthetic(
            "i.jpg", "C", source=1, query=9, score=0.8, box=(6, 0, 7, 1), original_index=12
        ),
        AuditRawDetection.synthetic(
            "i.jpg", "C", source=1, query=2, score=0.8, box=(9, 0, 10, 1), original_index=13
        ),
        AuditRawDetection.synthetic(
            "i.jpg", "C", source=1, query=2, score=0.8, box=(12, 0, 13, 1), original_index=14
        ),
    ]

    result = reconstruct_c_clusters(detections)

    assert result.cluster_members == ((11,), (13,), (14,), (12,), (10,))
    assert result.standard_predictions == result.pre_cap_predictions


def test_reconstructed_predictions_keep_pre_cap_rows_and_apply_final_top300():
    detections = [
        AuditRawDetection.synthetic(
            "i.jpg",
            "C",
            source=1,
            query=0,
            score=0.5,
            box=(3 * index, 0, 3 * index + 1, 1),
            width=1000,
            original_index=500 + index,
        )
        for index in range(302)
    ]

    result = reconstruct_c_clusters(detections)

    assert len(result.pre_cap_predictions) == 302
    assert len(result.standard_predictions) == 300
    assert result.cluster_members[:2] == ((500,), (501,))
    assert result.cluster_members[-2:] == ((800,), (801,))
    assert result.standard_predictions == result.pre_cap_predictions[:300]


def test_large_view_guard_uses_frozen_full_anchor_order_for_eligible_mixed_cluster():
    full_late = AuditRawDetection.synthetic(
        "i.jpg",
        "C",
        source=0,
        query=2,
        score=0.8,
        box=(20, 20, 120, 120),
        width=640,
        height=640,
        original_index=1,
    )
    full_first = AuditRawDetection.synthetic(
        "i.jpg",
        "C",
        source=0,
        query=1,
        score=0.8,
        box=(0, 0, 100, 100),
        width=640,
        height=640,
        original_index=20,
    )
    local = AuditRawDetection.synthetic(
        "i.jpg",
        "C",
        source=1,
        query=0,
        score=0.9,
        box=(10, 10, 110, 110),
        width=640,
        height=640,
        original_index=30,
    )
    raw = (local, full_late, full_first)
    standard = reconstruct_c_clusters(raw)

    guarded = audit_module.apply_large_view_guard(standard, raw)

    assert guarded.pre_cap_predictions[0].box == full_first.global_xyxy
    assert guarded.standard_predictions[0].box == full_first.global_xyxy
    assert guarded.pre_cap_predictions[0].score == standard.pre_cap_predictions[0].score
    assert guarded.pre_cap_predictions[0].class_id == standard.pre_cap_predictions[0].class_id
    assert guarded.pre_cap_predictions[0].source_order == standard.pre_cap_predictions[0].source_order
    assert guarded.pre_cap_predictions[0].query_index == standard.pre_cap_predictions[0].query_index
    assert guarded.cluster_members == standard.cluster_members


def test_large_view_guard_preserves_ineligible_mixed_and_single_source_clusters():
    raw = (
        AuditRawDetection.synthetic(
            "i.jpg", "C", source=0, query=0, score=0.9,
            box=(0, 0, 96, 96), width=640, height=640, original_index=0,
        ),
        AuditRawDetection.synthetic(
            "i.jpg", "C", source=1, query=0, score=0.8,
            box=(2, 2, 94, 94), width=640, height=640, original_index=1,
        ),
        AuditRawDetection.synthetic(
            "i.jpg", "C", source=0, query=1, score=0.7,
            box=(150, 0, 200, 50), width=640, height=640, original_index=2,
        ),
        AuditRawDetection.synthetic(
            "i.jpg", "C", source=1, query=1, score=0.6,
            box=(152, 2, 198, 48), width=640, height=640, original_index=3,
        ),
        AuditRawDetection.synthetic(
            "i.jpg", "C", source=1, query=2, score=0.5,
            box=(300, 0, 320, 20), width=640, height=640, original_index=4,
        ),
        AuditRawDetection.synthetic(
            "i.jpg", "C", source=0, query=3, score=0.4,
            box=(450, 0, 570, 120), width=640, height=640, original_index=5,
        ),
    )
    standard = reconstruct_c_clusters(raw)

    guarded = audit_module.apply_large_view_guard(standard, raw)

    assert guarded == standard


@pytest.mark.parametrize("malformed", ["missing", "duplicate", "wrong_arm"])
def test_large_view_guard_rejects_malformed_raw_provenance(malformed):
    full = AuditRawDetection.synthetic(
        "i.jpg", "C", source=0, score=0.8, box=(0, 0, 120, 120),
        width=640, height=640, original_index=10,
    )
    local = AuditRawDetection.synthetic(
        "i.jpg", "C", source=1, score=0.9, box=(10, 10, 110, 110),
        width=640, height=640, original_index=11,
    )
    standard = reconstruct_c_clusters((full, local))
    malformed_raw = {
        "missing": (full,),
        "duplicate": (full, replace(local, original_index=10)),
        "wrong_arm": (replace(full, arm="A"), local),
    }[malformed]

    with pytest.raises(ValueError):
        audit_module.apply_large_view_guard(standard, malformed_raw)


def test_large_view_guard_rejects_standard_not_rebuilt_from_supplied_raw():
    full = AuditRawDetection.synthetic(
        "i.jpg", "C", source=0, query=0, score=0.8,
        box=(0, 0, 120, 120), width=640, height=640, original_index=10,
    )
    local = AuditRawDetection.synthetic(
        "i.jpg", "C", source=1, query=0, score=0.9,
        box=(10, 10, 110, 110), width=640, height=640, original_index=11,
    )
    standard = reconstruct_c_clusters((full, local))
    mutated_full = replace(
        full,
        network_xyxy=(20, 20, 140, 140),
        view_xyxy=(20, 20, 140, 140),
        global_xyxy=(20, 20, 140, 140),
    )

    with pytest.raises(ValueError):
        audit_module.apply_large_view_guard(
            standard, (mutated_full, local)
        )


@pytest.mark.parametrize(
    "raw",
    [
        (
            AuditRawDetection.synthetic(
                "i.jpg", "C", source=5, original_index=10
            ),
        ),
        (
            AuditRawDetection.synthetic(
                "i.jpg", "C", source=99, original_index=10
            ),
        ),
        (
            AuditRawDetection.synthetic(
                "i.jpg", "C", source=0, tile_bounds=(0, 0, 100, 100),
                original_index=10,
            ),
        ),
        (
            AuditRawDetection.synthetic(
                "i.jpg", "C", source=1, tile_bounds=None, original_index=10
            ),
        ),
    ],
)
def test_c_reconstruction_rejects_invalid_source_provenance(raw):
    with pytest.raises(ValueError):
        reconstruct_c_clusters(raw)


def test_guard_invariants_pass_only_for_coordinate_only_guard_output():
    full = AuditRawDetection.synthetic(
        "i.jpg",
        "C",
        source=0,
        query=0,
        score=0.8,
        box=(0, 0, 120, 120),
        width=640,
        height=640,
        original_index=10,
    )
    local = AuditRawDetection.synthetic(
        "i.jpg",
        "C",
        source=1,
        query=0,
        score=0.9,
        box=(10, 10, 110, 110),
        width=640,
        height=640,
        original_index=11,
    )
    singleton = AuditRawDetection.synthetic(
        "i.jpg",
        "C",
        source=1,
        query=1,
        score=0.7,
        box=(300, 300, 320, 320),
        width=640,
        height=640,
        original_index=12,
    )
    raw = (full, local, singleton)
    standard = reconstruct_c_clusters(raw)
    guarded = audit_module.apply_large_view_guard(standard, raw)

    invariants = audit_module.verify_guard_invariants(
        standard, guarded, raw, guarded_raw_detections=raw
    )

    assert invariants == {
        "raw_hash_equal": True,
        "cluster_hash_equal": True,
        "cluster_count_equal": True,
        "scores_equal": True,
        "classes_equal": True,
        "selected_cluster_ids_equal": True,
        "singleton_preservation": 1.0,
        "passed": True,
    }
    assert isinstance(invariants["singleton_preservation"], float)


def test_guard_invariants_fail_closed_for_mutation_and_nonfinite_coordinates():
    first = AuditRawDetection.synthetic(
        "i.jpg",
        "C",
        source=1,
        query=0,
        score=0.8,
        box=(0, 0, 20, 20),
        width=640,
        height=640,
        original_index=10,
    )
    second = AuditRawDetection.synthetic(
        "i.jpg",
        "C",
        source=1,
        query=1,
        score=0.7,
        box=(300, 300, 320, 320),
        width=640,
        height=640,
        original_index=11,
    )
    raw = (first, second)
    standard = reconstruct_c_clusters(raw)
    malformed_prediction = replace(
        standard.pre_cap_predictions[0], box=(0.0, 0.0, float("nan"), 20.0)
    )
    mutated_prediction = replace(standard.pre_cap_predictions[1], score=0.6)
    guarded = type(standard)(
        pre_cap_predictions=(malformed_prediction, mutated_prediction),
        standard_predictions=(mutated_prediction, malformed_prediction),
        cluster_members=((10,), (999,)),
    )
    changed_raw = (replace(first, score=0.6), second)

    invariants = audit_module.verify_guard_invariants(
        standard, guarded, raw, guarded_raw_detections=changed_raw
    )

    assert set(invariants) == {
        "raw_hash_equal",
        "cluster_hash_equal",
        "cluster_count_equal",
        "scores_equal",
        "classes_equal",
        "selected_cluster_ids_equal",
        "singleton_preservation",
        "passed",
    }
    assert all(
        isinstance(value, bool)
        for key, value in invariants.items()
        if key != "singleton_preservation"
    )
    assert isinstance(invariants["singleton_preservation"], float)
    assert invariants["raw_hash_equal"] is False
    assert invariants["cluster_hash_equal"] is False
    assert invariants["scores_equal"] is False
    assert invariants["selected_cluster_ids_equal"] is False
    assert invariants["singleton_preservation"] == 0.0
    assert invariants["passed"] is False


def test_guard_invariants_reject_seed_provenance_changes_in_mixed_cluster():
    full = AuditRawDetection.synthetic(
        "i.jpg", "C", source=0, query=0, score=0.8,
        box=(0, 0, 120, 120), width=640, height=640, original_index=10,
    )
    local = AuditRawDetection.synthetic(
        "i.jpg", "C", source=1, query=0, score=0.9,
        box=(10, 10, 110, 110), width=640, height=640, original_index=11,
    )
    raw = (full, local)
    standard = reconstruct_c_clusters(raw)
    guarded = audit_module.apply_large_view_guard(standard, raw)
    mutated = replace(guarded.pre_cap_predictions[0], source_order=2)
    malformed_guard = type(guarded)(
        pre_cap_predictions=(mutated,),
        standard_predictions=(mutated,),
        cluster_members=guarded.cluster_members,
    )

    invariants = audit_module.verify_guard_invariants(
        standard, malformed_guard, raw
    )

    assert invariants["selected_cluster_ids_equal"] is False
    assert invariants["passed"] is False


def test_guard_invariants_reject_local_only_non_singleton_coordinate_change():
    raw = (
        AuditRawDetection.synthetic(
            "i.jpg", "C", source=1, query=0, score=0.9,
            box=(0, 0, 40, 40), width=640, height=640, original_index=10,
        ),
        AuditRawDetection.synthetic(
            "i.jpg", "C", source=2, query=0, score=0.8,
            box=(2, 2, 38, 38), width=640, height=640, original_index=11,
        ),
    )
    standard = reconstruct_c_clusters(raw)
    guarded = audit_module.apply_large_view_guard(standard, raw)
    wrong = replace(guarded.pre_cap_predictions[0], box=(1, 1, 39, 39))
    malformed_guard = type(guarded)(
        pre_cap_predictions=(wrong,),
        standard_predictions=(wrong,),
        cluster_members=guarded.cluster_members,
    )

    invariants = audit_module.verify_guard_invariants(
        standard, malformed_guard, raw
    )

    assert invariants["cluster_hash_equal"] is False
    assert invariants["passed"] is False


def test_guard_invariants_reject_size_96_mixed_coordinate_change():
    raw = (
        AuditRawDetection.synthetic(
            "i.jpg", "C", source=0, query=0, score=0.9,
            box=(0, 0, 96, 96), width=640, height=640, original_index=10,
        ),
        AuditRawDetection.synthetic(
            "i.jpg", "C", source=1, query=0, score=0.8,
            box=(2, 2, 94, 94), width=640, height=640, original_index=11,
        ),
    )
    standard = reconstruct_c_clusters(raw)
    guarded = audit_module.apply_large_view_guard(standard, raw)
    wrong = replace(guarded.pre_cap_predictions[0], box=(0, 0, 96, 96))
    malformed_guard = type(guarded)(
        pre_cap_predictions=(wrong,),
        standard_predictions=(wrong,),
        cluster_members=guarded.cluster_members,
    )

    invariants = audit_module.verify_guard_invariants(
        standard, malformed_guard, raw
    )

    assert invariants["cluster_hash_equal"] is False
    assert invariants["passed"] is False


def test_guard_invariants_reject_wrong_full_anchor_for_eligible_mixed_cluster():
    preferred = AuditRawDetection.synthetic(
        "i.jpg", "C", source=0, query=1, score=0.8,
        box=(0, 0, 120, 120), width=640, height=640, original_index=20,
    )
    wrong_anchor = AuditRawDetection.synthetic(
        "i.jpg", "C", source=0, query=2, score=0.8,
        box=(20, 20, 140, 140), width=640, height=640, original_index=10,
    )
    local = AuditRawDetection.synthetic(
        "i.jpg", "C", source=1, query=0, score=0.9,
        box=(10, 10, 130, 130), width=640, height=640, original_index=30,
    )
    raw = (preferred, wrong_anchor, local)
    standard = reconstruct_c_clusters(raw)
    guarded = audit_module.apply_large_view_guard(standard, raw)
    wrong = replace(
        guarded.pre_cap_predictions[0], box=wrong_anchor.global_xyxy
    )
    malformed_guard = type(guarded)(
        pre_cap_predictions=(wrong,),
        standard_predictions=(wrong,),
        cluster_members=guarded.cluster_members,
    )

    invariants = audit_module.verify_guard_invariants(
        standard, malformed_guard, raw
    )

    assert guarded.pre_cap_predictions[0].box == preferred.global_xyxy
    assert invariants["cluster_hash_equal"] is False
    assert invariants["passed"] is False


def test_singleton_preservation_rejects_global_frame_provenance_change():
    only = AuditRawDetection.synthetic(
        "i.jpg", "C", source=0, query=0, score=0.8,
        box=(0, 0, 40, 40), width=640, height=640, original_index=10,
    )
    raw = (only,)
    standard = reconstruct_c_clusters(raw)
    guarded = audit_module.apply_large_view_guard(standard, raw)
    wrong = replace(
        guarded.pre_cap_predictions[0], global_xyxy=(1, 1, 41, 41)
    )
    malformed_guard = type(guarded)(
        pre_cap_predictions=(wrong,),
        standard_predictions=(wrong,),
        cluster_members=guarded.cluster_members,
    )

    invariants = audit_module.verify_guard_invariants(
        standard, malformed_guard, raw
    )

    assert invariants["singleton_preservation"] == 0.0
    assert invariants["passed"] is False


def test_large_view_guard_changes_rank_301_pre_cap_without_changing_top300():
    distractors = tuple(
        AuditRawDetection.synthetic(
            "i.jpg",
            "C",
            source=1,
            query=index,
            score=0.9,
            box=(1000 + 10 * index, 0, 1001 + 10 * index, 1),
            width=5000,
            height=5000,
            original_index=1000 + index,
        )
        for index in range(300)
    )
    full = AuditRawDetection.synthetic(
        "i.jpg", "C", source=0, query=400, score=0.1,
        box=(0, 0, 800, 800), width=5000, height=5000, original_index=10,
    )
    local = AuditRawDetection.synthetic(
        "i.jpg", "C", source=1, query=400, score=0.1,
        box=(10, 10, 790, 790), width=5000, height=5000, original_index=11,
    )
    raw = (*distractors, full, local)
    standard = reconstruct_c_clusters(raw)

    guarded = audit_module.apply_large_view_guard(standard, raw)

    assert len(standard.pre_cap_predictions) == 301
    assert standard.pre_cap_predictions[300].box != full.global_xyxy
    assert guarded.pre_cap_predictions[300].box == full.global_xyxy
    assert guarded.standard_predictions == standard.standard_predictions
    assert guarded.cluster_members[:300] == standard.cluster_members[:300]


def _evidence_row(
    image_id, *, predicts=True, source=0, prediction_box=None
):
    gt_box = (0, 0, 120, 120)
    resolved_prediction_box = (
        gt_box if prediction_box is None else prediction_box
    )
    return {
        "image_id": image_id,
        "width": 640,
        "height": 640,
        "pred_boxes": [resolved_prediction_box] if predicts else [],
        "pred_scores": [0.9] if predicts else [],
        "pred_classes": [0] if predicts else [],
        "pred_source": [source] if predicts else [],
        "pred_query": [0] if predicts else [],
        "gt_boxes": [gt_box],
        "gt_classes": [0],
        "ignore_boxes": [],
        "effective_gain": 1.0,
    }


def _upper_bound_evidence(count=1, *, c_recovers=False):
    a_rows = [
        _evidence_row(f"i{index}.jpg", predicts=True, source=0)
        for index in range(count)
    ]
    c_rows = [
        _evidence_row(
            f"i{index}.jpg",
            predicts=True,
            source=0,
            prediction_box=(
                (0, 0, 120, 120)
                if c_recovers
                else (30, 30, 150, 150)
            ),
        )
        for index in range(count)
    ]
    v2_rows = [
        _evidence_row(f"i{index}.jpg", predicts=True, source=0)
        for index in range(count)
    ]
    return a_rows, c_rows, v2_rows


def test_guard_upper_bound_evaluates_real_a_c_v2_rows_and_fixed_gates():
    a_rows, c_rows, v2_rows = _upper_bound_evidence()
    invariants = {
        "raw_hash_equal": True,
        "cluster_hash_equal": True,
        "cluster_count_equal": True,
        "scores_equal": True,
        "classes_equal": True,
        "selected_cluster_ids_equal": True,
        "singleton_preservation": 1.0,
        "passed": True,
    }

    report = audit_module.evaluate_guard_upper_bound(
        a_rows,
        c_rows,
        v2_rows,
        mixed_localization_unique_large_gt=1,
        a_tp_to_c_fn_unique_large_gt=1,
        invariants=invariants,
    )

    assert set(report) == {
        "mechanism_share_ap75",
        "mechanism_gate",
        "a_metrics",
        "c_metrics",
        "v2_metrics",
        "v2_minus_a",
        "v2_minus_c",
        "recoverable_upper_bound_gate",
        "invariants",
    }
    assert report["mechanism_share_ap75"] == pytest.approx(1.0)
    assert report["mechanism_gate"] == "PASS"
    assert report["recoverable_upper_bound_gate"] == "PASS"
    assert report["invariants"] == invariants
    assert set(report["v2_minus_a"]) == {
        "AP-tiny-SBR",
        "mAP50-95",
        "tiny_recall",
        "AP75",
        "AP-large-SBR",
    }
    assert set(report["v2_minus_c"]) == set(report["v2_minus_a"])
    assert report["v2_metrics"]["AP-large-SBR"] == pytest.approx(
        report["a_metrics"]["AP-large-SBR"]
    )


def test_guard_upper_bound_reports_nonzero_deltas_as_v2_minus_baseline(monkeypatch):
    a_metrics = {
        "AP-tiny-SBR": 0.10,
        "mAP50-95": 0.20,
        "tiny_recall": 0.30,
        "AP75": 0.40,
        "AP-large-SBR": 0.50,
    }
    c_metrics = {
        "AP-tiny-SBR": 0.15,
        "mAP50-95": 0.25,
        "tiny_recall": 0.35,
        "AP75": 0.45,
        "AP-large-SBR": 0.55,
    }
    v2_metrics = {
        "AP-tiny-SBR": 0.30,
        "mAP50-95": 0.40,
        "tiny_recall": 0.50,
        "AP75": 0.60,
        "AP-large-SBR": 0.70,
    }
    responses = iter((a_metrics, c_metrics, v2_metrics))
    monkeypatch.setattr(
        audit_module, "evaluate_dataset", lambda _rows: next(responses)
    )
    a_rows, c_rows, v2_rows = _upper_bound_evidence()

    report = audit_module.evaluate_guard_upper_bound(
        a_rows, c_rows, v2_rows,
        mixed_localization_unique_large_gt=1,
        a_tp_to_c_fn_unique_large_gt=1,
        invariants={"singleton_preservation": 1.0, "passed": True},
    )

    for key in {
        "AP-tiny-SBR",
        "mAP50-95",
        "tiny_recall",
        "AP75",
        "AP-large-SBR",
    }:
        assert report["v2_minus_a"][key] == pytest.approx(
            v2_metrics[key] - a_metrics[key]
        )
        assert report["v2_minus_c"][key] == pytest.approx(
            v2_metrics[key] - c_metrics[key]
        )


def _guard_metrics(large):
    return {
        "AP-tiny-SBR": 0.4,
        "mAP50-95": 0.5,
        "tiny_recall": 0.6,
        "AP75": 0.5,
        "AP-large-SBR": large,
    }


def _patch_guard_metrics(monkeypatch, a_large, c_large, v2_large):
    responses = iter(
        (_guard_metrics(a_large), _guard_metrics(c_large), _guard_metrics(v2_large))
    )
    monkeypatch.setattr(
        audit_module, "evaluate_dataset", lambda _rows: next(responses)
    )


def test_guard_upper_bound_gate_boundaries_are_independent(monkeypatch):
    a_rows, c_rows, v2_rows = _upper_bound_evidence(5)
    _patch_guard_metrics(monkeypatch, 0.5, 0.4, 0.495)
    report = audit_module.evaluate_guard_upper_bound(
        a_rows, c_rows, v2_rows,
        mixed_localization_unique_large_gt=3,
        a_tp_to_c_fn_unique_large_gt=5,
        invariants={"passed": False},
    )
    assert report["mechanism_gate"] == "PASS"
    assert report["recoverable_upper_bound_gate"] == "PASS"
    assert report["invariants"]["passed"] is False

    _patch_guard_metrics(monkeypatch, 0.5, 0.4, 0.494999999)
    report = audit_module.evaluate_guard_upper_bound(
        a_rows, c_rows, v2_rows,
        mixed_localization_unique_large_gt=2,
        a_tp_to_c_fn_unique_large_gt=5,
        invariants={"passed": True},
    )
    assert report["mechanism_gate"] == "FAIL"
    assert report["recoverable_upper_bound_gate"] == "FAIL"


def test_guard_upper_bound_zero_denominator_is_finite_and_fails_closed(monkeypatch):
    a_rows, c_rows, v2_rows = _upper_bound_evidence(c_recovers=True)
    _patch_guard_metrics(monkeypatch, 0.5, 0.4, 0.5)

    report = audit_module.evaluate_guard_upper_bound(
        a_rows, c_rows, v2_rows,
        mixed_localization_unique_large_gt=0,
        a_tp_to_c_fn_unique_large_gt=0,
        invariants={"passed": True},
    )

    assert report["mechanism_share_ap75"] == 0.0
    assert report["mechanism_gate"] == "FAIL"


def test_guard_upper_bound_rejects_empty_evidence_rows():
    with pytest.raises(ValueError):
        audit_module.evaluate_guard_upper_bound(
            (), (), (),
            mixed_localization_unique_large_gt=0,
            a_tp_to_c_fn_unique_large_gt=0,
            invariants={"passed": True},
        )


def test_guard_upper_bound_rejects_forged_loss_denominator():
    a_rows, c_rows, v2_rows = _upper_bound_evidence()

    with pytest.raises(ValueError):
        audit_module.evaluate_guard_upper_bound(
            a_rows, c_rows, v2_rows,
            mixed_localization_unique_large_gt=1,
            a_tp_to_c_fn_unique_large_gt=5,
            invariants={"passed": True},
        )


def test_guard_upper_bound_rejects_v2_prediction_metadata_changes():
    a_rows, c_rows, v2_rows = _upper_bound_evidence()
    v2_rows[0]["pred_scores"] = [0.99]
    v2_rows[0]["pred_source"] = [4]
    v2_rows[0]["pred_query"] = [99]

    with pytest.raises(ValueError):
        audit_module.evaluate_guard_upper_bound(
            a_rows, c_rows, v2_rows,
            mixed_localization_unique_large_gt=1,
            a_tp_to_c_fn_unique_large_gt=1,
            invariants={
                "raw_hash_equal": True,
                "cluster_hash_equal": True,
                "cluster_count_equal": True,
                "scores_equal": True,
                "classes_equal": True,
                "selected_cluster_ids_equal": True,
                "singleton_preservation": 1.0,
                "passed": True,
            },
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "unequal_lengths",
        "wrong_order",
        "missing_key",
        "duplicate_image_id",
        "gt_mismatch",
        "gain_mismatch",
        "a_local_source",
        "c_source_out_of_range",
        "boolean_query",
    ],
)
def test_guard_upper_bound_rejects_malformed_evidence_contract(mutation):
    count = 2 if mutation in {"wrong_order", "duplicate_image_id"} else 1
    a_rows, c_rows, v2_rows = _upper_bound_evidence(count)
    if mutation == "unequal_lengths":
        c_rows.clear()
    elif mutation == "wrong_order":
        c_rows.reverse()
    elif mutation == "missing_key":
        del v2_rows[0]["pred_source"]
    elif mutation == "duplicate_image_id":
        a_rows[1]["image_id"] = a_rows[0]["image_id"]
    elif mutation == "gt_mismatch":
        c_rows[0]["gt_boxes"] = [(1, 1, 121, 121)]
    elif mutation == "gain_mismatch":
        v2_rows[0]["effective_gain"] = 0.5
    elif mutation == "a_local_source":
        a_rows[0]["pred_source"] = [1]
    elif mutation == "c_source_out_of_range":
        c_rows[0] = _evidence_row("i0.jpg", predicts=True, source=5)
    elif mutation == "boolean_query":
        v2_rows[0]["pred_query"] = [True]

    with pytest.raises(ValueError):
        audit_module.evaluate_guard_upper_bound(
            a_rows, c_rows, v2_rows,
            mixed_localization_unique_large_gt=0,
            a_tp_to_c_fn_unique_large_gt=count,
            invariants={"passed": True},
        )


@pytest.mark.parametrize("bad_count", [True, -1, 1.0, float("nan")])
def test_guard_upper_bound_rejects_non_strict_counts(bad_count):
    with pytest.raises(ValueError):
        audit_module.evaluate_guard_upper_bound(
            (), (), (),
            mixed_localization_unique_large_gt=bad_count,
            a_tp_to_c_fn_unique_large_gt=1,
            invariants={"passed": True},
        )


def test_guard_upper_bound_rejects_mechanism_numerator_over_denominator():
    with pytest.raises(ValueError):
        audit_module.evaluate_guard_upper_bound(
            (), (), (),
            mixed_localization_unique_large_gt=2,
            a_tp_to_c_fn_unique_large_gt=1,
            invariants={"passed": True},
        )


def test_guard_upper_bound_rejects_nonfinite_metrics(monkeypatch):
    a_rows, c_rows, v2_rows = _upper_bound_evidence()
    _patch_guard_metrics(monkeypatch, 0.5, 0.4, float("nan"))

    with pytest.raises(ValueError):
        audit_module.evaluate_guard_upper_bound(
            a_rows, c_rows, v2_rows,
            mixed_localization_unique_large_gt=1,
            a_tp_to_c_fn_unique_large_gt=1,
            invariants={"passed": True},
        )
