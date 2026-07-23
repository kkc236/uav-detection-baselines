from dataclasses import FrozenInstanceError, replace

import pytest
from src.sbr_fusion import Detection

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
