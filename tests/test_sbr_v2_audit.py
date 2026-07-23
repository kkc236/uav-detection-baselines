from dataclasses import FrozenInstanceError, replace

import pytest

from src.sbr_v2_audit import (
    AuditRawDetection,
    group_relevant_raw_rows,
    map_full_a_to_c,
    reconstruct_c_clusters,
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
