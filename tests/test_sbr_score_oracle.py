import math
from dataclasses import replace

from src.sbr_score_oracle import (
    apply_group_overlay,
    find_aggressor_groups,
    replay_overlay,
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
