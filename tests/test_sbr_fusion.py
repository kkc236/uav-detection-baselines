import math

import numpy as np
import pytest

from src.sbr_fusion import (
    Detection,
    border_reliability,
    fuse_sp_brf,
    fuse_sp_brf_from_clusters,
    fuse_standard,
    greedy_ios_clusters,
    intersection_over_smaller,
)
from src.sbr_geometry import Tile


def det(box, score, cls=0, source=0, query=0, **kwargs):
    cls = kwargs.pop("class_id", cls)
    source = kwargs.pop("source_order", kwargs.pop("source", source))
    query = kwargs.pop("query_index", kwargs.pop("query", query))
    return Detection(
        box=box,
        score=score,
        class_id=cls,
        source_order=source,
        query_index=query,
        **kwargs,
    )


def test_detection_is_immutable_and_preserves_provenance():
    d = det((1, 2, 3, 4), 0.9, source=2, query=7, view_xyxy=(2, 3, 4, 5), global_xyxy=(1, 2, 3, 4))
    assert d.box == (1.0, 2.0, 3.0, 4.0)
    assert d.view_xyxy == (2.0, 3.0, 4.0, 5.0)
    assert d.global_xyxy == (1.0, 2.0, 3.0, 4.0)
    with pytest.raises((AttributeError, TypeError)):
        d.score = 0.1


def test_ios_uses_smaller_area_not_iou():
    assert intersection_over_smaller((0, 0, 10, 10), (0, 0, 20, 20)) == pytest.approx(1.0)


def test_ios_exact_half_does_not_match():
    a = det((0, 0, 10, 10), 0.9)
    b = det((5, 0, 15, 10), 0.8)
    assert intersection_over_smaller(a.box, b.box) == pytest.approx(0.5)
    assert greedy_ios_clusters([a, b]) == ((a,), (b,))


def test_cross_class_boxes_never_match():
    a = det((0, 0, 10, 10), 0.9, cls=1)
    b = det((1, 1, 9, 9), 0.8, cls=2)
    assert greedy_ios_clusters([a, b]) == ((a,), (b,))


def test_seed_order_and_nontransitive_seed_only_matching():
    seed = det((0, 0, 10, 10), 0.9, source=1, query=5)
    absorbed = det((1, 1, 9, 9), 0.8, source=0, query=1)
    bridge = det((8, 0, 18, 10), 0.7, source=0, query=2)
    clusters = greedy_ios_clusters([bridge, absorbed, seed])
    assert clusters == ((seed, absorbed), (bridge,))


def test_equal_scores_use_source_then_query_tie_break():
    a = det((0, 0, 2, 2), 0.5, source=2, query=1)
    b = det((0, 0, 2, 2), 0.5, source=1, query=9)
    assert greedy_ios_clusters([a, b])[0][0] is b


def test_standard_fusion_is_score_weighted_and_uses_seed_class_and_max_score():
    a = det((0, 0, 10, 10), 0.8, cls=3, source=0, query=0)
    b = det((2, 2, 12, 12), 0.4, cls=3, source=1, query=1)
    out = fuse_standard([a, b])
    assert len(out) == 1
    assert out[0].box == pytest.approx((2 / 3, 2 / 3, 32 / 3, 32 / 3))
    assert out[0].score == pytest.approx(0.8)
    assert out[0].class_id == 3
    assert tuple(out[0].members) == (a, b)


def test_singletons_are_identity_and_final_sort_is_stable_with_max_det():
    detections = [det((i, 0, i + 1, 1), 0.1, source=i // 2, query=i) for i in range(301)]
    out = fuse_standard(detections)
    assert len(out) == 300
    assert all(x is y for x, y in zip(out, detections[:300]))


def test_nonfinite_or_invalid_inputs_fail_closed():
    good = det((0, 0, 2, 2), 0.9)
    bad_score = det((0, 0, 2, 2), math.nan)
    bad_box = det((0, 0, math.nan, 2), 0.8)
    assert fuse_standard([good, bad_score, bad_box]) == (good,)
    assert intersection_over_smaller((0, 0, 1, 1), (0, 0, math.nan, 1)) == 0.0


def test_score_types_fail_closed_without_rejecting_real_numeric_scores():
    integer_score = det((0, 0, 2, 2), 1)
    numpy_score = det((3, 0, 5, 2), np.float32(0.5))
    boolean_score = det((6, 0, 8, 2), True)
    string_score = det((9, 0, 11, 2), "0.8")

    assert fuse_standard(
        [integer_score, numpy_score, boolean_score, string_score]
    ) == (integer_score, numpy_score)


def test_protocol_rejects_nonfrozen_threshold_and_max_det():
    d = det((0, 0, 2, 2), 0.9)
    with pytest.raises(ValueError):
        greedy_ios_clusters([d], ios_threshold=0.4)
    with pytest.raises(ValueError):
        fuse_standard([d], max_det=10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"view_xyxy": (0, 0, math.nan, 1)},
        {"global_xyxy": (2, 0, 1, 1)},
        {"network_xyxy": (2, 2, 1, 3)},
        {"tile_local_box": (0, 0, math.inf, 1)},
        {"global_box": (0, 0, 0, 1)},
        {"class_id": "bad"},
        {"source": "bad"},
        {"query": "bad"},
    ],
)
def test_invalid_provenance_or_indices_are_dropped(kwargs):
    good = det((0, 0, 2, 2), 0.9)
    bad = det((0, 0, 2, 2), 0.8, **kwargs)
    assert fuse_standard([good, bad]) == (good,)


def test_sp_brf_full_view_reliability_is_one():
    full = det((0, 0, 10, 10), 0.8)
    assert border_reliability(full, None, (100, 100)) == pytest.approx(1.0)


def test_sp_brf_each_internal_edge_uses_exact_overlap_over_two():
    tile = Tile(20, 20, 80, 80, 1)
    # 60x60 tile over 100x100 image gives exact 20px overlap, denominator 10.
    assert border_reliability(
        det((0, 0, 10, 10), 0.8, tile_local_box=(2, 5, 58, 55), tile_bounds=tile),
        tile,
        (100, 100),
    ) == pytest.approx(0.2)
    assert border_reliability(
        det((0, 0, 10, 10), 0.8, tile_local_box=(2, 5, 50, 55), tile_bounds=tile),
        tile,
        (100, 100),
    ) == pytest.approx(0.2)
    assert border_reliability(
        det((0, 0, 10, 10), 0.8, tile_local_box=(25, 1, 50, 55), tile_bounds=tile),
        tile,
        (100, 100),
    ) == pytest.approx(0.1)
    assert border_reliability(
        det((0, 0, 10, 10), 0.8, tile_local_box=(25, 5, 50, 59), tile_bounds=tile),
        tile,
        (100, 100),
    ) == pytest.approx(0.1)


def test_sp_brf_ignores_real_image_edges_and_takes_minimum_across_edges():
    tile = Tile(0, 0, 60, 60, 0)
    # Left/top are real boundaries; right/bottom are artificial. Minimum is
    # the bottom reliability (2 / (40 / 2) = 0.1).
    local = det(
        (0, 0, 10, 10),
        0.8,
        tile_local_box=(0, 0, 58, 58),
        tile_bounds=tile,
    )
    assert border_reliability(local, tile, (80, 80)) == pytest.approx(0.1)


def test_sp_brf_reliability_has_no_applicable_edge_fallback_and_is_clipped():
    tile = Tile(0, 0, 100, 100, 0)
    local = det((0, 0, 2, 2), 0.8, tile_local_box=(0, 0, 100, 100), tile_bounds=tile)
    assert border_reliability(local, tile, (100, 100)) == pytest.approx(1.0)


def test_sp_brf_rejects_missing_local_metadata_and_zero_overlap():
    tile = Tile(20, 0, 80, 100, 1)
    missing = det((0, 0, 2, 2), 0.8)
    with pytest.raises(ValueError):
        border_reliability(missing, tile, (100, 100))
    zero_overlap_tile = Tile(50, 0, 100, 100, 1)
    local = det(
        (0, 0, 2, 2),
        0.8,
        tile_local_box=(0, 0, 50, 100),
        tile_bounds=zero_overlap_tile,
    )
    with pytest.raises(ValueError):
        border_reliability(local, zero_overlap_tile, (100, 100))


def test_sp_brf_two_and_three_member_weighted_coordinates_and_seed_class():
    tile = Tile(0, 0, 60, 100, 0)
    a = det((0, 0, 10, 10), 0.8, cls=3, tile_local_box=(10, 10, 30, 30), tile_bounds=tile)
    b = det((2, 2, 12, 12), 0.4, cls=3, tile_local_box=(58, 10, 60, 30), tile_bounds=tile)
    c = det((4, 4, 14, 14), 0.2, cls=3, tile_local_box=(30, 10, 50, 30), tile_bounds=tile)
    # Tile overlap is 20, so b has right reliability 0 and a/c have r=1.
    out = fuse_sp_brf((a, b, c), full_shape=(100, 100))
    wa, wb, wc = 0.8 * 2.0, 0.4 * 1.0, 0.2 * 2.0
    total = wa + wb + wc
    assert out.box == pytest.approx(tuple(
        (wa * a.box[i] + wb * b.box[i] + wc * c.box[i]) / total for i in range(4)
    ))
    assert out.score == pytest.approx(0.8)
    assert out.class_id == 3
    assert out.members == (a, b, c)


def test_sp_brf_singleton_is_identity_and_clusters_match_standard():
    a = det((0, 0, 10, 10), 0.8, source=1)
    b = det((1, 1, 9, 9), 0.7, source=2)
    standard_clusters = greedy_ios_clusters([a, b])
    sp = fuse_sp_brf_from_clusters(standard_clusters, full_shape=(100, 100))
    assert sp[0].members == (a, b)
    singleton = det((20, 20, 30, 30), 0.2, source=4)
    assert fuse_sp_brf((singleton,), full_shape=(100, 100)) is singleton


def test_sp_brf_from_clusters_is_deterministic_and_capped_at_300():
    detections = [det((i, 0, i + 1, 1), 0.1, source=i // 2, query=i) for i in range(301)]
    clusters = greedy_ios_clusters(detections)
    out = fuse_sp_brf_from_clusters(clusters, full_shape=(1000, 10))
    assert len(out) == 300
    assert all(x is y for x, y in zip(out, detections[:300]))


def test_negative_tile_index_is_invalid():
    good = det((0, 0, 2, 2), 0.9)
    bad = det((0, 0, 2, 2), 0.8, tile_index=-1)
    assert fuse_standard([good, bad]) == (good,)


def test_sp_brf_rejects_tile_and_local_box_outside_declared_geometry():
    local = det(
        (0, 0, 2, 2),
        0.8,
        tile_local_box=(0, 0, 60, 60),
        tile_bounds=Tile(0, 0, 60, 60, 0),
    )
    with pytest.raises(ValueError):
        border_reliability(local, Tile(0, 0, 60, 60, 0), (50, 50))
    with pytest.raises(ValueError):
        border_reliability(
            det(
                (0, 0, 2, 2),
                0.8,
                tile_local_box=(-1, 0, 10, 10),
                tile_bounds=Tile(0, 0, 60, 60, 0),
            ),
            Tile(0, 0, 60, 60, 0),
            (100, 100),
        )
    with pytest.raises(ValueError):
        border_reliability(
            det(
                (0, 0, 2, 2),
                0.8,
                tile_local_box=(0, 0, 61, 10),
                tile_bounds=Tile(0, 0, 60, 60, 0),
            ),
            Tile(0, 0, 60, 60, 0),
            (100, 100),
        )


def test_sp_brf_validates_full_shape_and_singleton_before_identity():
    singleton = det((0, 0, 2, 2), 0.8)
    with pytest.raises(ValueError):
        fuse_sp_brf((singleton,), full_shape=(0, 100))
    invalid = det((0, 0, 2, 2), math.nan)
    with pytest.raises(ValueError):
        fuse_sp_brf((invalid,), full_shape=(100, 100))


def test_sp_brf_rejects_mixed_class_precomputed_cluster():
    a = det((0, 0, 10, 10), 0.8, cls=0)
    b = det((1, 1, 9, 9), 0.7, cls=1)
    with pytest.raises(ValueError):
        fuse_sp_brf((a, b), full_shape=(100, 100))
