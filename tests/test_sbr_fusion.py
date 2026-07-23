import math

import pytest

from src.sbr_fusion import Detection, fuse_standard, greedy_ios_clusters, intersection_over_smaller


def det(box, score, cls=0, source=0, query=0, **kwargs):
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
