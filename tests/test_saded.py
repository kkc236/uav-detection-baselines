import inspect
import math

import pytest

from src import saded
from src.sbr_fusion import Detection


def test_saded_constants_are_frozen():
    assert saded.CONF_THRESHOLD == 0.001
    assert saded.MAX_DET == 300
    assert saded.TINY_EFFECTIVE_SIZE == 16.0
    assert saded.LARGE_EFFECTIVE_SIZE == 96.0
    assert saded.MATCH_IOU == 0.5
    assert saded.FRAGMENT_IOS == 0.5
    assert saded.ROUTER_K == math.log(9.0) / 8.0


def test_router_public_api_has_no_ground_truth_inputs():
    forbidden = {"gt", "target", "label", "annotation"}
    names = set(inspect.signature(saded.route_saded_image).parameters)
    assert not any(
        any(token in name.lower() for token in forbidden)
        for name in names
    )


def _candidate(
    box,
    *,
    class_id=0,
    source=0,
    query=0,
    index=0,
    score=0.5,
    image_id="image.jpg",
    global_box=True,
):
    return saded.ExpertCandidate(
        detection=Detection(
            box=box,
            global_xyxy=box if global_box else None,
            score=score,
            class_id=class_id,
            source_order=source,
            query_index=query,
        ),
        image_id=image_id,
        original_index=index,
    )


@pytest.mark.parametrize(
    ("size", "expected"),
    ((8.0, 0.9), (16.0, 0.5), (24.0, 0.1)),
)
def test_local_weight_has_analytic_transition_anchors(size, expected):
    assert saded.local_weight(size) == pytest.approx(expected)


def test_cross_expert_matching_is_strict_class_aware_and_one_to_one():
    baseline = (
        _candidate((0, 0, 10, 10), index=0),
        _candidate((20, 0, 30, 10), index=1),
        _candidate((40, 0, 43, 1), index=2),
    )
    local = (
        _candidate((1, 0, 11, 10), source=1, query=3, index=0),
        _candidate((20, 0, 30, 10), source=2, query=2, index=1),
        _candidate(
            (0, 0, 10, 10),
            class_id=1,
            source=3,
            query=1,
            index=2,
        ),
        _candidate((41, 0, 44, 1), source=4, query=0, index=3),
    )

    assert saded.match_cross_expert(baseline, local) == ((1, 1), (0, 0))


def test_cross_expert_matching_breaks_equal_iou_by_frozen_provenance():
    baseline = (_candidate((0, 0, 10, 10), index=0),)
    local = (
        _candidate((1, 0, 11, 10), source=2, query=0, index=1),
        _candidate((1, 0, 11, 10), source=1, query=9, index=2),
    )

    assert saded.match_cross_expert(baseline, local) == ((0, 1),)


def _route(baseline, local, *, width=640, height=640):
    return saded.route_saded_image(
        image_id="image.jpg",
        width=width,
        height=height,
        baseline=baseline,
        local_fused=local,
    )


def test_unmatched_baseline_is_retained():
    baseline = (_candidate((0, 0, 10, 10), score=0.4),)

    result = _route(baseline, ())

    assert result.predictions == (baseline[0].detection,)


def test_matched_non_tiny_baseline_is_immutable():
    baseline = (_candidate((0, 0, 20, 20), score=0.4),)
    local = (
        _candidate(
            (1, 1, 20, 20),
            source=1,
            query=2,
            score=0.9,
        ),
    )

    result = _route(baseline, local)

    assert result.predictions == (baseline[0].detection,)
    assert result.protected_baseline == (baseline[0].detection,)


def test_matched_tiny_uses_local_box_and_analytic_blended_score():
    baseline = (_candidate((0, 0, 10, 10), score=0.4),)
    local = (
        _candidate(
            (0, 0, 12, 12),
            source=1,
            query=2,
            score=0.9,
        ),
    )

    result = _route(baseline, local)

    prediction = result.predictions[0]
    alpha = saded.local_weight(10.0)
    assert prediction.box == local[0].detection.box
    assert prediction.global_xyxy == local[0].detection.global_xyxy
    assert prediction.score == pytest.approx(
        (1.0 - alpha) * 0.4 + alpha * 0.9
    )


def test_matched_local_non_tiny_cannot_replace_tiny_baseline():
    baseline = (_candidate((0, 0, 15, 15), score=0.4),)
    local = (
        _candidate(
            (0, 0, 18, 18),
            source=1,
            query=2,
            score=0.9,
        ),
    )

    result = _route(baseline, local)

    assert result.predictions == (baseline[0].detection,)
    assert result.invariants["no_local_non_tiny_leak"] is True


def test_unmatched_local_requires_tiny_size_and_complete_provenance():
    non_tiny = _candidate(
        (0, 0, 20, 20),
        source=1,
        query=1,
        score=0.9,
    )
    incomplete = _candidate(
        (30, 0, 40, 10),
        source=2,
        query=2,
        score=0.8,
        global_box=False,
    )

    result = _route((), (non_tiny, incomplete))

    assert result.predictions == ()


def test_unmatched_local_size_uses_the_actual_fused_output_box():
    local = saded.ExpertCandidate(
        detection=Detection(
            box=(0, 0, 20, 20),
            global_xyxy=(0, 0, 10, 10),
            score=0.9,
            class_id=0,
            source_order=1,
            query_index=1,
        ),
        image_id="image.jpg",
        original_index=0,
    )

    result = _route((), (local,))

    assert result.predictions == ()
    assert result.coverage["local_non_tiny_rejected"] == 1
    assert result.invariants["no_local_non_tiny_leak"] is True


@pytest.mark.parametrize(
    "actual_box",
    (
        (0, 0, 0, 10),
        (0, 0, math.nan, 10),
    ),
)
def test_candidate_rejects_invalid_actual_box_despite_valid_provenance(
    actual_box,
):
    local = saded.ExpertCandidate(
        detection=Detection(
            box=actual_box,
            global_xyxy=(0, 0, 10, 10),
            score=0.9,
            class_id=0,
            source_order=1,
            query_index=1,
        ),
        image_id="image.jpg",
        original_index=0,
    )

    result = _route((), (local,))

    assert result.predictions == ()
    assert result.coverage["incomplete_local_rejected"] == 1


def test_fragment_inside_protected_non_tiny_is_rejected():
    baseline = (_candidate((0, 0, 20, 20), score=0.7),)
    local = (
        _candidate(
            (2, 2, 12, 12),
            source=1,
            query=2,
            score=0.9,
        ),
    )

    result = _route(baseline, local)

    assert result.predictions == (baseline[0].detection,)
    assert result.coverage["fragment_rejected"] == 1


def test_protected_prefix_occupies_top300_before_tiny_candidates():
    protected = tuple(
        _candidate(
            (0, 0, 20, 20),
            query=i,
            index=i,
            score=0.1 + i / 1000,
        )
        for i in range(300)
    )
    local = (
        _candidate(
                (30, 30, 40, 40),
            source=1,
            query=0,
            index=0,
            score=0.99,
        ),
    )

    result = _route(protected, local)

    assert result.predictions == tuple(item.detection for item in protected)
    assert result.coverage["capacity_rejected"] == 1
    assert result.coverage["remaining_tiny_slots"] == 0


def test_effective_size_is_consistent_in_the_frozen_640_frame():
    first = _route(
        (),
        (_candidate((0, 0, 10, 10), source=1, score=0.9),),
        width=640,
        height=640,
    )
    second = _route(
        (),
        (_candidate((0, 0, 20, 20), source=1, score=0.9),),
        width=1280,
        height=1280,
    )

    assert len(first.predictions) == len(second.predictions) == 1
