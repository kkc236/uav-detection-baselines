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
):
    return saded.ExpertCandidate(
        detection=Detection(
            box=box,
            global_xyxy=box,
            score=0.5,
            class_id=class_id,
            source_order=source,
            query_index=query,
        ),
        image_id="image.jpg",
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
