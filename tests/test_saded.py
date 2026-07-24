import inspect
import math

from src import saded


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
