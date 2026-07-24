"""Pure prediction-only primitives for the frozen SADED router."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from src.sbr_fusion import Detection


CONF_THRESHOLD = 0.001
MAX_DET = 300
TINY_EFFECTIVE_SIZE = 16.0
LARGE_EFFECTIVE_SIZE = 96.0
MATCH_IOU = 0.5
FRAGMENT_IOS = 0.5
ROUTER_K = math.log(9.0) / 8.0


@dataclass(frozen=True)
class ExpertCandidate:
    detection: Detection
    image_id: str
    original_index: int


@dataclass(frozen=True)
class SADEDImageResult:
    predictions: tuple[Detection, ...]
    protected_baseline: tuple[Detection, ...]
    selected_matches: tuple[tuple[int, int], ...]
    coverage: Mapping[str, int]
    invariants: Mapping[str, bool]


def route_saded_image(
    *,
    image_id: str,
    width: int,
    height: int,
    baseline: Sequence[ExpertCandidate],
    local_fused: Sequence[ExpertCandidate],
) -> SADEDImageResult:
    """Route one image after both experts have emitted prediction candidates."""

    raise NotImplementedError
