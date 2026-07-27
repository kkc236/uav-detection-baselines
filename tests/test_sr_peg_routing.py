import torch
import pytest

from src.gcqf_cache import GCQFEvidenceRecord
from src.gcqf_routing import route_gcqf_record
from src.gcte_types import QueryEvidence
from src.gcte_views import build_frozen_view_geometry
from src.sr_peg_routing import SRPEGThresholds, route_sr_peg_record


def _evidence(query_count: int) -> QueryEvidence:
    return QueryEvidence(
        queries=torch.zeros(1, query_count, 8),
        logits=torch.zeros(1, query_count, 2),
        boxes=torch.full((1, query_count, 4), 0.25),
        quality=torch.full((1, query_count, 1), 0.5),
    )


def _record(
    *,
    global_size: float = 20.0,
    local_center: float = 0.7,
    local_class: int = 1,
) -> GCQFEvidenceRecord:
    postprocessed = torch.zeros(5, 300, 6)
    postprocessed[..., :4] = torch.tensor([0.5, 0.5, 0.01, 0.01])
    postprocessed[0, 0] = torch.tensor(
        [0.5, 0.5, global_size / 640, global_size / 640, 0.8, 0]
    )
    postprocessed[1, 0] = torch.tensor(
        [
            local_center,
            local_center,
            10 / 640,
            10 / 640,
            0.9,
            local_class,
        ]
    )
    return GCQFEvidenceRecord(
        image_id="val/a.jpg",
        global_evidence=_evidence(300),
        local_evidence=_evidence(1200),
        geometry=build_frozen_view_geometry(source_shapes=[(640, 640)]),
        anchor_mask=torch.ones(1, 1200, 1, dtype=torch.bool),
        quality_targets=torch.zeros(1, 1200, 1),
        equivariance_pairs=torch.empty(0, 2, dtype=torch.long),
        fixed_anchor_payload={
            "view_order": ("global", "TL", "TR", "BL", "BR"),
            "source_shape": [640, 640],
            "postprocessed": postprocessed,
            "selected_query_indices": torch.arange(300).repeat(5, 1),
        },
    )


def _route(
    record: GCQFEvidenceRecord,
    *,
    utility: float = 1.0,
    risk: float = 0.0,
    retain: float = 0.0,
):
    return route_sr_peg_record(
        record,
        score_residual=torch.zeros(1, 1200, 1),
        tiny_utility=torch.full((1, 1200, 1), utility),
        non_tiny_risk=torch.full((1, 1200, 1), risk),
        global_retain=torch.full((1, 300, 1), retain),
        thresholds=SRPEGThresholds(0.5, 0.5, 0.5),
        residual_enabled=True,
    )


def test_protected_global_rejects_class_conflicting_local_fragment():
    routed = _route(_record(local_class=1))

    assert routed.invariants["protected_identity_exact"]
    assert routed.invariants["no_class_conflicting_fragment"]
    assert routed.coverage["fragment_rejected"] >= 1
    assert len(routed.output) <= 300
    assert routed.output == routed.control


def test_small_global_can_be_learned_as_protected_non_tiny_evidence():
    routed = _route(_record(global_size=10.0), retain=1.0)

    assert routed.coverage["learned_protected_global"] == 1
    assert routed.output[0] == routed.control[0]
    assert routed.coverage["fragment_rejected"] >= 1


def test_local_is_rejected_by_utility_or_non_tiny_risk():
    away = _record(local_center=0.3)

    assert _route(away, utility=0.4).coverage["utility_rejected"] >= 1
    assert _route(away, risk=0.5).coverage["risk_rejected"] >= 1


def test_same_class_overlap_keeps_higher_stable_candidate():
    record = _record(global_size=10.0, local_class=0)
    routed = _route(record)

    assert len(routed.output) == 1
    assert routed.output[0].score == pytest.approx(0.9)
    assert routed.invariants["deterministic_tie_break"]


def test_missing_learned_outputs_restores_fixed_saded_exactly():
    record = _record(local_center=0.3)

    learned_off = route_sr_peg_record(record, learned_outputs=None)
    fixed = route_gcqf_record(record, score_residual=None)

    assert learned_off.output == fixed.output
    assert learned_off.control == fixed.control
    assert learned_off.invariants == fixed.invariants


def test_thresholds_fail_closed_outside_probability_range():
    try:
        SRPEGThresholds(0.5, 1.1, 0.5)
    except ValueError as error:
        assert "[0,1]" in str(error)
    else:
        raise AssertionError("invalid thresholds must fail closed")
