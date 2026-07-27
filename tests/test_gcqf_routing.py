import torch

from src.gcqf_cache import GCQFEvidenceRecord
from src.gcqf_routing import (
    rescore_postprocessed,
    route_gcqf_record,
)
from src.gcte_types import QueryEvidence
from src.gcte_views import build_frozen_view_geometry


def _evidence(query_count: int) -> QueryEvidence:
    return QueryEvidence(
        queries=torch.zeros(1, query_count, 8),
        logits=torch.zeros(1, query_count, 2),
        boxes=torch.full((1, query_count, 4), 0.25),
        quality=torch.full((1, query_count, 1), 0.5),
    )


def _record() -> GCQFEvidenceRecord:
    postprocessed = torch.zeros(5, 300, 6)
    postprocessed[..., :4] = torch.tensor([0.5, 0.5, 0.01, 0.01])
    # One protected 20x20 global result.
    postprocessed[0, 0] = torch.tensor(
        [0.5, 0.5, 20 / 640, 20 / 640, 0.8, 0]
    )
    # One local tiny result away from the protected box.
    postprocessed[1, 0] = torch.tensor(
        [0.1, 0.1, 10 / 640, 10 / 640, 0.4, 1]
    )
    return GCQFEvidenceRecord(
        image_id="val/a.jpg",
        global_evidence=_evidence(300),
        local_evidence=_evidence(1200),
        geometry=build_frozen_view_geometry(
            source_shapes=[(640, 640)]
        ),
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


def test_residual_off_returns_exact_original_postprocessed_tensor():
    record = _record()
    original = record.fixed_anchor_payload["postprocessed"]

    output = rescore_postprocessed(record, score_residual=None)

    assert output is original
    assert torch.equal(output, original)


def test_residual_rescores_local_selected_query_but_never_global():
    record = _record()
    residual = torch.zeros(1, 1200, 1)
    residual[0, 0, 0] = 1.0

    output = rescore_postprocessed(record, score_residual=residual)

    assert output[0, 0, 4] == 0.8
    assert output[1, 0, 4] > 0.4
    assert output[2:, :, 4].sum() == 0


def test_route_preserves_global_non_tiny_prediction_exactly():
    record = _record()

    routed = route_gcqf_record(record, score_residual=None)

    assert routed.control[0].box == routed.output[0].box
    assert routed.control[0].score == routed.output[0].score
    assert routed.control[0].class_id == routed.output[0].class_id
    assert routed.invariants["protected_identity_exact"] is True
