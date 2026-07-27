from types import MappingProxyType

import pytest
import torch

from src.gcte_types import (
    GCTENetworkOutput,
    GCTEStageOutput,
    QueryEvidence,
    ViewGeometry,
)


def _evidence(*, batch: int = 2, queries: int = 8, channels: int = 256) -> QueryEvidence:
    return QueryEvidence(
        queries=torch.zeros(batch, queries, channels),
        logits=torch.zeros(batch, queries, 10),
        boxes=torch.full((batch, queries, 4), 0.5),
        quality=torch.ones(batch, queries, 1),
    )


def test_query_evidence_accepts_matching_shapes():
    value = _evidence()

    assert value.batch_size == 2
    assert value.query_count == 8
    assert value.query_dim == 256
    assert value.num_classes == 10


def test_query_evidence_rejects_nonfinite_boxes():
    boxes = torch.zeros(1, 2, 4)
    boxes[0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        QueryEvidence(
            queries=torch.zeros(1, 2, 256),
            logits=torch.zeros(1, 2, 10),
            boxes=boxes,
            quality=torch.ones(1, 2, 1),
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("queries", torch.zeros(2, 8), r"queries must be \[B,Q,C\]"),
        ("logits", torch.zeros(2, 7, 10), r"logits must share \[B,Q\]"),
        ("boxes", torch.zeros(2, 8, 5), r"boxes must be normalized xywh"),
        ("quality", torch.zeros(2, 8), r"quality must be \[B,Q,1\]"),
    ),
)
def test_query_evidence_rejects_shape_mismatch(field, replacement, message):
    values = {
        "queries": torch.zeros(2, 8, 256),
        "logits": torch.zeros(2, 8, 10),
        "boxes": torch.full((2, 8, 4), 0.5),
        "quality": torch.ones(2, 8, 1),
    }
    values[field] = replacement

    with pytest.raises(ValueError, match=message):
        QueryEvidence(**values)


def test_query_evidence_requires_one_device_and_floating_dtype():
    with pytest.raises(ValueError, match="floating point"):
        QueryEvidence(
            queries=torch.zeros(1, 2, 4, dtype=torch.int64),
            logits=torch.zeros(1, 2, 3),
            boxes=torch.zeros(1, 2, 4),
            quality=torch.ones(1, 2, 1),
        )


def test_view_geometry_validates_per_query_contract():
    geometry = ViewGeometry(
        homography=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1),
        crop_metadata=torch.tensor(
            [[[0.0, 0.0, 0.6, 0.6, 1.0, 1.0], [0.4, 0.0, 0.6, 0.6, 1.0, 1.0]]]
        ),
        view_index=torch.tensor([[0, 1]], dtype=torch.long),
        valid_mask=torch.tensor([[True, True]]),
    )

    assert geometry.batch_size == 1
    assert geometry.query_count == 2


def test_view_geometry_rejects_singular_transform():
    with pytest.raises(ValueError, match="invertible"):
        ViewGeometry(
            homography=torch.zeros(1, 1, 3, 3),
            crop_metadata=torch.tensor(
                [[[0.0, 0.0, 0.6, 0.6, 1.0, 1.0]]]
            ),
            view_index=torch.tensor([[0]], dtype=torch.long),
            valid_mask=torch.tensor([[True]]),
        )


def test_stage_and_network_outputs_preserve_explicit_contracts():
    evidence = _evidence(batch=1, queries=2)
    stage = GCTEStageOutput(
        evidence=evidence,
        diagnostics=MappingProxyType({"delta": torch.zeros(1, 2, 1)}),
    )
    output = GCTENetworkOutput(
        unified_predictions=evidence,
        local_predictions=evidence,
        canonical_queries=evidence,
        gate_outputs=stage,
        losses=MappingProxyType({}),
        diagnostics=MappingProxyType({"enabled": torch.tensor(True)}),
    )

    assert output.gate_outputs is stage
    assert output.unified_predictions is evidence
