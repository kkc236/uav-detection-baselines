import torch

from src.gcqf import GCQF
from src.gcqf_cache import GCQFEvidenceRecord
from src.gcqf_training import (
    GCQF_BATCH_SIZE,
    GCQF_EPOCHS,
    GCQF_FIXED_AMP_SCALE,
    build_module_optimizer,
    collate_evidence_records,
)
from src.gcte_types import QueryEvidence, ViewGeometry


def _evidence(query_count: int) -> QueryEvidence:
    return QueryEvidence(
        queries=torch.zeros(1, query_count, 8),
        logits=torch.zeros(1, query_count, 3),
        boxes=torch.full((1, query_count, 4), 0.25),
        quality=torch.full((1, query_count, 1), 0.5),
    )


def _record(name: str, pair) -> GCQFEvidenceRecord:
    local_count = 1200
    return GCQFEvidenceRecord(
        image_id=name,
        global_evidence=_evidence(300),
        local_evidence=_evidence(local_count),
        geometry=ViewGeometry(
            homography=torch.eye(3)
            .reshape(1, 1, 3, 3)
            .repeat(1, local_count, 1, 1),
            crop_metadata=torch.tensor(
                [0.0, 0.0, 0.6, 0.6, 1.0, 1.0]
            )
            .reshape(1, 1, 6)
            .repeat(1, local_count, 1),
            view_index=torch.arange(4)
            .repeat_interleave(300)
            .reshape(1, local_count),
            valid_mask=torch.ones(1, local_count, dtype=torch.bool),
        ),
        anchor_mask=torch.ones(1, local_count, 1, dtype=torch.bool),
        quality_targets=torch.zeros(1, local_count, 1),
        equivariance_pairs=torch.tensor([pair], dtype=torch.long),
        fixed_anchor_payload={},
    )


def test_frozen_training_constants_match_screen_protocol():
    assert GCQF_BATCH_SIZE == 8
    assert GCQF_EPOCHS == 10
    assert GCQF_FIXED_AMP_SCALE == 128.0


def test_collate_adds_batch_indices_to_equivariance_pairs():
    batch = collate_evidence_records(
        [_record("a.jpg", (0, 300)), _record("b.jpg", (1, 301))]
    )

    assert batch.global_evidence.batch_size == 2
    assert batch.local_evidence.batch_size == 2
    assert batch.equivariance_pairs.tolist() == [
        [0, 0, 300],
        [1, 1, 301],
    ]
    assert batch.image_ids == ("a.jpg", "b.jpg")


class _RecordingOptimizer:
    def __init__(self, params, **kwargs):
        self.param_groups = params
        self.kwargs = kwargs


def test_module_optimizer_contains_every_and_only_gcqf_parameter():
    module = GCQF(
        query_dim=8,
        num_classes=3,
        num_heads=2,
        num_views=4,
    )

    optimizer = build_module_optimizer(
        module,
        optimizer_class=_RecordingOptimizer,
    )

    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected = {id(parameter) for parameter in module.parameters()}
    assert optimized == expected
    assert all(group["momentum"] == 0.937 for group in optimizer.param_groups)
    assert any(group["use_muon"] for group in optimizer.param_groups)
    assert all(
        parameter.ndim >= 2
        for group in optimizer.param_groups
        if group["use_muon"]
        for parameter in group["params"]
    )
