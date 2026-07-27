import torch

from src.gcqf import GCQF
from src.gcqf_cache import GCQFEvidenceRecord
from src.gcqf_training import (
    GCQF_BATCH_SIZE,
    GCQF_EPOCHS,
    GCQF_FIXED_AMP_SCALE,
    build_module_optimizer,
    collate_evidence_records,
    compute_positive_weights,
    split_seed0_records,
)
from src.gcte_types import QueryEvidence, ViewGeometry
from src.sr_peg_targets import SRPEGTargets


def _evidence(query_count: int) -> QueryEvidence:
    return QueryEvidence(
        queries=torch.zeros(1, query_count, 8),
        logits=torch.zeros(1, query_count, 3),
        boxes=torch.full((1, query_count, 4), 0.25),
        quality=torch.full((1, query_count, 1), 0.5),
    )


def _record(name: str, pair, *, supervised: bool = False) -> GCQFEvidenceRecord:
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
        sr_peg_targets=(
            SRPEGTargets(
                local_tiny_utility=torch.cat(
                    (
                        torch.ones(1, 1, 1),
                        torch.zeros(1, local_count - 1, 1),
                    ),
                    dim=1,
                ),
                local_non_tiny_risk=torch.cat(
                    (
                        torch.ones(1, 2, 1),
                        torch.zeros(1, local_count - 2, 1),
                    ),
                    dim=1,
                ),
                global_retain=torch.cat(
                    (
                        torch.ones(1, 3, 1),
                        torch.zeros(1, 297, 1),
                    ),
                    dim=1,
                ),
            )
            if supervised
            else None
        ),
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


def test_collate_requires_and_stacks_sr_peg_targets_when_requested():
    batch = collate_evidence_records(
        [
            _record("a.jpg", (0, 300), supervised=True),
            _record("b.jpg", (1, 301), supervised=True),
        ],
        require_sr_peg_targets=True,
    )

    assert batch.local_tiny_utility_targets is not None
    assert batch.local_tiny_utility_targets.shape == (2, 1200, 1)
    assert batch.local_non_tiny_risk_targets is not None
    assert batch.global_retain_targets is not None
    assert batch.global_retain_targets.shape == (2, 300, 1)


def test_collate_rejects_unsupervised_record_when_targets_required():
    try:
        collate_evidence_records(
            [_record("a.jpg", (0, 300))],
            require_sr_peg_targets=True,
        )
    except ValueError as error:
        assert "SR-PEG" in str(error)
    else:
        raise AssertionError("unsupervised records must fail closed")


def test_positive_weights_are_independent_and_clipped():
    records = [_record("a.jpg", (0, 300), supervised=True)]

    assert compute_positive_weights(records) == {
        "tiny": 20.0,
        "risk": 20.0,
        "retain": 20.0,
    }


def test_seed0_split_is_exact_stable_and_disjoint():
    ids = [f"train/{index:04d}.jpg" for index in range(647)]

    train, calibration = split_seed0_records(ids)

    assert len(train) == 518
    assert len(calibration) == 129
    assert set(train).isdisjoint(calibration)
    assert split_seed0_records(list(reversed(ids))) == (train, calibration)


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
