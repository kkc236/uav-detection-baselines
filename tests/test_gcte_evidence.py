import torch

from src.gcte_evidence import (
    build_local_match_assignments,
    split_five_view_extraction,
)
from src.gcte_types import QueryEvidence
from src.rtdetr_gcqf import DecoderEvidenceExtraction


def _extraction() -> DecoderEvidenceExtraction:
    batch, queries, query_dim, classes = 5, 4, 8, 3
    logits = torch.zeros(batch, queries, classes)
    logits[..., 0] = 5.0
    evidence = QueryEvidence(
        queries=torch.arange(
            batch * queries * query_dim,
            dtype=torch.float32,
        ).reshape(batch, queries, query_dim),
        logits=logits,
        boxes=torch.full((batch, queries, 4), 0.1),
        quality=logits.sigmoid().amax(dim=-1, keepdim=True),
    )
    return DecoderEvidenceExtraction(
        evidence=evidence,
        postprocessed=torch.zeros(batch, queries, 6),
        selected_query_indices=torch.arange(queries).repeat(batch, 1),
    )


def test_split_five_view_extraction_preserves_view_major_query_order():
    result = split_five_view_extraction(
        _extraction(),
        source_shape=(500, 1000),
        queries_per_view=4,
    )

    assert result.global_evidence.queries.shape == (1, 4, 8)
    assert result.local_evidence.queries.shape == (1, 16, 8)
    torch.testing.assert_close(
        result.local_evidence.queries[0, :4],
        _extraction().evidence.queries[1],
    )
    torch.testing.assert_close(
        result.local_evidence.queries[0, 12:],
        _extraction().evidence.queries[4],
    )
    assert result.geometry.view_index.tolist() == [
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    ]


class _FirstQueryMatcher:
    def __call__(
        self,
        pred_boxes,
        pred_logits,
        gt_boxes,
        gt_classes,
        groups,
    ):
        del pred_boxes, pred_logits, gt_classes
        if groups == [0]:
            return [
                (
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long),
                )
            ]
        assert gt_boxes.shape[0] == groups[0]
        return [(torch.tensor([0]), torch.tensor([0]))]


def test_match_assignments_keep_original_gt_identity_across_views():
    result = split_five_view_extraction(
        _extraction(),
        source_shape=(500, 1000),
        queries_per_view=4,
    )
    gt_boxes = torch.tensor([[0.5, 0.5, 0.05, 0.05]])
    gt_classes = torch.tensor([0], dtype=torch.long)

    assignments = build_local_match_assignments(
        matcher=_FirstQueryMatcher(),
        local_evidence=result.local_evidence,
        geometry=result.geometry,
        gt_boxes=gt_boxes,
        gt_classes=gt_classes,
        queries_per_view=4,
    )

    assert assignments.tolist() == [
        0,
        -1,
        -1,
        -1,
        0,
        -1,
        -1,
        -1,
        0,
        -1,
        -1,
        -1,
        0,
        -1,
        -1,
        -1,
    ]
