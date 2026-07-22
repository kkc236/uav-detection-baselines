import pytest
import torch

from src.ebc_qp_queries import (
    QuerySet,
    compete_queries,
    p2_diversity_statistics,
    quality_gated_ranking,
    replacement_statistics,
    stable_rank_indices,
)


def test_competition_keeps_budget_prefers_stock_on_ties_and_gathers_complete_records():
    stock = _make_query_set(scores=[0.9, 0.5, 0.4], source=0)
    p2 = _make_query_set(scores=[0.8, 0.5], source=1)

    mixed = compete_queries(stock, p2, budget=3)

    assert mixed.source[0].tolist() == [0, 1, 0]
    assert mixed.source_index[0].tolist() == [0, 0, 1]
    assert mixed.features[0, :, 0].tolist() == [0.0, 100.0, 1.0]
    assert mixed.boxes[0, :, 0].tolist() == [20.0, 120.0, 21.0]
    assert mixed.logits[0, :, 0].tolist() == [30.0, 130.0, 31.0]


def test_duplicate_p2_candidates_count_one_new_gt():
    stats = replacement_statistics(
        stock_centers=torch.tensor([[0.1, 0.1]]),
        final_centers=torch.tensor([[0.5, 0.5], [0.51, 0.5]]),
        gt_boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        tiny_mask=torch.tensor([True]),
    )

    assert stats.n_gain == 1
    assert stats.n_loss == 0
    assert stats.value == 1


def test_loss_counts_non_tiny_gt_removed_by_competition():
    stats = replacement_statistics(
        stock_centers=torch.tensor([[0.2, 0.2]]),
        final_centers=torch.tensor([[0.5, 0.5]]),
        gt_boxes=torch.tensor([[0.5, 0.5, 0.1, 0.1], [0.2, 0.2, 0.4, 0.4]]),
        tiny_mask=torch.tensor([True, False]),
    )

    assert stats.n_gain == 1
    assert stats.n_loss == 1
    assert stats.value == 0


def test_p2_diversity_separates_duplicates_from_background():
    stats = p2_diversity_statistics(
        p2_centers=torch.tensor([[0.49, 0.5], [0.51, 0.5], [0.9, 0.9]]),
        tiny_boxes=torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    )

    assert stats.foreground_at_50 == 2
    assert stats.unique_gt_at_50 == 1
    assert stats.duplicate_rate_at_50 == pytest.approx(0.5)
    assert stats.background_rate_at_50 == pytest.approx(1 / 3)


def test_stable_ranking_does_not_copy_scores_to_cpu():
    scores = torch.tensor([[0.5, 0.9, 0.5, 0.9]]).as_subclass(_NoCpuTensor)
    source = torch.tensor([[1, 1, 0, 0]])
    source_index = torch.tensor([[1, 0, 1, 0]])

    selected = stable_rank_indices(scores, source, source_index, k=4)

    assert selected.tolist() == [[3, 1, 2, 0]]


def test_quality_gated_ranking_uses_joint_class_quality_probability():
    class_logits = torch.tensor([[[2.0, -1.0], [0.0, 0.0]]])
    quality_logits = torch.tensor([[0.0, -2.0]])

    ranking = quality_gated_ranking(class_logits, quality_logits, epsilon=1e-6)
    expected_probability = class_logits.max(-1).values.sigmoid() * quality_logits.sigmoid()

    torch.testing.assert_close(ranking, torch.logit(expected_probability.clamp(1e-6, 1 - 1e-6)))
    assert ranking[0, 0] < class_logits[0, 0].max()
    assert ranking[0, 1] < ranking[0, 0]


def _make_query_set(scores: list[float], source: int) -> QuerySet:
    count = len(scores)
    offset = 100.0 * source
    index = torch.arange(count, dtype=torch.float32).reshape(1, count, 1)
    return QuerySet(
        features=torch.cat((index + offset, index + offset + 10), dim=-1),
        reference_logits=torch.cat((index + offset + 10,) * 4, dim=-1),
        boxes=torch.cat((index + offset + 20,) * 4, dim=-1),
        logits=torch.cat((index + offset + 30,) * 2, dim=-1),
        ranking_score=torch.tensor([scores]),
        centers=torch.cat((index / 10 + 0.1, index / 10 + 0.1), dim=-1),
        source=torch.full((1, count), source, dtype=torch.long),
        source_level=torch.full((1, count), 2 if source else 3, dtype=torch.long),
        source_index=torch.arange(count, dtype=torch.long).reshape(1, count),
    )


class _NoCpuTensor(torch.Tensor):
    def cpu(self, *args, **kwargs):
        raise AssertionError("stable ranking must remain on the training device")
