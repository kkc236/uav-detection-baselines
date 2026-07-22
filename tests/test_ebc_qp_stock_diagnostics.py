import pytest
import torch

from src.ebc_qp_stock_diagnostics import StockQueryProbe, StockTinyQueryAccumulator


def test_stock_tiny_query_accumulator_uses_best_global_rank_and_missing_sentinel():
    accumulator = StockTinyQueryAccumulator(query_budget=1, tiny_radius=16.0)
    scores = torch.tensor([[0.1, 0.9, 0.8, 0.2]])
    centers = torch.tensor([[[0.5, 0.5], [0.1, 0.1], [0.51, 0.5], [0.9, 0.9]]])
    batch = {
        "img": torch.zeros(1, 3, 640, 640),
        "batch_idx": torch.tensor([0.0, 0.0]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.02, 0.02], [0.3, 0.3, 0.02, 0.02]]),
    }

    accumulator.update(scores, centers, batch)
    result = accumulator.compute()

    assert result["tiny_gt"] == 2
    assert result["stock_top300_coverage"] == 0.0
    assert result["best_rank_mean"] == pytest.approx(3.5)
    assert result["best_rank_median"] == pytest.approx(3.5)
    assert result["normalized_best_rank_mean"] == pytest.approx(0.875)
    assert result["candidate_count_values"] == [4]


def test_stock_query_probe_captures_exact_encoder_scores_and_anchor_centers():
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc_score_head = torch.nn.Linear(2, 3, bias=False)
            self.anchors = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]]])

    decoder = Decoder()
    probe = StockQueryProbe(decoder)
    features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    raw_scores = decoder.enc_score_head(features)
    scores, centers = probe.consume()

    torch.testing.assert_close(scores, raw_scores.max(-1).values)
    torch.testing.assert_close(centers, decoder.anchors.sigmoid()[..., :2])
    with pytest.raises(RuntimeError, match="did not observe"):
        probe.consume()
    probe.close()
