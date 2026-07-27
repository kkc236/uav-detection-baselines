import pytest
import torch
from torch import nn

from src.rtdetr_gcqf import (
    extract_decoder_query_evidence,
    freeze_detector,
    stock_postprocess_with_query_indices,
)


class _FakeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nc = 3
        self.num_queries = 4
        self.decoder = nn.Module()
        self.decoder.eval_idx = 1
        self.decoder.layers = nn.ModuleList(
            [nn.Linear(8, 8), nn.Linear(8, 8)]
        )

    def postprocess(self, boxes, scores):
        scores, index = scores.flatten(1).topk(self.num_queries)
        query_idx = torch.div(index, self.nc, rounding_mode="floor")
        boxes = boxes.gather(
            1,
            query_idx.unsqueeze(-1).expand(-1, -1, 4),
        )
        classes = index - query_idx * self.nc
        return torch.cat(
            (boxes, scores.unsqueeze(-1), classes.unsqueeze(-1).float()),
            dim=-1,
        )


class _FakeDetector(nn.Module):
    def __init__(self, *, mutate_bn: bool = False) -> None:
        super().__init__()
        self.stem = nn.Linear(8, 8)
        self.bn = nn.BatchNorm1d(8)
        self.head = _FakeHead()
        self.model = nn.ModuleList([self.stem, self.head])
        self.mutate_bn = mutate_bn

    def predict(self, images, batch=None):
        del batch
        query = self.stem(images)
        for layer in self.head.decoder.layers:
            query = layer(query)
        boxes = torch.sigmoid(query[..., :4])
        logits = query[..., :3]
        if self.mutate_bn:
            self.bn.num_batches_tracked.add_(1)
        dec_boxes = boxes.unsqueeze(0)
        dec_logits = logits.unsqueeze(0)
        auxiliary = (
            dec_boxes,
            dec_logits,
            boxes,
            logits,
            None,
        )
        prediction = self.head.postprocess(boxes, logits.sigmoid())
        return prediction, auxiliary


def test_stock_postprocess_recovers_decoder_query_indices_exactly():
    boxes = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
            ]
        ]
    )

    postprocessed, query_indices = stock_postprocess_with_query_indices(
        boxes,
        logits,
        num_queries=4,
    )
    expected = _FakeHead().postprocess(boxes, logits.sigmoid())

    assert torch.equal(postprocessed, expected)
    assert query_indices.tolist() == [[3, 3, 3, 2]]


def test_query_extraction_is_read_only_detached_and_matches_stock_output():
    model = freeze_detector(_FakeDetector())
    images = torch.randn(2, 4, 8)

    result = extract_decoder_query_evidence(
        model,
        images,
        expected_query_count=4,
    )

    assert result.evidence.queries.shape == (2, 4, 8)
    assert result.evidence.logits.shape == (2, 4, 3)
    assert result.evidence.queries.requires_grad is False
    assert result.evidence.boxes.requires_grad is False
    assert torch.equal(
        result.postprocessed,
        model.predict(images)[0],
    )
    expected_quality = result.evidence.logits.sigmoid().amax(
        dim=-1,
        keepdim=True,
    )
    torch.testing.assert_close(result.evidence.quality, expected_quality)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model.training is False


def test_extraction_captures_the_configured_final_decoder_layer():
    model = freeze_detector(_FakeDetector())
    images = torch.randn(1, 4, 8)
    expected = model.stem(images)
    expected = model.head.decoder.layers[0](expected)
    expected = model.head.decoder.layers[1](expected)

    result = extract_decoder_query_evidence(
        model,
        images,
        expected_query_count=4,
    )

    torch.testing.assert_close(result.evidence.queries, expected)


def test_extraction_fails_closed_on_batchnorm_buffer_mutation():
    model = freeze_detector(_FakeDetector(mutate_bn=True))

    with pytest.raises(RuntimeError, match="BatchNorm"):
        extract_decoder_query_evidence(
            model,
            torch.randn(1, 4, 8),
            expected_query_count=4,
        )


def test_extraction_requires_fully_frozen_eval_detector():
    model = _FakeDetector()

    with pytest.raises(RuntimeError, match="eval"):
        extract_decoder_query_evidence(
            model,
            torch.randn(1, 4, 8),
            expected_query_count=4,
        )
