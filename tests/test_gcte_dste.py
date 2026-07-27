import torch

from src.gcte_dste import DetectionSupervisedTinyExpert
from src.gcte_types import QueryEvidence


def _make_query_evidence(
    *,
    batch: int = 2,
    queries: int = 12,
    query_dim: int = 256,
    num_classes: int = 10,
    requires_grad: bool = False,
) -> QueryEvidence:
    return QueryEvidence(
        queries=torch.randn(
            batch,
            queries,
            query_dim,
            requires_grad=requires_grad,
        ),
        logits=torch.randn(batch, queries, num_classes),
        boxes=torch.full((batch, queries, 4), 0.5),
        quality=torch.full((batch, queries, 1), 0.75),
    )


def test_dste_zero_initialization_is_identity():
    module = DetectionSupervisedTinyExpert(query_dim=256, num_classes=10)
    evidence = _make_query_evidence()

    output = module(evidence)

    torch.testing.assert_close(output.queries, evidence.queries)
    torch.testing.assert_close(output.logits, evidence.logits)
    torch.testing.assert_close(output.boxes, evidence.boxes)
    torch.testing.assert_close(output.quality, evidence.quality)


def test_dste_prediction_heads_receive_gradient():
    module = DetectionSupervisedTinyExpert(query_dim=256, num_classes=10)
    evidence = _make_query_evidence(batch=1, queries=4, requires_grad=True)

    output = module(evidence)
    (
        output.logits.sum()
        + output.boxes.sum()
        + output.quality.sum()
    ).backward()

    assert module.class_head.weight.grad is not None
    assert module.box_head.weight.grad is not None
    assert module.quality_head.weight.grad is not None


def test_dste_adapter_receives_gradient_from_query_supervision():
    module = DetectionSupervisedTinyExpert(query_dim=256, num_classes=10)
    evidence = _make_query_evidence(batch=1, queries=4)

    module(evidence).queries.square().mean().backward()

    assert module.adapter[-1].weight.grad is not None
    assert module.adapter[-1].weight.grad.abs().sum() > 0


def test_dste_query_and_prediction_residuals_are_bounded():
    module = DetectionSupervisedTinyExpert(
        query_dim=256,
        num_classes=10,
        residual_cap=0.2,
    )
    with torch.no_grad():
        for layer in (
            module.adapter[-1],
            module.class_head,
            module.box_head,
            module.quality_head,
        ):
            layer.weight.fill_(100.0)
            layer.bias.fill_(100.0)
    evidence = _make_query_evidence(batch=1, queries=4)

    output = module(evidence)

    assert (output.queries - evidence.queries).abs().max() <= 0.2 + 1e-6
    assert (output.logits - evidence.logits).abs().max() <= 0.2 + 1e-6
    assert (output.boxes - evidence.boxes).abs().max() <= 0.2 + 1e-6
    assert (output.quality - evidence.quality).abs().max() <= 0.2 + 1e-6


def test_dste_rejects_incompatible_evidence_dimensions():
    module = DetectionSupervisedTinyExpert(query_dim=32, num_classes=10)
    evidence = _make_query_evidence(query_dim=16)

    try:
        module(evidence)
    except ValueError as error:
        assert "query_dim" in str(error)
    else:
        raise AssertionError("incompatible evidence must fail closed")
