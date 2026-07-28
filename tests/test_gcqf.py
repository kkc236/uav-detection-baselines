import pytest
import torch

from src.gcqf import GCQF
from src.gcte_types import QueryEvidence, ViewGeometry


def _evidence(
    *,
    batch: int = 1,
    queries: int = 4,
    query_dim: int = 32,
    num_classes: int = 3,
    boxes: torch.Tensor | None = None,
) -> QueryEvidence:
    return QueryEvidence(
        queries=torch.randn(batch, queries, query_dim),
        logits=torch.randn(batch, queries, num_classes),
        boxes=boxes if boxes is not None else torch.full((batch, queries, 4), 0.25),
        quality=torch.full((batch, queries, 1), 0.5),
    )


def _geometry(
    matrix: torch.Tensor,
    *,
    queries: int = 1,
    view_index: int = 0,
    requires_grad: bool = False,
) -> ViewGeometry:
    matrix = matrix.to(torch.float32).reshape(1, 1, 3, 3).repeat(1, queries, 1, 1)
    matrix.requires_grad_(requires_grad)
    metadata = torch.tensor(
        [[[0.0, 0.0, 0.6, 0.6, 1.0, 1.0]]],
        requires_grad=requires_grad,
    ).repeat(1, queries, 1)
    return ViewGeometry(
        homography=matrix,
        crop_metadata=metadata,
        view_index=torch.full((1, queries), view_index, dtype=torch.long),
        valid_mask=torch.ones(1, queries, dtype=torch.bool),
    )


def test_gcqf_registers_exactly_three_stages_with_sr_peg_third():
    module = GCQF(
        query_dim=32,
        num_classes=3,
        num_heads=4,
        num_views=4,
    )

    assert tuple(dict(module.named_children())) == (
        "geometry_projector",
        "query_interaction",
        "sr_peg",
    )


def test_geometry_projection_identity_and_zero_initialized_query_adapter():
    module = GCQF(
        query_dim=32,
        num_classes=3,
        num_heads=4,
        num_views=4,
    )
    local = _evidence(
        queries=1,
        boxes=torch.tensor([[[0.25, 0.75, 0.10, 0.20]]]),
    )

    output = module(
        _evidence(queries=3),
        local,
        _geometry(torch.eye(3)),
        anchor_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        residual_enabled=False,
    )

    torch.testing.assert_close(output.canonical_local.boxes, local.boxes)
    torch.testing.assert_close(
        output.canonical_local.queries,
        module.geometry_projector.output_norm(local.queries),
    )


@pytest.mark.parametrize(
    ("matrix", "expected"),
    (
        (
            torch.tensor(
                [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]
            ),
            (0.25, 0.25, 0.10, 0.10),
        ),
        (
            torch.tensor(
                [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]
            ),
            (0.75, 0.75, 0.10, 0.10),
        ),
        (
            torch.tensor(
                [[-0.5, 0.0, 1.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]
            ),
            (0.75, 0.25, 0.10, 0.10),
        ),
    ),
)
def test_geometry_projection_supports_crop_translation_and_flip(matrix, expected):
    module = GCQF(
        query_dim=32,
        num_classes=3,
        num_heads=4,
        num_views=4,
    )
    local = _evidence(
        queries=1,
        boxes=torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
    )

    output = module(
        _evidence(queries=3),
        local,
        _geometry(matrix),
        anchor_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        residual_enabled=False,
    )

    torch.testing.assert_close(
        output.canonical_local.boxes,
        torch.tensor([[expected]]),
        atol=1e-6,
        rtol=0,
    )


def test_explicit_residual_bypass_returns_original_score_tensor_bitwise():
    module = GCQF(
        query_dim=32,
        num_classes=3,
        num_heads=4,
        num_views=4,
    )
    global_evidence = _evidence(queries=3)
    local = _evidence(queries=2)

    output = module(
        global_evidence,
        local,
        _geometry(torch.eye(3), queries=2),
        anchor_mask=torch.ones(1, 2, 1, dtype=torch.bool),
        residual_enabled=False,
    )

    assert output.adjusted_local_scores is local.quality
    assert output.global_evidence is global_evidence
    assert torch.equal(output.adjusted_local_scores, local.quality)


def test_zero_initialized_enabled_residual_is_numerical_noop():
    module = GCQF(
        query_dim=32,
        num_classes=3,
        num_heads=4,
        num_views=4,
    )
    local = _evidence(queries=2)

    output = module(
        _evidence(queries=3),
        local,
        _geometry(torch.eye(3), queries=2),
        anchor_mask=torch.ones(1, 2, 1, dtype=torch.bool),
        residual_enabled=True,
    )

    torch.testing.assert_close(output.adjusted_local_scores, local.quality)
    torch.testing.assert_close(output.score_residual, torch.zeros_like(local.quality))


def test_residual_is_bounded_and_only_changes_anchor_eligible_candidates():
    module = GCQF(
        query_dim=32,
        num_classes=3,
        num_heads=4,
        num_views=4,
        residual_eta=0.2,
    )
    with torch.no_grad():
        module.sr_peg.score_residual_head.weight.fill_(100.0)
        module.sr_peg.score_residual_head.bias.fill_(100.0)
    local = _evidence(queries=2)
    mask = torch.tensor([[[True], [False]]])

    output = module(
        _evidence(queries=3),
        local,
        _geometry(torch.eye(3), queries=2),
        anchor_mask=mask,
        residual_enabled=True,
    )

    assert output.score_residual.abs().max() <= 1.0
    assert output.adjusted_local_scores[0, 0] > local.quality[0, 0]
    torch.testing.assert_close(
        output.adjusted_local_scores[0, 1],
        local.quality[0, 1],
    )


def test_trainable_stages_receive_gradients_but_frozen_evidence_and_geometry_do_not():
    module = GCQF(
        query_dim=32,
        num_classes=3,
        num_heads=4,
        num_views=4,
    )
    global_evidence = _evidence(queries=3)
    local = _evidence(queries=2)
    geometry = _geometry(torch.eye(3), queries=2, requires_grad=True)

    output = module(
        global_evidence,
        local,
        geometry,
        anchor_mask=torch.ones(1, 2, 1, dtype=torch.bool),
        residual_enabled=True,
    )
    (
        output.adjusted_local_scores.sum()
        + output.canonical_local.queries.square().mean()
        + output.global_context.square().mean()
        + output.tiny_utility_logits.sum()
        + output.non_tiny_risk_logits.sum()
        + output.global_retain_logits.sum()
    ).backward()

    assert module.geometry_projector.query_adapter[-1].weight.grad is not None
    assert module.query_interaction.attention.in_proj_weight.grad is not None
    assert module.sr_peg.score_residual_head.weight.grad is not None
    assert module.sr_peg.tiny_utility_head.weight.grad is not None
    assert module.sr_peg.non_tiny_risk_head.weight.grad is not None
    assert module.sr_peg.global_retain_head[-1].weight.grad is not None
    assert global_evidence.queries.grad is None
    assert local.queries.grad is None
    assert geometry.homography.grad is None


def test_gcqf_exposes_anchor_admission_as_third_stage_output():
    module = GCQF(
        query_dim=32,
        num_classes=3,
        num_heads=4,
        num_views=4,
    )
    output = module(
        _evidence(queries=3),
        _evidence(queries=2),
        _geometry(torch.eye(3), queries=2),
        anchor_mask=torch.tensor([[[True], [False]]]),
        residual_enabled=True,
    )

    assert output.anchor_admission_logits.shape == (1, 2, 1)
