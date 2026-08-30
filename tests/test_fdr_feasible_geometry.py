from __future__ import annotations

import pytest
import torch
from torch import nn

from src.fdr_head import FDRDeformableTransformerDecoder
from src.fdr_math import project_feasible_fdr_distances


def test_projection_makes_pair_extents_positive_and_preserves_center() -> None:
    raw = torch.tensor([[-3.0, -2.5, -2.0, -3.0]], requires_grad=True)

    safe, stats = project_feasible_fdr_distances(raw, reg_scale=4.0)

    safe_x = 4.0 + safe[:, 0] + safe[:, 2]
    safe_y = 4.0 + safe[:, 1] + safe[:, 3]
    assert torch.all(safe_x > 0)
    assert torch.all(safe_y > 0)
    torch.testing.assert_close(safe_x, torch.full_like(safe_x, 1e-3), atol=1e-6, rtol=0)
    torch.testing.assert_close(safe_y, torch.full_like(safe_y, 1e-3), atol=1e-6, rtol=0)
    torch.testing.assert_close(safe[:, 2] - safe[:, 0], raw[:, 2] - raw[:, 0])
    torch.testing.assert_close(safe[:, 3] - safe[:, 1], raw[:, 3] - raw[:, 1])
    assert stats["horizontal_infeasible"].item() == 1
    assert stats["vertical_infeasible"].item() == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_projection_is_exact_identity_for_feasible_values(dtype: torch.dtype) -> None:
    raw = torch.tensor([[-1.0, -0.5, 0.25, 0.75]], dtype=dtype)

    safe, stats = project_feasible_fdr_distances(raw, reg_scale=4.0)

    assert torch.equal(safe, raw)
    assert safe.dtype == raw.dtype
    assert stats["horizontal_infeasible"].item() == 0
    assert stats["vertical_infeasible"].item() == 0


def test_projection_keeps_identity_gradient_when_extent_is_invalid() -> None:
    raw = torch.tensor([[-4.0, -4.0, -4.0, -4.0]], requires_grad=True)

    safe, _ = project_feasible_fdr_distances(raw, reg_scale=4.0)
    safe.sum().backward()

    torch.testing.assert_close(raw.grad, torch.ones_like(raw))


@pytest.mark.parametrize("minimum", [0.0, -1.0, float("nan"), float("inf")])
def test_projection_rejects_invalid_minimum_extent(minimum: float) -> None:
    with pytest.raises(ValueError, match="minimum_extent"):
        project_feasible_fdr_distances(
            torch.zeros(1, 4), reg_scale=4.0, minimum_extent=minimum
        )


def test_projection_rejects_wrong_edge_shape() -> None:
    with pytest.raises(ValueError, match="four FDR edges"):
        project_feasible_fdr_distances(torch.zeros(1, 3), reg_scale=4.0)


class _IdentityLayer(nn.Module):
    def forward(
        self,
        output: torch.Tensor,
        reference: torch.Tensor,
        feats: torch.Tensor,
        shapes: list,
        padding_mask: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        query_pos: torch.Tensor,
    ) -> torch.Tensor:
        del reference, feats, shapes, padding_mask, attn_mask, query_pos
        return output


class _StockDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_IdentityLayer() for _ in range(6)])
        self.hidden_dim = 8
        self.num_layers = 6
        self.eval_idx = 5


class _QueryPosition(nn.Module):
    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*reference.shape[:-1], 8), device=reference.device)


class _InvalidIntegral(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits[..., :4] * 0.0 - 4.0


def _invalid_extent_decoder() -> FDRDeformableTransformerDecoder:
    decoder = FDRDeformableTransformerDecoder.from_stock(
        _StockDecoder(), pre_bbox_head=nn.Linear(8, 4)
    )
    decoder.integral = _InvalidIntegral()
    return decoder


def _run_decoder(
    decoder: FDRDeformableTransformerDecoder,
) -> tuple[torch.Tensor, torch.Tensor]:
    embed = torch.zeros(1, 2, 8)
    reference_logits = torch.zeros(1, 2, 4)
    feats = torch.zeros(1, 2, 8)
    distribution_heads = nn.ModuleList([nn.Linear(8, 132) for _ in range(6)])
    score_heads = nn.ModuleList([nn.Linear(8, 3) for _ in range(6)])
    return decoder(
        embed,
        reference_logits,
        feats,
        [[1, 2]],
        distribution_heads,
        score_heads,
        _QueryPosition(),
    )


def test_training_and_eval_decode_use_projection_without_new_state_keys() -> None:
    decoder = _invalid_extent_decoder()
    before = tuple(decoder.state_dict())
    decoder.train()

    train_boxes, _ = _run_decoder(decoder)

    assert torch.all(train_boxes[..., 2:] > 0)
    assert decoder.last_geometry_statistics["horizontal_infeasible"].item() > 0
    assert decoder.last_geometry_statistics["vertical_infeasible"].item() > 0
    assert tuple(decoder.state_dict()) == before

    decoder.eval()
    with torch.no_grad():
        eval_boxes, _ = _run_decoder(decoder)

    assert torch.all(eval_boxes[..., 2:] > 0)
    assert decoder.last_geometry_statistics["horizontal_infeasible"].item() > 0
    assert tuple(decoder.state_dict()) == before
