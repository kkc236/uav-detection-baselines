from __future__ import annotations

import pytest
import torch

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
