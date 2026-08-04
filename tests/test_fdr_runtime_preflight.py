from __future__ import annotations

import torch

import src.rtdetr_fdr as integration


def test_default_preflight_gate_symbols_are_callable() -> None:
    for name in (
        "run_f1_preflight",
        "run_f2_preflight",
        "run_f3_preflight",
        "run_f4_representation_preflight",
    ):
        assert callable(getattr(integration, name))


def test_adjacent_target_interpolation_reconstructs_continuous_distances() -> None:
    from src.fdr_runtime_preflight import interpolate_target_distances

    project = torch.tensor([-4.0, -1.0, 0.0, 2.0, 4.0])
    left = torch.tensor([0.0, 1.0, 2.0, 3.0])
    weight_right = torch.tensor([0.25, 0.5, 0.75, 1.0])
    weight_left = 1.0 - weight_right

    actual = interpolate_target_distances(
        project,
        left,
        weight_right,
        weight_left,
    )

    expected = torch.tensor([-3.25, -0.5, 1.5, 4.0])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_adjacent_target_interpolation_rejects_invalid_shapes_and_indices() -> None:
    from src.fdr_runtime_preflight import interpolate_target_distances

    project = torch.arange(5, dtype=torch.float32)
    with torch.no_grad():
        for left in (torch.tensor([[0.0]]), torch.tensor([4.0])):
            try:
                interpolate_target_distances(
                    project,
                    left,
                    torch.ones_like(left),
                    torch.zeros_like(left),
                )
            except ValueError:
                pass
            else:
                raise AssertionError("invalid FDR target interpolation was accepted")
