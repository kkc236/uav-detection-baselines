from itertools import product

import torch

import src.ebc_qp_matching as matching
from src.ebc_qp_matching import assign_local_p2, match_centers_inside_boxes


def test_matching_maximizes_gt_coverage_before_distance():
    centers = torch.tensor([[0.20, 0.20], [0.28, 0.20]])
    boxes = torch.tensor([[0.24, 0.20, 0.10, 0.10], [0.28, 0.20, 0.04, 0.04]])

    result = match_centers_inside_boxes(centers, boxes)

    assert result.pairs.tolist() == [[0, 0], [1, 1]]


def test_equal_cost_tie_prefers_low_gt_then_low_token():
    centers = torch.tensor([[0.45, 0.50], [0.55, 0.50]])
    boxes = torch.tensor([[0.50, 0.50, 0.20, 0.20], [0.50, 0.50, 0.20, 0.20]])

    for _ in range(5):
        result = match_centers_inside_boxes(centers, boxes)
        assert result.pairs.tolist() == [[0, 0], [1, 1]]


def test_local_assignment_uses_unique_cells_and_reports_unassigned():
    result = assign_local_p2(
        height=4,
        width=4,
        boxes=torch.tensor([[0.375, 0.375, 0.05, 0.05], [0.375, 0.375, 0.05, 0.05]]),
        valid_mask=torch.ones(16, dtype=torch.bool),
        radius=1,
    )

    assert result.pairs.shape == (2, 2)
    assert result.pairs[:, 1].unique().numel() == 2
    assert result.unassigned_gt.numel() == 0


def test_empty_and_unreachable_inputs_return_typed_empty_tensors():
    result = match_centers_inside_boxes(
        torch.empty(0, 2),
        torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
    )

    assert result.pairs.shape == (0, 2)
    assert result.pairs.dtype == torch.long
    assert result.unassigned_gt.tolist() == [0]


def test_assignment_prunes_illegal_columns_and_solves_hungarian_once(monkeypatch):
    costs = torch.arange(300, dtype=torch.float32).reshape(3, 100)
    legal = torch.zeros_like(costs, dtype=torch.bool)
    legal[0, 95] = True
    legal[1, 96] = True
    legal[2, 95] = True
    calls = []
    original = matching.linear_sum_assignment

    def recording_solver(matrix):
        calls.append(matrix.shape)
        return original(matrix)

    monkeypatch.setattr(matching, "linear_sum_assignment", recording_solver)
    pairs = matching._lexicographic_assignment(costs, legal)

    assert pairs.tolist() == [[0, 95], [1, 96]]
    assert calls == [(3, 5)]


def test_tie_with_too_few_tokens_prefers_lower_gt_indices():
    centers = torch.tensor([[0.45, 0.50], [0.55, 0.50]])
    boxes = torch.tensor(
        [
            [0.50, 0.50, 0.20, 0.20],
            [0.50, 0.50, 0.20, 0.20],
            [0.50, 0.50, 0.20, 0.20],
        ]
    )

    result = match_centers_inside_boxes(centers, boxes)

    assert result.pairs.tolist() == [[0, 0], [1, 1]]
    assert result.unassigned_gt.tolist() == [2]


def test_single_solve_assignment_matches_brute_force_lexicographic_objective():
    generator = torch.Generator().manual_seed(11)
    for _ in range(20):
        costs = torch.randint(0, 5, (3, 4), generator=generator, dtype=torch.int64).float()
        legal = torch.rand((3, 4), generator=generator) > 0.45

        actual = matching._lexicographic_assignment(costs, legal).tolist()
        choices = [[index for index in range(4) if legal[row, index]] + [None] for row in range(3)]
        valid = (candidate_tuple for candidate_tuple in product(*choices) if _real_values_are_unique(candidate_tuple))
        best = min(valid, key=lambda value: _assignment_key(value, costs))
        expected = [[row, candidate] for row, candidate in enumerate(best) if candidate is not None]

        assert actual == expected


def _real_values_are_unique(values):
    real = [value for value in values if value is not None]
    return len(real) == len(set(real))


def _assignment_key(values, costs):
    real = [(row, candidate) for row, candidate in enumerate(values) if candidate is not None]
    distance = sum(float(costs[row, candidate]) for row, candidate in real)
    lexicographic = tuple(4 if candidate is None else candidate for candidate in values)
    return -len(real), distance, lexicographic
