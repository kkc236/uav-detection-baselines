from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class CenterMatch:
    pairs: torch.Tensor
    unassigned_gt: torch.Tensor

    @property
    def covered_gt(self) -> torch.Tensor:
        return self.pairs[:, 0]


def normalized_center_cost(centers: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    scale = boxes[:, 2:].prod(-1).sqrt().clamp_min(1e-6)
    return torch.cdist(boxes[:, :2].float(), centers.float()) / scale[:, None]


def p2_grid_centers(height: int, width: int, device: torch.device) -> torch.Tensor:
    rows = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    columns = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)


def match_centers_inside_boxes(centers: torch.Tensor, boxes: torch.Tensor) -> CenterMatch:
    half = boxes[:, None, 2:] / 2
    delta = (centers[None] - boxes[:, None, :2]).abs()
    legal = (delta <= half).all(-1)
    costs = normalized_center_cost(centers, boxes)
    pairs = _lexicographic_assignment(costs, legal)
    return _build_match(pairs, len(boxes), boxes.device)


def assign_local_p2(
    height: int,
    width: int,
    boxes: torch.Tensor,
    valid_mask: torch.Tensor,
    radius: int = 1,
) -> CenterMatch:
    if valid_mask.numel() != height * width:
        raise ValueError("valid_mask must contain one value per P2 cell")

    centers = p2_grid_centers(height, width, boxes.device)
    cell_x = (boxes[:, 0] * width).floor().long().clamp(0, width - 1)
    cell_y = (boxes[:, 1] * height).floor().long().clamp(0, height - 1)
    legal = torch.zeros((len(boxes), height * width), dtype=torch.bool, device=boxes.device)
    valid_mask = valid_mask.reshape(-1).to(device=boxes.device, dtype=torch.bool)

    for gt_index, (x, y) in enumerate(zip(cell_x.tolist(), cell_y.tolist())):
        for row in range(max(0, y - radius), min(height, y + radius + 1)):
            for column in range(max(0, x - radius), min(width, x + radius + 1)):
                flat = row * width + column
                legal[gt_index, flat] = valid_mask[flat]

    costs = normalized_center_cost(centers, boxes)
    pairs = _lexicographic_assignment(costs, legal)
    return _build_match(pairs, len(boxes), boxes.device)


def _build_match(pairs: torch.Tensor, gt_count: int, device: torch.device) -> CenterMatch:
    covered = set(pairs[:, 0].tolist())
    unassigned = torch.tensor(
        [index for index in range(gt_count) if index not in covered],
        dtype=torch.long,
        device=device,
    )
    return CenterMatch(pairs=pairs, unassigned_gt=unassigned)


def _lexicographic_assignment(costs: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    if costs.shape != legal.shape:
        raise ValueError("costs and legal must have the same shape")

    gt_count, candidate_count = costs.shape
    if gt_count == 0:
        return torch.empty((0, 2), dtype=torch.long, device=costs.device)

    costs_cpu = costs.detach().double().cpu().numpy()
    legal_cpu = legal.detach().cpu().numpy().astype(bool, copy=False)
    all_gt = list(range(gt_count))
    all_candidates = list(range(candidate_count))
    target_matches, target_distance = _solve_objective(costs_cpu, legal_cpu, all_gt, all_candidates)

    used_candidates: set[int] = set()
    selected_pairs: list[tuple[int, int]] = []
    prefix_matches = 0
    prefix_distance = 0.0

    for gt_index in all_gt:
        available = [index for index in all_candidates if index not in used_candidates]
        legal_options = [index for index in available if legal_cpu[gt_index, index]]
        remaining_gt = list(range(gt_index + 1, gt_count))
        chosen: int | None | object = _NO_CHOICE

        for candidate in [*legal_options, None]:
            remaining_candidates = available if candidate is None else [i for i in available if i != candidate]
            remaining_matches, remaining_distance = _solve_objective(
                costs_cpu,
                legal_cpu,
                remaining_gt,
                remaining_candidates,
            )
            option_matches = prefix_matches + (candidate is not None) + remaining_matches
            option_distance = prefix_distance + remaining_distance
            if candidate is not None:
                option_distance += float(costs_cpu[gt_index, candidate])

            if option_matches == target_matches and abs(option_distance - target_distance) <= 1e-10:
                chosen = candidate
                break

        if chosen is _NO_CHOICE:
            raise RuntimeError("failed to construct deterministic optimal assignment")
        if chosen is not None:
            candidate_index = int(chosen)
            selected_pairs.append((gt_index, candidate_index))
            used_candidates.add(candidate_index)
            prefix_matches += 1
            prefix_distance += float(costs_cpu[gt_index, candidate_index])

    if not selected_pairs:
        return torch.empty((0, 2), dtype=torch.long, device=costs.device)
    return torch.tensor(selected_pairs, dtype=torch.long, device=costs.device)


def _solve_objective(
    costs: np.ndarray,
    legal: np.ndarray,
    gt_indices: list[int],
    candidate_indices: list[int],
) -> tuple[int, float]:
    row_count = len(gt_indices)
    if row_count == 0:
        return 0, 0.0

    column_count = len(candidate_indices)
    sub_costs = costs[np.ix_(gt_indices, candidate_indices)]
    sub_legal = legal[np.ix_(gt_indices, candidate_indices)]
    max_legal_cost = float(sub_costs[sub_legal].max()) if sub_legal.any() else 0.0
    penalty = (max_legal_cost + 1.0) * (row_count + 1)
    forbidden = 2.0 * penalty
    assignment_costs = np.full((row_count, column_count + row_count), penalty, dtype=np.float64)
    if column_count:
        assignment_costs[:, :column_count] = np.where(sub_legal, sub_costs, forbidden)

    rows, columns = linear_sum_assignment(assignment_costs)
    real = columns < column_count
    real_rows = rows[real]
    real_columns = columns[real]
    if real_columns.size:
        selected_legal = sub_legal[real_rows, real_columns]
        real_rows = real_rows[selected_legal]
        real_columns = real_columns[selected_legal]

    matched = int(real_columns.size)
    distance = float(sub_costs[real_rows, real_columns].sum()) if matched else 0.0
    return matched, distance


_NO_CHOICE = object()
