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
    candidate_indices = np.flatnonzero(legal_cpu.any(axis=0))
    if candidate_indices.size == 0:
        return torch.empty((0, 2), dtype=torch.long, device=costs.device)

    sub_costs = costs_cpu[:, candidate_indices]
    sub_legal = legal_cpu[:, candidate_indices]
    row_count, column_count = sub_costs.shape
    max_legal_cost = float(sub_costs[sub_legal].max()) if sub_legal.any() else 0.0
    penalty = (max_legal_cost + 1.0) * (row_count + 1)
    forbidden = 2.0 * penalty
    assignment_costs = np.full((row_count, column_count + row_count), penalty, dtype=np.float64)
    if column_count:
        assignment_costs[:, :column_count] = np.where(sub_legal, sub_costs, forbidden)

    tie_rank = np.concatenate((np.arange(column_count, dtype=np.float64), np.full(row_count, column_count)))
    row_weight = np.power(column_count + 1.0, -np.arange(row_count, dtype=np.float64))
    assignment_costs += 1e-12 * row_weight[:, None] * tie_rank[None, :]

    rows, columns = linear_sum_assignment(assignment_costs)
    real = columns < column_count
    real_rows = rows[real]
    real_columns = columns[real]
    if real_columns.size:
        selected_legal = sub_legal[real_rows, real_columns]
        real_rows = real_rows[selected_legal]
        real_columns = real_columns[selected_legal]

    if real_columns.size == 0:
        return torch.empty((0, 2), dtype=torch.long, device=costs.device)
    selected_candidates = candidate_indices[real_columns]
    pairs = np.column_stack((real_rows, selected_candidates))
    return torch.as_tensor(pairs, dtype=torch.long, device=costs.device)
