from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.sqda_geometry_checkpoint_selection import (
    select_earliest_passing_candidate,
    select_earliest_feasible_candidate,
    select_trainable_candidates,
)
from src.sqda_sgc import SQDASGCAdapter


class _CheckpointCarrier(nn.Module):
    def __init__(self, fill: float) -> None:
        super().__init__()
        self.sqda_sgc = SQDASGCAdapter()
        with torch.no_grad():
            for parameter in self.sqda_sgc.geometry_agreement.parameters():
                parameter.fill_(fill)


def _save_checkpoint(path: Path, fill: float) -> None:
    torch.save({"model": _CheckpointCarrier(fill)}, path)


def test_updated_checkpoint_inventory_excludes_initial_best_payload(tmp_path: Path) -> None:
    _save_checkpoint(tmp_path / "epoch0.pt", 0.0)
    _save_checkpoint(tmp_path / "best.pt", 0.0)
    _save_checkpoint(tmp_path / "epoch1.pt", 1.0)
    _save_checkpoint(tmp_path / "last.pt", 2.0)

    assert select_trainable_candidates(tmp_path) == [
        tmp_path / "epoch1.pt",
        tmp_path / "last.pt",
    ]


def test_checkpoint_inventory_selects_the_earliest_strictly_passing_snapshot(
    tmp_path: Path,
) -> None:
    epoch1 = tmp_path / "epoch1.pt"
    epoch2 = tmp_path / "epoch2.pt"

    assert select_earliest_passing_candidate(
        [(epoch1, {"passed": False}), (epoch2, {"passed": True})]
    ) == epoch2
    assert select_earliest_passing_candidate([(epoch1, {"passed": False})]) is None


def test_checkpoint_inventory_selects_a_bounded_decline_for_g2_feasibility(
    tmp_path: Path,
) -> None:
    epoch1 = tmp_path / "epoch1.pt"
    epoch2 = tmp_path / "epoch2.pt"

    assert select_earliest_feasible_candidate(
        [
            (epoch1, {"passed": False, "g2_feasibility": {"eligible": False}}),
            (epoch2, {"passed": False, "g2_feasibility": {"eligible": True}}),
        ]
    ) == epoch2
