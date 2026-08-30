from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.rtdetr_fdr_bpdd import _parse_bpdd_options


ROOT = Path(__file__).resolve().parents[1]
FDR = ROOT / "configs" / "rtdetr-l-fdr.yaml"
BPDD = ROOT / "configs" / "rtdetr-l-fdr-bpdd.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_bpdd_yaml_preserves_the_complete_fdr_graph_and_loss() -> None:
    fdr = _load(FDR)
    bpdd = _load(BPDD)
    bpdd_without_loss = deepcopy(bpdd)
    bpdd_options = bpdd_without_loss.pop("bpdd_loss")

    assert bpdd_without_loss == fdr
    assert bpdd_options == {
        "enabled": True,
        "weight": 0.5,
        "temperature": 0.5,
        "margin": 0.02,
        "eps": 1.0e-6,
        "matched_layer": "final",
        "include_dn": False,
    }


def test_bpdd_parser_rejects_two_assignment_authorities() -> None:
    with pytest.raises(ValueError, match="one authority"):
        _parse_bpdd_options(
            {
                "matched_layer": "final",
                "assignment_mode": "consistent",
            }
        )

