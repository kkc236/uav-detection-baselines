from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from scripts.derive_qg_p2_initial_state import derive_qg_initial_state
from src.ebc_qp_protocol import state_fingerprint


def _parent() -> dict:
    common = {"stock.weight": torch.tensor([[1.0, 2.0]])}
    innovation = {
        "head.p2_adapter.weight": torch.tensor([[3.0, 4.0]]),
        "head.p2_fusion_gamma": torch.tensor(1.0),
    }
    return {
        "format_version": 1,
        "common_state": common,
        "innovation_state": innovation,
        "metadata": {"seed": 0},
        "fingerprints": {
            "common": state_fingerprint(common),
            "innovation": state_fingerprint(innovation),
        },
    }


def test_qg_derivation_preserves_parent_tensors_and_adds_only_zero_quality_state():
    parent = _parent()
    before = deepcopy(parent)
    quality = {
        "head.p2_quality_head.weight": torch.zeros(1, 2),
        "head.p2_quality_head.bias": torch.zeros(1),
    }

    derived = derive_qg_initial_state(parent, quality, parent_sha256="ABC123")

    assert parent["fingerprints"] == before["fingerprints"]
    assert derived["fingerprints"]["common"] == parent["fingerprints"]["common"]
    for name, value in parent["innovation_state"].items():
        torch.testing.assert_close(derived["innovation_state"][name], value)
    assert set(derived["innovation_state"]) == set(parent["innovation_state"]) | set(quality)
    assert derived["metadata"]["parent_initial_state_sha256"] == "ABC123"
    assert derived["metadata"]["variant"] == "qg-p2-v1"


def test_qg_derivation_rejects_nonzero_or_nonquality_additions():
    with pytest.raises(ValueError, match="zero initialized"):
        derive_qg_initial_state(
            _parent(),
            {"head.p2_quality_head.weight": torch.ones(1, 2)},
            parent_sha256="ABC123",
        )
    with pytest.raises(ValueError, match="quality-head"):
        derive_qg_initial_state(
            _parent(),
            {"head.unapproved.weight": torch.zeros(1, 2)},
            parent_sha256="ABC123",
        )
