from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from scripts.derive_qg_p2_initial_state import derive_qg_initial_state, derive_qg_protocol_manifest
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


def test_qg_protocol_derivation_preserves_data_and_records_new_state_lineage(tmp_path):
    parent = {
        "format_version": 1,
        "seed": 0,
        "dataset": {"sha256": "DATA"},
        "subset": {"path": "/data/d2.txt", "sha256": "SUBSET"},
        "data": {"path": "/data/d2.yaml", "sha256": "YAML"},
        "initial_state": {"path": "/old/init.pt", "sha256": "PARENT"},
        "signature": "OLD-SIGNATURE",
    }

    manifest = derive_qg_protocol_manifest(
        parent,
        initial_state_path=tmp_path / "qg-init.pt",
        initial_state_sha256="QG-STATE",
        common_fingerprint="COMMON",
        git_commit="abc123",
    )

    assert manifest["data"] == parent["data"]
    assert manifest["subset"] == parent["subset"]
    assert manifest["dataset"] == parent["dataset"]
    assert manifest["initial_state"]["sha256"] == "QG-STATE"
    assert manifest["lineage"]["parent_protocol_signature"] == "OLD-SIGNATURE"
    assert manifest["lineage"]["parent_initial_state_sha256"] == "PARENT"
    assert manifest["common_fingerprint"] == "COMMON"
    assert manifest["git_commit"] == "abc123"
    assert manifest["signature"] != "OLD-SIGNATURE"
