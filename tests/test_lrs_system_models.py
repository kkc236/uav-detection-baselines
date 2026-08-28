from __future__ import annotations

from pathlib import Path

import yaml

from src.rtdetr_fdr_bpdd import FDRBPDDDetectionModel


ROOT = Path(__file__).resolve().parents[1]


def test_bpdd_criterion_preserves_lrs_alpha() -> None:
    with (ROOT / "configs" / "rtdetr-l-lrs-fdr.yaml").open(
        encoding="utf-8"
    ) as stream:
        payload = yaml.safe_load(stream)
    payload["bpdd_loss"] = {
        "enabled": True,
        "weight": 0.5,
        "temperature": 0.5,
        "margin": 0.02,
        "eps": 1e-6,
        "matched_layer": "final",
        "include_dn": False,
    }

    model = FDRBPDDDetectionModel(payload, nc=10, verbose=False)
    criterion = model.init_criterion()

    assert criterion.reliability_shrinkage_alpha == 0.25
    assert criterion.supervise_dn_fdr is False
