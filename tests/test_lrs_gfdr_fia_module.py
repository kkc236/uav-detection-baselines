from __future__ import annotations

from pathlib import Path

import yaml

from src.fia import FIA
from src.rtdetr_lrs_system import (
    FIA_MODEL_INDEX,
    LRS_GFDR_FIA_CONFIG,
    LRSGFDRFIADetectionModel,
)


ROOT = Path(__file__).resolve().parents[1]


def test_lrs_gfdr_fia_config_keeps_geometry_and_excludes_bpdd() -> None:
    payload = yaml.safe_load(LRS_GFDR_FIA_CONFIG.read_text(encoding="utf-8"))
    decoder_options = payload["head"][-1][3][-1]

    assert LRS_GFDR_FIA_CONFIG == ROOT / "configs" / "rtdetr-l-lrs-gfdr-fia.yaml"
    assert decoder_options["feasible_geometry"] is True
    assert payload["fdr_loss"]["reliability_shrinkage_alpha"] == 0.25
    assert "bpdd_loss" not in payload
    assert sum(layer[2] == "FIA" for layer in payload["head"]) == 1


def test_lrs_gfdr_fia_model_has_only_the_p3_fia_branch() -> None:
    model = LRSGFDRFIADetectionModel(nc=10, verbose=False)

    assert len(model.model) == 30
    assert isinstance(model.model[FIA_MODEL_INDEX], FIA)
    assert model.fia.f == 21
    assert model.model[23].f == 21
    assert model.model[-1].f == [22, 25, 28]
    assert model.fia.residual_scale.item() == 0.0
    assert model.model[-1].decoder.feasible_geometry is True
    assert not hasattr(model, "bpdd_options")
    assert model.init_criterion().reliability_shrinkage_alpha == 0.25
