from __future__ import annotations

import inspect
from pathlib import Path

from src.rtdetr_btdse import BTDSEDetectionModel
from src.rtdetr_ioqc_sa import IOQCSADetectionModel
from src.rtdetr_vsf_rmr import VSFRMRDetectionModel


ROOT = Path(__file__).resolve().parents[1]
BTDSE_CONFIG = ROOT / "configs" / "rtdetr-l-btdse.yaml"


def module_names(model) -> set[str]:
    return {module.__class__.__name__ for module in model.modules()}


def test_vsf_rmr_model_is_stock_rtdetr_plus_only_vsf_innovation():
    model = VSFRMRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    names = module_names(model)
    source = inspect.getsource(__import__("src.rtdetr_vsf_rmr", fromlist=["*"]))

    assert "VSFRMR" in names
    assert "BTDSE" not in names
    assert not hasattr(model, "ioqc_probe")
    assert "src.btd_se" not in source
    assert "src.ioqc" not in source


def test_btdse_model_does_not_gain_vsf_or_ioqc_components():
    model = BTDSEDetectionModel(BTDSE_CONFIG, ch=3, nc=10, verbose=False)
    names = module_names(model)

    assert "BTDSE" in names
    assert "VSFRMR" not in names
    assert not hasattr(model, "ioqc_probe")


def test_ioqc_model_remains_stock_graph_with_probe_only():
    model = IOQCSADetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    names = module_names(model)

    assert "BTDSE" not in names
    assert "VSFRMR" not in names
    assert hasattr(model, "ioqc_probe")


def test_three_innovations_have_distinct_loss_contracts():
    vsf = VSFRMRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    btdse = BTDSEDetectionModel(BTDSE_CONFIG, ch=3, nc=10, verbose=False)
    ioqc = IOQCSADetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)

    assert tuple(vsf.loss_names[-2:]) == ("vsf_local_loss", "vsf_global_loss")
    assert tuple(btdse.loss_names[-2:]) == ("background_loss", "saliency_loss")
    assert tuple(ioqc.loss_names[-2:]) == ("ioqc_comp_loss", "ioqc_align_loss")

