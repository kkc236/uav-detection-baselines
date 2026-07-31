from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

from ultralytics.nn.tasks import RTDETRDetectionModel

from scripts.benchmark_lpr import parameter_counts
from src.rtdetr_btdse import BTDSEDetectionModel
from src.rtdetr_ioqc_sa import IOQCSADetectionModel
from src.rtdetr_lpr import LPRRTDETRDetectionModel
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


def test_lpr_adds_parameters_only_inside_decoder_refiners() -> None:
    stock = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    lpr = LPRRTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=10, verbose=False)
    stock_names = set(stock.state_dict())
    lpr_names = set(lpr.state_dict())
    added = lpr_names - stock_names
    names = module_names(lpr)

    assert added
    assert all("decoder.lpr_refiners" in name for name in added)
    assert not (stock_names - lpr_names)
    assert names.isdisjoint({"BTDSE", "VSFRMR", "P3SamplingProbe", "NWD"})
    assert not hasattr(lpr, "ioqc_probe")

    stock_counts = parameter_counts(stock)
    lpr_counts = parameter_counts(lpr)
    added_parameter_count = sum(lpr.state_dict()[name].numel() for name in added)
    assert lpr_counts["total"] - stock_counts["total"] == added_parameter_count
    assert 100 * added_parameter_count / stock_counts["total"] < 1.0


def test_benchmark_script_runs_as_a_direct_cli() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "benchmark_lpr.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout

