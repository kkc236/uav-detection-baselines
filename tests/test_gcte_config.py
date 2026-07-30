from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = ROOT / "configs" / "rtdetr-l-acr-eg.yaml"


def test_gcte_yaml_freezes_single_gcqf_module_and_protocol():
    payload = yaml.safe_load(
        FORMAL_CONFIG.read_text(encoding="utf-8")
    )

    # Keep the stock COCO graph declaration; the VisDrone dataset overrides
    # detector nc to 10 at model construction time.
    assert payload["nc"] == 80
    assert payload["gcte"] == {
        "enabled": True,
        "forward_integration": True,
        "query_dim": 256,
        "num_classes": 10,
        "num_heads": 8,
        "num_views": 4,
        "residual_eta": 0.2,
        "residual_enabled": True,
        "acr_eg_off": False,
        "gcte_off": False,
    }


def test_gcte_yaml_preserves_stock_rtdetr_l_graph():
    payload = yaml.safe_load(
        FORMAL_CONFIG.read_text(encoding="utf-8")
    )
    reference = yaml.safe_load(
        (ROOT / "configs" / "rtdetr-l-gcmv-plec.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert payload["backbone"] == reference["backbone"]
    assert payload["head"] == reference["head"]
    assert payload["head"][-1][2] == "RTDETRDecoder"
