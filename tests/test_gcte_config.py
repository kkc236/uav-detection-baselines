from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_gcte_yaml_freezes_single_gcqf_module_and_protocol():
    payload = yaml.safe_load(
        (ROOT / "configs" / "rtdetr-l-gcte.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert payload["nc"] == 10
    assert payload["gcte"] == {
        "enabled": True,
        "module": "GCQF",
        "integration": "decoder-output-wrapper",
        "num_local_views": 4,
        "queries_per_view": 300,
        "query_dim": 256,
        "num_heads": 8,
        "tile_ratio": 0.6,
        "residual_cap": 0.2,
        "residual_eta": 0.2,
        "stages": [
            "GeometryQueryProjector",
            "GlobalLocalQueryInteraction",
            "AnchorConditionedResidualEvidenceGate",
        ],
        "sr_peg": {
            "local_trunk": [770, 256, 256],
            "global_box_embedding": 64,
            "tiny_utility_head": True,
            "non_tiny_risk_head": True,
            "anchor_admission_head": True,
            "global_retain_head": True,
        },
        "protect_global_non_tiny": True,
        "exact_anchor_fallback": True,
        "loss": {
            "quality": 1.0,
            "equivariance": 0.1,
            "residual": 0.01,
            "tiny_utility": 1.0,
            "non_tiny_risk": 2.0,
            "global_retain": 2.0,
            "anchor_admission": 1.0,
        },
    }


def test_gcte_yaml_preserves_stock_rtdetr_l_graph():
    payload = yaml.safe_load(
        (ROOT / "configs" / "rtdetr-l-gcte.yaml").read_text(
            encoding="utf-8"
        )
    )
    reference = yaml.safe_load(
        (ROOT / "configs" / "rtdetr-l-gcmv-plec.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert payload["backbone"] == reference["backbone"]
    assert payload["head"] == reference["head"]
    assert payload["head"][-1][2] == "RTDETRDecoder"
