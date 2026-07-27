from __future__ import annotations

from pathlib import Path

import torch
import yaml

from src.rtdetr_gcmv_plec import (
    GCMVPLECDetectionModel,
    PLECReferenceAdapter,
)
from src.sbr_geometry import LetterboxTransform, overlapping_tiles


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "rtdetr-l-gcmv-plec.yaml"


def test_model_yaml_exposes_plec_at_the_frozen_predecoder_boundary():
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert payload["gcmv"] == {
        "enabled": True,
        "module": "PLEC",
        "semantic_p3_index": 21,
        "decoder_feature_indices": [21, 24, 27],
        "global_imgsz": 640,
        "local_imgsz": 1088,
        "tile_ratio": 0.6,
        "views": ["TL", "TR", "BL", "BR"],
        "reference_adapter": {
            "projection": "Conv2d-1x1",
            "gamma_init": 0.0,
        },
    }
    assert payload["head"][-1][0] == [21, 24, 27]
    assert payload["head"][-1][2] == "RTDETRDecoder"


def test_reference_adapter_is_exact_stock_identity_at_zero_gamma():
    adapter = PLECReferenceAdapter(channels=8)
    global_p3 = torch.randn(2, 8, 5, 7)
    local = torch.randn_like(global_p3)

    fused = adapter(global_p3, local)

    assert adapter.gamma_ref.item() == 0.0
    assert torch.equal(fused, global_p3)


def test_reference_adapter_projection_and_scalar_receive_gradients():
    adapter = PLECReferenceAdapter(channels=8)
    adapter.gamma_ref.data.fill_(1.0)
    global_p3 = torch.randn(2, 8, 5, 7, requires_grad=True)
    local = torch.randn_like(global_p3, requires_grad=True)

    adapter(global_p3, local).square().mean().backward()

    assert adapter.gamma_ref.grad is not None
    assert adapter.gamma_ref.grad.abs().sum() > 0
    assert adapter.project.weight.grad is not None
    assert adapter.project.weight.grad.abs().sum() > 0
    assert local.grad is not None
    assert local.grad.abs().sum() > 0


def test_geometry_builder_uses_source_shapes_and_actual_feature_shapes():
    model = object.__new__(GCMVPLECDetectionModel)
    model.global_imgsz = 640
    model.local_imgsz = 1088
    source_shapes = torch.tensor([[1000, 2000], [720, 960]])
    global_p3 = torch.randn(2, 256, 80, 80)
    locals_p3 = [torch.randn(2, 256, 136, 136) for _ in range(4)]

    geometry = GCMVPLECDetectionModel._build_plec_geometry(
        model,
        source_shapes=source_shapes,
        global_p3=global_p3,
        local_p3=locals_p3,
    )

    assert geometry.global_feature_shape == (80, 80)
    assert geometry.local_feature_shape == (136, 136)
    assert geometry.sample_grid.shape == (2, 4, 9, 80, 80, 2)
    first_tiles = overlapping_tiles(width=2000, height=1000)
    assert [tile.bounds for tile in first_tiles] == [
        (0, 0, 1200, 600),
        (800, 0, 2000, 600),
        (0, 400, 1200, 1000),
        (800, 400, 2000, 1000),
    ]
    assert LetterboxTransform.from_view(
        width=2000, height=1000, imgsz=640
    ).network_shape == (640, 640)


def test_model_direct_injection_backpropagates_through_every_plec_family():
    model = GCMVPLECDetectionModel(CONFIG, nc=10, verbose=False)
    model.reference_adapter.gamma_ref.data.fill_(1.0)
    global_p3 = torch.randn(1, 256, 20, 20, requires_grad=True)
    local_p3 = [
        torch.randn(1, 256, 34, 34, requires_grad=True) for _ in range(4)
    ]

    fused = model.inject_local_p3(
        global_p3=global_p3,
        local_p3=local_p3,
        source_shapes=torch.tensor([[1000, 2000]]),
    )
    fused.square().mean().backward()

    prefixes = {
        "view_embedding",
        "phase_mlp",
        "metadata_mlp",
        "phase_reducer",
        "spatial_mixer",
        "pointwise",
        "overlap_head",
        "output_norm",
    }
    for prefix in prefixes:
        gradients = [
            parameter.grad
            for name, parameter in model.plec.named_parameters()
            if name.startswith(prefix)
        ]
        assert gradients
        assert all(gradient is not None for gradient in gradients)
        assert sum(gradient.abs().sum() for gradient in gradients) > 0
    assert all(feature.grad is not None for feature in local_p3)
    assert all(feature.grad.abs().sum() > 0 for feature in local_p3)
    assert model.model[-1].f == [21, 24, 27]
