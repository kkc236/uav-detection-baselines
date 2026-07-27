from __future__ import annotations

from pathlib import Path

import torch
import yaml

from src.rtdetr_gcmv_plec import (
    GCMVPLECDetectionModel,
    GCMVPLECTrainer,
    batchnorm_buffer_fingerprint,
)
from src.sbr_geometry import LetterboxTransform, overlapping_tiles


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "rtdetr-l-gcmv-plec.yaml"


def test_model_yaml_exposes_plec_at_the_frozen_predecoder_boundary():
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert payload["gcmv"] == {
        "enabled": True,
        "module": "GCMV-EI",
        "semantic_p3_index": 21,
        "decoder_feature_indices": [21, 24, 27],
        "global_imgsz": 640,
        "local_imgsz": 1088,
        "tile_ratio": 0.6,
        "views": ["TL", "TR", "BL", "BR"],
        "gglf": {
            "interaction_channels": 64,
            "num_heads": 4,
            "window_size": 3,
        },
        "peg": {
            "residual_scalar_init": 0.0,
            "gate_logit_init": 0.0,
        },
    }
    assert payload["head"][-1][0] == [21, 24, 27]
    assert payload["head"][-1][2] == "RTDETRDecoder"


def test_trainer_owns_amp_setup_instead_of_running_download_admission():
    assert "_setup_train" in GCMVPLECTrainer.__dict__


def test_geometry_builder_uses_source_shapes_and_actual_feature_shapes():
    model = object.__new__(GCMVPLECDetectionModel)
    model.global_imgsz = 640
    model.local_imgsz = 1088
    source_shapes = torch.tensor([[1000, 2000], [720, 960]])
    global_to_source = torch.stack(
        [
            torch.tensor(
                [[3.125, 0.0, 0.0], [0.0, 3.125, -500.0], [0.0, 0.0, 1.0]]
            ),
            torch.tensor(
                [[1.5, 0.0, 0.0], [0.0, 1.5, -120.0], [0.0, 0.0, 1.0]]
            ),
        ]
    )
    global_p3 = torch.randn(2, 256, 80, 80)
    locals_p3 = [torch.randn(2, 256, 136, 136) for _ in range(4)]

    geometry = GCMVPLECDetectionModel._build_plec_geometry(
        model,
        source_shapes=source_shapes,
        global_to_source=global_to_source,
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
    model.gcmv_injector.peg.rho.data.fill_(1.0)
    global_p3 = torch.randn(1, 256, 20, 20, requires_grad=True)
    local_p3 = [
        torch.randn(1, 256, 34, 34, requires_grad=True) for _ in range(4)
    ]

    fused = model.inject_gcmv_evidence(
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
    for prefix in ("gglf.", "peg."):
        gradients = [
            parameter.grad
            for name, parameter in model.gcmv_injector.named_parameters()
            if name.startswith(prefix)
        ]
        assert gradients
        assert all(gradient is not None for gradient in gradients)
        assert sum(gradient.abs().sum() for gradient in gradients) > 0
    assert all(feature.grad is not None for feature in local_p3)
    assert all(feature.grad.abs().sum() > 0 for feature in local_p3)
    assert model.model[-1].f == [21, 24, 27]


def test_local_feature_passes_are_detached_and_preserve_batchnorm():
    model = GCMVPLECDetectionModel(CONFIG, nc=10, verbose=False)
    model.local_imgsz = 16
    model.train()
    model.audit_local_batchnorm = True
    model._extract_local_p3 = lambda image: image.mean(
        dim=1, keepdim=True
    ).expand(-1, 256, -1, -1)
    local_views = torch.randn(2, 4, 3, 16, 16, requires_grad=True)
    before = batchnorm_buffer_fingerprint(model)

    local_p3 = model._local_feature_passes(local_views)

    assert all(not feature.requires_grad for feature in local_p3)
    assert all(feature.grad_fn is None for feature in local_p3)
    assert model.last_local_bn_preserved
    assert batchnorm_buffer_fingerprint(model) == before


def test_integrated_zero_guard_is_exact_stock_p3_identity():
    model = GCMVPLECDetectionModel(CONFIG, nc=10, verbose=False)
    global_p3 = torch.randn(1, 256, 20, 20)
    local_p3 = [torch.randn(1, 256, 34, 34) for _ in range(4)]

    enhanced = model.inject_gcmv_evidence(
        global_p3=global_p3,
        local_p3=local_p3,
        source_shapes=torch.tensor([[1000, 2000]]),
    )

    assert torch.equal(enhanced, global_p3)


def test_integrated_model_exposes_detection_and_auxiliary_loss_names():
    model = GCMVPLECDetectionModel(CONFIG, nc=10, verbose=False)

    assert model.calibration_only is False
    assert model.loss_names == (
        "giou_loss",
        "cls_loss",
        "l1_loss",
        "gcmv_tiny_loss",
        "gcmv_gate_loss",
        "gcmv_protect_loss",
    )
