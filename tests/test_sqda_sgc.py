from __future__ import annotations

import math

import pytest
import torch

from src.sqda_sgc import ROLE_NAMES, SQDASGCAdapter, SQDASGCConfig


def _inputs(
    *,
    batch: int = 2,
    queries: int = 300,
    height: int = 32,
    width: int = 40,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    object_queries = torch.randn(batch, queries, 256, requires_grad=True)
    boxes = torch.rand(batch, queries, 4, requires_grad=True)
    with torch.no_grad():
        boxes[..., 2:].mul_(0.2).add_(0.005)
    raw_c2 = torch.randn(batch, 128, height, width, requires_grad=True)
    return object_queries, boxes, raw_c2


def test_frozen_configuration_and_parameter_budget() -> None:
    module = SQDASGCAdapter()
    assert module.config == SQDASGCConfig()
    assert ROLE_NAMES == ("C", "L", "R", "T", "B", "O")
    assert module.role_point_counts == (4, 2, 2, 2, 2, 8)
    assert sum(module.role_point_counts) == 20
    assert sum(parameter.numel() for parameter in module.parameters()) < 1_000_000


def test_fixed_point_templates_and_c2_cell_floor() -> None:
    module = SQDASGCAdapter()
    boxes = torch.tensor([[[0.5, 0.5, 1e-5, 1e-5]]])
    base_points, radii = module.fixed_sampling_points(boxes, height=20, width=40)

    assert base_points.shape == (1, 1, 20, 2)
    assert radii[0, 0, 0].item() == pytest.approx(1 / 40)
    assert radii[0, 0, 1].item() == pytest.approx(1 / 20)

    ux, uy = radii[0, 0]
    center_offsets = base_points[0, 0, :4] - boxes[0, 0, :2]
    expected_center = torch.tensor(
        [
            [-0.5 * ux, -0.5 * uy],
            [-0.5 * ux, 0.5 * uy],
            [0.5 * ux, -0.5 * uy],
            [0.5 * ux, 0.5 * uy],
        ]
    )
    assert torch.allclose(center_offsets, expected_center)

    outer_offsets = base_points[0, 0, 12:] - boxes[0, 0, :2]
    expected_outer = torch.tensor(
        [
            [-1.5 * ux, 0.0],
            [1.5 * ux, 0.0],
            [0.0, -1.5 * uy],
            [0.0, 1.5 * uy],
            [-1.25 * ux, -1.25 * uy],
            [-1.25 * ux, 1.25 * uy],
            [1.25 * ux, -1.25 * uy],
            [1.25 * ux, 1.25 * uy],
        ]
    )
    assert torch.allclose(outer_offsets, expected_outer)


def test_offsets_start_at_zero_and_remain_bounded() -> None:
    module = SQDASGCAdapter()
    for layer in module.point_offset_heads:
        assert torch.count_nonzero(layer.weight) == 0
        assert torch.count_nonzero(layer.bias) == 0

    queries, boxes, raw_c2 = _inputs(batch=1, queries=3)
    role_tokens = module.build_role_tokens(queries, boxes)
    base_points, radii = module.fixed_sampling_points(boxes, raw_c2.shape[-2], raw_c2.shape[-1])
    learned_points = module.apply_point_offsets(base_points, radii, role_tokens)
    assert torch.equal(learned_points, base_points)

    with torch.no_grad():
        for layer in module.point_offset_heads:
            layer.weight.fill_(1000)
            layer.bias.fill_(1000)
    learned_points = module.apply_point_offsets(base_points, radii, role_tokens)
    relative = (learned_points - base_points).abs()
    limits = (0.1 * radii).unsqueeze(2)
    assert torch.all(relative <= limits + 1e-6)


def test_forward_shape_dtype_diagnostics_and_reference_detach() -> None:
    module = SQDASGCAdapter()
    queries, boxes, raw_c2 = _inputs()
    output, diagnostics = module(queries, boxes, raw_c2)

    assert output.shape == queries.shape
    assert output.dtype == queries.dtype
    assert output.device == queries.device
    assert diagnostics["sampling_validity"].shape == (2, 300, 6)
    assert diagnostics["point_attention"].shape == (2, 300, 20)
    assert diagnostics["edge_attention"].shape == (2, 300, 4)
    assert diagnostics["group_gates"].shape == (2, 300, 2, 16)
    assert diagnostics["context_reliability"].shape == (2, 300)
    assert diagnostics["geometry_features"].shape == (2, 300, 5)
    assert diagnostics["semantic_component"].shape == (2, 300, 256)
    assert diagnostics["geometry_component"].shape == (2, 300, 256)
    assert diagnostics["raw_fusion"].shape == (2, 300, 256)
    assert diagnostics["pre_saturation_rms"].shape == (2, 300)
    assert diagnostics["post_saturation_rms"].shape == (2, 300)
    assert diagnostics["semantic_budget"].shape == (2, 300, 1)
    assert diagnostics["geometry_budget"].shape == (2, 300, 1)
    assert all(not value.requires_grad for value in diagnostics.values() if torch.is_tensor(value))

    output.square().mean().backward()
    assert queries.grad is not None
    assert raw_c2.grad is not None
    assert boxes.grad is None


def test_context_strength_and_layerscale_have_frozen_bounds() -> None:
    module = SQDASGCAdapter()
    assert module.context_strength.item() == pytest.approx(0.05, abs=1e-7)
    assert module.layer_scale.item() == pytest.approx(1e-3, abs=1e-8)

    with torch.no_grad():
        module.context_logit.fill_(-100)
        module.layer_scale_logit.fill_(-100)
    assert 0 <= module.context_strength.item() < 1e-10
    assert 0 <= module.layer_scale.item() < 1e-10

    with torch.no_grad():
        module.context_logit.fill_(100)
        module.layer_scale_logit.fill_(100)
    assert 0.249999 <= module.context_strength.item() <= 0.25
    assert 0.049999 <= module.layer_scale.item() <= 0.05


def test_safe_context_modulation_is_near_neutral_and_invalid_context_is_neutral() -> None:
    module = SQDASGCAdapter()
    similarity = torch.zeros(1, 4)
    valid = torch.tensor([[True, True, False, False]])
    context_similarity = torch.tensor([[0.0, 1.0, -1.0, 1.0]])
    modulation = module.context_modulation(similarity, context_similarity, valid)

    assert modulation[0, 0].item() == pytest.approx(0.975, abs=1e-6)
    assert 0.95 < modulation[0, 1].item() < 1.0
    assert modulation[0, 2].item() == 1.0
    assert modulation[0, 3].item() == 1.0
    assert torch.all((modulation > 0.75) & (modulation <= 1.0))


def test_invalid_writeback_roles_force_an_exact_zero_residual_without_nan() -> None:
    module = SQDASGCAdapter()
    queries, _, raw_c2 = _inputs(batch=1, queries=5, height=8, width=8)
    boxes = torch.tensor(
        [[[-10.0, -10.0, 0.1, 0.1]] * 5],
        dtype=queries.dtype,
        requires_grad=True,
    )
    output, diagnostics = module(queries, boxes, raw_c2)

    assert torch.equal(output, queries)
    assert torch.count_nonzero(diagnostics["writeback_valid"]) == 0
    assert torch.count_nonzero(diagnostics["residual_norm"]) == 0
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(value).all() for value in diagnostics.values() if torch.is_tensor(value))


def test_context_is_read_only_and_legacy_fusion_is_retained() -> None:
    module = SQDASGCAdapter()
    assert module.fusion.in_features == 512
    assert module.fusion.out_features == 256
    assert set(("fusion.weight", "fusion.bias")).issubset(module.state_dict())
    assert not hasattr(module, "semantic_projector")
    assert not hasattr(module, "geometry_projector")
    assert not hasattr(module, "agreement_gate")
    assert module.context_projector is module.value_projector


def test_geometry_trust_initialization_and_group_gate_layout() -> None:
    module = SQDASGCAdapter()
    assert module.gate[-1].out_features == 32
    assert torch.count_nonzero(module.gate[-1].bias) == 0
    assert module.gate[0].in_features == 5 * 256 + 4
    assert module.geometry_trust[0].in_features == 5
    assert module.geometry_trust[-1].out_features == 1
    assert module.geometry_trust[-1].bias.item() == pytest.approx(
        math.log(0.90 / 0.10), abs=1e-6
    )
    assert module.geometry_trust[-1].weight.std().item() == pytest.approx(0.01, rel=0.35)

    queries, boxes, raw_c2 = _inputs(batch=1, queries=4)
    _, diagnostics = module(queries, boxes, raw_c2)
    assert torch.allclose(
        diagnostics["group_gates"].sum(dim=-2),
        torch.ones_like(diagnostics["group_gates"][:, :, 0]),
    )
    expanded = module.expand_group_gate(diagnostics["group_gates"][:, :, 0])
    assert expanded.shape == (1, 4, 256)
    for group in range(16):
        values = expanded[..., group * 16 : (group + 1) * 16]
        assert torch.all(values == values[..., :1])


def test_residual_rms_is_bounded_by_layer_scale() -> None:
    module = SQDASGCAdapter()
    with torch.no_grad():
        module.fusion.weight.fill_(100.0)
    queries, boxes, raw_c2 = _inputs(batch=1, queries=4)

    _, diagnostics = module(queries, boxes, raw_c2)

    maximum_norm = module.layer_scale * math.sqrt(module.config.hidden_dim)
    assert torch.all(diagnostics["residual_norm"] <= maximum_norm + 1e-6)


def test_geometry_trust_gate_starts_near_one_and_has_strict_bounds() -> None:
    module = SQDASGCAdapter()
    queries, boxes, raw_c2 = _inputs(batch=1, queries=4)
    _, diagnostics = module(queries, boxes, raw_c2)

    semantic_budget = diagnostics["semantic_budget"]
    geometry_budget = diagnostics["geometry_budget"]
    assert torch.all((geometry_budget > 0.80) & (geometry_budget < 1.0))
    assert torch.equal(semantic_budget, torch.ones_like(semantic_budget))
    assert geometry_budget.mean().item() == pytest.approx(0.98, abs=0.01)


def test_full_counterfactual_uses_the_retained_fusion_module() -> None:
    module = SQDASGCAdapter()
    queries, boxes, raw_c2 = _inputs(batch=1, queries=5)
    captured: list[torch.Tensor] = []
    handle = module.fusion.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    _, diagnostics = module(queries, boxes, raw_c2, residual_mode="full")
    handle.remove()

    assert len(captured) == 1
    assert torch.allclose(
        diagnostics["raw_fusion"],
        module.fusion(captured[0]),
        atol=2e-6,
        rtol=1e-5,
    )


def test_counterfactual_modes_are_local_and_identity_is_bitwise_exact() -> None:
    module = SQDASGCAdapter()
    queries, boxes, raw_c2 = _inputs(batch=1, queries=5)
    outputs = {}
    diagnostics = {}
    for mode in ("full", "semantic_only", "geometry_only", "identity"):
        outputs[mode], diagnostics[mode] = module(
            queries,
            boxes,
            raw_c2,
            residual_mode=mode,
        )
        assert outputs[mode].shape == queries.shape

    assert torch.equal(outputs["identity"], queries)
    assert torch.count_nonzero(diagnostics["semantic_only"]["geometry_component"]) == 0
    assert torch.count_nonzero(diagnostics["geometry_only"]["semantic_component"]) == 0
    with pytest.raises(ValueError, match="residual_mode"):
        module(queries, boxes, raw_c2, residual_mode="not-a-mode")


def test_half_precision_caps_do_not_depend_on_nextafter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = SQDASGCAdapter().half()

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("nextafter is unavailable for this device/dtype")

    monkeypatch.setattr(torch, "nextafter", unavailable)

    assert module.context_strength.dtype == torch.float16
    assert 0 < module.context_strength < module.config.context_cap
    assert module.layer_scale.dtype == torch.float16
    assert 0 < module.layer_scale < module.config.residual_cap


@pytest.mark.parametrize("enabled,identity_override", [(False, False), (True, True)])
def test_identity_modes_are_bitwise_exact(enabled: bool, identity_override: bool) -> None:
    module = SQDASGCAdapter(enabled=enabled)
    queries, boxes, raw_c2 = _inputs(batch=1, queries=7)
    output, diagnostics = module(
        queries,
        boxes,
        raw_c2,
        identity_override=identity_override,
    )
    assert torch.equal(output, queries)
    assert diagnostics["identity_override"].item()


def test_configuration_rejects_non_frozen_dimensions() -> None:
    with pytest.raises(ValueError, match="hidden_dim"):
        SQDASGCAdapter(hidden_dim=128)
    with pytest.raises(ValueError, match="gate_groups"):
        SQDASGCAdapter(gate_groups=8)
    with pytest.raises(ValueError, match="query"):
        SQDASGCAdapter(query_count=100)


def test_logit_initializers_match_registered_values() -> None:
    module = SQDASGCAdapter()
    expected_context = math.log(0.2 / 0.8)
    expected_residual = math.log(0.02 / 0.98)
    assert module.context_logit.item() == pytest.approx(expected_context, abs=1e-6)
    assert module.layer_scale_logit.item() == pytest.approx(expected_residual, abs=1e-6)
