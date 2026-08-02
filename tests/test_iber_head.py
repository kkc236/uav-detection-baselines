from __future__ import annotations

import ast
import inspect

import pytest
import torch
from torch import nn

import src.iber_head as iber_head
from src.iber_head import IBEROutput, IBERRefiner, PROBES
from src.iber_protocol import PROBES as PROTOCOL_PROBES
from src.iber_protocol import module_state_sha256
from src.iber_sampling import (
    sample_f3_boundary_evidence,
    sample_rgb_boundary_evidence,
)
from src.itber_geometry import apply_edge_update, xyxy_to_cxcywh
from src.itber_loss import itber_private_loss


def _refiner(probe: str = "b3") -> IBERRefiner:
    return IBERRefiner(
        hidden_dim=8,
        f3_channels=4,
        private_seed=17,
        probe=probe,
        image_size=640,
        rho=0.05,
    )


def _inputs(
    *,
    batch: int = 2,
    queries: int = 3,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(1234)
    hidden = torch.randn(batch, queries, 8, generator=generator)
    template = torch.tensor(
        (
            (0.50, 0.50, 0.20, 0.16),
            (0.12, 0.82, 0.08, 0.12),
            (0.88, 0.18, 0.10, 0.06),
        )
    )
    stock_boxes = template[:queries].unsqueeze(0).expand(batch, -1, -1).clone()
    stock_scores = torch.randn(batch, queries, 5, generator=generator)
    f3 = torch.randn(batch, 4, 12, 10, generator=generator)

    y = torch.linspace(-1.0, 1.0, 18).view(1, 1, 18, 1)
    x = torch.linspace(-0.75, 1.25, 20).view(1, 1, 1, 20)
    image_rgb = torch.cat(
        (
            x.expand(batch, 1, 18, 20),
            y.expand(batch, 1, 18, 20),
            (x + y).expand(batch, 1, 18, 20),
        ),
        dim=1,
    ).clone()
    values = (hidden, stock_boxes, stock_scores, f3, image_rgb)
    if requires_grad:
        for value in values:
            value.requires_grad_()
    return values


def _assert_linear(module: nn.Module, in_features: int, out_features: int) -> None:
    assert isinstance(module, nn.Linear)
    assert module.in_features == in_features
    assert module.out_features == out_features


def test_probe_initialization_is_equal_capacity_reproducible_and_rng_private() -> None:
    assert PROBES is PROTOCOL_PROBES
    assert PROBES == frozenset(("b0", "b1", "b2", "b3"))
    torch.manual_seed(91)
    before = torch.random.get_rng_state().clone()

    refiners = {probe: _refiner(probe) for probe in sorted(PROBES)}

    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    counts = {
        probe: sum(parameter.numel() for parameter in model.parameters())
        for probe, model in refiners.items()
    }
    fingerprints = {
        probe: module_state_sha256(model) for probe, model in refiners.items()
    }
    assert len(set(counts.values())) == 1
    assert len(set(fingerprints.values())) == 1


def test_private_initialization_never_seeds_accelerators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator_seed_calls: list[int] = []
    before = torch.random.get_rng_state().clone()
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: accelerator_seed_calls.append(int(seed)),
    )

    _refiner()

    assert accelerator_seed_calls == []
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)


def test_exact_architecture_and_four_zero_initialized_final_heads() -> None:
    model = _refiner()

    assert len(model.query_path) == 3
    assert isinstance(model.query_path[0], nn.LayerNorm)
    assert model.query_path[0].normalized_shape == (8,)
    _assert_linear(model.query_path[1], 8, 64)
    assert isinstance(model.query_path[2], nn.SiLU)

    assert len(model.geometry_path) == 2
    _assert_linear(model.geometry_path[0], 8, 16)
    assert isinstance(model.geometry_path[1], nn.SiLU)
    assert isinstance(model.edge_embedding, nn.Embedding)
    assert (model.edge_embedding.num_embeddings, model.edge_embedding.embedding_dim) == (
        4,
        8,
    )

    assert len(model.base_trunk) == 4
    _assert_linear(model.base_trunk[0], 88, 64)
    assert isinstance(model.base_trunk[1], nn.SiLU)
    _assert_linear(model.base_trunk[2], 64, 64)
    assert isinstance(model.base_trunk[3], nn.SiLU)

    assert isinstance(model.f3_projection, nn.Conv2d)
    assert model.f3_projection.in_channels == 4
    assert model.f3_projection.out_channels == 32
    assert model.f3_projection.kernel_size == (1, 1)
    assert len(model.f3_encoder) == 2
    _assert_linear(model.f3_encoder[0], 96, 32)
    assert isinstance(model.f3_encoder[1], nn.SiLU)

    assert len(model.rgb_encoder) == 3
    _assert_linear(model.rgb_encoder[0], 15, 16)
    assert isinstance(model.rgb_encoder[1], nn.LayerNorm)
    assert model.rgb_encoder[1].normalized_shape == (16,)
    assert isinstance(model.rgb_encoder[2], nn.SiLU)

    assert len(model.boundary_encoder) == 2
    _assert_linear(model.boundary_encoder[0], 48, 32)
    assert isinstance(model.boundary_encoder[1], nn.SiLU)
    assert len(model.boundary_trunk) == 4
    _assert_linear(model.boundary_trunk[0], 72, 64)
    assert isinstance(model.boundary_trunk[1], nn.SiLU)
    _assert_linear(model.boundary_trunk[2], 64, 64)
    assert isinstance(model.boundary_trunk[3], nn.SiLU)

    head_names = {
        "base_gate_head",
        "boundary_gate_head",
        "base_residual_head",
        "boundary_residual_head",
    }
    assert {name for name, _ in model.named_children() if name.endswith("_head")} == head_names
    for name in head_names:
        head = getattr(model, name)
        _assert_linear(head, 64, 1)
        torch.testing.assert_close(head.weight, torch.zeros_like(head.weight), rtol=0, atol=0)
        torch.testing.assert_close(head.bias, torch.zeros_like(head.bias), rtol=0, atol=0)


def test_zero_initialization_is_exact_identity_even_outside_image() -> None:
    hidden, _, scores, f3, image = _inputs(batch=1)
    stock = torch.tensor(
        [[
            [1.02, -0.02, 0.08, 0.06],
            [-0.01, 1.01, 0.04, 0.03],
            [0.50, 0.50, 1e-12, 1e-12],
        ]],
        requires_grad=True,
    )

    output = _refiner()(hidden, stock, scores, f3, image)

    torch.testing.assert_close(output.refined_boxes, output.stock_boxes, rtol=0, atol=0)
    torch.testing.assert_close(
        output.boundary_off_boxes, output.stock_boxes, rtol=0, atol=0
    )
    torch.testing.assert_close(output.refined_edges, output.stock_edges, rtol=0, atol=0)
    torch.testing.assert_close(
        output.boundary_off_edges, output.stock_edges, rtol=0, atol=0
    )
    torch.testing.assert_close(
        output.effective_correction,
        torch.zeros_like(output.effective_correction),
        rtol=0,
        atol=0,
    )
    assert stock.grad is None


def test_exposed_stock_storage_is_isolated_from_detector_inputs() -> None:
    inputs = _inputs(requires_grad=True)
    detector_boxes = inputs[1].detach().clone()
    detector_scores = inputs[2].detach().clone()
    output = _refiner()(*inputs)

    output.stock_boxes.add_(7)
    output.stock_scores.mul_(-3)

    torch.testing.assert_close(inputs[1], detector_boxes, rtol=0, atol=0)
    torch.testing.assert_close(inputs[2], detector_scores, rtol=0, atol=0)
    assert output.stock_boxes.data_ptr() != inputs[1].data_ptr()
    assert output.stock_scores.data_ptr() != inputs[2].data_ptr()
    assert all(value.grad is None for value in inputs)


def test_probe_masks_zero_raw_evidence_before_the_encoders() -> None:
    inputs = tuple(value.detach() for value in _inputs(batch=1))
    outputs = {probe: _refiner(probe)(*inputs) for probe in sorted(PROBES)}

    projected = _refiner("b3").f3_projection(inputs[3])
    expected_f3 = sample_f3_boundary_evidence(
        projected, inputs[1], image_size=640
    )
    expected_rgb = sample_rgb_boundary_evidence(
        inputs[4], inputs[1], image_size=640
    )
    assert torch.count_nonzero(expected_f3) > 0
    assert torch.count_nonzero(expected_rgb) > 0

    for probe, output in outputs.items():
        expected_f3_for_probe = expected_f3 if probe in {"b1", "b3"} else torch.zeros_like(expected_f3)
        expected_rgb_for_probe = expected_rgb if probe in {"b2", "b3"} else torch.zeros_like(expected_rgb)
        torch.testing.assert_close(
            output.f3_boundary_evidence,
            expected_f3_for_probe,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            output.rgb_boundary_evidence,
            expected_rgb_for_probe,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    ("probe", "expected_f3_calls", "expected_rgb_calls"),
    [
        pytest.param("b0", 0, 0, id="b0-skips-both"),
        pytest.param("b1", 1, 0, id="b1-skips-rgb"),
        pytest.param("b2", 0, 1, id="b2-skips-f3"),
    ],
)
def test_disabled_modality_samplers_are_never_called(
    probe: str,
    expected_f3_calls: int,
    expected_rgb_calls: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"f3": 0, "rgb": 0}
    real_f3_sampler = iber_head.sample_f3_boundary_evidence
    real_rgb_sampler = iber_head.sample_rgb_boundary_evidence

    def counted_f3(*args: object, **kwargs: object) -> torch.Tensor:
        calls["f3"] += 1
        return real_f3_sampler(*args, **kwargs)  # type: ignore[arg-type]

    def counted_rgb(*args: object, **kwargs: object) -> torch.Tensor:
        calls["rgb"] += 1
        return real_rgb_sampler(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(iber_head, "sample_f3_boundary_evidence", counted_f3)
    monkeypatch.setattr(iber_head, "sample_rgb_boundary_evidence", counted_rgb)

    _refiner(probe)(*_inputs(batch=1))

    assert calls == {"f3": expected_f3_calls, "rgb": expected_rgb_calls}


def test_detector_inputs_are_detached_and_both_enabled_arms_receive_gradients() -> None:
    model = _refiner("b3")
    with torch.no_grad():
        model.boundary_residual_head.weight.fill_(0.125)
    inputs = _inputs(requires_grad=True)

    output = model(*inputs)
    target = output.stock_boxes + torch.tensor((0.01, -0.02, 0.015, 0.01))
    loss = (output.refined_boxes - target).square().sum()
    loss.backward()

    assert all(value.grad is None for value in inputs)
    parameter_groups = (
        tuple(model.f3_projection.parameters()),
        tuple(model.f3_encoder.parameters()),
        tuple(model.rgb_encoder.parameters()),
    )
    for parameters in parameter_groups:
        gradients = [parameter.grad for parameter in parameters]
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


def test_first_private_loss_step_updates_all_four_zero_initialized_heads() -> None:
    model = _refiner()
    head_names = (
        "base_gate_head",
        "boundary_gate_head",
        "base_residual_head",
        "boundary_residual_head",
    )
    for name in head_names:
        head = getattr(model, name)
        assert torch.count_nonzero(head.weight) == 0
        assert torch.count_nonzero(head.bias) == 0

    output = model(*_inputs(batch=2))
    losses = itber_private_loss(
        output,
        target_edges=torch.tensor(
            [[0.39, 0.40, 0.61, 0.58], [0.07, 0.75, 0.17, 0.87]]
        ),
        match_indices=[
            (torch.tensor([0]), torch.tensor([0])),
            (torch.tensor([1]), torch.tensor([1])),
        ],
        rho=0.05,
    )
    losses.total.backward()

    for name in head_names:
        gradients = [parameter.grad for parameter in getattr(model, name).parameters()]
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


def test_head_is_trajectory_independent_and_has_exact_public_signatures() -> None:
    constructor = inspect.signature(IBERRefiner)
    assert tuple(constructor.parameters) == (
        "hidden_dim",
        "f3_channels",
        "private_seed",
        "probe",
        "image_size",
        "rho",
    )
    assert constructor.parameters["probe"].default == "b3"
    assert constructor.parameters["image_size"].default == 640
    assert constructor.parameters["rho"].default == 0.05
    assert tuple(inspect.signature(IBERRefiner.forward).parameters) == (
        "self",
        "hidden",
        "stock_boxes",
        "stock_scores",
        "f3",
        "image_rgb",
    )

    source = inspect.getsource(iber_head)
    tree = ast.parse(source)
    imports = [
        node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    geometry_import = next(
        node
        for node in imports
        if isinstance(node, ast.ImportFrom) and node.module == "src.itber_geometry"
    )
    assert {alias.name for alias in geometry_import.names} == {
        "apply_edge_update",
        "cxcywh_to_xyxy",
        "xyxy_to_cxcywh",
    }
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module in {"src.itber_head", "src.rtdetr_itber"}
        for node in imports
    )
    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert identifiers.isdisjoint(
        {"ITBERRefiner", "ITBERRecordingDecoder", "trajectory_state", "trajectory", "box_l1", "box_l2"}
    )
    assert not any("trajectory" in name for name, _ in _refiner().named_modules())
    assert not any("trajectory" in name for name, _ in _refiner().named_parameters())


def test_tiny_border_outside_boxes_and_extreme_scores_are_finite() -> None:
    hidden, _, _, f3, image = _inputs(batch=1)
    boxes = torch.tensor(
        [[
            [0.0, 0.0, 1e-12, 1e-12],
            [1.0, 1.0, 1e-12, 1e-12],
            [1.05, -0.05, 0.10, 0.08],
        ]]
    )
    scores = torch.tensor(
        [[[1000.0, -1000.0], [-1000.0, 1000.0], [1000.0, 1000.0]]]
    )

    output = _refiner()(hidden, boxes, scores, f3, image)

    for value in vars(output).values():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()


@pytest.mark.parametrize("score_dtype", [torch.float16, torch.bfloat16])
def test_low_precision_extreme_scores_keep_outputs_and_private_loss_finite(
    score_dtype: torch.dtype,
) -> None:
    hidden, boxes, _, f3, image = _inputs(batch=1)
    scores = torch.tensor(
        [[[1000.0, -1000.0], [-1000.0, 1000.0], [1000.0, 1000.0]]],
        dtype=score_dtype,
    )

    output = _refiner()(hidden, boxes, scores, f3, image)

    assert output.quality.dtype == torch.float32
    assert output.entropy.dtype == torch.float32
    for value in vars(output).values():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()
    torch.testing.assert_close(output.refined_boxes, boxes, rtol=0, atol=0)
    torch.testing.assert_close(output.boundary_off_boxes, boxes, rtol=0, atol=0)

    losses = itber_private_loss(
        output,
        target_edges=torch.tensor([[0.39, 0.40, 0.61, 0.58]]),
        match_indices=[(torch.tensor([0]), torch.tensor([0]))],
        rho=0.05,
    )
    for value in vars(losses).values():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value)


@pytest.mark.parametrize(("batch", "queries"), [(1, 0), (0, 2)])
def test_empty_batch_or_query_dimensions_return_well_shaped_outputs(
    batch: int, queries: int
) -> None:
    hidden = torch.empty(batch, queries, 8)
    boxes = torch.empty(batch, queries, 4)
    scores = torch.empty(batch, queries, 3)
    f3 = torch.empty(batch, 4, 8, 8)
    image = torch.empty(batch, 3, 16, 16)

    output = _refiner()(hidden, boxes, scores, f3, image)

    assert output.refined_boxes.shape == (batch, queries, 4)
    assert output.boundary_off_boxes.shape == (batch, queries, 4)
    assert output.f3_boundary_evidence.shape == (batch, queries, 4, 96)
    assert output.rgb_boundary_evidence.shape == (batch, queries, 4, 15)
    assert output.boundary_features.shape == (batch, queries, 4, 32)


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (lambda: IBERRefiner(0, 4, 17), ValueError, "positive"),
        (lambda: IBERRefiner(8, 0, 17), ValueError, "positive"),
        (lambda: IBERRefiner(8, 4, 17, probe="p3"), ValueError, "probe"),
        (lambda: IBERRefiner(8, 4, 17, image_size=0), ValueError, "positive"),
        (lambda: IBERRefiner(8, 4, 17, rho=0), ValueError, "positive"),
    ],
)
def test_constructor_rejects_invalid_configuration(
    factory: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    ("index", "replacement", "error", "message"),
    [
        (0, torch.ones(2, 3, 8, dtype=torch.int64), TypeError, "hidden"),
        (0, torch.ones(2, 3, 7), ValueError, "hidden"),
        (1, torch.ones(2, 2, 4), ValueError, "stock_boxes"),
        (2, torch.ones(2, 2, 5), ValueError, "stock_scores"),
        (2, torch.ones(2, 3), ValueError, "stock_scores"),
        (3, torch.ones(2, 5, 12, 10), ValueError, "F3"),
        (3, torch.ones(1, 4, 12, 10), ValueError, "F3"),
        (4, torch.ones(2, 4, 18, 20), ValueError, "image_rgb"),
        (4, torch.ones(1, 3, 18, 20), ValueError, "image_rgb"),
    ],
)
def test_forward_rejects_invalid_inputs_clearly(
    index: int,
    replacement: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    inputs = list(_inputs())
    inputs[index] = replacement

    with pytest.raises(error, match=message):
        _refiner()(*inputs)


def test_select_boxes_is_exact_and_stock_scores_are_never_modified() -> None:
    inputs = _inputs()
    output = _refiner()(*inputs)

    assert isinstance(output, IBEROutput)
    assert output.select_boxes("stock") is output.stock_boxes
    assert output.select_boxes("refined") is output.refined_boxes
    assert output.select_boxes("boundary_off") is output.boundary_off_boxes
    with pytest.raises(ValueError, match="mode"):
        output.select_boxes("unknown")
    torch.testing.assert_close(output.stock_scores, inputs[2].detach(), rtol=0, atol=0)
    assert output.stock_scores.requires_grad is False


def test_boundary_off_uses_only_nonzero_base_heads() -> None:
    first = _refiner()
    second = _refiner()
    with torch.no_grad():
        for model in (first, second):
            model.base_gate_head.weight.fill_(0.04)
            model.base_gate_head.bias.fill_(0.10)
            model.base_residual_head.weight.fill_(-0.03)
            model.base_residual_head.bias.fill_(0.20)
        first.boundary_gate_head.weight.fill_(0.02)
        first.boundary_gate_head.bias.fill_(-0.15)
        first.boundary_residual_head.weight.fill_(0.01)
        first.boundary_residual_head.bias.fill_(0.25)
        second.boundary_gate_head.weight.fill_(-0.05)
        second.boundary_gate_head.bias.fill_(0.35)
        second.boundary_residual_head.weight.fill_(-0.04)
        second.boundary_residual_head.bias.fill_(-0.30)
    inputs = _inputs(batch=1)

    first_output = first(*inputs)
    second_output = second(*inputs)
    expected_edges = apply_edge_update(
        first_output.stock_edges,
        first_output.base_gate_raw.sigmoid(),
        first_output.base_residual_raw.tanh(),
        rho=0.05,
    )
    expected_boxes = first_output.stock_boxes + (
        xyxy_to_cxcywh(expected_edges)
        - xyxy_to_cxcywh(first_output.stock_edges)
    )

    assert torch.count_nonzero(expected_edges - first_output.stock_edges) > 0
    torch.testing.assert_close(
        first_output.boundary_off_edges, expected_edges, rtol=0, atol=0
    )
    torch.testing.assert_close(
        first_output.boundary_off_boxes, expected_boxes, rtol=0, atol=0
    )
    torch.testing.assert_close(
        first_output.boundary_off_edges,
        second_output.boundary_off_edges,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        first_output.boundary_off_boxes,
        second_output.boundary_off_boxes,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(first_output.refined_edges, second_output.refined_edges)


def test_b0_disables_boundary_outputs_even_when_boundary_heads_are_nonzero() -> None:
    model = _refiner("b0")
    with torch.no_grad():
        model.boundary_gate_head.weight.fill_(0.07)
        model.boundary_gate_head.bias.fill_(0.13)
        model.boundary_residual_head.weight.fill_(-0.05)
        model.boundary_residual_head.bias.fill_(0.21)

    output = model(*_inputs(batch=1))

    torch.testing.assert_close(
        output.boundary_gate_raw,
        torch.zeros_like(output.boundary_gate_raw),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        output.boundary_residual_raw,
        torch.zeros_like(output.boundary_residual_raw),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(output.refined_edges, output.boundary_off_edges, rtol=0, atol=0)


@pytest.mark.parametrize("probe", sorted(PROBES))
def test_zero_boundary_evidence_has_no_counterfactual_boundary_delta(probe: str) -> None:
    model = _refiner(probe)
    with torch.no_grad():
        model.boundary_gate_head.weight.fill_(0.07)
        model.boundary_gate_head.bias.fill_(0.13)
        model.boundary_residual_head.weight.fill_(-0.05)
        model.boundary_residual_head.bias.fill_(0.21)

    hidden, boxes, scores, f3, image = _inputs(batch=1)
    output = model(hidden, boxes, scores, torch.zeros_like(f3), torch.zeros_like(image))

    torch.testing.assert_close(
        output.boundary_gate_raw,
        torch.zeros_like(output.boundary_gate_raw),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        output.boundary_residual_raw,
        torch.zeros_like(output.boundary_residual_raw),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(output.refined_edges, output.boundary_off_edges, rtol=0, atol=0)


def test_existing_private_loss_accepts_real_iber_output() -> None:
    model = _refiner()
    output = model(*_inputs(batch=2))
    target_edges = torch.tensor(
        [[0.39, 0.40, 0.61, 0.58], [0.07, 0.75, 0.17, 0.87]]
    )
    matches = [
        (torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([1]), torch.tensor([1])),
    ]

    losses = itber_private_loss(
        output,
        target_edges=target_edges,
        match_indices=matches,
        rho=0.05,
    )

    assert torch.isfinite(losses.total)
    assert losses.matched_queries == 2
    losses.total.backward()
    assert model.base_gate_head.weight.grad is not None
    assert model.boundary_residual_head.weight.grad is not None


def test_only_f3_projection_is_convolutional_and_rgb_is_pointwise_encoded() -> None:
    model = _refiner()
    convolutions = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    }
    assert convolutions == {"f3_projection": model.f3_projection}
    assert not any(isinstance(module, nn.Conv2d) for module in model.rgb_encoder.modules())
    assert not any(isinstance(module, nn.MultiheadAttention) for module in model.modules())

    source_without_docstrings = ast.parse(inspect.getsource(iber_head))
    for node in ast.walk(source_without_docstrings):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
    lowered = ast.unparse(source_without_docstrings).lower()
    for forbidden in ("attention", "learned_offset", "pyramid", "sobel", "p2"):
        assert forbidden not in lowered
