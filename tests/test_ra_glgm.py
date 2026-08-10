from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from scripts.train_rtdetr_ra_glgm import _private_losses
from src.ra_glgm import RAGLGM
from src.ra_glgm_head import RAFDRRTDETRDecoder
from src.ra_glgm_loss import (
    ResidualDifficultyTargets,
    SCALE_LOG_AREA_KNOTS,
    build_residual_difficulty_targets,
    log_area_to_empirical_cdf,
    residual_support_focal_loss,
    scale_conditioning_loss,
    scale_prediction_diagnostics,
)
from src.ra_glgm_protocol import (
    RA_GLGM_PRIVATE_PREFIX,
    build_ra_glgm_initial_state,
    load_ra_glgm_initial_state,
    partition_ra_glgm_state_dicts,
    validate_ra_glgm_initial_state,
)
from src.rtdetr_btdse import filter_detection_batch
from src.rtdetr_ra_glgm import (
    _RAEpochScaleStatistics,
    RAGLGMControlDetectionModel,
    RAGLGMControlTrainer,
    RAGLGMDetectionModel,
    RAGLGMTrainer,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PARAMETERS = 33_156_614
BASE_CFG = ROOT / "configs" / "rtdetr-l-fdr.yaml"
CONTROL_CFG = ROOT / "configs" / "rtdetr-l-fdr-ra-glgm-control.yaml"
METHOD_CFG = ROOT / "configs" / "rtdetr-l-fdr-ra-glgm.yaml"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _tiny_module() -> RAGLGM:
    return RAGLGM(channels=32, hidden_channels=24, route_groups=4, private_seed=91)


def test_initial_output_routes_and_public_gradient_are_exactly_baseline() -> None:
    module = _tiny_module().train()
    public = torch.randn(2, 32, 9, 11, requires_grad=True)
    baseline = public.detach().clone().requires_grad_(True)
    target = torch.zeros(2, 1, 9, 11)
    valid = torch.ones_like(target, dtype=torch.bool)

    output, routes, support, scales = module.forward_with_diagnostics(public)
    baseline_loss = baseline.square().mean()
    method_loss = output.square().mean() + 0.05 * residual_support_focal_loss(
        support,
        ResidualDifficultyTargets(
            heatmap=target,
            valid_mask=valid,
            difficulty=torch.empty(0),
        ),
    )
    baseline_loss.backward()
    method_loss.backward()

    assert torch.equal(output, public)
    assert routes.shape == (2, 2, 4, 9, 11)
    assert torch.equal(routes, torch.full_like(routes, 0.5))
    assert torch.equal(scales, torch.full_like(scales, 0.5))
    assert torch.equal(public.grad, baseline.grad)
    assert module.alpha.grad is not None and module.alpha.grad.abs().sum() > 0
    assert module.support_head.weight.grad is not None
    assert module.support_head.weight.grad.abs().sum() > 0
    assert module.router.weight.grad is not None
    assert module.router.weight.grad.abs().sum() > 0
    assert module.local_one[0].weight.grad is not None
    assert module.local_one[0].weight.grad.abs().sum() > 0
    assert module.output_projection.weight.grad is not None
    assert torch.count_nonzero(module.output_projection.weight.grad) == 0
    assert module.scale_head.weight.grad is not None
    assert torch.count_nonzero(module.scale_head.weight.grad) == 0
    assert module.scale_expert_slopes.grad is not None
    assert torch.count_nonzero(module.scale_expert_slopes.grad) == 0
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_second_step_opens_output_projection_gradient_and_residual_is_bounded() -> None:
    module = _tiny_module().train()
    x = torch.randn(2, 32, 8, 10)
    first = module(x).square().mean()
    first.backward()
    with torch.no_grad():
        module.alpha.add_(-0.1 * module.alpha.grad)
    module.zero_grad(set_to_none=True)

    output = module(x)
    output.square().mean().backward()

    assert module.output_projection.weight.grad is not None
    assert module.output_projection.weight.grad.abs().sum() > 0
    assert torch.isfinite(module.output_projection.weight.grad).all()
    assert (output - x).abs().max() <= module.max_residual_scale + 1e-7


def test_scale_supervision_opens_zero_initialized_expert_slopes_on_next_step() -> None:
    module = _tiny_module().train()
    x = torch.randn(1, 32, 8, 8)
    targets = ResidualDifficultyTargets(
        heatmap=torch.ones(1, 1, 8, 8),
        valid_mask=torch.ones(1, 1, 8, 8, dtype=torch.bool),
        difficulty=torch.ones(1),
        scale_boxes=torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
        scale_batch_idx=torch.zeros(1, dtype=torch.long),
        scale_values=torch.ones(1),
    )
    _, _, _, scale = module.forward_with_diagnostics(x)
    scale_conditioning_loss(scale, targets).backward()
    assert module.scale_head.bias.grad is not None
    assert module.scale_head.bias.grad.abs().sum() > 0
    with torch.no_grad():
        module.scale_head.bias.sub_(module.scale_head.bias.grad)
    module.zero_grad(set_to_none=True)

    _, _, support, opened_scale = module.forward_with_diagnostics(x)
    assert not torch.equal(opened_scale, torch.full_like(opened_scale, 0.5))
    support.mean().backward()
    assert module.scale_expert_slopes.grad is not None
    assert module.scale_expert_slopes.grad.abs().sum() > 0


def test_continuous_scale_modulation_is_bounded_and_residual_remains_bounded() -> None:
    module = RAGLGM(channels=32, hidden_channels=24, route_groups=4, private_seed=91)
    x = torch.randn(2, 32, 8, 10)

    output, routes, _, scales = module.forward_with_diagnostics(x)
    assert torch.equal(output, x)
    assert torch.equal(scales, torch.full_like(scales, 0.5))
    assert torch.equal(routes, torch.full_like(routes, 0.5))

    with torch.no_grad():
        module.alpha.fill_(10.0)
        module.scale_head.bias.fill_(20.0)
        module.scale_expert_slopes.fill_(20.0)
    opened, opened_routes, _, opened_scales = module.forward_with_diagnostics(x)
    assert torch.all(opened_scales > 0.999)
    assert torch.all(opened_routes[:, 1] < torch.sigmoid(torch.tensor(2.0)) + 1e-6)
    assert torch.all(opened_routes[:, 1] > 0.5)
    assert (opened - x).abs().max() <= module.max_residual_scale + 1e-7


@pytest.mark.parametrize("shape", [(1, 32, 7, 13), (2, 32, 16, 9)])
def test_dynamic_shapes_and_cpu_amp_remain_finite(shape: tuple[int, ...]) -> None:
    module = _tiny_module().train()
    x = torch.randn(shape)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output, routes, support, scales = module.forward_with_diagnostics(x)

    assert output.shape == x.shape
    assert routes.shape[-2:] == x.shape[-2:]
    assert support.shape == (shape[0], 1, shape[2], shape[3])
    assert scales.shape == (shape[0], 1, shape[2], shape[3])
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        routes.float().sum(dim=1),
        torch.ones_like(routes[:, 0].float()),
        rtol=0,
        atol=0,
    )


def test_private_initialization_does_not_advance_public_rng() -> None:
    torch.manual_seed(17)
    expected = torch.rand(5)
    torch.manual_seed(17)
    first = _tiny_module()
    actual = torch.rand(5)
    second = _tiny_module()

    assert torch.equal(actual, expected)
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name]), name


def test_parameter_budget_and_depthwise_large_kernels_are_frozen() -> None:
    module = RAGLGM()

    assert module.private_parameter_count == 813_018
    assert module.private_parameter_count / BASELINE_PARAMETERS == pytest.approx(
        0.0245205368
    )
    assert module.private_parameter_count <= 0.10 * BASELINE_PARAMETERS
    assert module.global_large[0].groups == 192
    assert module.global_large[0].kernel_size == (7, 7)
    assert module.global_dilated[0].groups == 192
    assert module.global_dilated[0].dilation == (3, 3)
    assert module.global_pool_projection.groups == 1
    assert module.global_pool_projection.bias is not None
    assert module.router.groups == 8
    assert torch.count_nonzero(module.router.weight) == 0
    assert torch.count_nonzero(module.router.bias) == 0
    assert torch.count_nonzero(module.scale_head.weight) == 0
    assert torch.count_nonzero(module.scale_head.bias) == 0
    assert torch.count_nonzero(module.scale_expert_slopes) == 0
    assert torch.count_nonzero(module.output_projection.weight) > 0
    assert torch.count_nonzero(module.support_head.weight) > 0


def test_grouped_router_preserves_each_groups_two_expert_channel_order() -> None:
    module = _tiny_module().eval()
    with torch.no_grad():
        module.router.bias.copy_(
            torch.tensor([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0])
        )
    local = torch.zeros(1, 24, 2, 3)
    context = torch.zeros_like(local)

    scale = torch.full((1, 1, 2, 3), 0.5)
    _, routes = module._route(torch.zeros_like(local), local, context, scale)

    expected_local = torch.tensor([1.0, 2.0, 3.0, 4.0]).mul(2).sigmoid()
    torch.testing.assert_close(routes[0, 0, :, 0, 0], expected_local)
    torch.testing.assert_close(routes.sum(dim=1), torch.ones_like(routes[:, 0]))


def test_router_reads_shared_reduced_feature_not_expert_sum() -> None:
    module = _tiny_module().eval()
    with torch.no_grad():
        module.router.weight.zero_()
        module.router.bias.zero_()
        # Each group's local logit reads its first reduced input channel.  The
        # experts stay zero, so routing through local+global would remain 0.5.
        module.router.weight[0::2, 0, 0, 0] = 1.0
    reduced = torch.ones(1, 24, 2, 3)
    local = torch.zeros_like(reduced)
    context = torch.zeros_like(reduced)

    scale = torch.full((1, 1, 2, 3), 0.5)
    fused, routes = module._route(reduced, local, context, scale)

    assert torch.count_nonzero(fused) == 0
    assert torch.all(routes[:, 0] > 0.73)
    assert torch.all(routes[:, 1] < 0.27)


def test_yaml_changes_only_decoder_type_without_changing_graph_or_p3_indices() -> None:
    base = _yaml(BASE_CFG)
    control = _yaml(CONTROL_CFG)
    method = _yaml(METHOD_CFG)

    assert control == base
    expected = deepcopy(base)
    expected["head"][-1][2] = "RAFDRRTDETRDecoder"
    assert method == expected
    assert method["head"][-1][0] == [21, 24, 27]
    assert len(method["backbone"] + method["head"]) == 29


def test_full_graph_preserves_layer28_public_state_and_isolates_only_ra_state() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        control = RAGLGMControlDetectionModel(nc=10, verbose=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        method = RAGLGMDetectionModel(nc=10, verbose=False)

    assert control.model[-1].i == method.model[-1].i == 28
    assert control.model[-1].f == method.model[-1].f == [21, 24, 27]
    assert isinstance(method.model[-1], RAFDRRTDETRDecoder)
    image = torch.rand(1, 3, 128, 128)
    control.eval()
    method.eval()
    with torch.no_grad():
        control_output = control.predict(image)
        method_output = method.predict(image)
    assert torch.equal(control_output[0], method_output[0])
    public, private = partition_ra_glgm_state_dicts(
        control.state_dict(), method.state_dict()
    )
    assert set(public) == set(control.state_dict())
    assert private and all(name.startswith(RA_GLGM_PRIVATE_PREFIX) for name in private)
    assert sum(parameter.numel() for parameter in method.parameters()) - sum(
        parameter.numel() for parameter in control.parameters()
    ) == 813_018

    artifact = build_ra_glgm_initial_state(
        control.state_dict(), method.state_dict(), metadata={"seed": 0}
    )
    validate_ra_glgm_initial_state(artifact)
    load_ra_glgm_initial_state(control, artifact, variant="baseline")
    load_ra_glgm_initial_state(method, artifact, variant="ra_glgm")


def test_real_training_loss_reuses_seven_stock_matches_and_opens_expected_gradients() -> None:
    torch.manual_seed(0)
    model = RAGLGMDetectionModel(nc=3, verbose=False).train()
    batch = {
        "img": torch.rand(1, 3, 128, 128),
        "cls": torch.tensor([[1.0], [-1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]]),
        "batch_idx": torch.tensor([0.0, 0.0]),
    }

    loss, displayed = model.loss(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(displayed).all()
    assert model.criterion.stock_match_calls == 7
    assert model.criterion.fgl_extra_match_calls == 0
    assert model.criterion.last_normal_decoder_assignment is not None
    assert model.last_ra_glgm_losses["loss_ra_support"] > 0
    assert model.last_ra_glgm_losses["loss_ra_scale"] > 0
    assert model.ra_glgm.alpha.grad is not None
    assert model.ra_glgm.alpha.grad.abs().sum() > 0
    assert model.ra_glgm.support_head.weight.grad is not None
    assert model.ra_glgm.support_head.weight.grad.abs().sum() > 0
    assert model.ra_glgm.output_projection.weight.grad is not None
    assert torch.count_nonzero(model.ra_glgm.output_projection.weight.grad) == 0
    assert model.ra_glgm.scale_head.weight.grad is not None
    assert model.ra_glgm.scale_head.weight.grad.abs().sum() > 0


def test_full_graph_first_step_public_gradients_match_control_within_fp_tolerance() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        control = RAGLGMControlDetectionModel(nc=3, verbose=False).train()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        method = RAGLGMDetectionModel(nc=3, verbose=False).train()
    control_state = control.state_dict()
    method_state = method.state_dict()
    for name, tensor in control_state.items():
        assert torch.equal(tensor, method_state[name]), name

    batch = {
        "img": torch.rand(1, 3, 128, 128),
        "cls": torch.tensor([[1.0], [-1.0]]),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]]
        ),
        "batch_idx": torch.tensor([0.0, 0.0]),
    }
    rng_state = torch.get_rng_state()
    control_loss, _ = control.loss(batch)
    torch.set_rng_state(rng_state)
    method_loss, _ = method.loss(batch)
    control_loss.backward()
    method_loss.backward()

    method_parameters = dict(method.named_parameters())
    for name, parameter in control.named_parameters():
        other = method_parameters[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad,
                other.grad,
                rtol=1e-6,
                atol=1e-8,
                msg=name,
            )


def test_trainer_partitions_every_parameter_once_and_both_arms_share_dataset_path() -> None:
    torch.manual_seed(0)
    model = RAGLGMDetectionModel(nc=3, verbose=False)
    trainer = object.__new__(RAGLGMTrainer)
    trainer.model = model

    groups = trainer.gradient_parameter_groups()
    identifiers = [id(parameter) for parameters in groups.values() for parameter in parameters]

    assert set(groups) == {
        "gradient_norm",
        "fdr_gradient_norm",
        "ra_glgm_gradient_norm",
    }
    assert len(identifiers) == len(set(identifiers))
    assert set(identifiers) == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert RAGLGMTrainer.build_dataset is RAGLGMControlTrainer.build_dataset


def test_residual_targets_use_matched_difficulty_and_unmatched_gt_one() -> None:
    pred_boxes = torch.tensor([[[0.25, 0.25, 0.20, 0.20], [0.8, 0.8, 0.1, 0.1]]])
    pred_scores = torch.full((1, 2, 3), -8.0)
    pred_scores[0, 0, 1] = 8.0
    boxes = torch.tensor([[0.25, 0.25, 0.20, 0.20], [0.75, 0.75, 0.1, 0.1]])
    classes = torch.tensor([[1.0], [2.0]])
    batch_idx = torch.tensor([0.0, 0.0])

    targets = build_residual_difficulty_targets(
        pred_bboxes=pred_boxes,
        pred_scores=pred_scores,
        detection_bboxes=boxes,
        detection_classes=classes,
        detection_batch_idx=batch_idx,
        match_indices=[(torch.tensor([0]), torch.tensor([0]))],
        all_bboxes=boxes,
        all_classes=classes,
        all_batch_idx=batch_idx,
        height=16,
        width=16,
        chunk_size=1,
    )

    assert targets.difficulty[0] == pytest.approx(0.25)
    assert targets.difficulty[1] == 1.0
    assert targets.heatmap[0, 0, 12, 12] == pytest.approx(1.0)
    assert targets.heatmap.dtype == torch.float32
    assert not targets.heatmap.requires_grad


def test_matched_difficulty_uses_target_class_probability_and_aligned_iou() -> None:
    scores = torch.tensor([[[8.0, 0.0, 8.0]]])
    box = torch.tensor([[0.75, 0.75, 0.1, 0.1]])
    targets = build_residual_difficulty_targets(
        pred_bboxes=torch.tensor([[[0.2, 0.2, 0.1, 0.1]]]),
        pred_scores=scores,
        detection_bboxes=box,
        detection_classes=torch.tensor([[1.0]]),
        detection_batch_idx=torch.tensor([0.0]),
        match_indices=[(torch.tensor([0]), torch.tensor([0]))],
        all_bboxes=box,
        all_classes=torch.tensor([[1.0]]),
        all_batch_idx=torch.tensor([0.0]),
        height=16,
        width=16,
    )

    # Target-class sigmoid(0)=0.5 and aligned IoU=0: 0.7*0.5 + 0.3*1.
    assert targets.difficulty[0] == pytest.approx(0.65)
    assert targets.heatmap[0, 0, 12, 12] == pytest.approx(0.65)


def test_scale_targets_use_frozen_empirical_cdf_and_train_feature_gate() -> None:
    knots = torch.tensor(SCALE_LOG_AREA_KNOTS)
    torch.testing.assert_close(
        log_area_to_empirical_cdf(knots),
        torch.linspace(0.0, 1.0, len(SCALE_LOG_AREA_KNOTS)),
        rtol=0,
        atol=2e-7,
    )
    assert log_area_to_empirical_cdf(torch.tensor([-10.0])).item() == 0.0
    assert log_area_to_empirical_cdf(torch.tensor([20.0])).item() == 1.0

    side_lengths = torch.tensor([8.0, 24.0, 32.0]) / 640.0
    boxes = torch.tensor([[0.25, 0.25], [0.50, 0.50], [0.75, 0.75]])
    boxes = torch.cat((boxes, side_lengths[:, None].expand(-1, 2)), dim=1)
    targets = build_residual_difficulty_targets(
        pred_bboxes=boxes.unsqueeze(0),
        pred_scores=torch.full((1, 3, 3), 8.0),
        detection_bboxes=boxes,
        detection_classes=torch.tensor([[0.0], [1.0], [2.0]]),
        detection_batch_idx=torch.zeros(3),
        match_indices=[(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))],
        all_bboxes=boxes,
        all_classes=torch.tensor([[0.0], [1.0], [2.0]]),
        all_batch_idx=torch.zeros(3),
        height=80,
        width=80,
    )

    assert targets.difficulty[0] == pytest.approx(0.25)
    assert targets.difficulty[1] == pytest.approx(0.25)
    assert targets.difficulty[2] == pytest.approx(0.25)
    assert targets.scale_values is not None
    expected = log_area_to_empirical_cdf(
        torch.tensor([64.0, 576.0, 1024.0]).log()
    )
    torch.testing.assert_close(targets.scale_values, expected)

    logits = torch.zeros(1, 1, 80, 80, requires_grad=True)
    predictions = logits.sigmoid()
    loss = scale_conditioning_loss(predictions, targets)
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    routes = torch.full((1, 2, 8, 80, 80), 0.5)
    diagnostics = scale_prediction_diagnostics(predictions.detach(), targets, routes)
    assert diagnostics["scale_instances"] == 3
    assert diagnostics["scale_prediction_mean"] == pytest.approx(0.5)
    assert diagnostics["route_entropy"] == pytest.approx(torch.log(torch.tensor(2.0)).item())


def _scale_targets(values: list[float]) -> ResidualDifficultyTargets:
    instances = len(values)
    mask = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    return ResidualDifficultyTargets(
        heatmap=torch.ones_like(mask, dtype=torch.float32),
        valid_mask=mask,
        difficulty=torch.ones(instances),
        scale_boxes=torch.tensor([[0.5, 0.5, 0.1, 0.1]]).repeat(instances, 1),
        scale_batch_idx=torch.zeros(instances, dtype=torch.long),
        scale_values=torch.tensor(values),
    )


def test_scale_loss_is_instance_balanced_instead_of_support_area_weighted() -> None:
    targets = ResidualDifficultyTargets(
        heatmap=torch.ones(1, 1, 16, 16),
        valid_mask=torch.ones(1, 1, 16, 16, dtype=torch.bool),
        difficulty=torch.ones(2),
        scale_boxes=torch.tensor(
            [[0.25, 0.5, 0.02, 0.02], [0.75, 0.5, 0.75, 0.75]]
        ),
        scale_batch_idx=torch.zeros(2, dtype=torch.long),
        scale_values=torch.tensor([0.0, 1.0]),
    )
    predictions = torch.zeros(1, 1, 16, 16, requires_grad=True)
    loss = scale_conditioning_loss(predictions, targets)
    loss.backward()

    # SmoothL1 errors are 0 and 0.5; per-instance normalization yields 0.25
    # even though the second target covers much more of the feature map.
    assert float(loss.detach()) == pytest.approx(0.25)
    assert predictions.grad is not None and predictions.grad.abs().sum() > 0


def test_epoch_scale_statistics_are_instance_weighted_not_last_batch() -> None:
    statistics = _RAEpochScaleStatistics()
    first = torch.full((1, 1, 1, 1), 0.2)
    second = torch.full((1, 1, 1, 1), 0.8)
    routes = torch.full((1, 2, 4, 1, 1), 0.5)
    slopes = torch.zeros(1, 4, 1, 1)
    statistics.update(first, _scale_targets([0.1]), torch.tensor(0.2), routes, slopes)
    statistics.update(
        second, _scale_targets([0.9, 0.9, 0.9]), torch.tensor(0.6), routes, slopes
    )

    values = statistics.values()
    assert values["scale_instances"] == 4
    assert values["loss_ra_scale"] == pytest.approx(0.5)
    assert values["scale_prediction_mean"] == pytest.approx(0.65)
    assert values["scale_target_mean"] == pytest.approx(0.70)
    assert values["scale_mae"] == pytest.approx(0.10)
    assert values["scale_pearson"] == pytest.approx(1.0)
    assert values["scale_spearman"] == pytest.approx(1.0)


def test_epoch_route_statistics_detect_balanced_and_noncollapsed_loads() -> None:
    uniform = _RAEpochScaleStatistics()
    uniform_routes = torch.full((1, 2, 4, 1, 1), 0.5)
    uniform.update(
        torch.full((1, 1, 1, 1), 0.5),
        _scale_targets([0.2, 0.5, 0.8]),
        torch.tensor(0.1),
        uniform_routes,
        torch.zeros(1, 4, 1, 1),
    )
    uniform_values = uniform.values()
    assert uniform_values["route_global_mean"] == pytest.approx(0.5)
    assert uniform_values["route_load_min"] == pytest.approx(0.5)
    assert uniform_values["route_load_max"] == pytest.approx(0.5)
    assert uniform_values["route_entropy"] == pytest.approx(torch.log(torch.tensor(2.0)).item())

    nontrivial = _RAEpochScaleStatistics()
    route_loads = torch.tensor([0.1, 0.3, 0.7, 0.9]).view(1, 1, 4, 1, 1)
    nontrivial_routes = torch.cat((1.0 - route_loads, route_loads), dim=1)
    nontrivial.update(
        torch.full((1, 1, 1, 1), 0.8),
        _scale_targets([0.2, 0.8]),
        torch.tensor(0.1),
        nontrivial_routes,
        torch.full((1, 4, 1, 1), 0.5),
    )
    nontrivial_values = nontrivial.values()
    assert nontrivial_values["route_load_min"] == pytest.approx(0.1)
    assert nontrivial_values["route_load_max"] == pytest.approx(0.9)
    assert nontrivial_values["route_global_std"] > 0
    assert nontrivial_values["scale_slope_rms"] > 0
    assert nontrivial_values["scale_modulation_route_delta_mean"] > 0


def test_epoch_evidence_uses_aggregated_scale_statistics() -> None:
    class FakeModel:
        last_fdr_losses: dict[str, torch.Tensor] = {}
        last_ra_glgm_losses = {
            "loss_ra_scale": torch.tensor(99.0),
            "scale_pearson": torch.tensor(-1.0),
        }

        @staticmethod
        def ra_glgm_epoch_statistics() -> dict[str, torch.Tensor]:
            return {
                "loss_ra_scale": torch.tensor(0.4),
                "scale_instances": torch.tensor(40.0),
                "scale_mae": torch.tensor(0.1),
                "scale_rmse": torch.tensor(0.2),
                "scale_prediction_mean": torch.tensor(0.45),
                "scale_prediction_std": torch.tensor(0.25),
                "scale_target_mean": torch.tensor(0.5),
                "scale_target_std": torch.tensor(0.3),
                "scale_pearson": torch.tensor(0.6),
                "scale_spearman": torch.tensor(0.55),
                "route_entropy": torch.tensor(0.65),
                "route_global_mean": torch.tensor(0.5),
                "route_global_std": torch.tensor(0.1),
                "route_load_min": torch.tensor(0.3),
                "route_load_max": torch.tensor(0.7),
                "scale_route_correlation_mean_abs": torch.tensor(0.2),
                "scale_route_correlation_max_abs": torch.tensor(0.4),
                "scale_slope_rms": torch.tensor(0.1),
                "scale_slope_max_abs": torch.tensor(0.2),
                "scale_modulation_route_delta_mean": torch.tensor(0.01),
                "scale_modulation_route_delta_max": torch.tensor(0.03),
            }

    losses = _private_losses(type("Trainer", (), {"model": FakeModel()})(), "ra_glgm")
    assert losses["loss_ra_scale"] == pytest.approx(0.4)
    assert losses["scale_instances"] == pytest.approx(40.0)
    assert losses["scale_pearson"] == pytest.approx(0.6)
    assert losses["scale_spearman"] == pytest.approx(0.55)
    assert losses["route_load_min"] == pytest.approx(0.3)
    assert losses["route_load_max"] == pytest.approx(0.7)
    assert losses["scale_slope_rms"] == pytest.approx(0.1)
    assert losses["scale_modulation_route_delta_mean"] == pytest.approx(0.01)


def test_ignored_boxes_are_neither_targets_nor_negative_support_pixels() -> None:
    raw = {
        "bboxes": torch.tensor([[0.25, 0.25, 0.1, 0.1], [0.75, 0.75, 0.2, 0.2]]),
        "cls": torch.tensor([[1.0], [-1.0]]),
        "batch_idx": torch.tensor([0.0, 0.0]),
    }
    detection = filter_detection_batch(raw)
    targets = build_residual_difficulty_targets(
        pred_bboxes=torch.tensor([[[0.25, 0.25, 0.1, 0.1]]]),
        pred_scores=torch.zeros(1, 1, 3),
        detection_bboxes=detection["bboxes"],
        detection_classes=detection["cls"],
        detection_batch_idx=detection["batch_idx"],
        match_indices=[(torch.tensor([0]), torch.tensor([0]))],
        all_bboxes=raw["bboxes"],
        all_classes=raw["cls"],
        all_batch_idx=raw["batch_idx"],
        height=16,
        width=16,
    )

    assert len(targets.difficulty) == 1
    assert targets.heatmap[0, 0, 12, 12] == 0
    assert not targets.valid_mask[0, 0, 12, 12]


def test_empty_gt_and_all_ignore_batch_produces_finite_masked_loss() -> None:
    targets = build_residual_difficulty_targets(
        pred_bboxes=torch.zeros(1, 2, 4),
        pred_scores=torch.zeros(1, 2, 3),
        detection_bboxes=torch.empty(0, 4),
        detection_classes=torch.empty(0, 1),
        detection_batch_idx=torch.empty(0),
        match_indices=[(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))],
        all_bboxes=torch.tensor([[0.5, 0.5, 1.0, 1.0]]),
        all_classes=torch.tensor([[-1.0]]),
        all_batch_idx=torch.tensor([0.0]),
        height=8,
        width=8,
    )
    support = torch.full((1, 1, 8, 8), 0.5, requires_grad=True)
    loss = residual_support_focal_loss(support, targets)
    loss.backward()

    assert targets.difficulty.numel() == 0
    assert torch.count_nonzero(targets.heatmap) == 0
    assert torch.count_nonzero(targets.valid_mask) == 0
    assert loss == 0
    assert support.grad is not None and torch.count_nonzero(support.grad) == 0
