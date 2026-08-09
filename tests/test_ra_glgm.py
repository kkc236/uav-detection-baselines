from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from src.ra_glgm import RAGLGM
from src.ra_glgm_head import RAFDRRTDETRDecoder
from src.ra_glgm_loss import (
    ResidualDifficultyTargets,
    build_residual_difficulty_targets,
    residual_support_focal_loss,
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

    output, routes, support = module.forward_with_diagnostics(public)
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


@pytest.mark.parametrize("shape", [(1, 32, 7, 13), (2, 32, 16, 9)])
def test_dynamic_shapes_and_cpu_amp_remain_finite(shape: tuple[int, ...]) -> None:
    module = _tiny_module().train()
    x = torch.randn(shape)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output, routes, support = module.forward_with_diagnostics(x)

    assert output.shape == x.shape
    assert routes.shape[-2:] == x.shape[-2:]
    assert support.shape == (shape[0], 1, shape[2], shape[3])
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

    assert module.private_parameter_count == 812_817
    assert module.private_parameter_count / BASELINE_PARAMETERS == pytest.approx(
        0.0245144755
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

    _, routes = module._route(torch.zeros_like(local), local, context)

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

    fused, routes = module._route(reduced, local, context)

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
    ) == 812_817

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
    assert model.ra_glgm.alpha.grad is not None
    assert model.ra_glgm.alpha.grad.abs().sum() > 0
    assert model.ra_glgm.support_head.weight.grad is not None
    assert model.ra_glgm.support_head.weight.grad.abs().sum() > 0
    assert model.ra_glgm.output_projection.weight.grad is not None
    assert torch.count_nonzero(model.ra_glgm.output_projection.weight.grad) == 0


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
