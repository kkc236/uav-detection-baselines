"""Synthetic mechanism checks, not evidence of a dataset-level gain."""

import pytest
import torch

from src.bpdd_loss import BPDDOptions, assignment_consistent_bpdd_loss
from src.fdr_head import cumulative_distribution_logits
from src.fdr_math import decode_feasible_fdr_boxes, weighting_function


STUDENT = [.009968596697947913, .9107677603797396, .07926364292231247]
TEACHER = [.185676236431577, .8113069378623573, .0030168257060655852]


def pair(student=STUDENT, teacher=TEACHER, indices=(16, 17, 32)):
    p = torch.full((2, 1, 1, 4, 33), 1e-8)
    p[0, ..., list(indices)] = torch.tensor(student)
    p[1, ..., list(indices)] = torch.tensor(teacher)
    logits = (p / p.sum(-1, keepdim=True)).log().requires_grad_()
    ref = torch.tensor([[[.5, .5, .2, .2]]], requires_grad=True)
    gt = ref.detach().reshape(1, 4)
    matches = [(torch.tensor([0]),) * 3] * 2
    return logits, ref, gt, matches


def options(**kwargs):
    return BPDDOptions(assignment_mode="consistent", weight=.15,
                       decoded_iou_gate=True, **kwargs)


def loss(inputs, **kwargs):
    return assignment_consistent_bpdd_loss(*inputs, options=options(**kwargs))


def centered_iou(logits, reference):
    mu = (logits.softmax(-1) * weighting_function().to(logits)).sum(-1)
    boxes, _ = decode_feasible_fdr_boxes(reference, mu)
    inter = torch.minimum(boxes[..., 2:], reference[..., 2:]).prod(-1)
    return inter / (boxes[..., 2:].prod(-1) + reference[..., 2:].prod(-1) - inter)


def test_existing_iou_gate_can_admit_a_locally_harmful_kl_step():
    inputs = pair()
    z, ref, _, _ = inputs
    result = loss(inputs)
    grad = torch.autograd.grad(result.loss, z)[0]
    before = centered_iou(z[0], ref)
    assert centered_iou(z[1], ref) > before
    assert result.statistics["active_edge_ratio"] == 1
    assert centered_iou(z[0] - grad[0], ref) < before


def test_mean_huber_improves_the_fixed_kl_counterexample():
    inputs = pair()
    z, ref, _, _ = inputs
    result = loss(inputs, distribution_objective="mean_huber")
    result.loss.backward()
    assert result.loss > 0
    assert centered_iou(z[0] - z.grad[0], ref) > centered_iou(z[0], ref)
    assert torch.count_nonzero(z.grad[1]) == 0
    assert ref.grad is None


def cumulative_case():
    p = torch.full((3, 1, 1, 4, 33), 1e-8)
    for i, vals in enumerate(([.3, .4, .3], [.6, .01, .39], [.95, .04, .01])):
        p[i, ..., [16, 17, 32]] = torch.tensor(vals)
    z = (p / p.sum(-1, keepdim=True)).log().reshape(3, 1, 1, 132)
    delta = torch.cat([z[:1], z[1:] - z[:-1]], 0).detach().requires_grad_()
    cumulative = cumulative_distribution_logits(delta).reshape(3, 1, 1, 4, 33)
    ref = torch.tensor([[[.5, .5, .2, .2]]])
    gt = torch.tensor([[.3, .5, .2, .2], [.5, .5, .2, .2]])
    zero, one = torch.tensor([0]), torch.tensor([1])
    matches = [(zero, zero, zero), (zero, zero, one), (zero, zero, one)]
    return delta, (cumulative, ref, gt, matches)


def test_existing_assignment_gate_does_not_isolate_cumulative_history():
    delta, inputs = cumulative_case()
    result = loss(inputs)
    result.loss.backward()
    assert result.statistics["stable_source_matches"] == 1
    assert delta.grad[0].norm() > 0
    torch.testing.assert_close(delta.grad[0], delta.grad[1])
    assert delta.grad[2].norm() == 0


@pytest.mark.parametrize("objective", ["kl", "mean_huber"])
def test_residual_only_preserves_loss_and_blocks_direct_history(objective):
    delta_old, inputs_old = cumulative_case()
    delta_new, inputs_new = cumulative_case()
    old = loss(inputs_old, distribution_objective=objective)
    new = loss(inputs_new, distribution_objective=objective, residual_gradient_only=True)
    torch.testing.assert_close(old.loss, new.loss, rtol=0, atol=0)
    old.loss.backward()
    new.loss.backward()
    assert delta_old.grad[0].norm() > 0
    assert delta_new.grad[0].norm() == 0
    assert delta_new.grad[1].norm() > 0
    assert delta_new.grad[2].norm() == 0
    torch.testing.assert_close(delta_old.grad[1], delta_new.grad[1])


def test_explicit_defaults_preserve_value_and_gradient():
    a, b = pair(), pair()
    first = loss(a)
    second = loss(b, distribution_objective="kl", residual_gradient_only=False)
    torch.testing.assert_close(first.loss, second.loss, rtol=0, atol=0)
    ga = torch.autograd.grad(first.loss, a[0])[0]
    gb = torch.autograd.grad(second.loss, b[0])[0]
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)


def test_mean_huber_rejects_better_teacher_that_crosses_gt():
    inputs = pair([.1, .6, .3], [.065, .9, .035], indices=(0, 16, 32))
    assert loss(inputs).loss > 0
    result = loss(inputs, distribution_objective="mean_huber")
    assert result.loss == 0
    result.loss.backward()
    assert torch.count_nonzero(inputs[0].grad) == 0


def test_mean_huber_rejects_unrepresentable_raw_gt():
    z, ref, _, matches = pair([.7, .25, .05], [.3, .2, .5])
    gt = torch.tensor([[.5, .5, 1., 1.]])  # raw distances +8, beyond support +4
    inputs = z, ref, gt, matches
    assert loss(inputs).loss > 0
    result = loss(inputs, distribution_objective="mean_huber")
    assert result.loss == 0
    result.loss.backward()
    assert torch.count_nonzero(z.grad) == 0


@pytest.mark.parametrize("kwargs", [
    {"distribution_objective": "bad"},
    {"residual_gradient_only": "false"},
    {"residual_gradient_only": 1},
])
def test_bad_joint_options_rejected(kwargs):
    with pytest.raises(ValueError):
        options(**kwargs)


def test_candidate_contracts_require_consistency_and_geometry_gate():
    with pytest.raises(ValueError):
        BPDDOptions(residual_gradient_only=True)
    with pytest.raises(ValueError):
        BPDDOptions(assignment_mode="consistent", distribution_objective="mean_huber")


@pytest.mark.parametrize("mode", ["disabled", "empty"])
def test_joint_candidate_zero_contract(mode):
    z, ref, gt, matches = pair()
    if mode == "empty":
        matches = [(torch.empty(0, dtype=torch.long),) * 3] * 2
    result = assignment_consistent_bpdd_loss(z, ref, gt, matches,
        options=options(enabled=mode != "disabled", residual_gradient_only=True,
                        distribution_objective="mean_huber"))
    assert result.loss == 0
    result.loss.backward()
    assert torch.count_nonzero(z.grad) == 0


def test_candidate_yaml_changes_no_model_state_and_preserves_lrs():
    from pathlib import Path
    from src.rtdetr_fdr_bpdd import FDRBPDDDetectionModel, _parse_bpdd_options
    root = Path(__file__).resolve().parents[1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(129)
        base = FDRBPDDDetectionModel(root / "configs/rtdetr-l-lrs-fdr-bpdd.yaml", nc=10, verbose=False)
        torch.manual_seed(129)
        candidate = FDRBPDDDetectionModel(root / "configs/rtdetr-l-lrs-fdr-bpdd-joint-candidate.yaml", nc=10, verbose=False)
    assert candidate.bpdd_options.residual_gradient_only
    assert candidate.bpdd_options.distribution_objective == "mean_huber"
    criterion = candidate.init_criterion()
    assert criterion.reliability_shrinkage_alpha == .25
    assert criterion.bpdd_options == candidate.bpdd_options
    old_state, new_state = base.state_dict(), candidate.state_dict()
    assert old_state.keys() == new_state.keys()
    for key in old_state:
        torch.testing.assert_close(old_state[key], new_state[key], rtol=0, atol=0)
    assert not _parse_bpdd_options({"assignment_mode": "consistent"}).residual_gradient_only
    with pytest.raises(ValueError):
        _parse_bpdd_options({"assignment_mode": "consistent", "residual_gradient_only": "false"})


def test_candidate_trainer_honors_explicit_model_config():
    from pathlib import Path
    from src.rtdetr_lrs_system import LRSFDRBPDDTrainer
    root = Path(__file__).resolve().parents[1]
    trainer = LRSFDRBPDDTrainer.__new__(LRSFDRBPDDTrainer)
    trainer.data = {"nc": 10, "channels": 3}
    trainer.experiment_seed = 0
    trainer.initial_state_path = None
    model = trainer.get_model(
        cfg=root / "configs/rtdetr-l-lrs-fdr-bpdd-joint-candidate.yaml",
        verbose=False,
    )
    assert model.bpdd_options.residual_gradient_only
    assert model.bpdd_options.distribution_objective == "mean_huber"


def test_eligible_mean_gradients_agree_with_gt_in_independent_logit_space():
    from src.bpdd_loss import _mean_distillation_terms
    generator = torch.Generator().manual_seed(823)
    z = torch.randn(512, 4, 33, generator=generator, requires_grad=True)
    teacher = torch.randn(512, 4, 33, generator=generator).log_softmax(-1)
    ref = torch.tensor([[.5, .5, .2, .2]]).repeat(512, 1)
    terms, eligible = _mean_distillation_terms(z.log_softmax(-1), teacher, ref, ref)
    assert eligible.sum() > 100
    gradient = torch.autograd.grad((terms * eligible).sum(), z)[0]
    p = z.softmax(-1)
    support = weighting_function()
    mu = (p * support).sum(-1)
    # GT distance is zero. Its local squared-error derivative under the update
    # must be nonpositive for every retained independent edge, not just on average.
    mean_jacobian = p * (support - mu[..., None])
    directional_derivative = -mu * (mean_jacobian * gradient).sum(-1)
    assert (directional_derivative[eligible] < 0).all()
    assert torch.count_nonzero(gradient[~eligible]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_joint_candidate_has_finite_loss_and_student_gradients(dtype):
    z, ref, gt, matches = pair()
    z = z.detach().to(device="cuda", dtype=dtype).requires_grad_()
    ref, gt = ref.detach().cuda(), gt.cuda()
    matches = [tuple(value.cuda() for value in triple) for triple in matches]
    with torch.autocast("cuda", dtype=torch.float16):
        result = loss((z, ref, gt, matches), residual_gradient_only=True,
                      distribution_objective="mean_huber")
    result.loss.backward()
    assert result.loss.dtype == torch.float32
    assert torch.isfinite(result.loss) and result.loss > 0
    assert torch.isfinite(z.grad).all()
    assert z.grad[0].norm() > 0
    assert z.grad[1].norm() == 0
