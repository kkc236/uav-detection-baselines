import pytest
import torch

from src.bpdd_loss import BPDDOptions, assignment_consistent_bpdd_loss, decoded_teacher_iou_gate
from src.fdr_math import Integral, decode_feasible_fdr_boxes, cxcywh_to_xyxy


def run_pair(student, teacher, gate):
    p = torch.full((2, 1, 1, 4, 33), 1e-8)
    for layer, values in enumerate((student, teacher)):
        p[layer, ..., [16, 17, 32]] = torch.tensor(values)
    p = p / p.sum(-1, keepdim=True)
    logits = p.log().requires_grad_()
    ref = torch.tensor([[[.5, .5, .2, .2]]], requires_grad=True)
    gt = ref.detach().reshape(1, 4)
    match = (torch.tensor([0]),) * 3
    result = assignment_consistent_bpdd_loss(
        logits, ref, gt, [match, match],
        options=BPDDOptions(assignment_mode='consistent', weight=.15,
                            decoded_iou_gate=gate),
    )
    return result, logits, ref


def test_nll_better_localization_worse_is_rejected():
    old, _, _ = run_pair([.60, .39, .01], [.70, .01, .29], False)
    new, logits, ref = run_pair([.60, .39, .01], [.70, .01, .29], True)
    assert old.loss > 0
    assert new.loss == 0
    new.loss.backward()
    assert torch.count_nonzero(logits.grad) == 0
    assert ref.grad is None


def test_better_localization_keeps_student_gradient_only():
    result, logits, ref = run_pair([.60, .01, .39], [.95, .04, .01], True)
    assert result.loss > 0
    result.loss.backward()
    assert torch.count_nonzero(logits.grad[0]) > 0
    assert torch.count_nonzero(logits.grad[1]) == 0
    assert ref.grad is None


@pytest.mark.parametrize('margin', [-1, float('nan'), float('inf'), 1.1])
def test_bad_iou_margin_rejected(margin):
    with pytest.raises(ValueError):
        BPDDOptions(assignment_mode='consistent', decoded_iou_gate=True, iou_margin=margin)


def test_gate_requires_assignment_consistency():
    with pytest.raises(ValueError):
        BPDDOptions(decoded_iou_gate=True)


def test_partial_edges_and_tiny_boxes_match_independent_iou():
    torch.manual_seed(17)
    source = torch.randn(64, 4, 33).log_softmax(-1)
    teacher = torch.randn(64, 4, 33).log_softmax(-1)
    ref = torch.rand(64, 4)
    ref[:, 2:] = torch.logspace(-6, -1, 64)[:, None]
    gt = ref.clone()
    active = torch.rand(64, 4) > .5
    keep, delta = decoded_teacher_iou_gate(source, teacher, ref, gt, active)
    hybrid = torch.where(active[..., None], teacher, source)
    integral = Integral()
    def oracle(logp):
        boxes, _ = decode_feasible_fdr_boxes(ref, integral(logp.reshape(64, 132)))
        a, b = cxcywh_to_xyxy(boxes.double()), cxcywh_to_xyxy(gt.double())
        inter = (torch.minimum(a[:, 2:], b[:, 2:]) - torch.maximum(a[:, :2], b[:, :2])).clamp_min(0).prod(-1)
        return inter / ((a[:, 2:] - a[:, :2]).prod(-1) + (b[:, 2:] - b[:, :2]).prod(-1) - inter)
    expected = oracle(hybrid) - oracle(source)
    torch.testing.assert_close(delta.double(), expected, atol=2e-6, rtol=2e-5)
    assert torch.equal(keep, (expected > 0) & active.any(-1))
    # Completely inactive edges cannot create a fictitious improving target.
    k, d = decoded_teacher_iou_gate(source, teacher, ref, gt, torch.zeros_like(active))
    assert not k.any()
    assert torch.count_nonzero(d) == 0


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_gate_input_fails(bad):
    logp = torch.zeros(1, 4, 33).log_softmax(-1)
    ref = torch.tensor([[.5, .5, .2, .2]])
    teacher = logp.clone()
    teacher[0, 0, 0] = bad
    with pytest.raises(ValueError, match='nonfinite'):
        decoded_teacher_iou_gate(logp, teacher, ref, ref, torch.ones(1, 4, dtype=torch.bool))


def test_yaml_options_and_candidate_model():
    from src.rtdetr_fdr_bpdd import FDRBPDDDetectionModel, _parse_bpdd_options
    from pathlib import Path
    cfg = Path(__file__).resolve().parents[1] / 'configs/rtdetr-l-lrs-fdr-bpdd-iou-candidate.yaml'
    model = FDRBPDDDetectionModel(cfg, nc=10, verbose=False)
    assert model.bpdd_options.decoded_iou_gate
    assert model.bpdd_options.weight == .15
    assert not _parse_bpdd_options({'assignment_mode': 'consistent'}).decoded_iou_gate
    with pytest.raises(ValueError):
        _parse_bpdd_options({'assignment_mode': 'consistent', 'decoded_iou_gate': 'false'})
