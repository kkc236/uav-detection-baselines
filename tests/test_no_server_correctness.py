from __future__ import annotations

import pytest
import torch

from src import fdr_math
from src.bpdd_loss import BPDDOptions, assignment_consistent_bpdd_loss


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("reference_size", [0.1, 1e-8, 0.0])
def test_feasible_decode_retains_positive_xyxy_extent(dtype, reference_size):
    decode = getattr(fdr_math, "decode_feasible_fdr_boxes", None)
    assert callable(decode), "a separate precision-safe decoder is required"
    points = torch.tensor([[0.5, 0.75, reference_size, reference_size]], dtype=dtype)
    raw = torch.full((1, 4), -4.0, dtype=dtype, requires_grad=True)
    boxes, stats = decode(points, raw)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert boxes.dtype == expected_dtype
    xyxy = fdr_math.cxcywh_to_xyxy(boxes)
    assert torch.all(xyxy[:, 2:] > xyxy[:, :2])
    torch.testing.assert_close(boxes[:, :2], points[:, :2].to(expected_dtype))
    assert stats["minimum_decoded_width"] > 0
    boxes.sum().backward()
    assert torch.isfinite(raw.grad).all()
    if reference_size == 0.1:
        assert torch.all(raw.grad != 0)


def test_feasible_decode_keeps_ordinary_boxes_and_preserves_asymmetric_center():
    decode = getattr(fdr_math, "decode_feasible_fdr_boxes", None)
    assert callable(decode)
    points = torch.tensor([[0.5, 0.6, 0.2, 0.3]], dtype=torch.float64)
    for distance in (torch.tensor([[0.1, -0.2, 0.3, 0.4]], dtype=torch.float64),
                     torch.tensor([[-4.0, -3.0, -2.0, -4.0]], dtype=torch.float64)):
        boxes, _ = decode(points, distance)
        original = fdr_math.distance2bbox(points, distance)
        torch.testing.assert_close(boxes[:, :2], original[:, :2], atol=1e-14, rtol=0)
        if torch.all(original[:, 2:] > 0):
            torch.testing.assert_close(boxes, original, atol=1e-14, rtol=0)


def _rare_target_case(nlls):
    reference = torch.tensor([[[0.5, 0.5, 0.25, 0.25]]])
    targets = reference[0].clone()
    # Identical reference/GT yields distance zero, exact support bin 16.
    logits = torch.full((len(nlls), 1, 1, 4, 33), -1000.0)
    logits[..., 0] = 0.0
    for layer, nll in enumerate(nlls):
        logits[layer, ..., 16] = -float(nll)
    logits.requires_grad_(True)
    matches = [(torch.tensor([0]), torch.tensor([0]), torch.tensor([0])) for _ in nlls]
    return logits, reference, targets, matches


@pytest.mark.parametrize("nlls", [(20, 40), (120, 140), (40, 40)])
def test_ac_bpdd_never_accepts_equal_or_worse_rare_target_teacher(nlls):
    logits, ref, gt, matches = _rare_target_case(nlls)
    result = assignment_consistent_bpdd_loss(
        logits, ref, gt, matches, options=BPDDOptions(assignment_mode="consistent"))
    assert result.statistics["mean_reliability"].item() == 0
    assert result.loss.item() == 0
    result.loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_ac_bpdd_underflow_teacher_has_true_improvement_and_detached_gradient():
    logits, ref, gt, matches = _rare_target_case((140, 120))
    result = assignment_consistent_bpdd_loss(
        logits, ref, gt, matches, options=BPDDOptions(assignment_mode="consistent"))
    assert result.statistics["mean_reliability"].item() == pytest.approx((20-0.02)/140)
    assert result.statistics["mean_teacher_improvement"].item() == pytest.approx(20)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[-1], torch.zeros_like(logits.grad[-1]))


def test_ac_bpdd_no_eligible_future_remains_finite_with_extreme_logits():
    logits, ref, gt, matches = _rare_target_case((140, 120, 160))
    empty = torch.empty(0, dtype=torch.long)
    matches[1:] = [(empty, empty, empty), (empty, empty, empty)]
    result = assignment_consistent_bpdd_loss(
        logits, ref, gt, matches, options=BPDDOptions(assignment_mode="consistent"))
    assert result.loss.item() == 0
    result.loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_current_launcher_has_f_baseline_and_versioned_four_arm_contract(tmp_path):
    from scripts import train_visdrone_lrs_system as launcher
    from src.rtdetr_lrs_system import ARM_CONFIGS, MODEL_TYPES, TRAINER_TYPES
    assert set(ARM_CONFIGS) == set(MODEL_TYPES) == set(TRAINER_TYPES) == set("fghi")
    assert set(launcher.ARM_METHODS) == set("fghi")
    settings = {arm: launcher.build_settings(arm, tmp_path/'data.yaml', tmp_path/'runs')
                for arm in "fghi"}
    for arm in "fghi":
        assert settings[arm]["name"].endswith("-v2")
        assert {k:v for k,v in settings[arm].items() if k not in {"model", "name"}} == {
            k:v for k,v in settings["f"].items() if k not in {"model", "name"}}
