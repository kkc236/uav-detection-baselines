from hashlib import sha256

import pytest
import torch

from src.pfcr import (
    PFCR_FEATURE_DIM,
    PFCRGate,
    _stable_unique_indices,
    one_to_one_union_teacher,
    pfcr_boundary_loss,
    pfcr_features,
    pfcr_split,
    protected_merge,
    stock_predictions,
)


NUM_QUERIES = 300
NUM_CLASSES = 10


def synthetic_pair(
    *, dtype: torch.dtype = torch.float32, requires_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes = torch.tensor([0.5, 0.5, 0.2, 0.1], dtype=dtype).repeat(
        NUM_QUERIES, 1
    )
    fdr_boxes = boxes.clone()
    cm_boxes = boxes.clone()
    fdr_logits = torch.linspace(
        -3.0, 3.0, NUM_QUERIES * NUM_CLASSES, dtype=dtype
    ).reshape(NUM_QUERIES, NUM_CLASSES)
    cm_logits = torch.flip(fdr_logits, dims=(0,)).clone()
    tensors = (fdr_boxes, fdr_logits, cm_boxes, cm_logits)
    if requires_grad:
        tensors = tuple(value.requires_grad_() for value in tensors)
    return tensors


def test_split_is_hash_deterministic_disjoint_and_uses_basename():
    image_ids = [f"img-{index:04d}.jpg" for index in range(100)]
    first = {name: pfcr_split(name) for name in image_ids}
    second = {name: pfcr_split(name) for name in reversed(image_ids)}

    assert first == second
    assert set(first.values()) == {"train", "dev"}
    assert all(
        (int(sha256(name.encode("utf-8")).hexdigest(), 16) % 5 == 0)
        == (split == "dev")
        for name, split in first.items()
    )
    assert pfcr_split("nested/directory/img-0007.jpg") == first["img-0007.jpg"]
    assert pfcr_split(r"nested\directory\img-0007.jpg") == first["img-0007.jpg"]


def test_pfcr_features_have_exact_shape_dtype_device_and_are_detached():
    detector_tensors = synthetic_pair(dtype=torch.float64, requires_grad=True)

    features = pfcr_features(*detector_tensors)

    assert PFCR_FEATURE_DIM == 35
    assert features.shape == (NUM_QUERIES, NUM_CLASSES, PFCR_FEATURE_DIM)
    assert features.dtype == torch.float64
    assert features.device == detector_tensors[0].device
    assert features.is_contiguous()
    assert not features.requires_grad
    assert features.grad_fn is None
    assert bool(torch.isfinite(features).all())


def test_flattened_ranks_are_stable_and_query_major():
    fdr_boxes, _, cm_boxes, _ = synthetic_pair()
    tied_logits = torch.zeros(NUM_QUERIES, NUM_CLASSES)

    features = pfcr_features(fdr_boxes, tied_logits, cm_boxes, tied_logits.clone())

    denominator = NUM_QUERIES * NUM_CLASSES - 1
    torch.testing.assert_close(features[0, 0, 5], torch.tensor(0.0))
    torch.testing.assert_close(features[0, 1, 5], torch.tensor(1 / denominator))
    torch.testing.assert_close(features[1, 0, 5], torch.tensor(10 / denominator))
    torch.testing.assert_close(features[0, 0, 17], torch.tensor(0.0))
    torch.testing.assert_close(features[0, 1, 17], torch.tensor(1 / denominator))


def test_matching_tie_uses_lowest_fdr_query_index():
    fdr_boxes, _, cm_boxes, cm_logits = synthetic_pair()
    fdr_logits = torch.zeros(NUM_QUERIES, NUM_CLASSES)
    fdr_logits[0, 1] = 4.0
    fdr_logits[1, 1] = 2.0

    features = pfcr_features(fdr_boxes, fdr_logits, cm_boxes, cm_logits)

    expected_query_max = torch.sigmoid(torch.tensor(4.0))
    torch.testing.assert_close(features[0, 0, 14], expected_query_max)


def test_invalid_width_is_preserved_but_all_overlap_contributions_are_zero():
    fdr_boxes, fdr_logits, cm_boxes, cm_logits = synthetic_pair()
    cm_boxes[0] = torch.tensor([0.4, 0.6, -0.2, 0.1])
    original_cm_boxes = cm_boxes.clone()
    fdr_logits[:, 0] = -4.0
    fdr_logits[0, 0] = -2.0
    fdr_logits[1, 0] = 8.0

    features = pfcr_features(fdr_boxes, fdr_logits, cm_boxes, cm_logits)

    assert torch.equal(cm_boxes, original_cm_boxes)
    torch.testing.assert_close(features[0, :, 8], cm_boxes[0, 2].expand(NUM_CLASSES))
    assert torch.equal(features[0, :, 18], torch.zeros(NUM_CLASSES))
    torch.testing.assert_close(features[0, 0, 12], fdr_logits[0, 0])


@pytest.mark.parametrize("tensor_index", range(4))
def test_pfcr_features_reject_non_finite_detector_evidence(tensor_index):
    detector_tensors = list(synthetic_pair())
    detector_tensors[tensor_index].reshape(-1)[0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        pfcr_features(*detector_tensors)


@pytest.mark.parametrize(
    ("tensor_index", "replacement"),
    [
        (0, torch.zeros(NUM_QUERIES - 1, 4)),
        (1, torch.zeros(NUM_QUERIES, NUM_CLASSES - 1)),
        (2, torch.zeros(NUM_QUERIES, 4, 1)),
        (3, torch.zeros(1, NUM_QUERIES, NUM_CLASSES)),
    ],
)
def test_pfcr_features_reject_shape_drift(tensor_index, replacement):
    detector_tensors = list(synthetic_pair())
    detector_tensors[tensor_index] = replacement

    with pytest.raises(ValueError, match="shape"):
        pfcr_features(*detector_tensors)


@pytest.mark.parametrize("tensor_index", range(4))
def test_pfcr_features_require_floating_point_inputs(tensor_index):
    detector_tensors = list(synthetic_pair())
    detector_tensors[tensor_index] = detector_tensors[tensor_index].to(torch.int64)

    with pytest.raises(TypeError, match="floating-point"):
        pfcr_features(*detector_tensors)


def test_pfcr_features_require_one_shared_device():
    detector_tensors = list(synthetic_pair())
    detector_tensors[0] = torch.empty((NUM_QUERIES, 4), device="meta")

    with pytest.raises(ValueError, match="device"):
        pfcr_features(*detector_tensors)


def test_gate_has_exact_zero_identity_at_initialization():
    model = PFCRGate()
    features = torch.randn(2, NUM_QUERIES, NUM_CLASSES, PFCR_FEATURE_DIM)

    residual = model(features)

    assert residual.shape == (2, NUM_QUERIES, NUM_CLASSES)
    assert torch.equal(residual, torch.zeros_like(residual))


@pytest.mark.parametrize(("bias", "sign"), [(1000.0, 1), (-1000.0, -1)])
def test_gate_residual_is_bounded(bias, sign):
    model = PFCRGate()
    features = torch.randn(2, 3, PFCR_FEATURE_DIM)
    with torch.no_grad():
        model.network[-1].bias.fill_(bias)

    residual = model(features)

    assert bool((residual >= -2.0).all())
    assert bool((residual <= 2.0).all())
    assert bool((sign * residual > 0).all())


def test_gate_detaches_features_but_trains_gate_parameters():
    model = PFCRGate()
    with torch.no_grad():
        model.network[-1].weight.fill_(0.1)
    features = torch.randn(4, PFCR_FEATURE_DIM, requires_grad=True)

    model(features).sum().backward()

    assert features.grad is None
    assert model.network[0].weight.grad is not None
    assert model.network[-1].weight.grad is not None


@pytest.mark.parametrize(
    ("features", "error", "message"),
    [
        (torch.zeros(2, PFCR_FEATURE_DIM - 1), ValueError, "shape"),
        (torch.zeros(2, PFCR_FEATURE_DIM, dtype=torch.int64), TypeError, "floating-point"),
        (
            torch.full((2, PFCR_FEATURE_DIM), float("inf")),
            ValueError,
            "finite",
        ),
    ],
)
def test_gate_rejects_invalid_features(features, error, message):
    with pytest.raises(error, match=message):
        PFCRGate()(features)


@pytest.mark.parametrize("slots", [0, 15, 30, 60])
def test_protected_merge_preserves_registered_fdr_prefix(slots):
    fdr_boxes, fdr_logits, cm_boxes, cm_logits = synthetic_pair()
    stock = stock_predictions(fdr_boxes, fdr_logits)
    merged = protected_merge(
        fdr_boxes,
        fdr_logits,
        cm_boxes,
        cm_logits,
        rescue_slots=slots,
    )

    assert merged.shape == (NUM_QUERIES, 6)
    if slots == 0:
        assert torch.equal(merged, stock)
    else:
        assert torch.equal(merged[: NUM_QUERIES - slots], stock[: NUM_QUERIES - slots])


def test_protected_merge_rejects_unregistered_budget():
    tensors = synthetic_pair()
    with pytest.raises(ValueError, match="rescue"):
        protected_merge(*tensors, rescue_slots=29)


def test_protected_merge_can_rescue_cm_without_modifying_fdr_inputs():
    fdr_boxes, fdr_logits, cm_boxes, cm_logits = synthetic_pair()
    originals = tuple(value.clone() for value in (fdr_boxes, fdr_logits, cm_boxes, cm_logits))
    fdr_logits.fill_(-8.0)
    cm_logits.fill_(-9.0)
    cm_logits[7, 3] = 8.0

    merged = protected_merge(
        fdr_boxes, fdr_logits, cm_boxes, cm_logits, rescue_slots=15
    )

    assert bool(((merged[:, 5] == 3) & (merged[:, 4] > 0.99)).any())
    assert torch.equal(fdr_boxes, originals[0])
    assert torch.equal(cm_boxes, originals[2])


def test_duplicate_union_candidates_receive_only_one_positive_teacher():
    fdr_boxes, fdr_logits, cm_boxes, cm_logits = synthetic_pair()
    fdr_logits.fill_(-10.0)
    cm_logits.fill_(-10.0)
    fdr_logits[0, 2] = 5.0
    cm_logits[0, 2] = 5.0
    target_boxes = fdr_boxes[:1].clone()
    target_classes = torch.tensor([2])

    teacher = one_to_one_union_teacher(
        fdr_boxes,
        fdr_logits,
        cm_boxes,
        cm_logits,
        target_boxes,
        target_classes,
    )

    assert teacher.fdr.shape == (NUM_QUERIES, NUM_CLASSES)
    assert teacher.frequencycm.shape == (NUM_QUERIES, NUM_CLASSES)
    assert int(((teacher.fdr > 0).sum() + (teacher.frequencycm > 0).sum()).item()) == 1
    assert not teacher.fdr.requires_grad
    assert not teacher.frequencycm.requires_grad


def test_invalid_candidate_never_receives_positive_teacher():
    fdr_boxes, fdr_logits, cm_boxes, cm_logits = synthetic_pair()
    cm_boxes[0, 2] = -0.2
    cm_logits.fill_(-10.0)
    cm_logits[0, 0] = 10.0
    teacher = one_to_one_union_teacher(
        fdr_boxes,
        fdr_logits,
        cm_boxes,
        cm_logits,
        fdr_boxes[:1],
        torch.tensor([0]),
    )
    assert torch.equal(teacher.frequencycm[0], torch.zeros(NUM_CLASSES))


def test_pfcr_boundary_loss_prefers_teacher_boundary_order():
    fdr_logits = torch.full((NUM_QUERIES, NUM_CLASSES), -4.0)
    fdr_teacher = torch.zeros_like(fdr_logits)
    cm_teacher = torch.zeros_like(fdr_logits)
    fdr_logits.reshape(-1)[299] = 0.0
    fdr_teacher.reshape(-1)[299] = 0.1
    cm_teacher.reshape(-1)[0] = 0.9
    aligned = torch.full_like(fdr_logits, -4.0, requires_grad=True)
    reversed_order = torch.full_like(fdr_logits, -4.0, requires_grad=True)
    aligned.data.reshape(-1)[0] = 2.0
    reversed_order.data.reshape(-1)[0] = -2.0

    aligned_loss = pfcr_boundary_loss(
        aligned, cm_teacher, fdr_logits, fdr_teacher, rescue_slots=15
    )
    reversed_loss = pfcr_boundary_loss(
        reversed_order, cm_teacher, fdr_logits, fdr_teacher, rescue_slots=15
    )

    assert aligned_loss < reversed_loss
    aligned_loss.backward()
    assert aligned.grad is not None
    assert torch.isfinite(aligned.grad).all()


def test_pfcr_boundary_loss_rejects_zero_or_unregistered_budget():
    values = torch.zeros(NUM_QUERIES, NUM_CLASSES)
    with pytest.raises(ValueError, match="rescue"):
        pfcr_boundary_loss(values, values, values, values, rescue_slots=0)
    with pytest.raises(ValueError, match="rescue"):
        pfcr_boundary_loss(values, values, values, values, rescue_slots=29)


def test_boundary_candidate_union_is_stably_deduplicated():
    values = torch.tensor([7, 2, 7, 3, 2, 9], dtype=torch.long)
    assert _stable_unique_indices(values).tolist() == [7, 2, 3, 9]
