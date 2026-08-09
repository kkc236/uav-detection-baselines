from hashlib import sha256

import pytest
import torch

from src.pfcr import PFCR_FEATURE_DIM, PFCRGate, pfcr_features, pfcr_split


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
