import pytest
import torch

from src.sr_peg_targets import build_sr_peg_targets


def _base_inputs() -> dict[str, object]:
    return {
        "global_boxes": torch.tensor([[[0.50, 0.50, 0.02, 0.02]]]),
        "global_logits": torch.tensor([[[9.0, -9.0]]]),
        "local_boxes": torch.tensor([[[0.50, 0.50, 0.02, 0.02]]]),
        "local_logits": torch.tensor([[[9.0, -9.0]]]),
        "gt_boxes": torch.tensor([[0.50, 0.50, 0.08, 0.08]]),
        "gt_classes": torch.tensor([0], dtype=torch.long),
        "source_shape": (640, 640),
    }


def test_targets_distinguish_true_tiny_from_underestimated_medium():
    targets = build_sr_peg_targets(**_base_inputs())

    assert targets.local_non_tiny_risk.item() == 1.0
    assert targets.local_tiny_utility.item() == 0.0
    assert targets.global_retain.item() == 1.0


def test_exact_twelve_pixel_gt_produces_soft_tiny_utility_only():
    inputs = _base_inputs()
    size = 12.0 / 640.0
    inputs["global_boxes"] = torch.tensor(
        [[[0.50, 0.50, size, size]]]
    )
    inputs["local_boxes"] = torch.tensor(
        [[[0.50, 0.50, size * 0.8, size * 0.8]]]
    )
    inputs["gt_boxes"] = torch.tensor([[0.50, 0.50, size, size]])

    targets = build_sr_peg_targets(**inputs)

    torch.testing.assert_close(
        targets.local_tiny_utility,
        torch.tensor([[[0.64]]]),
        atol=1e-5,
        rtol=0,
    )
    assert targets.local_non_tiny_risk.item() == 0.0
    assert targets.global_retain.item() == 0.0


def test_empty_ground_truth_yields_zero_targets():
    inputs = _base_inputs()
    inputs["gt_boxes"] = torch.empty((0, 4))
    inputs["gt_classes"] = torch.empty((0,), dtype=torch.long)

    targets = build_sr_peg_targets(**inputs)

    assert torch.count_nonzero(targets.local_tiny_utility) == 0
    assert torch.count_nonzero(targets.local_non_tiny_risk) == 0
    assert torch.count_nonzero(targets.global_retain) == 0


def test_wrong_class_global_evidence_does_not_receive_retain_target():
    inputs = _base_inputs()
    inputs["global_logits"] = torch.tensor([[[-9.0, 9.0]]])

    targets = build_sr_peg_targets(**inputs)

    assert targets.global_retain.item() == 0.0
    assert targets.local_non_tiny_risk.item() == 1.0


def test_large_predicted_global_box_is_not_given_a_learned_retain_target():
    inputs = _base_inputs()
    inputs["global_boxes"] = torch.tensor([[[0.50, 0.50, 0.08, 0.08]]])

    targets = build_sr_peg_targets(**inputs)

    assert targets.global_retain.item() == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("global_boxes", torch.zeros(2, 1, 4)),
        ("global_logits", torch.zeros(1, 2, 2)),
        ("local_boxes", torch.zeros(1, 1, 5)),
        ("gt_classes", torch.tensor([0], dtype=torch.int32)),
        ("source_shape", (0, 640)),
    ),
)
def test_invalid_target_inputs_fail_closed(field, value):
    inputs = _base_inputs()
    inputs[field] = value

    with pytest.raises(ValueError):
        build_sr_peg_targets(**inputs)
