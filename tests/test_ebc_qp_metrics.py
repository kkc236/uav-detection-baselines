import torch

from src.ebc_qp_metrics import TinyDetectionMetrics, resized_radius


def test_radius_uses_actual_xy_gain_and_padding_changes_no_size():
    boxes_xyxy = torch.tensor([[10.0, 20.0, 30.0, 60.0]])

    radius = resized_radius(boxes_xyxy, gain=(0.5, 0.25))

    torch.testing.assert_close(radius, torch.tensor([10.0]))


def test_gt_larger_than_16_is_ignored_not_counted_as_false_positive():
    metric = TinyDetectionMetrics(iouv=torch.linspace(0.5, 0.95, 10))
    metric.update(
        pred_boxes=torch.tensor([[0.0, 0.0, 40.0, 40.0]]),
        pred_conf=torch.tensor([0.9]),
        pred_cls=torch.tensor([0]),
        gt_boxes=torch.tensor([[0.0, 0.0, 40.0, 40.0]]),
        gt_cls=torch.tensor([0]),
        radius=torch.tensor([20.0]),
    )

    result = metric.compute()

    assert result.target_count == 0
    assert result.false_positive_count == 0


def test_custom_groups_are_r_lt_8_and_8_through_16():
    metric = TinyDetectionMetrics(iouv=torch.linspace(0.5, 0.95, 10))
    radii = torch.tensor([7.99, 8.0, 16.0, 16.01])
    boxes = torch.stack(
        [torch.tensor([0.0, 0.0, float(radius), float(radius)]) for radius in radii]
    )
    metric.update(
        pred_boxes=torch.empty(0, 4),
        pred_conf=torch.empty(0),
        pred_cls=torch.empty(0, dtype=torch.long),
        gt_boxes=boxes,
        gt_cls=torch.zeros(4, dtype=torch.long),
        radius=radii,
    )

    assert metric.extreme_tiny_count == 1
    assert metric.tiny_8_16_count == 2


def test_perfect_tiny_detection_accumulates_ap_and_recall():
    metric = TinyDetectionMetrics(iouv=torch.linspace(0.5, 0.95, 10))
    box = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    metric.update(
        pred_boxes=box.clone(),
        pred_conf=torch.tensor([0.9]),
        pred_cls=torch.tensor([2]),
        gt_boxes=box,
        gt_cls=torch.tensor([2]),
        radius=torch.tensor([10.0]),
    )

    result = metric.compute()

    assert result.map > 0.99
    assert result.recall == 1.0
    assert result.target_count == 1
