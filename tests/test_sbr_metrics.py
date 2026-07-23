import numpy as np
import pytest

from src.sbr_metrics import (
    IOU_THRESHOLDS,
    evaluate_dataset,
    evaluate_sbr,
    tiny_recall,
)


def _run(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, **kwargs):
    return evaluate_sbr(
        pred_boxes=np.asarray(pred_boxes, dtype=float).reshape((-1, 4)),
        pred_scores=np.asarray(pred_scores, dtype=float),
        pred_classes=np.asarray(pred_classes, dtype=int),
        gt_boxes=np.asarray(gt_boxes, dtype=float).reshape((-1, 4)),
        gt_classes=np.asarray(gt_classes, dtype=int),
        **kwargs,
    )


def test_perfect_prediction_has_ap50_ap75_and_map_one():
    m = _run([[0, 0, 10, 10]], [0.9], [1], [[0, 0, 10, 10]], [1])
    # Ultralytics' 101-point interpolation integrates the right endpoint as
    # zero, yielding 0.995 for a one-point perfect curve.
    assert m["AP50"] == pytest.approx(0.995)
    assert m["AP75"] == pytest.approx(0.995)
    assert m["mAP50-95"] == pytest.approx(0.995)
    assert m["counts"]["tp"] == 1
    assert m["counts"]["fp"] == 0
    assert m["counts"]["fn"] == 0


def test_empty_predictions_and_wrong_class_are_counted():
    empty = _run([], [], [], [[0, 0, 10, 10]], [1])
    assert empty["mAP50-95"] == 0.0
    assert empty["counts"]["fn"] == 1
    wrong = _run([[0, 0, 10, 10]], [0.9], [2], [[0, 0, 10, 10]], [1])
    assert wrong["counts"]["fp"] == 1
    assert wrong["counts"]["fn"] == 1


def test_duplicate_prediction_is_false_positive():
    m = _run(
        [[0, 0, 10, 10], [0, 0, 10, 10]],
        [0.9, 0.8],
        [1, 1],
        [[0, 0, 10, 10]],
        [1],
    )
    assert m["counts"]["tp"] == 1
    assert m["counts"]["fp"] == 1


def test_ignore_region_neutralizes_by_prediction_ioa():
    m = _run(
        [[0, 0, 10, 10]],
        [0.9],
        [99],
        [],
        [],
        ignore_boxes=[[0, 0, 10, 10]],
    )
    assert m["counts"]["fp"] == 0
    assert m["counts"]["neutralized"] == 1


def test_out_of_bin_neutralization_is_threshold_dependent():
    # IoU = 0.6: neutral at AP50 but false positive at AP75.
    m = _run(
        [[0, 0, 6, 10]],
        [0.9],
        [1],
        [[0, 0, 10, 10]],
        [1],
        size_bin="tiny",
        effective_gain=10.0,
    )
    assert m["AP50-tiny-SBR"] == pytest.approx(0.0)
    assert m["AP75-tiny-SBR"] == pytest.approx(0.0)
    # Out-of-bin GT has no target; at AP50 prediction is neutral, AP75 FP.
    assert m["per_threshold"]["tiny"][0.50]["neutralized"] == 1
    assert m["per_threshold"]["tiny"][0.75]["fp"] == 1


def test_size_bin_boundaries_are_inclusive_on_upper_edge():
    # Effective square-root area exactly 16, 32 and 96.
    boxes = [[0, 0, 4, 64], [0, 0, 8, 128], [0, 0, 12, 768]]
    classes = [1, 1, 1]
    m = _run(boxes, [0.9, 0.8, 0.7], classes, boxes, classes)
    assert m["AP-tiny-SBR"] == pytest.approx(0.995)
    assert m["AP-small-SBR"] == pytest.approx(0.995)
    assert m["AP-medium-SBR"] == pytest.approx(0.995)


def test_max_det_truncates_after_stable_confidence_source_query_sort():
    boxes = [[0, 0, 10, 10]] * 301
    m = _run(
        boxes,
        [0.9] + [0.8] * 300,
        [1] * 301,
        [[0, 0, 10, 10]],
        [1],
        max_det=300,
        pred_source=[0] + [1] * 300,
        pred_query=list(range(301)),
    )
    assert m["counts"]["predictions"] == 300


def test_tiny_recall_is_class_aware_and_uses_iou_half():
    rec = tiny_recall(
        pred_boxes=[[0, 0, 10, 10], [0, 0, 10, 10]],
        pred_scores=[0.9, 0.8],
        pred_classes=[2, 1],
        gt_boxes=[[0, 0, 10, 10]],
        gt_classes=[1],
        effective_gain=1.0,
    )
    assert rec["recall"] == pytest.approx(1.0)
    assert rec["matched"] == 1
    assert rec["targets"] == 1


def test_threshold_grid_is_frozen():
    assert np.array_equal(IOU_THRESHOLDS, np.arange(0.50, 0.951, 0.05))


def test_invalid_nonfinite_or_illegal_boxes_fail_closed():
    with pytest.raises(ValueError):
        _run([[0, 0, np.nan, 1]], [0.9], [1], [], [])
    with pytest.raises(ValueError):
        _run([[2, 0, 1, 1]], [0.9], [1], [], [])


def test_dataset_ap_is_pooled_across_images_not_mean_of_image_aps():
    images = [
        dict(
            pred_boxes=[[0, 0, 10, 10], [0, 0, 10, 10]],
            pred_scores=[0.9, 0.8],
            pred_classes=[1, 1],
            gt_boxes=[[0, 0, 10, 10]],
            gt_classes=[1],
        ),
        dict(
            pred_boxes=[[0, 0, 10, 10]],
            pred_scores=[0.1],
            pred_classes=[1],
            gt_boxes=[[0, 0, 10, 10]],
            gt_classes=[1],
        ),
    ]
    pooled = evaluate_dataset(images)
    mean_image_ap = np.mean([evaluate_sbr(**x)["AP50"] for x in images])
    assert pooled["AP50"] < mean_image_ap
    assert pooled["AP50"] == pytest.approx(0.8283333333)


def test_frozen_limits_reject_nonprotocol_max_det_and_confidence():
    with pytest.raises(ValueError):
        _run([[0, 0, 1, 1]], [0.9], [1], [[0, 0, 1, 1]], [1], max_det=10)
    with pytest.raises(ValueError):
        _run([[0, 0, 1, 1]], [0.9], [1], [[0, 0, 1, 1]], [1], conf_threshold=0.01)
