from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class, box_iou


@dataclass(frozen=True)
class TinyDetectionResult:
    map: float
    recall: float
    extreme_map: float
    tiny_8_16_map: float
    target_count: int
    false_positive_count: int
    extreme_tiny_count: int
    tiny_8_16_count: int


@dataclass
class _MetricAccumulator:
    iou_threshold_count: int
    true_positives: list[np.ndarray] = field(default_factory=list)
    confidence: list[np.ndarray] = field(default_factory=list)
    predicted_class: list[np.ndarray] = field(default_factory=list)
    target_class: list[np.ndarray] = field(default_factory=list)
    target_count: int = 0
    false_positive_count: int = 0

    def append(
        self,
        correct: torch.Tensor,
        confidence: torch.Tensor,
        predicted_class: torch.Tensor,
        target_class: torch.Tensor,
    ) -> None:
        self.true_positives.append(correct.detach().cpu().numpy())
        self.confidence.append(confidence.detach().float().cpu().numpy())
        self.predicted_class.append(predicted_class.detach().float().cpu().numpy())
        self.target_class.append(target_class.detach().float().cpu().numpy())
        self.target_count += len(target_class)
        if correct.numel():
            self.false_positive_count += int((~correct[:, 0]).sum())

    def compute(self) -> tuple[float, float]:
        if self.target_count == 0:
            return 0.0, 0.0
        true_positives = np.concatenate(self.true_positives, axis=0)
        confidence = np.concatenate(self.confidence)
        predicted_class = np.concatenate(self.predicted_class)
        target_class = np.concatenate(self.target_class)
        _, _, _, recall, _, average_precision, *_ = ap_per_class(
            true_positives,
            confidence,
            predicted_class,
            target_class,
        )
        mean_ap = float(average_precision.mean()) if average_precision.size else 0.0
        mean_recall = float(recall.mean()) if recall.size else 0.0
        return mean_ap, mean_recall


def resized_radius(boxes_xyxy: torch.Tensor, gain: tuple[float, float]) -> torch.Tensor:
    width = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * gain[0]
    height = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) * gain[1]
    return (width.clamp_min(0) * height.clamp_min(0)).sqrt()


class TinyDetectionMetrics:
    def __init__(self, iouv: torch.Tensor):
        self.iouv = iouv.detach().float().cpu()
        self._all = _MetricAccumulator(len(self.iouv))
        self._extreme = _MetricAccumulator(len(self.iouv))
        self._eight_to_sixteen = _MetricAccumulator(len(self.iouv))

    @property
    def extreme_tiny_count(self) -> int:
        return self._extreme.target_count

    @property
    def tiny_8_16_count(self) -> int:
        return self._eight_to_sixteen.target_count

    def update(
        self,
        pred_boxes: torch.Tensor,
        pred_conf: torch.Tensor,
        pred_cls: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_cls: torch.Tensor,
        radius: torch.Tensor,
    ) -> None:
        pred_boxes = pred_boxes.detach().float()
        pred_conf = pred_conf.detach().float()
        pred_cls = pred_cls.detach().long()
        gt_boxes = gt_boxes.detach().float()
        gt_cls = gt_cls.detach().long()
        radius = radius.detach().float()

        self._update_group(self._all, radius <= 16.0, pred_boxes, pred_conf, pred_cls, gt_boxes, gt_cls)
        self._update_group(self._extreme, radius < 8.0, pred_boxes, pred_conf, pred_cls, gt_boxes, gt_cls)
        self._update_group(
            self._eight_to_sixteen,
            (radius >= 8.0) & (radius <= 16.0),
            pred_boxes,
            pred_conf,
            pred_cls,
            gt_boxes,
            gt_cls,
        )

    def update_from_prepared(
        self,
        pred: dict[str, torch.Tensor],
        prepared: dict,
        gain: tuple[float, float],
    ) -> None:
        original_boxes = _remove_padding_and_gain(prepared["bboxes"], prepared["ratio_pad"], gain)
        radius = resized_radius(original_boxes, gain)
        self.update(
            pred_boxes=pred["bboxes"],
            pred_conf=pred["conf"],
            pred_cls=pred["cls"],
            gt_boxes=prepared["bboxes"],
            gt_cls=prepared["cls"],
            radius=radius,
        )

    def compute(self) -> TinyDetectionResult:
        mean_ap, mean_recall = self._all.compute()
        extreme_ap, _ = self._extreme.compute()
        middle_ap, _ = self._eight_to_sixteen.compute()
        return TinyDetectionResult(
            map=mean_ap,
            recall=mean_recall,
            extreme_map=extreme_ap,
            tiny_8_16_map=middle_ap,
            target_count=self._all.target_count,
            false_positive_count=self._all.false_positive_count,
            extreme_tiny_count=self._extreme.target_count,
            tiny_8_16_count=self._eight_to_sixteen.target_count,
        )

    def clear(self) -> None:
        self._all = _MetricAccumulator(len(self.iouv))
        self._extreme = _MetricAccumulator(len(self.iouv))
        self._eight_to_sixteen = _MetricAccumulator(len(self.iouv))

    def state_dict(self) -> dict:
        return {
            "all": _accumulator_state(self._all),
            "extreme": _accumulator_state(self._extreme),
            "eight_to_sixteen": _accumulator_state(self._eight_to_sixteen),
        }

    def merge_state_dict(self, state: dict) -> None:
        _merge_accumulator_state(self._all, state["all"])
        _merge_accumulator_state(self._extreme, state["extreme"])
        _merge_accumulator_state(self._eight_to_sixteen, state["eight_to_sixteen"])

    def _update_group(
        self,
        accumulator: _MetricAccumulator,
        target_mask: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_conf: torch.Tensor,
        pred_cls: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_cls: torch.Tensor,
    ) -> None:
        ignored_mask = ~target_mask
        keep_predictions = torch.ones(len(pred_boxes), dtype=torch.bool, device=pred_boxes.device)
        if ignored_mask.any() and len(pred_boxes):
            ignored_iou = box_iou(gt_boxes[ignored_mask], pred_boxes)
            ignored_matches = _match_predictions(
                pred_cls,
                gt_cls[ignored_mask],
                ignored_iou,
                self.iouv[:1].to(pred_boxes.device),
            )
            keep_predictions = ~ignored_matches[:, 0]

        selected_boxes = pred_boxes[keep_predictions]
        selected_conf = pred_conf[keep_predictions]
        selected_cls = pred_cls[keep_predictions]
        target_boxes = gt_boxes[target_mask]
        target_cls = gt_cls[target_mask]
        if len(target_boxes) and len(selected_boxes):
            correct = _match_predictions(
                selected_cls,
                target_cls,
                box_iou(target_boxes, selected_boxes),
                self.iouv.to(selected_boxes.device),
            )
        else:
            correct = torch.zeros(
                (len(selected_boxes), len(self.iouv)),
                dtype=torch.bool,
                device=selected_boxes.device,
            )
        accumulator.append(correct, selected_conf, selected_cls, target_cls)


def validation_xy_gain(
    original_shape: tuple[int, int],
    image_shape: tuple[int, int],
    ratio_pad,
) -> tuple[float, float]:
    if ratio_pad is not None:
        ratio = ratio_pad[0]
        if isinstance(ratio, torch.Tensor):
            ratio = ratio.detach().cpu().reshape(-1).tolist()
        if isinstance(ratio, (tuple, list)):
            return float(ratio[0]), float(ratio[1] if len(ratio) > 1 else ratio[0])
        return float(ratio), float(ratio)
    return image_shape[1] / original_shape[1], image_shape[0] / original_shape[0]


def _remove_padding_and_gain(
    boxes: torch.Tensor,
    ratio_pad,
    gain: tuple[float, float],
) -> torch.Tensor:
    original = boxes.detach().float().clone()
    pad_x, pad_y = ratio_pad[1] if ratio_pad is not None else (0.0, 0.0)
    original[:, [0, 2]] = (original[:, [0, 2]] - float(pad_x)) / gain[0]
    original[:, [1, 3]] = (original[:, [1, 3]] - float(pad_y)) / gain[1]
    return original


def _match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    correct = np.zeros((pred_classes.shape[0], thresholds.shape[0]), dtype=bool)
    class_aware_iou = (iou * (true_classes[:, None] == pred_classes)).detach().cpu().numpy()
    for threshold_index, threshold in enumerate(thresholds.detach().cpu().tolist()):
        matches = np.array(np.nonzero(class_aware_iou >= threshold)).T
        if matches.shape[0] > 1:
            matches = matches[class_aware_iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        if matches.shape[0]:
            correct[matches[:, 1].astype(int), threshold_index] = True
    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)


def _accumulator_state(accumulator: _MetricAccumulator) -> dict:
    return {
        "true_positives": accumulator.true_positives,
        "confidence": accumulator.confidence,
        "predicted_class": accumulator.predicted_class,
        "target_class": accumulator.target_class,
        "target_count": accumulator.target_count,
        "false_positive_count": accumulator.false_positive_count,
    }


def _merge_accumulator_state(accumulator: _MetricAccumulator, state: dict) -> None:
    accumulator.true_positives.extend(state["true_positives"])
    accumulator.confidence.extend(state["confidence"])
    accumulator.predicted_class.extend(state["predicted_class"])
    accumulator.target_class.extend(state["target_class"])
    accumulator.target_count += int(state["target_count"])
    accumulator.false_positive_count += int(state["false_positive_count"])
