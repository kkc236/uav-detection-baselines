from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src.sqda_diagnostics import (
    ProposalDiagnosticAccumulator,
    pairwise_iou_xywh,
    size_bin_masks,
)
from src.sqda_preflight import (
    dataset_signature,
    parse_yolo_label,
    validate_visdrone_dataset,
    write_dataset_yaml,
)


def _make_split(root: Path, split: str, count: int) -> None:
    images = root / "images" / split
    labels = root / "labels" / split
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index in range(count):
        (images / f"{index:06d}.jpg").write_bytes(b"not-decoded-by-preflight")
        (labels / f"{index:06d}.txt").write_text(
            "0 0.500000 0.500000 0.100000 0.200000\n",
            encoding="utf-8",
        )


def test_dataset_validation_counts_pairs_labels_and_is_deterministic(tmp_path: Path) -> None:
    _make_split(tmp_path, "train", 3)
    _make_split(tmp_path, "val", 2)
    expected = {"train": 3, "val": 2}
    first = validate_visdrone_dataset(tmp_path, expected_counts=expected)
    second = validate_visdrone_dataset(tmp_path, expected_counts=expected)

    assert first == second
    assert first["total_files"] == 10
    assert first["total_boxes"] == 5
    assert first["signature"] == dataset_signature(tmp_path)


def test_dataset_validation_rejects_count_and_pair_mismatches(tmp_path: Path) -> None:
    _make_split(tmp_path, "train", 2)
    _make_split(tmp_path, "val", 1)
    with pytest.raises(RuntimeError, match="count mismatch"):
        validate_visdrone_dataset(
            tmp_path,
            expected_counts={"train": 3, "val": 1},
        )

    (tmp_path / "labels" / "train" / "000001.txt").rename(
        tmp_path / "labels" / "train" / "different.txt"
    )
    with pytest.raises(RuntimeError, match="stem mismatch"):
        validate_visdrone_dataset(
            tmp_path,
            expected_counts={"train": 2, "val": 1},
        )


@pytest.mark.parametrize(
    "row,message",
    [
        ("10 0.5 0.5 0.1 0.1", "class"),
        ("0 nan 0.5 0.1 0.1", "non-finite"),
        ("0 1.1 0.5 0.1 0.1", "center"),
        ("0 0.5 0.5 0 0.1", "size"),
        ("0 0.5 0.5 0.1", "five"),
    ],
)
def test_label_parser_rejects_invalid_rows(
    tmp_path: Path,
    row: str,
    message: str,
) -> None:
    label = tmp_path / "bad.txt"
    label.write_text(row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        parse_yolo_label(label)


def test_dataset_yaml_uses_the_actual_absolute_root(tmp_path: Path) -> None:
    root = tmp_path / "VisDrone"
    destination = tmp_path / "protocol" / "data.yaml"
    output = write_dataset_yaml(root, destination)
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert Path(payload["path"]) == root.resolve()
    assert payload["train"] == "images/train"
    assert payload["val"] == "images/val"
    assert len(payload["names"]) == 10


def test_pairwise_iou_and_size_bins_are_pre_registered() -> None:
    first = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
    second = torch.tensor(
        [
            [0.5, 0.5, 0.1, 0.1],
            [0.0, 0.0, 0.1, 0.1],
        ]
    )
    iou = pairwise_iou_xywh(first, second)
    assert torch.allclose(iou, torch.tensor([[1.0, 0.0]]))

    boxes = torch.tensor(
        [
            [0.5, 0.5, 10 / 640, 10 / 640],
            [0.5, 0.5, 24 / 640, 24 / 640],
            [0.5, 0.5, 64 / 640, 64 / 640],
            [0.5, 0.5, 120 / 640, 120 / 640],
        ]
    )
    masks = size_bin_masks(boxes)
    assert masks["tiny_lt_16sq"].tolist() == [True, False, False, False]
    assert masks["coco_small_lt_32sq"].tolist() == [True, True, False, False]
    assert masks["coco_medium_32_96sq"].tolist() == [False, False, True, False]
    assert masks["coco_large_ge_96sq"].tolist() == [False, False, False, True]


def test_recoverable_missed_statistics_are_deterministic() -> None:
    proposals = torch.zeros(300, 4)
    proposals[0] = torch.tensor([0.5, 0.5, 0.1, 0.1])
    gt_boxes = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
    gt_classes = torch.tensor([3])
    final_predictions = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.9, 3.0]])

    first = ProposalDiagnosticAccumulator()
    second = ProposalDiagnosticAccumulator()
    first.update(proposals, gt_boxes, gt_classes, final_predictions)
    second.update(proposals, gt_boxes, gt_classes, final_predictions)
    assert first.report() == second.report()
    overall = first.report()["bins"]["all"]
    assert overall["objects"] == 1
    assert overall["proposal_recall"]["0.5"] == 1.0
    assert overall["final_missed_at_conf_0.25_iou_0.5"] == 1
    assert overall["recoverable_missed_proposal_iou_ge_0.5"] == 1
    assert overall["recoverable_missed_ratio"] == 1.0
