from __future__ import annotations

import math

import numpy as np
import torch

from src.gcmv_data import (
    GCMVRTDETRDataset,
    build_local_view_tensor,
)


def test_local_views_are_cut_from_the_source_image_before_letterboxing():
    height, width = 100, 200
    source = np.zeros((height, width, 3), dtype=np.uint8)
    source[:, :120] = (11, 22, 33)
    source[:, 80:] = (44, 55, 66)

    views = build_local_view_tensor(source, local_imgsz=128)

    assert views.shape == (4, 3, 128, 128)
    assert views.dtype == torch.uint8
    tile_width = math.ceil(0.60 * width)
    tile_height = math.ceil(0.60 * height)
    assert (tile_width, tile_height) == (120, 60)
    # LetterBox is centered and does not upscale. BGR is converted to RGB.
    top = (128 - tile_height) // 2
    left = (128 - tile_width) // 2
    assert views[0, :, top, left].tolist() == [33, 22, 11]
    assert views[1, :, top, left].tolist() == [66, 55, 44]
    assert views[0, :, 0, 0].tolist() == [114, 114, 114]


def test_gcmv_collate_stacks_view_and_source_metadata():
    samples = []
    for index in range(2):
        samples.append(
            {
                "img": torch.full((3, 32, 32), index, dtype=torch.uint8),
                "local_views": torch.full(
                    (4, 3, 64, 64), index, dtype=torch.uint8
                ),
                "source_shape": torch.tensor([100 + index, 200 + index]),
                "batch_idx": torch.tensor([0.0]),
                "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                "cls": torch.tensor([[0.0]]),
                "im_file": f"{index}.jpg",
            }
        )

    batch = GCMVRTDETRDataset.collate_fn(samples)

    assert batch["img"].shape == (2, 3, 32, 32)
    assert batch["local_views"].shape == (2, 4, 3, 64, 64)
    assert batch["source_shape"].tolist() == [[100, 200], [101, 201]]
    assert batch["batch_idx"].tolist() == [0.0, 1.0]

