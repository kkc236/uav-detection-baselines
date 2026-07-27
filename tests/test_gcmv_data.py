from __future__ import annotations

from copy import deepcopy
import inspect
import math
import random

import numpy as np
import pytest
import torch
from ultralytics.data.augment import (
    Compose,
    LetterBox,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
)
from ultralytics.utils.instance import Instances

import src.gcmv_data as gcmv_data
from src.gcmv_data import (
    GCMVRTDETRDataset,
    build_local_view_tensor,
)


def empty_labels(image: np.ndarray) -> dict:
    return {
        "img": image.copy(),
        "cls": np.empty((0, 1), dtype=np.float32),
        "instances": Instances(
            bboxes=np.empty((0, 4), dtype=np.float32),
            segments=np.empty((0, 0, 2), dtype=np.float32),
            keypoints=None,
            bbox_format="xyxy",
            normalized=False,
        ),
    }


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
                "source_to_global": torch.eye(3),
                "global_to_source": torch.eye(3),
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
    assert batch["source_to_global"].shape == (2, 3, 3)
    assert batch["global_to_source"].shape == (2, 3, 3)
    assert batch["batch_idx"].tolist() == [0.0, 1.0]


def test_traced_random_perspective_preserves_stock_pixels_and_labels():
    image = np.arange(72 * 96 * 3, dtype=np.uint8).reshape(72, 96, 3)
    stock = RandomPerspective(
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        size=(64, 64),
    )
    traced = gcmv_data.GCMVRandomPerspective(
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        size=(64, 64),
    )

    random.seed(123)
    stock_result = stock(empty_labels(image))
    random.seed(123)
    traced_result = traced(empty_labels(image))

    assert np.array_equal(traced_result["img"], stock_result["img"])
    assert np.array_equal(
        traced_result["instances"].bboxes,
        stock_result["instances"].bboxes,
    )
    assert traced_result["_gcmv_affine_matrix"].shape == (3, 3)
    assert traced_result["_gcmv_pre_affine_shape"] == (72, 96)


def test_traced_random_flip_preserves_stock_result_and_records_decision():
    image = np.arange(12 * 20 * 3, dtype=np.uint8).reshape(12, 20, 3)
    stock = RandomFlip(direction="horizontal", p=0.5, flip_idx=[])
    traced = gcmv_data.GCMVRandomFlip(
        direction="horizontal", p=0.5, flip_idx=[]
    )

    random.seed(7)
    stock_result = stock(empty_labels(image))
    random.seed(7)
    traced_result = traced(empty_labels(image))

    assert np.array_equal(traced_result["img"], stock_result["img"])
    assert np.array_equal(
        traced_result["instances"].bboxes,
        stock_result["instances"].bboxes,
    )
    assert traced_result["_gcmv_flip_horizontal"] is True


def test_paired_hsv_replays_identical_color_transform_on_raw_source():
    global_image = np.full((32, 48, 3), (10, 80, 160), dtype=np.uint8)
    source_image = np.full((80, 120, 3), (10, 80, 160), dtype=np.uint8)
    traced = gcmv_data.GCMVRandomHSV(hgain=0.015, sgain=0.7, vgain=0.4)
    stock_for_source = RandomHSV(hgain=0.015, sgain=0.7, vgain=0.4)
    labels = empty_labels(global_image)
    labels["_gcmv_source_image"] = source_image.copy()

    np.random.seed(99)
    traced_result = traced(deepcopy(labels))
    np.random.seed(99)
    expected_source = stock_for_source(
        {"img": source_image.copy()}
    )["img"]

    assert np.array_equal(
        traced_result["_gcmv_source_image"],
        expected_source,
    )


def test_source_to_global_matrix_composes_resize_affine_and_flip():
    affine = np.array(
        [[1.2, 0.0, 10.0], [0.0, 1.2, 20.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    actual = gcmv_data.compose_source_to_global(
        source_shape=(100, 200),
        pre_affine_shape=(50, 100),
        affine_matrix=affine,
        network_shape=(640, 640),
        flip_horizontal=True,
        flip_vertical=False,
    )
    resize = np.diag([0.5, 0.5, 1.0]).astype(np.float32)
    horizontal = np.array(
        [[-1.0, 0.0, 640.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    assert np.allclose(actual, horizontal @ affine @ resize)


def test_trace_transform_pipeline_replaces_only_geometry_sensitive_transforms():
    stock = Compose(
        [
            RandomPerspective(
                degrees=0.0,
                translate=0.1,
                scale=0.5,
                shear=0.0,
                perspective=0.0,
                size=(640, 640),
            ),
            RandomHSV(hgain=0.015, sgain=0.7, vgain=0.4),
            RandomFlip(direction="vertical", p=0.0, flip_idx=[]),
            RandomFlip(direction="horizontal", p=0.5, flip_idx=[]),
        ]
    )

    traced = gcmv_data.trace_gcmv_transforms(stock)

    assert isinstance(traced.transforms[0], gcmv_data.GCMVRandomPerspective)
    assert isinstance(traced.transforms[1], gcmv_data.GCMVRandomHSV)
    assert isinstance(traced.transforms[2], gcmv_data.GCMVRandomFlip)
    assert isinstance(traced.transforms[3], gcmv_data.GCMVRandomFlip)


def test_traced_letterbox_preserves_stock_result_and_records_affine():
    image = np.arange(40 * 80 * 3, dtype=np.uint8).reshape(40, 80, 3)
    stock = LetterBox(new_shape=(64, 64), scaleup=False)
    traced = gcmv_data.GCMVLetterBox(
        new_shape=(64, 64),
        scaleup=False,
    )

    stock_result = stock(empty_labels(image))
    traced_result = traced(empty_labels(image))

    assert np.array_equal(traced_result["img"], stock_result["img"])
    assert np.array_equal(
        traced_result["instances"].bboxes,
        stock_result["instances"].bboxes,
    )
    assert traced_result["_gcmv_pre_affine_shape"] == (40, 80)
    expected = np.array(
        [[0.8, 0.0, 0.0], [0.0, 0.8, 16.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    assert np.allclose(traced_result["_gcmv_affine_matrix"], expected)


def test_finalize_provenance_inverts_source_to_global_matrix():
    transformed = {
        "_gcmv_affine_matrix": np.array(
            [[1.0, 0.0, 7.0], [0.0, 1.0, 11.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        "_gcmv_pre_affine_shape": (50, 100),
        "_gcmv_flip_vertical": False,
        "_gcmv_flip_horizontal": True,
        "img": torch.zeros(3, 640, 640, dtype=torch.uint8),
    }

    source_to_global, global_to_source = gcmv_data.finalize_gcmv_provenance(
        transformed,
        source_shape=(100, 200),
    )

    assert source_to_global.shape == (3, 3)
    assert global_to_source.shape == (3, 3)
    assert torch.allclose(
        global_to_source @ source_to_global,
        torch.eye(3),
        atol=1e-6,
    )


def test_finalize_provenance_rejects_missing_affine_metadata():
    with pytest.raises(ValueError, match="affine provenance"):
        gcmv_data.finalize_gcmv_provenance(
            {"img": torch.zeros(3, 640, 640, dtype=torch.uint8)},
            source_shape=(100, 200),
        )


def test_dataset_keeps_raw_source_available_through_transform_pipeline():
    dataset = object.__new__(GCMVRTDETRDataset)
    dataset.local_imgsz = 32
    source = np.full((32, 32, 3), 17, dtype=np.uint8)
    dataset.get_image_and_label = lambda index: {
        "img": source.copy(),
        "_gcmv_source_image": source.copy(),
    }

    def transform(label):
        assert "_gcmv_source_image" in label
        label["img"] = torch.from_numpy(
            label["img"].transpose(2, 0, 1).copy()
        )
        label["_gcmv_affine_matrix"] = np.eye(3, dtype=np.float32)
        label["_gcmv_pre_affine_shape"] = (32, 32)
        label["_gcmv_flip_horizontal"] = False
        label["_gcmv_flip_vertical"] = False
        return label

    dataset.transforms = transform

    sample = GCMVRTDETRDataset.__getitem__(dataset, 0)

    assert sample["local_views"].shape == (4, 3, 32, 32)
    assert sample["source_shape"].tolist() == [32, 32]
    assert sample["global_to_source"].shape == (3, 3)


def test_dataset_uses_stock_long_side_resize_default():
    parameter = inspect.signature(
        GCMVRTDETRDataset.load_image
    ).parameters["rect_mode"]

    assert parameter.default is True
