"""Paired source-resolution views for the first GCMV PLEC screen."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.rtdetr.val import RTDETRDataset
from ultralytics.utils.patches import imread

from src.sbr_geometry import overlapping_tiles


GCMV_GLOBAL_IMAGE_SIZE = 640
GCMV_LOCAL_IMAGE_SIZE = 1088


def _validate_source_image(image: np.ndarray) -> tuple[int, int]:
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
    ):
        raise TypeError("source image must be a uint8 HxWx3 array")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("source image dimensions must be positive")
    return height, width


def _letterbox_rgb_chw(image: np.ndarray, *, imgsz: int) -> torch.Tensor:
    if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
        raise ValueError("imgsz must be a positive integer")
    network_image = LetterBox(
        new_shape=(imgsz, imgsz),
        auto=False,
        scale_fill=False,
        scaleup=False,
        center=True,
        padding_value=114,
    )(image=image)
    rgb_chw = np.ascontiguousarray(network_image[:, :, ::-1].transpose(2, 0, 1))
    return torch.from_numpy(rgb_chw)


def build_local_view_tensor(
    source_image: np.ndarray,
    *,
    local_imgsz: int = GCMV_LOCAL_IMAGE_SIZE,
) -> torch.Tensor:
    """Cut TL/TR/BL/BR views from a raw source image and letterbox each view."""

    height, width = _validate_source_image(source_image)
    views = []
    for tile in overlapping_tiles(width=width, height=height):
        crop = source_image[tile.top : tile.bottom, tile.left : tile.right]
        views.append(_letterbox_rgb_chw(crop, imgsz=local_imgsz))
    return torch.stack(views)


class GCMVRTDETRDataset(RTDETRDataset):
    """RT-DETR labels plus four unaugmented source-resolution local views."""

    def __init__(
        self,
        *args: Any,
        local_imgsz: int = GCMV_LOCAL_IMAGE_SIZE,
        **kwargs: Any,
    ) -> None:
        kwargs["augment"] = False
        self.local_imgsz = int(local_imgsz)
        if self.local_imgsz <= 0:
            raise ValueError("local_imgsz must be positive")
        super().__init__(*args, **kwargs)

    def load_image(self, i: int, rect_mode: bool = False):
        """Load the original source pixels without RT-DETR's square pre-resize."""

        image = imread(self.im_files[i], flags=self.cv2_flag)
        if image is None:
            raise FileNotFoundError(f"Image Not Found {self.im_files[i]}")
        if image.ndim == 2:
            image = image[..., None]
        height, width = image.shape[:2]
        return image, (height, width), (height, width)

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        label = super().get_image_and_label(index)
        label["_gcmv_source_image"] = label["img"].copy()
        return label

    def __getitem__(self, index: int) -> dict[str, Any]:
        label = self.get_image_and_label(index)
        source_image = label.pop("_gcmv_source_image")
        transformed = self.transforms(label)
        transformed["local_views"] = build_local_view_tensor(
            source_image,
            local_imgsz=self.local_imgsz,
        )
        transformed["source_shape"] = torch.tensor(
            source_image.shape[:2], dtype=torch.long
        )
        return transformed

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        local_views = torch.stack(
            [sample.pop("local_views") for sample in batch]
        )
        source_shapes = torch.stack(
            [sample.pop("source_shape") for sample in batch]
        )
        collated = YOLODataset.collate_fn(batch)
        collated["local_views"] = local_views
        collated["source_shape"] = source_shapes
        return collated

