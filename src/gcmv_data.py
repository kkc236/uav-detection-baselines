"""Paired source-resolution views for the first GCMV-EI screen."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ultralytics.data.augment import (
    Compose,
    LetterBox,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
)
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.rtdetr.val import RTDETRDataset
from ultralytics.utils.patches import imread

from src.sbr_geometry import overlapping_tiles


GCMV_GLOBAL_IMAGE_SIZE = 640
GCMV_LOCAL_IMAGE_SIZE = 1088


class GCMVLetterBox(LetterBox):
    """Ultralytics LetterBox with exact affine provenance."""

    def __call__(
        self,
        labels: dict[str, Any] | None = None,
        image: np.ndarray | None = None,
    ) -> dict[str, Any] | np.ndarray:
        if labels is None:
            labels = {}
        return_image_only = len(labels) == 0
        if image is not None:
            labels["img"] = image
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        if not return_image_only:
            labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        if return_image_only:
            return labels["img"]
        ratio_x, ratio_y = params["ratio"]
        labels["_gcmv_affine_matrix"] = np.array(
            [
                [ratio_x, 0.0, float(params["left"])],
                [0.0, ratio_y, float(params["top"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        labels["_gcmv_pre_affine_shape"] = tuple(params["orig_shape"])
        return labels


class GCMVRandomPerspective(RandomPerspective):
    """Ultralytics RandomPerspective with exact matrix provenance."""

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        labels["_gcmv_affine_matrix"] = params["M"].copy()
        labels["_gcmv_pre_affine_shape"] = tuple(params["orig_shape"])
        return labels


class GCMVRandomFlip(RandomFlip):
    """Ultralytics RandomFlip with the sampled decision retained."""

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        labels[f"_gcmv_flip_{self.direction}"] = bool(params["flip"])
        return labels


class GCMVRandomHSV(RandomHSV):
    """Replay the global HSV draw on the raw source without RNG drift."""

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        source = labels.get("_gcmv_source_image")
        if source is None:
            return super().__call__(labels)
        state_before = np.random.get_state()
        labels = super().__call__(labels)
        state_after = np.random.get_state()
        try:
            np.random.set_state(state_before)
            replayed = super().__call__({"img": source.copy()})
            labels["_gcmv_source_image"] = replayed["img"]
        finally:
            np.random.set_state(state_after)
        return labels


def trace_gcmv_transforms(transform: Any) -> Any:
    """Replace stochastic geometry/color transforms with RNG-equivalent tracers."""

    if isinstance(
        transform,
        (
            GCMVLetterBox,
            GCMVRandomPerspective,
            GCMVRandomFlip,
            GCMVRandomHSV,
        ),
    ):
        return transform
    if isinstance(transform, LetterBox):
        return GCMVLetterBox(
            new_shape=transform.new_shape,
            auto=transform.auto,
            scale_fill=transform.scale_fill,
            scaleup=transform.scaleup,
            center=transform.center,
            stride=transform.stride,
            padding_value=transform.padding_value,
            interpolation=transform.interpolation,
        )
    if isinstance(transform, RandomPerspective):
        return GCMVRandomPerspective(
            degrees=transform.degrees,
            translate=transform.translate,
            scale=transform.scale,
            shear=transform.shear,
            perspective=transform.perspective,
            size=transform.size,
        )
    if isinstance(transform, RandomHSV):
        return GCMVRandomHSV(
            hgain=transform.hgain,
            sgain=transform.sgain,
            vgain=transform.vgain,
        )
    if isinstance(transform, RandomFlip):
        return GCMVRandomFlip(
            p=transform.p,
            direction=transform.direction,
            flip_idx=transform.flip_idx,
        )
    if isinstance(transform, Compose):
        transform.transforms = [
            trace_gcmv_transforms(child) for child in transform.transforms
        ]
        return transform
    pre_transform = getattr(transform, "pre_transform", None)
    if pre_transform is not None:
        transform.pre_transform = trace_gcmv_transforms(pre_transform)
    return transform


def compose_source_to_global(
    *,
    source_shape: tuple[int, int],
    pre_affine_shape: tuple[int, int],
    affine_matrix: np.ndarray,
    network_shape: tuple[int, int],
    flip_horizontal: bool,
    flip_vertical: bool,
) -> np.ndarray:
    """Compose the exact continuous source-edge to global-network transform."""

    source_height, source_width = source_shape
    resized_height, resized_width = pre_affine_shape
    network_height, network_width = network_shape
    if min(
        source_height,
        source_width,
        resized_height,
        resized_width,
        network_height,
        network_width,
    ) <= 0:
        raise ValueError("all source, resized, and network dimensions must be positive")
    matrix = np.asarray(affine_matrix, dtype=np.float32)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("affine_matrix must be a finite 3x3 matrix")
    resize = np.diag(
        [
            resized_width / source_width,
            resized_height / source_height,
            1.0,
        ]
    ).astype(np.float32)
    composed = matrix @ resize
    if flip_vertical:
        vertical = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, float(network_height)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        composed = vertical @ composed
    if flip_horizontal:
        horizontal = np.array(
            [
                [-1.0, 0.0, float(network_width)],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        composed = horizontal @ composed
    return composed


def finalize_gcmv_provenance(
    transformed: dict[str, Any],
    *,
    source_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build and invert the recorded source-to-global transform."""

    if (
        "_gcmv_affine_matrix" not in transformed
        or "_gcmv_pre_affine_shape" not in transformed
    ):
        raise ValueError("GCMV affine provenance is missing")
    image = transformed.get("img")
    if not isinstance(image, torch.Tensor) or image.ndim != 3:
        raise TypeError("transformed image must be a CHW tensor")
    matrix = compose_source_to_global(
        source_shape=source_shape,
        pre_affine_shape=tuple(transformed["_gcmv_pre_affine_shape"]),
        affine_matrix=transformed["_gcmv_affine_matrix"],
        network_shape=tuple(int(value) for value in image.shape[-2:]),
        flip_horizontal=bool(
            transformed.get("_gcmv_flip_horizontal", False)
        ),
        flip_vertical=bool(transformed.get("_gcmv_flip_vertical", False)),
    )
    source_to_global = torch.from_numpy(matrix.copy())
    try:
        global_to_source = torch.linalg.inv(
            source_to_global.to(dtype=torch.float64)
        ).to(dtype=source_to_global.dtype)
    except RuntimeError as error:
        raise ValueError("GCMV affine provenance is singular") from error
    if not torch.isfinite(global_to_source).all():
        raise ValueError("GCMV affine inverse is nonfinite")
    return source_to_global, global_to_source


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
    """Stock RT-DETR samples plus four source-resolution local views."""

    def __init__(
        self,
        *args: Any,
        local_imgsz: int = GCMV_LOCAL_IMAGE_SIZE,
        **kwargs: Any,
    ) -> None:
        self.local_imgsz = int(local_imgsz)
        if self.local_imgsz <= 0:
            raise ValueError("local_imgsz must be positive")
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp: Any = None) -> Compose:
        return trace_gcmv_transforms(super().build_transforms(hyp))

    def load_image(self, i: int, rect_mode: bool = True):
        """Retain the stock RT-DETR long-side resize."""

        return super().load_image(i, rect_mode=rect_mode)

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        label = super().get_image_and_label(index)
        source = imread(self.im_files[index], flags=self.cv2_flag)
        if source is None:
            raise FileNotFoundError(f"Image Not Found {self.im_files[index]}")
        if source.ndim == 2:
            source = source[..., None]
        if tuple(source.shape[:2]) != tuple(label["ori_shape"]):
            raise RuntimeError("GCMV source shape drift")
        label["_gcmv_source_image"] = source
        return label

    def __getitem__(self, index: int) -> dict[str, Any]:
        label = self.get_image_and_label(index)
        source_image = label["_gcmv_source_image"]
        source_shape = tuple(int(value) for value in source_image.shape[:2])
        transformed = self.transforms(label)
        source_to_global, global_to_source = finalize_gcmv_provenance(
            transformed,
            source_shape=source_shape,
        )
        transformed["local_views"] = build_local_view_tensor(
            transformed.pop("_gcmv_source_image"),
            local_imgsz=self.local_imgsz,
        )
        transformed["source_shape"] = torch.tensor(
            source_shape, dtype=torch.long
        )
        transformed["source_to_global"] = source_to_global
        transformed["global_to_source"] = global_to_source
        for key in tuple(transformed):
            if key.startswith("_gcmv_"):
                transformed.pop(key)
        return transformed

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        local_views = torch.stack(
            [sample.pop("local_views") for sample in batch]
        )
        source_shapes = torch.stack(
            [sample.pop("source_shape") for sample in batch]
        )
        source_to_global = torch.stack(
            [sample.pop("source_to_global") for sample in batch]
        )
        global_to_source = torch.stack(
            [sample.pop("global_to_source") for sample in batch]
        )
        collated = YOLODataset.collate_fn(batch)
        collated["local_views"] = local_views
        collated["source_shape"] = source_shapes
        collated["source_to_global"] = source_to_global
        collated["global_to_source"] = global_to_source
        return collated
