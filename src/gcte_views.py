"""Frozen full-plus-four-view image and homography construction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from src.gcte_types import ViewGeometry
from src.sbr_geometry import (
    LetterboxTransform,
    Tile,
    overlapping_tiles,
)


IMAGE_SIZE = 640
LOCAL_VIEWS = 4
TILE_RATIO = 0.6


def transform_xywh_homography(
    boxes: torch.Tensor,
    homography: torch.Tensor,
    *,
    clip: bool,
) -> torch.Tensor:
    """Transform all four corners and return axis-aligned normalized ``xywh``."""

    if boxes.ndim < 2 or boxes.shape[-1] != 4:
        raise ValueError("boxes must end in xywh")
    if homography.shape != (*boxes.shape[:-1], 3, 3):
        raise ValueError("homography must match every input box")
    x, y, width, height = boxes.unbind(dim=-1)
    half_width = width * 0.5
    half_height = height * 0.5
    left, right = x - half_width, x + half_width
    top, bottom = y - half_height, y + half_height
    corners = torch.stack(
        (
            torch.stack((left, top), dim=-1),
            torch.stack((right, top), dim=-1),
            torch.stack((right, bottom), dim=-1),
            torch.stack((left, bottom), dim=-1),
        ),
        dim=-2,
    )
    homogeneous = torch.cat(
        (corners, torch.ones_like(corners[..., :1])),
        dim=-1,
    )
    mapped = torch.matmul(
        homography.detach().unsqueeze(-3),
        homogeneous.unsqueeze(-1),
    ).squeeze(-1)
    divisor = mapped[..., 2:3]
    if bool((divisor.detach().abs() <= 1e-12).any()):
        raise ValueError("homography maps a box corner to infinity")
    mapped_xy = mapped[..., :2] / divisor
    if clip:
        mapped_xy = mapped_xy.clamp(0.0, 1.0)
    minimum = mapped_xy.amin(dim=-2)
    maximum = mapped_xy.amax(dim=-2)
    return torch.cat(
        ((minimum + maximum) * 0.5, maximum - minimum),
        dim=-1,
    )


def build_local_to_global_homography(
    *,
    tile: Tile,
    global_transform: LetterboxTransform,
    local_transform: LetterboxTransform,
) -> torch.Tensor:
    """Map normalized local-network points to normalized global-network points."""

    if (
        global_transform.network_width != IMAGE_SIZE
        or global_transform.network_height != IMAGE_SIZE
        or local_transform.network_width != IMAGE_SIZE
        or local_transform.network_height != IMAGE_SIZE
    ):
        raise ValueError("GCTE homography requires 640x640 view networks")
    if min(
        global_transform.gain_x,
        global_transform.gain_y,
        local_transform.gain_x,
        local_transform.gain_y,
    ) <= 0:
        raise ValueError("letterbox gains must be positive")
    local_normalized_to_network = np.array(
        [
            [IMAGE_SIZE, 0.0, 0.0],
            [0.0, IMAGE_SIZE, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    local_network_to_tile = np.array(
        [
            [
                1.0 / local_transform.gain_x,
                0.0,
                -local_transform.pad_x / local_transform.gain_x,
            ],
            [
                0.0,
                1.0 / local_transform.gain_y,
                -local_transform.pad_y / local_transform.gain_y,
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    tile_to_source = np.array(
        [
            [1.0, 0.0, float(tile.left)],
            [0.0, 1.0, float(tile.top)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    source_to_global_network = np.array(
        [
            [
                global_transform.gain_x,
                0.0,
                global_transform.pad_x,
            ],
            [
                0.0,
                global_transform.gain_y,
                global_transform.pad_y,
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    global_network_to_normalized = np.array(
        [
            [1.0 / IMAGE_SIZE, 0.0, 0.0],
            [0.0, 1.0 / IMAGE_SIZE, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    matrix = (
        global_network_to_normalized
        @ source_to_global_network
        @ tile_to_source
        @ local_network_to_tile
        @ local_normalized_to_network
    )
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) <= 1e-12:
        raise ValueError("local-to-global homography is invalid")
    return torch.from_numpy(matrix.astype(np.float32))


def build_frozen_view_geometry(
    *,
    source_shapes: Sequence[tuple[int, int]],
    queries_per_view: int = 300,
) -> ViewGeometry:
    """Build repeated TL/TR/BL/BR geometry for decoder-query batches."""

    if not source_shapes:
        raise ValueError("source_shapes must be nonempty")
    if queries_per_view <= 0:
        raise ValueError("queries_per_view must be positive")
    all_matrices: list[torch.Tensor] = []
    all_metadata: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    for source_height, source_width in source_shapes:
        if source_height <= 0 or source_width <= 0:
            raise ValueError("source dimensions must be positive")
        global_transform = LetterboxTransform.from_view(
            width=source_width,
            height=source_height,
            imgsz=IMAGE_SIZE,
        )
        matrices: list[torch.Tensor] = []
        metadata: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        for view_index, tile in enumerate(
            overlapping_tiles(source_width, source_height)
        ):
            local_transform = LetterboxTransform.from_view(
                width=tile.width,
                height=tile.height,
                imgsz=IMAGE_SIZE,
            )
            matrix = build_local_to_global_homography(
                tile=tile,
                global_transform=global_transform,
                local_transform=local_transform,
            )
            meta = torch.tensor(
                [
                    tile.left / source_width,
                    tile.top / source_height,
                    tile.width / source_width,
                    tile.height / source_height,
                    local_transform.gain_x,
                    local_transform.gain_y,
                ],
                dtype=torch.float32,
            )
            matrices.append(
                matrix.unsqueeze(0).repeat(queries_per_view, 1, 1)
            )
            metadata.append(
                meta.unsqueeze(0).repeat(queries_per_view, 1)
            )
            indices.append(
                torch.full(
                    (queries_per_view,),
                    view_index,
                    dtype=torch.long,
                )
            )
        all_matrices.append(torch.cat(matrices, dim=0))
        all_metadata.append(torch.cat(metadata, dim=0))
        all_indices.append(torch.cat(indices, dim=0))
    homography = torch.stack(all_matrices)
    crop_metadata = torch.stack(all_metadata)
    view_index = torch.stack(all_indices)
    return ViewGeometry(
        homography=homography,
        crop_metadata=crop_metadata,
        view_index=view_index,
        valid_mask=torch.ones_like(view_index, dtype=torch.bool),
    )


def build_local_view_tensor(
    source_image: np.ndarray,
    *,
    imgsz: int = IMAGE_SIZE,
) -> torch.Tensor:
    """Build exact SADED local images with ``scaleup=False`` letterboxing."""

    if (
        not isinstance(source_image, np.ndarray)
        or source_image.ndim != 3
        or source_image.shape[2] != 3
        or source_image.dtype != np.uint8
    ):
        raise TypeError("source_image must be uint8 HxWx3 BGR")
    if imgsz != IMAGE_SIZE:
        raise ValueError("GCTE freezes local view size at 640")
    try:
        from ultralytics.data.augment import LetterBox
    except Exception as error:
        raise RuntimeError("Ultralytics LetterBox is required") from error
    height, width = source_image.shape[:2]
    views = []
    for tile in overlapping_tiles(width, height):
        crop = source_image[
            tile.top : tile.bottom,
            tile.left : tile.right,
        ]
        network = LetterBox(
            new_shape=(IMAGE_SIZE, IMAGE_SIZE),
            auto=False,
            scale_fill=False,
            scaleup=False,
            center=True,
            padding_value=114,
        )(image=crop)
        rgb_chw = np.ascontiguousarray(
            network[:, :, ::-1].transpose(2, 0, 1)
        )
        views.append(torch.from_numpy(rgb_chw))
    return torch.stack(views)


__all__ = [
    "IMAGE_SIZE",
    "LOCAL_VIEWS",
    "TILE_RATIO",
    "build_frozen_view_geometry",
    "build_local_to_global_homography",
    "build_local_view_tensor",
    "transform_xywh_homography",
]
