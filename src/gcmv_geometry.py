"""Exact feature-lattice geometry for GCMV's PLEC module."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from src.sbr_geometry import LetterboxTransform, Tile


_PHASE_OFFSETS = (
    (-1.0 / 3.0, -1.0 / 3.0),
    (0.0, -1.0 / 3.0),
    (1.0 / 3.0, -1.0 / 3.0),
    (-1.0 / 3.0, 0.0),
    (0.0, 0.0),
    (1.0 / 3.0, 0.0),
    (-1.0 / 3.0, 1.0 / 3.0),
    (0.0, 1.0 / 3.0),
    (1.0 / 3.0, 1.0 / 3.0),
)


@dataclass(frozen=True)
class PLECGeometry:
    sample_grid: torch.Tensor
    sample_valid: torch.Tensor
    center_valid: torch.Tensor
    subcell_offset: torch.Tensor
    magnification: torch.Tensor
    edge_distance: torch.Tensor
    local_feature_shape: tuple[int, int]
    global_feature_shape: tuple[int, int]

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> PLECGeometry:
        if dtype is not None and not dtype.is_floating_point:
            raise TypeError("geometry dtype must be floating point")

        def move(value: torch.Tensor) -> torch.Tensor:
            target_dtype = dtype if value.is_floating_point() else value.dtype
            return value.to(device=device, dtype=target_dtype).detach()

        return PLECGeometry(
            sample_grid=move(self.sample_grid),
            sample_valid=move(self.sample_valid),
            center_valid=move(self.center_valid),
            subcell_offset=move(self.subcell_offset),
            magnification=move(self.magnification),
            edge_distance=move(self.edge_distance),
            local_feature_shape=self.local_feature_shape,
            global_feature_shape=self.global_feature_shape,
        )


def _validate_shape(name: str, shape: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(shape, (tuple, list))
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
    ):
        raise TypeError(f"{name} shape must contain two integers")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    return height, width


def _validate_transform(
    transform: LetterboxTransform,
    *,
    source_height: int,
    source_width: int,
    label: str,
) -> None:
    if not isinstance(transform, LetterboxTransform):
        raise TypeError(f"{label} must be a LetterboxTransform")
    if (
        transform.source_height != source_height
        or transform.source_width != source_width
    ):
        raise ValueError(f"{label} source dimensions do not match its source frame")
    if transform.network_height is None or transform.network_width is None:
        raise ValueError(f"{label} network dimensions are required")
    if transform.network_height <= 0 or transform.network_width <= 0:
        raise ValueError(f"{label} network dimensions must be positive")
    if (
        transform.gain_x <= 0
        or transform.gain_y <= 0
        or not math.isfinite(transform.gain_x)
        or not math.isfinite(transform.gain_y)
    ):
        raise ValueError(f"{label} letterbox gain must be finite and positive")


def _validate_arguments(
    *,
    source_shapes: Sequence[tuple[int, int]],
    tiles: Sequence[Sequence[Tile]],
    global_transforms: Sequence[LetterboxTransform],
    local_transforms: Sequence[Sequence[LetterboxTransform]],
    global_to_source: Sequence[torch.Tensor] | None,
    global_feature_shape: tuple[int, int],
    local_feature_shape: tuple[int, int],
    dtype: torch.dtype,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("geometry dtype must be floating point")
    global_shape = _validate_shape("global feature", global_feature_shape)
    local_shape = _validate_shape("local feature", local_feature_shape)
    batch_size = len(source_shapes)
    if batch_size == 0:
        raise ValueError("batch must contain at least one image")
    if not (
        len(tiles)
        == len(global_transforms)
        == len(local_transforms)
        == batch_size
        ):
            raise ValueError("batch metadata lengths must match")
    if global_to_source is not None and len(global_to_source) != batch_size:
        raise ValueError("global_to_source batch metadata length must match")

    for image_index, source_shape in enumerate(source_shapes):
        source_height, source_width = _validate_shape("source", source_shape)
        image_tiles = tiles[image_index]
        image_local_transforms = local_transforms[image_index]
        if len(image_tiles) != 4 or len(image_local_transforms) != 4:
            raise ValueError("PLEC geometry requires exactly four local views")
        if [tile.index for tile in image_tiles] != [0, 1, 2, 3]:
            raise ValueError("tile order must be TL/TR/BL/BR with indices 0,1,2,3")
        _validate_transform(
            global_transforms[image_index],
            source_height=source_height,
            source_width=source_width,
            label="global transform",
        )
        for tile, transform in zip(image_tiles, image_local_transforms):
            if tile.right > source_width or tile.bottom > source_height:
                raise ValueError("tile bounds exceed the source image")
            _validate_transform(
                transform,
                source_height=tile.height,
                source_width=tile.width,
                label=f"local transform {tile.index}",
            )
        if global_to_source is not None:
            matrix = global_to_source[image_index]
            if not isinstance(matrix, torch.Tensor) or matrix.shape != (3, 3):
                raise TypeError("global_to_source entries must be 3x3 tensors")
            if not torch.isfinite(matrix).all():
                raise ValueError("global_to_source matrix must be finite")
            if matrix.requires_grad:
                raise ValueError("global_to_source matrix must be detached")
            matrix_cpu = matrix.detach().to(dtype=torch.float64, device="cpu")
            if not torch.allclose(
                matrix_cpu[2],
                torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError("global_to_source matrix must be affine")
            if (
                abs(float(matrix_cpu[0, 1])) > 1e-8
                or abs(float(matrix_cpu[1, 0])) > 1e-8
            ):
                raise ValueError(
                    "global_to_source matrix must be axis-aligned"
                )
            if abs(float(torch.linalg.det(matrix_cpu))) <= 1e-12:
                raise ValueError("global_to_source matrix is singular")
    return global_shape, local_shape


def build_plec_geometry(
    *,
    source_shapes: Sequence[tuple[int, int]],
    tiles: Sequence[Sequence[Tile]],
    global_transforms: Sequence[LetterboxTransform],
    local_transforms: Sequence[Sequence[LetterboxTransform]],
    global_to_source: Sequence[torch.Tensor] | None = None,
    global_feature_shape: tuple[int, int],
    local_feature_shape: tuple[int, int],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> PLECGeometry:
    (global_height, global_width), (local_height, local_width) = _validate_arguments(
        source_shapes=source_shapes,
        tiles=tiles,
        global_transforms=global_transforms,
        local_transforms=local_transforms,
        global_to_source=global_to_source,
        global_feature_shape=global_feature_shape,
        local_feature_shape=local_feature_shape,
        dtype=dtype,
    )
    phase_offsets = torch.tensor(_PHASE_OFFSETS, device=device, dtype=dtype)
    y_index, x_index = torch.meshgrid(
        torch.arange(global_height, device=device, dtype=dtype),
        torch.arange(global_width, device=device, dtype=dtype),
        indexing="ij",
    )

    batch_grids: list[torch.Tensor] = []
    batch_sample_valid: list[torch.Tensor] = []
    batch_center_valid: list[torch.Tensor] = []
    batch_subcell: list[torch.Tensor] = []
    batch_magnification: list[torch.Tensor] = []
    batch_edge_distance: list[torch.Tensor] = []

    for image_index, (
        image_tiles,
        global_transform,
        image_local_transforms,
    ) in enumerate(zip(tiles, global_transforms, local_transforms)):
        inverse_matrix = (
            None
            if global_to_source is None
            else global_to_source[image_index]
            .detach()
            .to(device=device, dtype=dtype)
        )
        view_grids: list[torch.Tensor] = []
        view_sample_valid: list[torch.Tensor] = []
        view_center_valid: list[torch.Tensor] = []
        view_subcell: list[torch.Tensor] = []
        view_magnification: list[torch.Tensor] = []
        view_edge_distance: list[torch.Tensor] = []

        for tile, local_transform in zip(image_tiles, image_local_transforms):
            phase_x = x_index.unsqueeze(0) + phase_offsets[:, 0, None, None]
            phase_y = y_index.unsqueeze(0) + phase_offsets[:, 1, None, None]
            global_network_x = (
                (phase_x + 0.5) * global_transform.network_width / global_width
            )
            global_network_y = (
                (phase_y + 0.5) * global_transform.network_height / global_height
            )
            if inverse_matrix is None:
                source_x = (
                    global_network_x - global_transform.pad_x
                ) / global_transform.gain_x
                source_y = (
                    global_network_y - global_transform.pad_y
                ) / global_transform.gain_y
                global_source_scale_x = 1.0 / global_transform.gain_x
                global_source_scale_y = 1.0 / global_transform.gain_y
            else:
                source_x = (
                    inverse_matrix[0, 0] * global_network_x
                    + inverse_matrix[0, 2]
                )
                source_y = (
                    inverse_matrix[1, 1] * global_network_y
                    + inverse_matrix[1, 2]
                )
                global_source_scale_x = abs(
                    float(inverse_matrix[0, 0])
                )
                global_source_scale_y = abs(
                    float(inverse_matrix[1, 1])
                )
            local_network_x = (
                (source_x - tile.left) * local_transform.gain_x
                + local_transform.pad_x
            )
            local_network_y = (
                (source_y - tile.top) * local_transform.gain_y
                + local_transform.pad_y
            )
            local_feature_x = (
                local_network_x * local_width / local_transform.network_width - 0.5
            )
            local_feature_y = (
                local_network_y * local_height / local_transform.network_height - 0.5
            )
            grid_x = 2.0 * (local_feature_x + 0.5) / local_width - 1.0
            grid_y = 2.0 * (local_feature_y + 0.5) / local_height - 1.0
            grid = torch.stack((grid_x, grid_y), dim=-1)

            sample_valid = (
                (source_x >= tile.left)
                & (source_x < tile.right)
                & (source_y >= tile.top)
                & (source_y < tile.bottom)
                & (grid_x >= -1.0)
                & (grid_x <= 1.0)
                & (grid_y >= -1.0)
                & (grid_y <= 1.0)
            )
            subcell_x = torch.remainder(local_feature_x + 0.5, 1.0) - 0.5
            subcell_y = torch.remainder(local_feature_y + 0.5, 1.0) - 0.5
            subcell = torch.stack((subcell_x, subcell_y), dim=1)

            mag_x = (
                global_transform.network_width
                / global_width
                * global_source_scale_x
                * local_transform.gain_x
                * local_width
                / local_transform.network_width
            )
            mag_y = (
                global_transform.network_height
                / global_height
                * global_source_scale_y
                * local_transform.gain_y
                * local_height
                / local_transform.network_height
            )
            magnification = torch.empty(
                (2, global_height, global_width), device=device, dtype=dtype
            )
            magnification[0].fill_(mag_x)
            magnification[1].fill_(mag_y)

            center_x = source_x[4]
            center_y = source_y[4]
            distances = torch.stack(
                (
                    center_x - tile.left,
                    tile.right - center_x,
                    center_y - tile.top,
                    tile.bottom - center_y,
                )
            )
            edge_distance = (
                distances.amin(dim=0)
                / (0.5 * min(tile.width, tile.height))
            ).clamp_(0.0, 1.0)
            center_valid = sample_valid[4].unsqueeze(0)
            edge_distance = edge_distance.unsqueeze(0) * center_valid

            view_grids.append(grid)
            view_sample_valid.append(sample_valid)
            view_center_valid.append(center_valid)
            view_subcell.append(subcell)
            view_magnification.append(magnification)
            view_edge_distance.append(edge_distance)

        batch_grids.append(torch.stack(view_grids))
        batch_sample_valid.append(torch.stack(view_sample_valid))
        batch_center_valid.append(torch.stack(view_center_valid))
        batch_subcell.append(torch.stack(view_subcell))
        batch_magnification.append(torch.stack(view_magnification))
        batch_edge_distance.append(torch.stack(view_edge_distance))

    return PLECGeometry(
        sample_grid=torch.stack(batch_grids),
        sample_valid=torch.stack(batch_sample_valid),
        center_valid=torch.stack(batch_center_valid),
        subcell_offset=torch.stack(batch_subcell),
        magnification=torch.stack(batch_magnification),
        edge_distance=torch.stack(batch_edge_distance),
        local_feature_shape=local_feature_shape,
        global_feature_shape=global_feature_shape,
    )
