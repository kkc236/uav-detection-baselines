"""Pure, frozen geometry helpers for SBR-RTDETR views."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Tile:
    """A half-open image rectangle in ``[left, top, right, bottom)`` pixels."""

    left: int
    top: int
    right: int
    bottom: int
    index: int = 0

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom, self.index)
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
            raise TypeError("tile coordinates and index must be integers")
        if self.left < 0 or self.top < 0 or self.right <= self.left or self.bottom <= self.top:
            raise ValueError("tile must be a non-empty half-open rectangle")

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    @property
    def x1(self) -> int:
        return self.left

    @property
    def x(self) -> int:
        return self.left

    @property
    def y1(self) -> int:
        return self.top

    @property
    def y(self) -> int:
        return self.top

    @property
    def x2(self) -> int:
        return self.right

    @property
    def y2(self) -> int:
        return self.bottom

    @property
    def id(self) -> int:
        return self.index

    @property
    def origin(self) -> tuple[int, int]:
        return self.left, self.top


@dataclass(frozen=True, init=False)
class LetterboxTransform:
    """Metadata for Ultralytics' explicit, centered square LetterBox transform.

    ``gain`` and ``pad`` are accepted as scalar/2-tuples for convenient
    construction in tests; ``gain_x``, ``gain_y``, ``pad_x`` and ``pad_y`` are
    exposed as stable scalar metadata for inverse mapping.
    """

    source_width: int | None
    source_height: int | None
    network_width: int | None
    network_height: int | None
    gain_x: float
    gain_y: float
    pad_x: float
    pad_y: float
    resized_width: int | None
    resized_height: int | None
    auto: bool
    scale_fill: bool
    scaleup: bool
    center: bool
    padding_value: int

    def __init__(
        self,
        source_width: int | None = None,
        source_height: int | None = None,
        imgsz: int | None = None,
        *,
        source_shape: tuple[int, int] | None = None,
        input_shape: tuple[int, int] | None = None,
        network_shape: tuple[int, int] | None = None,
        new_shape: tuple[int, int] | int | None = None,
        gain: float | tuple[float, float] = 1.0,
        pad: tuple[float, float] | float = (0.0, 0.0),
        gain_x: float | None = None,
        gain_y: float | None = None,
        pad_x: float | None = None,
        pad_y: float | None = None,
        resized_width: int | None = None,
        resized_height: int | None = None,
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = False,
        center: bool = True,
        padding_value: int = 114,
    ) -> None:
        # Shape conventions follow Ultralytics: (height, width).
        shape = source_shape if source_shape is not None else input_shape
        if shape is not None:
            if len(shape) != 2:
                raise ValueError("source_shape must be (height, width)")
            source_height, source_width = int(shape[0]), int(shape[1])
        target = network_shape if network_shape is not None else new_shape
        if target is not None:
            if isinstance(target, int):
                network_height = network_width = target
            else:
                if len(target) != 2:
                    raise ValueError("network_shape must be (height, width)")
                network_height, network_width = int(target[0]), int(target[1])
        elif imgsz is not None:
            network_height = network_width = int(imgsz)
        else:
            network_width = network_height = None
        if gain_x is None or gain_y is None:
            if isinstance(gain, Iterable) and not isinstance(gain, (str, bytes)):
                gains = tuple(gain)
                if len(gains) != 2:
                    raise ValueError("gain must be a scalar or (x, y)")
                gx, gy = float(gains[0]), float(gains[1])
            else:
                gx = gy = float(gain)
            gain_x = gx if gain_x is None else float(gain_x)
            gain_y = gy if gain_y is None else float(gain_y)
        if pad_x is None or pad_y is None:
            if isinstance(pad, Iterable) and not isinstance(pad, (str, bytes)):
                pads = tuple(pad)
                if len(pads) != 2:
                    raise ValueError("pad must be a scalar or (x, y)")
                px, py = float(pads[0]), float(pads[1])
            else:
                px = py = float(pad)
            pad_x = px if pad_x is None else float(pad_x)
            pad_y = py if pad_y is None else float(pad_y)
        for name, value in (
            ("source_width", source_width),
            ("source_height", source_height),
            ("network_width", network_width),
            ("network_height", network_height),
            ("resized_width", resized_width),
            ("resized_height", resized_height),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, None if value is None else int(value))
        for name, value in (("gain_x", gain_x), ("gain_y", gain_y), ("pad_x", pad_x), ("pad_y", pad_y)):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        for name, value in (
            ("auto", bool(auto)),
            ("scale_fill", bool(scale_fill)),
            ("scaleup", bool(scaleup)),
            ("center", bool(center)),
            ("padding_value", int(padding_value)),
        ):
            object.__setattr__(self, name, value)

    @classmethod
    def from_view(cls, *, width: int, height: int, imgsz: int) -> "LetterboxTransform":
        if width <= 0 or height <= 0 or imgsz <= 0:
            raise ValueError("view dimensions and imgsz must be positive")
        gain = min(float(imgsz) / width, float(imgsz) / height, 1.0)
        resized_width = max(1, int(round(width * gain)))
        resized_height = max(1, int(round(height * gain)))
        pad_x = (imgsz - resized_width) / 2.0
        pad_y = (imgsz - resized_height) / 2.0
        return cls(
            source_width=width,
            source_height=height,
            imgsz=imgsz,
            gain_x=gain,
            gain_y=gain,
            pad_x=pad_x,
            pad_y=pad_y,
            resized_width=resized_width,
            resized_height=resized_height,
            auto=False,
            scale_fill=False,
            scaleup=False,
            center=True,
            padding_value=114,
        )

    # Common metadata spellings used by callers.
    @property
    def gain(self) -> tuple[float, float]:
        return self.gain_x, self.gain_y

    @property
    def pad(self) -> tuple[float, float]:
        return self.pad_x, self.pad_y

    @property
    def source_shape(self) -> tuple[int, int] | None:
        if self.source_height is None or self.source_width is None:
            return None
        return self.source_height, self.source_width

    @property
    def network_shape(self) -> tuple[int, int] | None:
        if self.network_height is None or self.network_width is None:
            return None
        return self.network_height, self.network_width

    @property
    def new_shape(self) -> tuple[int, int] | None:
        return self.network_shape

    @property
    def imgsz(self) -> int | None:
        if self.network_width == self.network_height:
            return self.network_width
        return None


def _validate_dimensions(width: int, height: int) -> None:
    if isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer)):
        raise TypeError("width and height must be integers")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")


def overlapping_tiles(width: int, height: int) -> tuple[Tile, Tile, Tile, Tile]:
    """Return ordered TL, TR, BL, BR tiles with 60% dimensions."""
    _validate_dimensions(width, height)
    tile_w = int(math.ceil(0.60 * width))
    tile_h = int(math.ceil(0.60 * height))
    x_origins = (0, width - tile_w)
    y_origins = (0, height - tile_h)
    overlap_x = tile_w - x_origins[1]
    overlap_y = tile_h - y_origins[1]
    if overlap_x <= 0 or overlap_y <= 0:
        raise ValueError("tile overlap must be positive")
    return (
        Tile(x_origins[0], y_origins[0], x_origins[0] + tile_w, y_origins[0] + tile_h, 0),
        Tile(x_origins[1], y_origins[0], x_origins[1] + tile_w, y_origins[0] + tile_h, 1),
        Tile(x_origins[0], y_origins[1], x_origins[0] + tile_w, y_origins[1] + tile_h, 2),
        Tile(x_origins[1], y_origins[1], x_origins[1] + tile_w, y_origins[1] + tile_h, 3),
    )


def non_overlapping_tiles(width: int, height: int) -> tuple[Tile, Tile, Tile, Tile]:
    """Return ordered four-way Arm F tiles, assigning odd remainders right/bottom."""
    _validate_dimensions(width, height)
    mid_x, mid_y = width // 2, height // 2
    if mid_x <= 0 or mid_y <= 0:
        raise ValueError("non-overlapping partition requires dimensions >= 2")
    return (
        Tile(0, 0, mid_x, mid_y, 0),
        Tile(mid_x, 0, width, mid_y, 1),
        Tile(0, mid_y, mid_x, height, 2),
        Tile(mid_x, mid_y, width, height, 3),
    )


def _boxes_array(boxes: object) -> tuple[np.ndarray, bool]:
    array = np.asarray(boxes, dtype=float)
    was_one = array.ndim == 1
    if was_one:
        array = array.reshape(1, 4) if array.size == 4 else array
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("boxes must have shape (4,) or (N, 4)")
    if not np.isfinite(array).all():
        raise ValueError("boxes must contain only finite values")
    if np.any(array[:, 2] < array[:, 0]) or np.any(array[:, 3] < array[:, 1]):
        raise ValueError("boxes must be valid xyxy rectangles")
    return array, was_one


def inverse_letterbox_xyxy(
    boxes: object,
    transform: LetterboxTransform,
    *,
    normalized: bool = False,
) -> np.ndarray:
    """Map network-input xyxy boxes back to the source view pixel frame."""
    array, was_one = _boxes_array(boxes)
    if transform.gain_x <= 0 or transform.gain_y <= 0:
        raise ValueError("letterbox gain must be positive")
    if normalized:
        if transform.network_width is None or transform.network_height is None:
            raise ValueError("normalized boxes require network dimensions")
        array = array.copy()
        array[:, [0, 2]] *= transform.network_width
        array[:, [1, 3]] *= transform.network_height
    restored = array.copy()
    restored[:, [0, 2]] = (restored[:, [0, 2]] - transform.pad_x) / transform.gain_x
    restored[:, [1, 3]] = (restored[:, [1, 3]] - transform.pad_y) / transform.gain_y
    return restored[0] if was_one else restored


def tile_to_global_xyxy(boxes: object, tile: Tile, width: int, height: int) -> np.ndarray:
    """Offset view-frame boxes exactly once and clip to full-image bounds."""
    _validate_dimensions(width, height)
    if tile.right > width or tile.bottom > height:
        raise ValueError("tile exceeds image bounds")
    array, was_one = _boxes_array(boxes)
    mapped = array.copy()
    mapped[:, [0, 2]] += tile.left
    mapped[:, [1, 3]] += tile.top
    mapped[:, [0, 2]] = np.clip(mapped[:, [0, 2]], 0.0, float(width))
    mapped[:, [1, 3]] = np.clip(mapped[:, [1, 3]], 0.0, float(height))
    return mapped[0] if was_one else mapped
