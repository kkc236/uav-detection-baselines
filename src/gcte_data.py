"""Lazy Ultralytics dataset construction for deterministic GCQF caching."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.gcte_views import IMAGE_SIZE, build_local_view_tensor


GCQF_CACHE_AUGMENT = False


def build_gcqf_dataset(
    data: Mapping[str, Any],
    *,
    split: str,
    batch_size: int = 1,
    image_path: str | None = None,
):
    """Build a no-augmentation RT-DETR dataset with exact SADED local views."""

    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if split not in data and image_path is None:
        raise ValueError(f"dataset has no {split} split")
    try:
        from ultralytics.cfg import get_cfg
        from ultralytics.utils.patches import imread

        from src.gcmv_data import GCMVRTDETRDataset
    except Exception as error:
        raise RuntimeError(
            "Ultralytics 8.4.90 dataset dependencies are required"
        ) from error

    class GCQFRTDETRDataset(GCMVRTDETRDataset):
        def cache_labels(self, path: Path = Path("./labels.cache")):
            # Ultralytics writes its label index even when image caching is
            # disabled.  The sealed VisDrone mount can be read-only or full,
            # so build the same index in a disposable local directory.
            with TemporaryDirectory(prefix="gcqf-label-cache-") as directory:
                temporary = Path(directory) / Path(path).name
                return super().cache_labels(temporary)

        def __getitem__(self, index: int) -> dict[str, Any]:
            sample = super().__getitem__(index)
            source = imread(self.im_files[index], flags=self.cv2_flag)
            if source is None:
                raise FileNotFoundError(self.im_files[index])
            if source.ndim == 2:
                source = source[..., None]
            if source.shape[2] != 3:
                raise RuntimeError("GCQF source image must have three channels")
            sample["local_views"] = build_local_view_tensor(source)
            return sample

    hyp = get_cfg(
        overrides={
            "imgsz": IMAGE_SIZE,
            "rect": False,
            "cache": False,
            "single_cls": False,
            "classes": None,
            "mask_ratio": 4,
            "overlap_mask": True,
            "bgr": 0.0,
        }
    )
    return GCQFRTDETRDataset(
        img_path=image_path or data[split],
        imgsz=IMAGE_SIZE,
        local_imgsz=IMAGE_SIZE,
        batch_size=batch_size,
        augment=GCQF_CACHE_AUGMENT,
        hyp=hyp,
        rect=False,
        cache=None,
        single_cls=False,
        prefix=f"gcqf-cache-{split}: ",
        classes=None,
        data=dict(data),
        fraction=1.0,
    )


__all__ = ["GCQF_CACHE_AUGMENT", "build_gcqf_dataset"]
