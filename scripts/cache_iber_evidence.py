"""Build the fixed no-augmentation IBER-BE Gate-1 evidence cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.iber_cache import DEFAULT_SHARD_SIZE, write_evidence_cache  # noqa: E402
from src.iber_protocol import (  # noqa: E402
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    PRIVATE_SEED,
    RUNTIME_AMENDMENT_SHA256,
)
from src.lpr_protocol import (  # noqa: E402
    CATEGORY_NAMES,
    category_mapping_sha256,
    dataset_signature,
    file_sha256,
    select_hashed_subset,
    subset_signature,
)
from src.rtdetr_iber import FrozenIBERAdapter  # noqa: E402


IMAGE_SIZE = 640
SHARD_SIZE = DEFAULT_SHARD_SIZE
TRAIN_COUNT = 647
VAL_COUNT = 548
EXPECTED_CATEGORY_SHA256 = (
    "1455A1F5D9FA9988799815B36CF2CE6D5044B5BD7CAD7CC614D6B5E5059EF2A6"
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    return parser.parse_args(argv)


def _letterbox(
    image_path: Path, label_path: Path
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unreadable VisDrone image: {image_path}")
    height, width = image.shape[:2]
    scale = min(IMAGE_SIZE / height, IMAGE_SIZE / width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    pad_x = (IMAGE_SIZE - resized_width) / 2
    pad_y = (IMAGE_SIZE - resized_height) / 2
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    rgb = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1)
    image_tensor = rgb.float().div(255).contiguous()

    boxes: list[list[float]] = []
    classes: list[int] = []
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"invalid VisDrone label row: {label_path}")
            category, center_x, center_y, box_width, box_height = fields
            boxes.append(
                [
                    (float(center_x) * width * scale + left) / IMAGE_SIZE,
                    (float(center_y) * height * scale + top) / IMAGE_SIZE,
                    float(box_width) * width * scale / IMAGE_SIZE,
                    float(box_height) * height * scale / IMAGE_SIZE,
                ]
            )
            classes.append(int(category))
    return (
        image_tensor,
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(classes, dtype=torch.long),
    )


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center = boxes[..., :2]
    half_size = boxes[..., 2:].mul(0.5)
    return torch.cat((center - half_size, center + half_size), dim=-1)


def _source_commit() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip().lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("source commit must be exactly 40 hexadecimal characters")
    return commit


def _split_image_paths(dataset_root: Path, split: str) -> list[Path]:
    image_paths = sorted((dataset_root / "images" / split).glob("*.jpg"))
    if split == "train":
        image_paths = select_hashed_subset(
            image_paths,
            root=dataset_root,
            fraction=0.10,
        )
        actual_subset_sha = subset_signature(image_paths, root=dataset_root)
        if len(image_paths) != TRAIN_COUNT or actual_subset_sha != EXPECTED_SUBSET_SHA256:
            raise ValueError(
                "IBER train subset authority mismatch: "
                f"count={len(image_paths)}, sha256={actual_subset_sha}"
            )
    elif split == "val":
        if len(image_paths) != VAL_COUNT:
            raise ValueError(
                f"IBER validation count mismatch: expected={VAL_COUNT}, actual={len(image_paths)}"
            )
    else:
        raise ValueError(f"unsupported cache split: {split}")
    return image_paths


def _cache_split(
    adapter: FrozenIBERAdapter,
    dataset_root: Path,
    split: str,
    *,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, object]]:
    image_paths = _split_image_paths(dataset_root, split)
    records: list[dict[str, object]] = []
    for start in range(0, len(image_paths), batch_size):
        selected = image_paths[start : start + batch_size]
        samples = [
            _letterbox(
                path,
                dataset_root / "labels" / split / f"{path.stem}.txt",
            )
            for path in selected
        ]
        images = torch.stack([sample[0] for sample in samples]).to(
            device, non_blocking=True
        )
        boxes = [sample[1].to(device, non_blocking=True) for sample in samples]
        classes = [sample[2].to(device, non_blocking=True) for sample in samples]
        groups = [len(value) for value in boxes]
        target_boxes = (
            torch.cat(boxes)
            if sum(groups)
            else torch.empty((0, 4), dtype=torch.float32, device=device)
        )
        target_classes = (
            torch.cat(classes)
            if sum(groups)
            else torch.empty((0,), dtype=torch.long, device=device)
        )
        amp_enabled = device.type == "cuda"
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            output = adapter.forward_evidence(images)
        decoder = adapter.detector.model[-1].decoder
        hidden = decoder.last_hidden
        stock_boxes = decoder.last_stock_boxes
        stock_scores = decoder.last_stock_scores
        f3 = adapter._last_f3
        if hidden is None or stock_boxes is None or stock_scores is None or f3 is None:
            raise RuntimeError("IBER cache evidence is incomplete")
        matches = adapter.criterion.matcher(
            stock_boxes.detach(),
            stock_scores.detach(),
            target_boxes,
            target_classes,
            groups,
        )
        offset = 0
        for local_index, image_path in enumerate(selected):
            source, destination = matches[local_index]
            local_destination = destination.to(device=device, dtype=torch.long) - offset
            records.append(
                {
                    "index": len(records),
                    "image_id": image_path.relative_to(dataset_root).as_posix(),
                    "hidden": hidden[local_index].float().cpu(),
                    "stock_boxes": stock_boxes[local_index].float().cpu(),
                    "stock_scores": stock_scores[local_index].float().cpu(),
                    "f3": f3[local_index].half().cpu(),
                    "image_rgb": images[local_index].mul(255).round().clamp(0, 255).to(torch.uint8).cpu(),
                    "target_edges": _cxcywh_to_xyxy(boxes[local_index]).float().cpu(),
                    "match_source": source.long().cpu(),
                    "match_target": local_destination.long().cpu(),
                }
            )
            offset += groups[local_index]
    return records


def _validate_public_authority(
    baseline_checkpoint: Path, dataset_root: Path
) -> dict[str, str]:
    baseline_sha = file_sha256(baseline_checkpoint)
    dataset_sha = str(dataset_signature(dataset_root)["sha256"])
    category_sha = category_mapping_sha256(CATEGORY_NAMES)
    actual = {
        "baseline_sha256": baseline_sha,
        "dataset_sha256": dataset_sha,
        "category_sha256": category_sha,
    }
    expected = {
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "category_sha256": EXPECTED_CATEGORY_SHA256,
    }
    if actual != expected:
        raise ValueError(
            "IBER cache authority mismatch: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )
    return actual


def _device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{value}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.batch < 1 or args.workers < 0:
        raise ValueError("batch must be positive and workers nonnegative")
    if args.output_root.exists() and (
        not args.output_root.is_dir() or any(args.output_root.iterdir())
    ):
        raise FileExistsError(
            f"refusing to overwrite non-empty cache root: {args.output_root}"
        )
    authority = _validate_public_authority(
        args.baseline_checkpoint, args.dataset_root
    )

    from ultralytics import RTDETR

    device = _device(args.device)
    detector = RTDETR(str(args.baseline_checkpoint)).model.to(device).eval()
    with FrozenIBERAdapter.from_detector(
        detector,
        private_seed=PRIVATE_SEED,
        probe="b3",
        image_size=IMAGE_SIZE,
    ) as adapter:
        adapter.to(device).eval()
        train_records = _cache_split(
            adapter,
            args.dataset_root,
            "train",
            batch_size=args.batch,
            device=device,
        )
        val_records = _cache_split(
            adapter,
            args.dataset_root,
            "val",
            batch_size=args.batch,
            device=device,
        )
    write_evidence_cache(
        args.output_root,
        train_records=train_records,
        val_records=val_records,
        authority={
            **authority,
            "subset_sha256": EXPECTED_SUBSET_SHA256,
            "source_commit": _source_commit(),
            "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
        },
        shard_size=SHARD_SIZE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
