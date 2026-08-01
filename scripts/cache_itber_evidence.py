"""Build the fixed no-augmentation I-TBER Gate 1 evidence cache."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.itber_cache import write_evidence_cache  # noqa: E402
from src.itber_geometry import cxcywh_to_xyxy  # noqa: E402
from src.itber_protocol import (  # noqa: E402
    EXPECTED_BASELINE_SHA256,
    EXPECTED_CATEGORY_SHA256,
    EXPECTED_DATASET_SHA256,
)
from src.lpr_protocol import (  # noqa: E402
    CATEGORY_NAMES,
    category_mapping_sha256,
    dataset_signature,
    file_sha256,
)
from src.rtdetr_itber import FrozenITBERAdapter  # noqa: E402


IMAGE_SIZE = 640
SHARD_SIZE = 16


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def _letterbox(image_path: Path, label_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    tensor = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255)

    boxes: list[list[float]] = []
    classes: list[int] = []
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"invalid VisDrone label row: {label_path}")
            cls, cx, cy, box_width, box_height = fields
            boxes.append(
                [
                    (float(cx) * width * scale + left) / IMAGE_SIZE,
                    (float(cy) * height * scale + top) / IMAGE_SIZE,
                    float(box_width) * width * scale / IMAGE_SIZE,
                    float(box_height) * height * scale / IMAGE_SIZE,
                ]
            )
            classes.append(int(cls))
    return (
        tensor,
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(classes, dtype=torch.long),
    )


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _cache_split(
    adapter: FrozenITBERAdapter,
    dataset_root: Path,
    split: str,
    *,
    batch_size: int,
    device: torch.device,
) -> list[dict]:
    image_paths = sorted((dataset_root / "images" / split).glob("*.jpg"))
    records: list[dict] = []
    for start in range(0, len(image_paths), batch_size):
        selected = image_paths[start : start + batch_size]
        samples = [
            _letterbox(path, dataset_root / "labels" / split / f"{path.stem}.txt")
            for path in selected
        ]
        images = torch.stack([sample[0] for sample in samples]).to(device, non_blocking=True)
        boxes = [sample[1].to(device) for sample in samples]
        classes = [sample[2].to(device) for sample in samples]
        groups = [len(value) for value in boxes]
        target_boxes = torch.cat(boxes) if sum(groups) else torch.empty(0, 4, device=device)
        target_classes = torch.cat(classes) if sum(groups) else torch.empty(0, dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            output = adapter.forward_evidence(images)
        decoder = adapter.detector.model[-1].decoder
        scores = decoder.last_stock_scores
        f3 = adapter._last_f3
        if scores is None or f3 is None or decoder.last_hidden is None or decoder.last_three_boxes is None:
            raise RuntimeError("I-TBER cache evidence is incomplete")
        matches = adapter.criterion.matcher(
            output.stock_boxes.detach(),
            scores.detach(),
            target_boxes,
            target_classes,
            groups,
        )
        offset = 0
        box_l2, box_l1, stock_boxes = decoder.last_three_boxes.unbind(0)
        for local_index, image_path in enumerate(selected):
            source, destination = matches[local_index]
            local_destination = destination.to(device=device, dtype=torch.long) - offset
            records.append(
                {
                    "index": len(records),
                    "image_id": image_path.relative_to(dataset_root).as_posix(),
                    "hidden": decoder.last_hidden[local_index].float().cpu(),
                    "box_l2": box_l2[local_index].float().cpu(),
                    "box_l1": box_l1[local_index].float().cpu(),
                    "stock_boxes": stock_boxes[local_index].float().cpu(),
                    "stock_scores": scores[local_index].float().cpu(),
                    "f3": f3[local_index].half().cpu(),
                    "target_edges": cxcywh_to_xyxy(boxes[local_index]).float().cpu(),
                    "match_source": source.long().cpu(),
                    "match_target": local_destination.long().cpu(),
                }
            )
            offset += groups[local_index]
    return records


def main() -> int:
    args = _parse_args()
    if args.batch < 1 or args.workers < 0:
        raise ValueError("batch must be positive and workers nonnegative")
    baseline_sha = file_sha256(args.baseline_checkpoint)
    dataset_sha = str(dataset_signature(args.dataset_root)["sha256"])
    category_sha = category_mapping_sha256(CATEGORY_NAMES)
    actual = (baseline_sha, dataset_sha, category_sha)
    expected = (EXPECTED_BASELINE_SHA256, EXPECTED_DATASET_SHA256, EXPECTED_CATEGORY_SHA256)
    if actual != expected:
        raise ValueError(f"I-TBER cache authority mismatch: expected={expected}, actual={actual}")

    from ultralytics import RTDETR

    device = torch.device(f"cuda:{args.device}")
    detector = RTDETR(str(args.baseline_checkpoint)).model.to(device).eval()
    adapter = FrozenITBERAdapter.from_detector(
        detector,
        private_seed=10_000,
        probe="p3",
        image_size=IMAGE_SIZE,
    ).to(device).eval()
    train = _cache_split(adapter, args.dataset_root, "train", batch_size=args.batch, device=device)
    val = _cache_split(adapter, args.dataset_root, "val", batch_size=args.batch, device=device)
    write_evidence_cache(
        args.output_root,
        train_records=train,
        val_records=val,
        authority={
            "baseline_sha256": baseline_sha,
            "dataset_sha256": dataset_sha,
            "category_sha256": category_sha,
            "source_commit": _source_commit(),
        },
        shard_size=SHARD_SIZE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
