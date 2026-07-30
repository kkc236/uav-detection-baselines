from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import RTDETRDetectionModel

from src.rtdetr_sqda_sgc import (
    BASELINE_SHA256,
    SQDASGCDetectionModel,
    load_mature_baseline,
    sha256_file,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prove exact G0 identity between stock RT-DETR-L and SQDA-SGC identity mode."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path)
    return parser


def _resolve_root(data_yaml: Path, config: dict) -> Path:
    configured = Path(str(config.get("path", data_yaml.parent))).expanduser()
    return configured.resolve() if configured.is_absolute() else (data_yaml.parent / configured).resolve()


def validation_images(data_yaml: Path, limit: int = 2) -> list[Path]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = _resolve_root(data_yaml, config)
    split = config.get("val")
    if not isinstance(split, (str, list)):
        raise ValueError("dataset YAML must define a validation split")
    split_entries = split if isinstance(split, list) else [split]
    images: list[Path] = []
    for entry in split_entries:
        path = Path(str(entry)).expanduser()
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if path.is_dir():
            images.extend(
                candidate
                for candidate in sorted(path.rglob("*"))
                if candidate.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                candidate = Path(line.strip())
                if not candidate.is_absolute():
                    candidate = (path.parent / candidate).resolve()
                images.append(candidate)
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
    images = sorted(dict.fromkeys(images))
    if len(images) < limit:
        raise RuntimeError(f"validation split contains only {len(images)} readable image paths")
    return images[:limit]


def load_fixed_batch(image_paths: list[Path], size: int = 640) -> torch.Tensor:
    tensors = []
    for path in image_paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            scale = min(size / rgb.width, size / rgb.height)
            resized = rgb.resize(
                (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
                Image.Resampling.BILINEAR,
            )
            canvas = Image.new("RGB", (size, size), (114, 114, 114))
            left = (size - resized.width) // 2
            top = (size - resized.height) // 2
            canvas.paste(resized, (left, top))
            array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
            tensors.append(torch.from_numpy(array))
    return torch.stack(tensors)


def _tensor_leaves(value) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        leaves: list[torch.Tensor] = []
        for item in value:
            leaves.extend(_tensor_leaves(item))
        return leaves
    if isinstance(value, dict):
        leaves = []
        for key in sorted(value):
            leaves.extend(_tensor_leaves(value[key]))
        return leaves
    return []


def _device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    index = int(value.split(",")[0])
    return torch.device(f"cuda:{index}")


def _class_count(data_yaml: Path) -> int:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = config.get("names")
    if isinstance(names, (list, tuple, dict)):
        return len(names)
    if isinstance(config.get("nc"), int):
        return int(config["nc"])
    raise ValueError("dataset YAML must define class names or nc")


def run_g0(
    checkpoint: Path,
    data_yaml: Path,
    device_name: str,
) -> dict:
    checkpoint = checkpoint.expanduser().resolve()
    data_yaml = data_yaml.expanduser().resolve()
    actual_sha = sha256_file(checkpoint)
    if actual_sha != BASELINE_SHA256:
        raise ValueError(
            f"baseline SHA256 mismatch: expected {BASELINE_SHA256}, got {actual_sha}"
        )
    image_paths = validation_images(data_yaml)
    inputs = load_fixed_batch(image_paths)
    device = _device(device_name)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    nc = _class_count(data_yaml)

    stock = RTDETRDetectionModel("rtdetr-l.yaml", nc=nc, verbose=False)
    load_mature_baseline(stock, checkpoint, expected_sha256=BASELINE_SHA256)
    stock = stock.to(device).eval()
    captured_stock: dict[str, torch.Tensor] = {}

    def capture_reference(_module, args, _kwargs):
        captured_stock["boxes"] = args[1][:, -300:, :].sigmoid().detach().cpu()

    handle = stock.model[-1].decoder.register_forward_pre_hook(
        capture_reference,
        with_kwargs=True,
    )
    with torch.inference_mode():
        stock_output = stock(inputs.to(device))
    handle.remove()
    stock_tensors = [tensor.detach().cpu() for tensor in _tensor_leaves(stock_output)]
    del stock, stock_output
    if device.type == "cuda":
        torch.cuda.empty_cache()

    enhanced = SQDASGCDetectionModel("rtdetr-l.yaml", nc=nc, verbose=False)
    load_mature_baseline(enhanced, checkpoint, expected_sha256=BASELINE_SHA256)
    enhanced.identity_override = True
    enhanced = enhanced.to(device).eval()
    with torch.inference_mode():
        enhanced_output = enhanced(inputs.to(device))
    enhanced_tensors = [tensor.detach().cpu() for tensor in _tensor_leaves(enhanced_output)]
    enhanced_references = enhanced.last_sqda_reference_boxes
    if enhanced_references is None:
        raise RuntimeError("SQDA-SGC did not expose its G0 reference boxes")
    enhanced_references = enhanced_references.cpu()

    if len(stock_tensors) != len(enhanced_tensors):
        raise AssertionError(
            f"G0 tensor structure differs: stock={len(stock_tensors)}, SQDA={len(enhanced_tensors)}"
        )
    differences = []
    for index, (stock_tensor, enhanced_tensor) in enumerate(
        zip(stock_tensors, enhanced_tensors)
    ):
        if stock_tensor.shape != enhanced_tensor.shape:
            raise AssertionError(
                f"G0 tensor {index} shape differs: {stock_tensor.shape} vs {enhanced_tensor.shape}"
            )
        differences.append((stock_tensor - enhanced_tensor).abs().max().item())
    reference_difference = (
        captured_stock["boxes"] - enhanced_references
    ).abs().max().item()
    maximum_difference = max(differences + [reference_difference])
    if maximum_difference != 0.0:
        raise AssertionError(f"G0 is not bitwise exact; max absolute difference={maximum_difference}")

    return {
        "gate": "G0",
        "passed": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "data_yaml": str(data_yaml),
        "images": [str(path) for path in image_paths],
        "tensor_leaves": len(stock_tensors),
        "query_count": int(enhanced_references.shape[1]),
        "max_abs_difference": maximum_difference,
        "reference_box_max_abs_difference": reference_difference,
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_g0(args.checkpoint, args.data, args.device)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
