"""Extract frozen RT-DETR P2 evidence and run the predeclared boundary oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.cache_iber_evidence import _device, _letterbox, _validate_public_authority  # noqa: E402
from src.iber_p2_oracle import (  # noqa: E402
    ORACLE_EPOCHS,
    P2_NORMAL_OFFSETS_PX,
    P2_TANGENT_FRACTIONS,
    correction_direction_targets,
    decide_p2_viability,
    load_p2_oracle_cache,
    sample_p2_edge_profiles,
    train_p2_oracles,
    write_p2_oracle_cache,
)
from src.iber_protocol import (  # noqa: E402
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
    PRIVATE_SEED,
    RUNTIME_AMENDMENT_SHA256,
)
from src.itber_geometry import cxcywh_to_xyxy  # noqa: E402
from src.itber_metrics import area_bucket  # noqa: E402
from src.lpr_protocol import select_hashed_subset, subset_signature  # noqa: E402
from src.rtdetr_iber import FrozenIBERAdapter  # noqa: E402


P2_LAYER_INDEX = 1
IMAGE_SIZE = 640
BATCH_SIZE = 8
TRAIN_COUNT = 647
VAL_COUNT = 548


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser.parse_args(argv)


def _source_commit() -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip().lower()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("source commit must be exactly 40 hexadecimal characters")
    return value


def _schema_sha256() -> str:
    payload = {
        "identity": "iber-p2-boundary-oracle-v1",
        "layer": P2_LAYER_INDEX,
        "normal_offsets_px": P2_NORMAL_OFFSETS_PX,
        "tangent_fractions": P2_TANGENT_FRACTIONS,
        "epochs": ORACLE_EPOCHS,
        "seed": PRIVATE_SEED,
        "selection": "final_epoch_only",
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(serialized).hexdigest().upper()


def _split_image_paths(dataset_root: Path, split: str) -> list[Path]:
    paths = sorted((dataset_root / "images" / split).glob("*.jpg"))
    if split == "train":
        paths = select_hashed_subset(paths, root=dataset_root, fraction=0.10)
        actual_sha = subset_signature(paths, root=dataset_root)
        if len(paths) != TRAIN_COUNT or actual_sha != EXPECTED_SUBSET_SHA256:
            raise ValueError(f"P2 oracle subset authority mismatch: count={len(paths)}, sha256={actual_sha}")
    elif split == "val":
        if len(paths) != VAL_COUNT:
            raise ValueError(f"P2 oracle validation count mismatch: {len(paths)}")
    else:
        raise ValueError(f"unsupported split: {split}")
    return paths


def _geometry_features(boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    width = boxes[:, 2].clamp_min(1e-6)
    height = boxes[:, 3].clamp_min(1e-6)
    probability = scores.sigmoid()
    quality = probability.max(dim=-1).values
    normalized = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    entropy = -(normalized * normalized.clamp_min(1e-9).log()).sum(dim=-1).div(math.log(scores.shape[-1]))
    return torch.stack(
        (
            boxes[:, 0],
            boxes[:, 1],
            width.log(),
            height.log(),
            (width * height).log(),
            (width / height).log(),
            quality,
            entropy,
        ),
        dim=-1,
    )


def _extract_split(
    adapter: FrozenIBERAdapter,
    dataset_root: Path,
    split: str,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    paths = _split_image_paths(dataset_root, split)
    records: list[dict[str, Any]] = []
    captured: dict[str, torch.Tensor | None] = {"p2": None}

    def capture_p2(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("RT-DETR layer-1 P2 output is not a tensor")
        captured["p2"] = output.detach()

    hook = adapter.detector.model[P2_LAYER_INDEX].register_forward_hook(capture_p2)
    try:
        for start in range(0, len(paths), BATCH_SIZE):
            selected = paths[start : start + BATCH_SIZE]
            samples = [
                _letterbox(path, dataset_root / "labels" / split / f"{path.stem}.txt")
                for path in selected
            ]
            images = torch.stack([sample[0] for sample in samples]).to(device, non_blocking=True)
            target_boxes = [sample[1].to(device, non_blocking=True) for sample in samples]
            target_classes = [sample[2].to(device, non_blocking=True) for sample in samples]
            groups = [len(boxes) for boxes in target_boxes]
            joined_boxes = torch.cat(target_boxes) if sum(groups) else torch.empty((0, 4), device=device)
            joined_classes = torch.cat(target_classes) if sum(groups) else torch.empty((0,), dtype=torch.long, device=device)
            captured["p2"] = None
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                adapter.forward_evidence(images)
            decoder = adapter.detector.model[-1].decoder
            hidden = decoder.last_hidden
            stock_boxes = decoder.last_stock_boxes
            stock_scores = decoder.last_stock_scores
            p2 = captured["p2"]
            if hidden is None or stock_boxes is None or stock_scores is None or p2 is None:
                raise RuntimeError("P2 oracle evidence capture is incomplete")
            matches = adapter.criterion.matcher(
                stock_boxes.detach(),
                stock_scores.detach(),
                joined_boxes,
                joined_classes,
                groups,
            )
            offset = 0
            for image_index, path in enumerate(selected):
                source, destination = matches[image_index]
                local_destination = destination.to(device=device, dtype=torch.long) - offset
                source = source.to(device=device, dtype=torch.long)
                matched_stock = stock_boxes[image_index, source].float()
                matched_scores = stock_scores[image_index, source].float()
                matched_targets = target_boxes[image_index][local_destination]
                stock_edges = cxcywh_to_xyxy(matched_stock)
                target_edges = cxcywh_to_xyxy(matched_targets)
                labels, valid = correction_direction_targets(stock_edges, target_edges, image_size=IMAGE_SIZE)
                profiles = sample_p2_edge_profiles(
                    p2[image_index : image_index + 1],
                    matched_stock[None].to(dtype=p2.dtype),
                    image_size=IMAGE_SIZE,
                )[0]
                records.append(
                    {
                        "image_id": path.relative_to(dataset_root).as_posix(),
                        "profiles": profiles.half().cpu(),
                        "hidden": hidden[image_index, source].half().cpu(),
                        "geometry": _geometry_features(matched_stock, matched_scores).float().cpu(),
                        "labels": labels.float().cpu(),
                        "valid": valid.bool().cpu(),
                        "buckets": area_bucket(matched_targets.float(), image_size=IMAGE_SIZE).long().cpu(),
                    }
                )
                offset += groups[image_index]
    finally:
        hook.remove()
    if any(parameter.grad is not None for parameter in adapter.detector.parameters()):
        raise RuntimeError("frozen detector received a gradient during P2 extraction")
    return records


def _write_json_create_only(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.report_root.exists() and (
        not args.report_root.is_dir() or any(args.report_root.iterdir())
    ):
        raise FileExistsError(f"refusing to overwrite non-empty report root: {args.report_root}")
    public_authority = _validate_public_authority(args.baseline_checkpoint, args.dataset_root)
    authority = {
        "baseline_sha256": public_authority["baseline_sha256"],
        "dataset_sha256": public_authority["dataset_sha256"],
        "subset_sha256": EXPECTED_SUBSET_SHA256,
        "runtime_amendment_sha256": RUNTIME_AMENDMENT_SHA256,
        "source_commit": _source_commit(),
        "schema_sha256": _schema_sha256(),
    }
    device = _device(args.device)
    if not (args.cache_root / "manifest.json").is_file():
        from ultralytics import RTDETR

        detector = RTDETR(str(args.baseline_checkpoint)).model.to(device).eval()
        with FrozenIBERAdapter.from_detector(
            detector,
            private_seed=PRIVATE_SEED,
            probe="b3",
            image_size=IMAGE_SIZE,
        ) as adapter:
            adapter.to(device).eval()
            train_records = _extract_split(adapter, args.dataset_root, "train", device=device)
            val_records = _extract_split(adapter, args.dataset_root, "val", device=device)
        write_p2_oracle_cache(
            args.cache_root,
            train=train_records,
            val=val_records,
            authority=authority,
        )
        del train_records, val_records, detector
        if device.type == "cuda":
            torch.cuda.empty_cache()
    cache = load_p2_oracle_cache(args.cache_root, authority=authority)
    report = train_p2_oracles(cache, device=device)
    decision = decide_p2_viability(report)
    args.report_root.mkdir(parents=True, exist_ok=False)
    _write_json_create_only(
        args.report_root / "p2-oracle-report.json",
        {"authority": authority, **report},
    )
    _write_json_create_only(
        args.report_root / "p2-oracle-decision.json",
        {"authority": authority, **decision},
    )
    return 0 if decision["status"] in {"passed", "scientific_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
