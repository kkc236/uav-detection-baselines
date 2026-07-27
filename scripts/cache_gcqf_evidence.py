"""Cache frozen RT-DETR global/local decoder evidence for GCQF G0."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterator

import torch

from src.gcqf_cache import GCQFEvidenceRecord, write_evidence_cache
from src.gcte_data import build_gcqf_dataset
from src.gcte_evidence import (
    build_local_match_assignments,
    split_five_view_extraction,
)
from src.gcte_targets import (
    build_equivariance_pairs,
    build_quality_targets,
    build_tiny_anchor_mask,
)
from src.gcte_views import transform_xywh_homography
from src.rtdetr_gcqf import (
    extract_decoder_query_evidence,
    freeze_detector,
)
from src.sr_peg_targets import SRPEGTargets, build_sr_peg_targets


EXPECTED_BASELINE_SHA256 = (
    "54CE60289DD34C6750B8BA5F7516EEF"
    "CF3AFEF6C174C6E4F3B1EF810C883099B"
)
EXPECTED_ULTRALYTICS = "8.4.90"
VIEW_ORDER = ("global", "TL", "TR", "BL", "BR")


def sr_peg_targets_for_split(
    *,
    split: str,
    global_boxes: torch.Tensor,
    global_logits: torch.Tensor,
    local_boxes: torch.Tensor,
    local_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    source_shape: tuple[int, int],
) -> SRPEGTargets | None:
    """Keep val cache v1-compatible and supervise only train10 records."""

    if split == "val":
        return None
    if split != "train":
        raise ValueError("split must be train or val")
    return build_sr_peg_targets(
        global_boxes=global_boxes,
        global_logits=global_logits,
        local_boxes=local_boxes,
        local_logits=local_logits,
        gt_boxes=gt_boxes,
        gt_classes=gt_classes,
        source_shape=source_shape,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal five-view RT-DETR decoder evidence for GCQF."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument(
        "--image-path",
        help="Sealed image list/path override; required for the train10 cache.",
    )
    parser.add_argument("--dataset-signature", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--records-per-shard", type=int, default=8)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument(
        "--expected-baseline-sha256",
        default=EXPECTED_BASELINE_SHA256,
    )
    parser.add_argument("--queries-per-view", type=int, default=300)
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_image_id(image: str | Path, image_root: str | Path) -> str:
    candidate = Path(image).resolve()
    root = Path(image_root).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("image is outside the sealed dataset image root")
    return candidate.relative_to(root).as_posix()


def _load_detector(
    checkpoint: Path,
    *,
    expected_sha256: str,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    observed = _sha256_file(checkpoint)
    if observed != expected_sha256.upper():
        raise ValueError(
            f"baseline checksum mismatch: expected={expected_sha256.upper()} "
            f"actual={observed}"
        )
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("baseline checkpoint must contain a mapping")
    detector = payload.get("ema") or payload.get("model")
    if not isinstance(detector, torch.nn.Module):
        raise ValueError("baseline checkpoint has no model or EMA module")
    return freeze_detector(detector.float().to(device))


def _record_stream(
    *,
    detector: torch.nn.Module,
    dataset,
    device: torch.device,
    workers: int,
    image_root: Path,
    amp: bool,
    split: str,
) -> Iterator[GCQFEvidenceRecord]:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    criterion = getattr(detector, "criterion", None)
    if criterion is None:
        criterion = detector.init_criterion()
    matcher = criterion.matcher
    for index, batch in enumerate(loader, start=1):
        global_image = (
            batch["img"].to(device, non_blocking=True).float() / 255.0
        )
        local_views = (
            batch["local_views"][0]
            .to(device, non_blocking=True)
            .float()
            / 255.0
        )
        view_batch = torch.cat((global_image, local_views), dim=0)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp,
        ):
            extraction = extract_decoder_query_evidence(
                detector,
                view_batch,
                expected_query_count=300,
            )
        source_height, source_width = (
            int(value) for value in batch["source_shape"][0].tolist()
        )
        five = split_five_view_extraction(
            extraction,
            source_shape=(source_height, source_width),
            queries_per_view=300,
        )
        batch_indices = batch["batch_idx"].reshape(-1).to(torch.long)
        selection = batch_indices == 0
        gt_boxes = batch["bboxes"][selection].to(
            device=device,
            dtype=five.local_evidence.boxes.dtype,
        )
        gt_classes = (
            batch["cls"][selection]
            .reshape(-1)
            .to(device=device, dtype=torch.long)
        )
        geometry_on_device = five.geometry.homography.to(
            device=device,
            dtype=five.local_evidence.boxes.dtype,
        )
        canonical_boxes = transform_xywh_homography(
            five.local_evidence.boxes,
            geometry_on_device,
            clip=True,
        )
        quality_targets = build_quality_targets(
            canonical_boxes,
            five.local_evidence.logits,
            gt_boxes,
            gt_classes,
        )
        matched_gt = build_local_match_assignments(
            matcher=matcher,
            local_evidence=five.local_evidence,
            geometry=five.geometry,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
            queries_per_view=300,
        )
        pairs = build_equivariance_pairs(
            matched_gt,
            five.geometry.view_index[0].to(matched_gt.device),
        )
        image_file = Path(str(batch["im_file"][0])).resolve()
        record = GCQFEvidenceRecord(
            image_id=canonical_image_id(image_file, image_root),
            global_evidence=five.global_evidence,
            local_evidence=five.local_evidence,
            geometry=five.geometry,
            anchor_mask=build_tiny_anchor_mask(canonical_boxes),
            quality_targets=quality_targets,
            equivariance_pairs=pairs,
            fixed_anchor_payload={
                "view_order": VIEW_ORDER,
                "source_shape": [source_height, source_width],
                "postprocessed": five.postprocessed.detach().cpu(),
                "selected_query_indices": (
                    five.selected_query_indices.detach().cpu()
                ),
            },
            sr_peg_targets=sr_peg_targets_for_split(
                split=split,
                global_boxes=five.global_evidence.boxes,
                global_logits=five.global_evidence.logits,
                local_boxes=canonical_boxes,
                local_logits=five.local_evidence.logits,
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                source_shape=(source_height, source_width),
            ),
        )
        yield record
        if index % 25 == 0 or index == len(dataset):
            print(
                f"GCQF_CACHE_PROGRESS {index}/{len(dataset)}",
                flush=True,
            )


def cache(args: argparse.Namespace) -> Path:
    if (
        args.batch != 1
        or args.views != 5
        or args.queries_per_view != 300
    ):
        raise ValueError(
            "GCQF cache freezes batch=1 five views and 300 queries/view"
        )
    if args.workers < 0 or args.records_per_shard <= 0:
        raise ValueError("workers and records_per_shard are invalid")
    if args.split == "train" and not args.image_path:
        raise ValueError("train cache requires the sealed 10% image list")
    if not torch.cuda.is_available():
        raise RuntimeError("GCQF evidence caching requires CUDA")
    if str(args.device) != "0":
        raise ValueError("GCQF evidence caching freezes device=0")
    try:
        import ultralytics
        from ultralytics.data.utils import check_det_dataset
    except Exception as error:
        raise RuntimeError("Ultralytics 8.4.90 is required") from error
    if ultralytics.__version__ != EXPECTED_ULTRALYTICS:
        raise RuntimeError("Ultralytics version drift")
    data = check_det_dataset(str(args.data), autodownload=False)
    image_root = Path(data["path"]).resolve() / "images"
    dataset = build_gcqf_dataset(
        data,
        split=args.split,
        batch_size=1,
        image_path=args.image_path,
    )
    expected_records = args.expected_records
    if expected_records is None:
        expected_records = 647 if args.split == "train" else 548
    if len(dataset) != expected_records:
        raise RuntimeError(
            f"dataset record count drift: expected={expected_records} "
            f"actual={len(dataset)}"
        )
    device = torch.device("cuda:0")
    detector = _load_detector(
        args.checkpoint,
        expected_sha256=args.expected_baseline_sha256,
        device=device,
    )
    manifest_path = write_evidence_cache(
        output=args.output,
        records=_record_stream(
            detector=detector,
            dataset=dataset,
            device=device,
            workers=args.workers,
            image_root=image_root,
            amp=args.amp,
            split=args.split,
        ),
        baseline_sha256=args.expected_baseline_sha256,
        dataset_signature=args.dataset_signature,
        split="train10" if args.split == "train" else "val",
        records_per_shard=args.records_per_shard,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["record_count"] != expected_records:
        raise RuntimeError("written cache record count drift")
    print(
        f"GCQF_CACHE_COMPLETE records={expected_records} "
        f"manifest_sha256={_sha256_file(manifest_path)}",
        flush=True,
    )
    return manifest_path


def main() -> None:
    print(cache(build_parser().parse_args()))


if __name__ == "__main__":
    main()
