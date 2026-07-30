from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr.train import RTDETRDataset
from ultralytics.nn.tasks import RTDETRDetectionModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rtdetr_sqda_sgc import BASELINE_SHA256, load_mature_baseline, sha256_file
from src.sqda_diagnostics import ProposalDiagnosticAccumulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure stock RT-DETR Top-300 proposal recall and missed-object recoverability."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(f"cuda:{int(value.split(',')[0])}")


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def run_diagnostic(args: argparse.Namespace) -> dict:
    checkpoint = args.checkpoint.expanduser().resolve()
    data_yaml = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if sha256_file(checkpoint) != BASELINE_SHA256:
        raise ValueError("mature baseline SHA256 mismatch")
    data = check_det_dataset(str(data_yaml), autodownload=False)
    config = get_cfg(
        overrides={
            "task": "detect",
            "mode": "val",
            "data": str(data_yaml),
            "imgsz": 640,
            "batch": 8,
            "workers": args.workers,
            "rect": False,
            "seed": 0,
            "deterministic": True,
            "nms": False,
            "max_det": 300,
        }
    )
    dataset = RTDETRDataset(
        img_path=data["val"],
        imgsz=640,
        batch_size=8,
        augment=False,
        hyp=config,
        rect=False,
        cache=None,
        single_cls=False,
        prefix="Top300 val: ",
        classes=None,
        data=data,
        fraction=1.0,
    )
    loader = build_dataloader(
        dataset,
        batch=8,
        workers=args.workers,
        shuffle=False,
        rank=-1,
        drop_last=False,
    )
    device = _device(args.device)
    model = RTDETRDetectionModel(
        "rtdetr-l.yaml",
        nc=int(data["nc"]),
        verbose=False,
    )
    load_mature_baseline(model, checkpoint, expected_sha256=BASELINE_SHA256)
    model = model.to(device).eval()
    captured: dict[str, torch.Tensor] = {}

    def capture_reference(_module, hook_args, _hook_kwargs):
        captured["boxes"] = hook_args[1][:, -300:, :].sigmoid().detach()

    handle = model.model[-1].decoder.register_forward_pre_hook(
        capture_reference,
        with_kwargs=True,
    )
    accumulator = ProposalDiagnosticAccumulator()
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                images = batch["img"].to(device, non_blocking=True).float() / 255.0
                predictions = model(images)
                decoded = predictions[0] if isinstance(predictions, tuple) else predictions
                proposal_boxes = captured.pop("boxes")
                batch_indices = batch["batch_idx"].reshape(-1).long()
                gt_boxes = batch["bboxes"]
                gt_classes = batch["cls"].reshape(-1).long()
                for image_index in range(images.shape[0]):
                    mask = batch_indices == image_index
                    accumulator.update(
                        proposal_boxes[image_index].float().cpu(),
                        gt_boxes[mask].float().cpu(),
                        gt_classes[mask].cpu(),
                        decoded[image_index].float().cpu(),
                    )
                if (batch_index + 1) % 10 == 0:
                    print(
                        f"Top300 diagnostic processed {accumulator.images}/{len(dataset)} images",
                        flush=True,
                    )
    finally:
        handle.remove()

    report = accumulator.report()
    report["identity"] = {
        "git_sha": _git_sha(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": BASELINE_SHA256,
        "data_yaml": str(data_yaml),
        "data_yaml_sha256": sha256_file(data_yaml),
        "seed": 0,
        "imgsz": 640,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run_diagnostic(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
