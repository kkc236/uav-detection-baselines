from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ebc_qp_config import EBCQPConfig
from src.rtdetr_ebc_qp import EBCQPTrainer, PairedControlTrainer


@dataclass(frozen=True)
class Stage:
    epochs: int
    fraction: float
    scratch: bool
    stock_frozen: bool
    lambda_ebc: float
    inject_p2: bool


STAGES = {
    "d1": Stage(epochs=3, fraction=0.10, scratch=False, stock_frozen=True, lambda_ebc=0.0, inject_p2=False),
    "d2": Stage(epochs=10, fraction=0.10, scratch=True, stock_frozen=False, lambda_ebc=0.05, inject_p2=True),
    "d3": Stage(epochs=10, fraction=0.10, scratch=True, stock_frozen=False, lambda_ebc=0.0, inject_p2=True),
    "a1": Stage(epochs=100, fraction=1.00, scratch=True, stock_frozen=False, lambda_ebc=0.0, inject_p2=True),
    "a2": Stage(epochs=100, fraction=1.00, scratch=True, stock_frozen=False, lambda_ebc=0.05, inject_p2=True),
    "qg-p2": Stage(epochs=10, fraction=0.10, scratch=True, stock_frozen=False, lambda_ebc=0.0, inject_p2=True),
}

DIAGNOSTIC_FIELDS = frozenset(
    {
        "epoch",
        "active_images",
        "tiny_gt",
        "ap_tiny",
        "tiny_recall",
        "stock_top300_coverage",
        "local_assign_rate",
        "p2_entry_count",
        "n_gain",
        "n_loss",
        "v_replace",
        "effective_p2_entry_rate",
        "boundary_gap_mean",
        "boundary_gap_positive_ratio",
        "p2_foreground_at_50",
        "p2_unique_gt_at_50",
        "p2_duplicate_rate_at_50",
        "p2_background_rate_at_50",
        "score_iou_spearman",
        "score_nwd_spearman",
        "score_quality_sample_count",
        "assigned_entry_mean_iou",
        "assigned_entry_mean_nwd",
        "unassigned_entry_rate",
        "low_quality_entry_rate",
        "c2_p3_rms_ratio",
        "p2_loss",
        "ebc_loss",
        "precision",
        "recall",
        "map50",
        "map50_95",
    }
)


class CompactDiagnosticsWriter:
    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, record: dict) -> None:
        unsupported = set(record) - DIAGNOSTIC_FIELDS
        if unsupported:
            raise ValueError(f"unsupported diagnostic fields: {sorted(unsupported)}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen EBC-QP D1-D3 and formal screening protocol.")
    parser.add_argument("--stage", required=True, choices=("d1", "d2", "d3", "formal"))
    parser.add_argument("--arm", choices=("control", "a1", "a2", "qg-p2"), default="a2")
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--create-initial-state", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--d2-manifest", type=Path)
    parser.add_argument("--a2-manifest", type=Path)
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--signature-file", type=Path)
    parser.add_argument("--protocol-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--device", default="0")
    parser.add_argument("--data", default="VisDrone.yaml")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "ebc-qp")
    parser.add_argument("--name")
    parser.add_argument("--quality-weighted-ebc", action="store_true")
    parser.add_argument("--learnable-fusion-gamma", action="store_true")
    parser.add_argument("--disable-query-injection", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def build_settings(args: argparse.Namespace) -> dict:
    stage_key = args.arm if args.stage == "formal" else args.stage
    stage = STAGES[stage_key]
    control = args.stage == "d2" and args.arm == "control"
    model = "rtdetr-l.yaml" if control else str(ROOT / "configs" / "rtdetr-l-ebc-qp.yaml")
    default_name = f"{args.stage}-{args.arm}-seed{args.seed}"
    settings = {
        "model": model,
        "data": args.data,
        "epochs": 1 if args.smoke else stage.epochs,
        "fraction": 0.01 if args.smoke else (1.0 if args.stage in {"d2", "d3"} else stage.fraction),
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "device": args.device,
        "project": str(args.project.resolve()),
        "name": args.name or (f"{default_name}-smoke" if args.smoke else default_name),
        "exist_ok": False,
        "resume": False,
        "pretrained": False,
        "cache": False,
        "amp": True,
        "deterministic": True,
        "seed": args.seed,
        "nbs": 64,
        "nms": False,
        "max_det": 300,
        "save": True,
        "save_period": 1,
        "optimizer": "auto",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "cos_lr": False,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "scale": 0.5,
        "translate": 0.1,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "plots": True,
        "val": True,
    }
    return settings


def build_ebc_config(args: argparse.Namespace) -> EBCQPConfig:
    stage_key = (
        args.arm
        if args.stage == "formal" or (args.stage == "d2" and args.arm in {"a1", "qg-p2"})
        else args.stage
    )
    stage = STAGES[stage_key]
    return EBCQPConfig(
        lambda_ebc=stage.lambda_ebc,
        quality_weighted_ebc=args.quality_weighted_ebc,
        learnable_fusion_gamma=args.learnable_fusion_gamma or args.arm == "qg-p2",
        query_injection_enabled=not args.disable_query_injection,
        quality_gated_p2=args.arm == "qg-p2",
    )


def validate_protocol(args: argparse.Namespace) -> None:
    if args.stage == "d2" and args.arm not in {"control", "a1", "a2", "qg-p2"}:
        raise SystemExit("D2 arm must be control, a1, a2, or qg-p2")
    if args.stage == "formal" and args.arm not in {"a1", "a2"}:
        raise SystemExit("formal arm must be a1 or a2")
    if args.arm == "qg-p2" and args.stage != "d2":
        raise SystemExit("qg-p2 is only valid for D2")
    if args.quality_weighted_ebc and not (args.arm == "a2" and args.stage in {"d2", "formal"}):
        raise SystemExit("quality-weighted EBC is only valid for the A2 arm")
    if args.learnable_fusion_gamma and not (args.arm in {"a1", "a2"} and args.stage in {"d2", "formal"}):
        raise SystemExit("learnable fusion gamma is only valid for the A1/A2 arms")
    if args.disable_query_injection and not (args.stage == "d2" and args.arm == "a1"):
        raise SystemExit("disabled query injection is only valid for the D2 A1 arm")
    if args.quality_weighted_ebc and args.learnable_fusion_gamma:
        raise SystemExit("quality-weighted EBC and fusion gamma are mutually exclusive")

    if args.stage == "d3":
        manifest = _read_json(args.d2_manifest, "passing D2 manifest")
        if not manifest.get("gate", {}).get("passed", False):
            raise SystemExit("D3 requires a passing D2 manifest")

    if args.stage == "formal" and args.arm == "a1":
        manifest = _read_json(args.a2_manifest, "completed A2 seed-0 manifest")
        record = manifest.get("formal_a2_seed0", {})
        if not record.get("complete"):
            raise SystemExit("A1 requires a completed A2 seed-0 manifest")
        if args.initial_state is None or not args.initial_state.exists():
            raise SystemExit("A1 requires the exact A2 seed-0 initial state")
        expected_path = str(args.initial_state.resolve())
        expected_hash = _file_sha256(args.initial_state)
        if record.get("initial_state") != expected_path or record.get("initial_state_sha256") != expected_hash:
            raise SystemExit("A1 requires the exact A2 seed-0 initial state")

    if args.seed > 0:
        frozen = _read_json(args.frozen_manifest, "frozen experiment signature")
        current = _read_json(args.signature_file, "current experiment signature")
        if frozen.get("signature") != current:
            raise SystemExit("seed 1/2 require the frozen experiment signature")


def main() -> None:
    args = build_parser().parse_args()
    validate_protocol(args)
    if args.create_initial_state is not None:
        raise SystemExit("initial-state creation is gated until the dataset/subset signature artifact is available")
    if args.stage == "d1":
        if args.weights is None or not args.weights.is_file():
            raise SystemExit("D1 requires the verified matched baseline best.pt artifact")
    if args.stage in {"d2", "d3", "formal"} and (args.initial_state is None or not args.initial_state.is_file()):
        raise SystemExit(f"{args.stage} requires a frozen initial-state artifact")
    if args.stage in {"d2", "d3"}:
        _validate_pair_artifacts(args)

    settings = build_settings(args)
    if args.stage == "d2" and args.arm == "control":
        trainer = PairedControlTrainer(overrides=settings, initial_state_path=args.initial_state)
    else:
        config = build_ebc_config(args)
        trainer = EBCQPTrainer(
            overrides=settings,
            ebc_config=config,
            initial_state_path=args.initial_state,
        )
    trainer.train()


def _validate_pair_artifacts(args: argparse.Namespace) -> None:
    manifest = _read_json(args.protocol_manifest, "paired protocol manifest")
    expected_data = str(Path(args.data).resolve())
    expected_state = str(args.initial_state.resolve())
    if manifest.get("data", {}).get("path") != expected_data:
        raise SystemExit("paired protocol manifest does not match the D2 data file")
    if manifest.get("initial_state", {}).get("path") != expected_state:
        raise SystemExit("paired protocol manifest does not match the initial state")
    if manifest.get("initial_state", {}).get("sha256") != _file_sha256(args.initial_state):
        raise SystemExit("paired protocol initial-state hash mismatch")
    if manifest.get("seed") != args.seed:
        raise SystemExit("paired protocol seed mismatch")


def _read_json(path: Path | None, label: str) -> dict:
    if path is None or not path.is_file():
        raise SystemExit(f"missing {label}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid {label}: {error}") from error


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


if __name__ == "__main__":
    main()
