from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import subprocess
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import torch
import ultralytics
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ebc_qp_config import EBCQPConfig, SOURCE_SHA256
from src.ebc_qp_protocol import dataset_signature, state_fingerprint, subset_signature
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
    "e1": Stage(epochs=10, fraction=1.00, scratch=True, stock_frozen=False, lambda_ebc=0.0, inject_p2=False),
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
    parser.add_argument("--stage", required=True, choices=("d1", "d2", "d3", "e1", "formal"))
    parser.add_argument("--arm", choices=("control", "a1", "a2", "qg-p2", "tsgr-p2"), default="a2")
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
    parser.add_argument("--controlled-amp-scale", type=float, choices=(256.0,))
    parser.add_argument("--smoke", action="store_true")
    return parser


def build_settings(args: argparse.Namespace) -> dict:
    stage_key = args.arm if args.stage == "formal" else args.stage
    stage = STAGES[stage_key]
    control = args.arm == "control" and args.stage in {"d2", "e1"}
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
    if args.stage == "e1" and args.arm == "tsgr-p2":
        return EBCQPConfig(
            lambda_p2=0.1,
            lambda_ebc=0.0,
            lambda_quality=0.0,
            query_injection_enabled=False,
            p2_c2_grad_scale=0.1,
            contribution_separated_aux_gradients=True,
        )
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
    if args.stage == "e1" and args.arm not in {"control", "tsgr-p2"}:
        raise SystemExit("E1 arm must be control or tsgr-p2")
    if args.stage == "formal" and args.arm not in {"a1", "a2"}:
        raise SystemExit("formal arm must be a1 or a2")
    if args.arm == "qg-p2" and args.stage != "d2":
        raise SystemExit("qg-p2 is only valid for D2")
    if args.arm == "tsgr-p2" and args.stage != "e1":
        raise SystemExit("tsgr-p2 is only valid for E1")
    if args.quality_weighted_ebc and not (args.arm == "a2" and args.stage in {"d2", "formal"}):
        raise SystemExit("quality-weighted EBC is only valid for the A2 arm")
    if args.learnable_fusion_gamma and not (args.arm in {"a1", "a2"} and args.stage in {"d2", "formal"}):
        raise SystemExit("learnable fusion gamma is only valid for the A1/A2 arms")
    if args.disable_query_injection and not (args.stage == "d2" and args.arm == "a1"):
        raise SystemExit("disabled query injection is only valid for the D2 A1 arm")
    if args.quality_weighted_ebc and args.learnable_fusion_gamma:
        raise SystemExit("quality-weighted EBC and fusion gamma are mutually exclusive")
    if args.stage == "e1":
        if args.controlled_amp_scale != 256.0:
            raise SystemExit("E1 requires controlled AMP scale 256")
        if args.smoke:
            raise SystemExit("E1 does not permit smoke settings")
        if args.device != "0":
            raise SystemExit("E1 requires the frozen single GPU device 0")
    elif args.controlled_amp_scale is not None:
        raise SystemExit("controlled AMP scale is only valid for E1")

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

    if args.seed > 0 and args.stage != "e1":
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
    if args.stage in {"d2", "d3", "e1", "formal"} and (args.initial_state is None or not args.initial_state.is_file()):
        raise SystemExit(f"{args.stage} requires a frozen initial-state artifact")
    if args.stage in {"d2", "d3", "e1"}:
        _validate_pair_artifacts(args)

    settings = build_settings(args)
    if args.stage == "e1":
        _assert_e1_launch_environment(settings)
    if args.arm == "control" and args.stage in {"d2", "e1"}:
        trainer = PairedControlTrainer(
            overrides=settings,
            initial_state_path=args.initial_state,
            controlled_amp_scale=args.controlled_amp_scale,
        )
    else:
        config = build_ebc_config(args)
        trainer = EBCQPTrainer(
            overrides=settings,
            ebc_config=config,
            initial_state_path=args.initial_state,
            controlled_amp_scale=args.controlled_amp_scale,
        )
    trainer.train()
    if args.stage == "e1":
        write_e1_run_manifest(args, settings, trainer)


def _validate_pair_artifacts(args: argparse.Namespace) -> None:
    manifest = _read_json(args.protocol_manifest, "paired protocol manifest")
    expected_data = str(Path(args.data).resolve())
    expected_state = str(args.initial_state.resolve())
    if manifest.get("data", {}).get("path") != expected_data:
        raise SystemExit("paired protocol manifest does not match the D2 data file")
    if manifest.get("data", {}).get("sha256") != _file_sha256(Path(args.data)):
        raise SystemExit("paired protocol data YAML hash mismatch")
    if manifest.get("initial_state", {}).get("path") != expected_state:
        raise SystemExit("paired protocol manifest does not match the initial state")
    if manifest.get("initial_state", {}).get("sha256") != _file_sha256(args.initial_state):
        raise SystemExit("paired protocol initial-state hash mismatch")
    if manifest.get("seed") != args.seed:
        raise SystemExit("paired protocol seed mismatch")
    if args.stage != "e1":
        return

    signed = dict(manifest)
    signature = signed.pop("signature", None)
    if signature != _json_sha256(signed):
        raise SystemExit("E1 protocol manifest signature mismatch")
    if manifest.get("source_sha256") != SOURCE_SHA256:
        raise SystemExit("E1 protocol source lock mismatch")
    if manifest.get("git_commit") != _git_commit():
        raise SystemExit("E1 protocol git commit mismatch")
    if manifest.get("environment") != _current_environment():
        raise SystemExit("E1 protocol environment mismatch")

    data = YAML.load(args.data)
    subset_path = Path(data["train"]).resolve()
    if str(subset_path) != manifest.get("subset", {}).get("path"):
        raise SystemExit("E1 subset path mismatch")
    if not subset_path.is_file():
        raise SystemExit("E1 subset file is missing")
    subset_lines = [Path(line) for line in subset_path.read_text(encoding="utf-8").splitlines() if line]
    subset = manifest.get("subset", {})
    if len(subset_lines) != subset.get("count"):
        raise SystemExit("E1 subset count mismatch")
    dataset_root = Path(data["path"]).resolve()
    if subset_signature(subset_lines, root=dataset_root) != subset.get("sha256"):
        raise SystemExit("E1 subset semantic hash mismatch")
    if dataset_signature(dataset_root) != manifest.get("dataset"):
        raise SystemExit("E1 dataset signature mismatch")
    if _json_sha256(data["names"]) != manifest.get("category_mapping_sha256"):
        raise SystemExit("E1 category mapping hash mismatch")

    frozen = _read_json(args.frozen_manifest, "frozen E1 experiment signature")
    frozen_payload = frozen.get("payload")
    frozen_signature = frozen.get("experiment_signature")
    if not isinstance(frozen_payload, dict) or frozen_signature != _json_sha256(frozen_payload):
        raise SystemExit("invalid frozen E1 experiment signature")
    if manifest.get("experiment_signature") != frozen_signature:
        raise SystemExit("E1 experiment signature mismatch")

    artifact = torch.load(args.initial_state, map_location="cpu", weights_only=False)
    if artifact.get("metadata", {}).get("experiment_signature") != frozen_signature:
        raise SystemExit("E1 initial-state experiment signature mismatch")
    fingerprints = artifact.get("fingerprints", {})
    if state_fingerprint(artifact.get("common_state", {})) != fingerprints.get("common"):
        raise SystemExit("E1 common initial-state fingerprint mismatch")
    if state_fingerprint(artifact.get("innovation_state", {})) != fingerprints.get("innovation"):
        raise SystemExit("E1 innovation initial-state fingerprint mismatch")


def _assert_e1_launch_environment(settings: dict) -> None:
    run_dir = Path(settings["project"]) / settings["name"]
    if run_dir.exists():
        raise SystemExit(f"E1 target run directory already exists: {run_dir}")
    _assert_tracked_worktree_clean()


def _assert_tracked_worktree_clean() -> None:
    for arguments in (("diff", "--quiet", "--"), ("diff", "--cached", "--quiet", "--")):
        result = subprocess.run(["git", *arguments], cwd=ROOT, check=False)
        if result.returncode != 0:
            raise SystemExit("E1 requires a tracked-clean Git worktree")


def write_e1_run_manifest(args: argparse.Namespace, settings: dict, trainer) -> dict:
    run_dir = Path(trainer.save_dir).resolve()
    expected_dir = (Path(settings["project"]) / settings["name"]).resolve()
    if run_dir != expected_dir:
        raise RuntimeError(f"E1 run directory drifted: expected {expected_dir}, got {run_dir}")

    evidence_path = run_dir / "optimizer-evidence.jsonl"
    records = []
    try:
        for line in evidence_path.read_text(encoding="utf-8").splitlines():
            records.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid E1 optimizer evidence: {error}") from error
    if not records or [record.get("optimizer_attempt") for record in records] != list(range(1, len(records) + 1)):
        raise RuntimeError("E1 optimizer evidence attempts are missing or non-contiguous")
    if any(record.get("amp_step_skipped") for record in records):
        raise RuntimeError("E1 optimizer evidence contains an AMP skip")
    if any(record.get("nonfinite_fields") for record in records):
        raise RuntimeError("E1 optimizer evidence contains non-finite fields")
    if any(record.get("runtime_violation") for record in records):
        raise RuntimeError("E1 optimizer evidence contains a runtime protocol violation")
    if any(float(record.get("amp_scale_before", -1)) != 256.0 for record in records):
        raise RuntimeError("E1 optimizer evidence contains a changed AMP scale")
    if args.arm == "tsgr-p2" and any(
        record.get("p2_entry_count") != 0 or record.get("ordinary_query_count") != 300 for record in records
    ):
        raise RuntimeError("E1 TSGR query isolation invariant failed")

    results_path = run_dir / "results.csv"
    try:
        with results_path.open(newline="", encoding="utf-8") as stream:
            results = list(csv.DictReader(stream))
    except OSError as error:
        raise RuntimeError(f"missing E1 results: {error}") from error
    if len(results) != 10:
        raise RuntimeError(f"E1 requires exactly 10 result rows, got {len(results)}")
    metric_keys = (
        "metrics/mAP50-95(B)",
        "metrics/mAP50(B)",
        "metrics/AP-tiny",
        "metrics/Recall-tiny",
        "metrics/AP-r<8",
        "metrics/AP-8<=r<=16",
    )
    if any(not math.isfinite(float(row[key])) for row in results for key in metric_keys):
        raise RuntimeError("E1 results contain a non-finite required metric")

    weights_dir = run_dir / "weights"
    required_checkpoints = [weights_dir / "last.pt", weights_dir / "best.pt"]
    if not all(path.is_file() for path in required_checkpoints):
        raise RuntimeError("E1 last/best checkpoints are incomplete")
    epoch_checkpoints = sorted(weights_dir.glob("epoch*.pt"))
    if len(epoch_checkpoints) < 10:
        raise RuntimeError(f"E1 requires 10 epoch checkpoints, found {len(epoch_checkpoints)}")

    route_ratios = [float(record["shallow_applied_ratio"]) for record in records]
    protocol = _read_json(args.protocol_manifest, "paired protocol manifest")
    _assert_tracked_worktree_clean()
    end_git_commit = _git_commit()
    if end_git_commit != protocol.get("git_commit"):
        raise RuntimeError("E1 Git HEAD changed after protocol validation")
    source_paths = {
        "config": ROOT / "src" / "ebc_qp_config.py",
        "decoder": ROOT / "src" / "ebc_qp_decoder.py",
        "trainer": ROOT / "src" / "rtdetr_ebc_qp.py",
        "launcher": ROOT / "scripts" / "train_rtdetr_ebc_qp.py",
    }
    artifacts = {
        "args": _artifact_record(run_dir / "args.yaml"),
        "results": _artifact_record(results_path),
        "optimizer_evidence": _artifact_record(evidence_path),
        "last_checkpoint": _artifact_record(required_checkpoints[0]),
        "best_checkpoint": _artifact_record(required_checkpoints[1]),
        "protocol_manifest": _artifact_record(args.protocol_manifest),
        "initial_state": _artifact_record(args.initial_state),
        **{f"source_{name}": _artifact_record(path) for name, path in source_paths.items()},
    }
    manifest = {
        "format_version": 1,
        "stage": "e1",
        "arm": args.arm,
        "seed": args.seed,
        "command": list(sys.argv),
        "git_commit_start": protocol["git_commit"],
        "git_commit_end": end_git_commit,
        "tracked_worktree_clean_at_start": True,
        "tracked_worktree_clean_at_end": True,
        "environment": _current_environment(),
        "settings": settings,
        "controlled_amp": {
            "init_scale": args.controlled_amp_scale,
            "growth_interval": 2**31 - 1,
            "optimizer_attempts": len(records),
            "skipped_attempts": 0,
        },
        "ebc_config": trainer.ebc_config.as_dict() if hasattr(trainer, "ebc_config") else None,
        "protocol": {
            "signature": protocol["signature"],
            "experiment_signature": protocol["experiment_signature"],
            "dataset": protocol["dataset"],
            "subset": protocol["subset"],
            "data_sha256": protocol["data"]["sha256"],
            "initial_state_sha256": protocol["initial_state"]["sha256"],
        },
        "optimizer_evidence": {
            "attempts": len(records),
            "p2_entry_count_max": max(int(record.get("p2_entry_count") or 0) for record in records),
            "ordinary_query_count_values": sorted(
                {record.get("ordinary_query_count") for record in records if record.get("ordinary_query_count") is not None}
            ),
            "shallow_applied_ratio_min": min(route_ratios),
            "shallow_applied_ratio_median": statistics.median(route_ratios),
            "shallow_applied_ratio_max": max(route_ratios),
        },
        "results": {
            "epochs": len(results),
            "final_map50_95": float(results[-1]["metrics/mAP50-95(B)"]),
            "tail3_map50_95": statistics.mean(
                float(row["metrics/mAP50-95(B)"]) for row in results[-3:]
            ),
            "final_map50": float(results[-1]["metrics/mAP50(B)"]),
            "final_ap_tiny": float(results[-1]["metrics/AP-tiny"]),
            "final_recall_tiny": float(results[-1]["metrics/Recall-tiny"]),
            "final_ap_r_lt_8": float(results[-1]["metrics/AP-r<8"]),
            "final_ap_r_8_to_16": float(results[-1]["metrics/AP-8<=r<=16"]),
        },
        "artifacts": artifacts,
        "epoch_checkpoints": [_artifact_record(path) for path in epoch_checkpoints],
    }
    manifest["signature"] = _json_sha256(manifest)
    destination = run_dir / "e1-run-manifest.json"
    if destination.exists():
        raise FileExistsError(f"refusing to replace E1 run manifest: {destination}")
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return manifest


def _read_json(path: Path | None, label: str) -> dict:
    if path is None or not path.is_file():
        raise SystemExit(f"missing {label}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid {label}: {error}") from error


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def _artifact_record(path: Path) -> dict[str, str | int]:
    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeError(f"required E1 artifact is missing: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _json_sha256(payload: object) -> str:
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256(content).hexdigest().upper()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _current_environment() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


if __name__ == "__main__":
    main()
