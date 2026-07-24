from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.ascv_loc_stage import ASCVStage, stage_policy


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one frozen ASCV-Loc training stage.")
    parser.add_argument("--stage", type=ASCVStage, choices=list(ASCVStage), required=True)
    parser.add_argument("--model", required=True, help="Frozen starting checkpoint or rtdetr-l.yaml for final scratch runs.")
    parser.add_argument("--data", type=Path, required=True, help="Protocol-generated train-only YAML.")
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--amp", type=parse_bool, default=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_forbidden(value: str | Path) -> None:
    normalized = str(value).replace("\\", "/").lower()
    if "test-dev" in normalized or "test_dev" in normalized:
        raise ValueError(f"test-dev is forbidden in ASCV-Loc training: {value}")


def validate_protocol_inputs(args: argparse.Namespace) -> dict:
    for value in (args.model, args.data, args.protocol_manifest, args.project):
        _reject_forbidden(value)
    manifest_path = args.protocol_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "ascv-loc-protocol/v1":
        raise ValueError("unexpected ASCV-Loc protocol schema")
    if manifest.get("ultralytics_version") != "8.4.90":
        raise ValueError("protocol is not frozen to Ultralytics 8.4.90")

    policy = stage_policy(args.stage)
    data_record = manifest["train_only_yaml"] if policy.uses_hashed_subset else manifest["full_train_only_yaml"]
    data_path = args.data.resolve()
    if data_path.as_posix() != Path(data_record["path"]).resolve().as_posix():
        raise ValueError("stage data path does not match the frozen protocol")
    if sha256_file(data_path) != data_record["sha256"]:
        raise ValueError("stage data checksum does not match the frozen protocol")

    model_path = Path(args.model)
    if args.stage in {ASCVStage.MECHANISM_500, ASCVStage.SCREEN_6, ASCVStage.FULL_20}:
        expected = manifest["checkpoint"]
        if model_path.resolve().as_posix() != Path(expected["path"]).resolve().as_posix():
            raise ValueError("development stage must start from the frozen mature checkpoint")
        if sha256_file(model_path.resolve()) != expected["sha256"]:
            raise ValueError("starting checkpoint checksum does not match the frozen protocol")
    elif str(args.model) != "rtdetr-l.yaml":
        raise ValueError("paper seed stages must start fresh from rtdetr-l.yaml")
    return manifest


def build_settings(args: argparse.Namespace) -> dict:
    policy = stage_policy(args.stage)
    return {
        "model": args.model,
        "data": str(args.data.resolve()),
        "epochs": policy.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "project": str(args.project.resolve()),
        "name": args.name,
        "exist_ok": False,
        "pretrained": False,
        "resume": False,
        "cache": False,
        "amp": args.amp,
        "compile": False,
        "deterministic": True,
        "seed": args.seed,
        "fraction": 1.0,
        "nbs": 64,
        "nms": False,
        "max_det": 300,
        "save": True,
        "save_period": 1,
        "optimizer": "AdamW",
        "lr0": 0.000714,
        "momentum": 0.9,
        "plots": False,
        "val": False,
    }
