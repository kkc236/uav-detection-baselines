"""Launch the two high-value AP-FDR formal ablations under one frozen protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sync_experiment_checkpoint import write_json_atomic  # noqa: E402
from scripts.train_rtdetr_fdr import (  # noqa: E402
    FORMAL_EPOCHS,
    FROZEN_SETTINGS,
    current_source_identity,
    prepare_data_yaml,
)
from src.fdr_protocol import validate_fdr_initial_state  # noqa: E402
from src.lpr_protocol import dataset_signature  # noqa: E402


VARIANT_CONFIGS = {
    "no_preliminary_reference": (
        ROOT / "configs" / "rtdetr-l-fdr-no-prebox.yaml"
    ),
    "no_dn_fdr": ROOT / "configs" / "rtdetr-l-fdr-no-dn.yaml",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one formal seed0 AP-FDR internal ablation."
    )
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_CONFIGS),
        required=True,
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_settings(
    variant: str,
    *,
    data_yaml: Path,
    output_root: Path,
    name: str | None = None,
) -> dict[str, Any]:
    if variant not in VARIANT_CONFIGS:
        raise ValueError(f"unknown AP-FDR ablation variant: {variant}")
    run_name = name or f"formal-seed0-ap-fdr-{variant.replace('_', '-')}"
    return {
        **FROZEN_SETTINGS,
        "model": str(VARIANT_CONFIGS[variant].resolve()),
        "data": str(Path(data_yaml).resolve()),
        "epochs": FORMAL_EPOCHS,
        "seed": 0,
        "project": str(Path(output_root).resolve()),
        "name": run_name,
        "exist_ok": False,
    }


def build_launch_record(
    *,
    variant: str,
    source_identity: Mapping[str, Any],
    config_path: Path,
    initial_state_path: Path,
    dataset: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    initial_state_path = Path(initial_state_path).resolve()
    return {
        "format_version": 1,
        "variant": variant,
        "source": dict(source_identity),
        "config": {
            "path": str(config_path),
            "sha256": _file_sha256(config_path),
        },
        "initial_state": {
            "path": str(initial_state_path),
            "sha256": _file_sha256(initial_state_path),
        },
        "dataset": dict(dataset),
        "settings": dict(settings),
    }


def require_clean_tracked_worktree(root: Path = ROOT) -> None:
    result = subprocess.run(
        ["git", "-C", str(Path(root).resolve()), "diff", "--quiet", "HEAD", "--"],
        check=False,
    )
    if result.returncode == 1:
        raise RuntimeError("tracked source differs from HEAD; commit before training")
    if result.returncode != 0:
        raise RuntimeError(f"git worktree check failed with exit code {result.returncode}")


def validate_initial_state_file(path: Path) -> Path:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"FDR initial state not found: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, Mapping):
        raise TypeError("FDR initial state must be a checkpoint mapping")
    validate_fdr_initial_state(artifact)
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_clean_tracked_worktree()

    output_root = args.output_root.resolve()
    authority_root = output_root / "authority"
    data_yaml = prepare_data_yaml(
        args.dataset_root.resolve(),
        "formal",
        authority_root / "data",
    )
    initial_state = validate_initial_state_file(args.initial_state)
    settings = build_settings(
        args.variant,
        data_yaml=data_yaml,
        output_root=output_root,
        name=args.name,
    )
    source = current_source_identity()
    dataset = dataset_signature(args.dataset_root.resolve())
    record = build_launch_record(
        variant=args.variant,
        source_identity=source,
        config_path=VARIANT_CONFIGS[args.variant],
        initial_state_path=initial_state,
        dataset=dataset,
        settings=settings,
    )
    record_path = authority_root / f"{settings['name']}.json"
    if record_path.exists():
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing != record:
            raise ValueError(f"launch authority already exists with different bytes: {record_path}")
    else:
        write_json_atomic(record_path, record)

    print(json.dumps(record, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    from src.rtdetr_fdr import FDRTrainer

    trainer = FDRTrainer(
        overrides=settings,
        initial_state_path=initial_state,
        experiment_seed=0,
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

