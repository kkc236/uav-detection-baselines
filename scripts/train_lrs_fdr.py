"""Launch Clean FDR plus frozen LRS-FGL under Formal100 seed 0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
LRS_FDR_CONFIG = ROOT / "configs" / "rtdetr-l-lrs-fdr.yaml"
sys.path.insert(0, str(ROOT))

from scripts.sync_experiment_checkpoint import write_json_atomic  # noqa: E402
from scripts.train_ace_fdr import (  # noqa: E402
    build_launch_record as build_base_launch_record,
    require_clean_tracked_worktree,
    validate_initial_state_file,
)
from scripts.train_rtdetr_fdr import (  # noqa: E402
    FORMAL_EPOCHS,
    FROZEN_SETTINGS,
    current_source_identity,
    prepare_data_yaml,
)
from src.lpr_protocol import dataset_signature  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Clean FDR plus LRS-FGL under Formal100 seed 0."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_settings(
    *, data_yaml: Path, output_root: Path, name: str | None = None
) -> dict[str, Any]:
    return {
        **FROZEN_SETTINGS,
        "model": str(LRS_FDR_CONFIG.resolve()),
        "save_period": -1,
        "data": str(Path(data_yaml).resolve()),
        "epochs": FORMAL_EPOCHS,
        "seed": 0,
        "project": str(Path(output_root).resolve()),
        "name": name or "formal-seed0-lrs-fdr-v1",
        "exist_ok": False,
    }


def build_method_record() -> dict[str, object]:
    return {
        "kind": "layerwise_reliability_shrinkage_v1",
        "alpha0": 0.25,
        "schedule": [0.25, 0.20, 0.15, 0.10, 0.05, 0.0],
        "scope": "normal_decoder_fgl_only",
        "grouping": "same_image_same_layer",
        "resume_policy": "restart_from_epoch_0",
    }


def build_launch_record(
    *,
    source_identity: Mapping[str, Any],
    config_path: Path,
    initial_state_path: Path,
    dataset: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    record = build_base_launch_record(
        source_identity=source_identity,
        config_path=config_path,
        initial_state_path=initial_state_path,
        dataset=dataset,
        settings=settings,
    )
    record["method"] = "lrs_fdr"
    record["lrs_fgl"] = build_method_record()
    return record


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_clean_tracked_worktree()
    output_root = args.output_root.resolve()
    authority_root = output_root / "authority"
    data_yaml = prepare_data_yaml(
        args.dataset_root.resolve(), "formal", authority_root / "data"
    )
    initial_state = validate_initial_state_file(args.initial_state)
    settings = build_settings(
        data_yaml=data_yaml, output_root=output_root, name=args.name
    )
    record = build_launch_record(
        source_identity=current_source_identity(),
        config_path=LRS_FDR_CONFIG,
        initial_state_path=initial_state,
        dataset=dataset_signature(args.dataset_root.resolve()),
        settings=settings,
    )
    record_path = authority_root / f"{settings['name']}.json"
    if record_path.exists():
        if json.loads(record_path.read_text(encoding="utf-8")) != record:
            raise ValueError(f"authority exists with different bytes: {record_path}")
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
