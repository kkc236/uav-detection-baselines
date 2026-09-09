"""Launch the local-expert capacity arms under Formal100 seed 0."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
EXPERT_CONFIG = ROOT / "configs" / "rtdetr-l-lrs-gfdr-capacity-expert.yaml"
BPDD_CONFIG = ROOT / "configs" / "rtdetr-l-lrs-gfdr-capacity-bpdd.yaml"
DEFAULT_RUN_NAME = "formal-seed0-lrs_gfdr_capacity_bpdd-v1"
sys.path.insert(0, str(ROOT))

from scripts.sync_experiment_checkpoint import write_json_atomic  # noqa: E402
from scripts.train_ace_fdr import require_clean_tracked_worktree  # noqa: E402
from scripts.train_rtdetr_fdr import (  # noqa: E402
    FORMAL_EPOCHS,
    FROZEN_SETTINGS,
    current_source_identity,
    prepare_data_yaml,
)
from scripts.train_visdrone_lrs_system import (  # noqa: E402
    validate_initial_state_file,
    validate_run_name,
)
from src.lpr_protocol import dataset_signature  # noqa: E402
from src.lrs_runtime_evidence import RuntimeEvidenceRecorder  # noqa: E402
from src.rtdetr_lrs_system import LRSFDRBPDDTrainer  # noqa: E402


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train local-expert capacity arms under Formal100 seed 0."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--arm", choices=("expert", "bpdd"), default="bpdd")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_settings(
    data_yaml: Path,
    output_root: Path,
    name: str = DEFAULT_RUN_NAME,
    arm: str = "bpdd",
) -> dict[str, Any]:
    config_path = BPDD_CONFIG if arm == "bpdd" else EXPERT_CONFIG
    return {
        **FROZEN_SETTINGS,
        "model": str(config_path.resolve()),
        "save_period": -1,
        "data": str(Path(data_yaml).resolve()),
        "epochs": FORMAL_EPOCHS,
        "seed": 0,
        "project": str(Path(output_root).resolve()),
        "name": validate_run_name(name),
        "exist_ok": False,
    }


def build_launch_record(
    *,
    source_identity: Mapping[str, Any],
    config_path: Path,
    arm: str,
    initial_state_path: Path,
    dataset: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    initial_state_path = Path(initial_state_path).resolve()
    return {
        "format_version": 1,
        "method_revision": "lrs-gfdr-capacity-v1",
        "method": f"lrs_gfdr_capacity_{arm}",
        "arm": arm,
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


def _write_authority(path: Path, record: Mapping[str, Any]) -> None:
    payload = dict(record)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"launch authority already exists with different bytes: {path}")
        return
    write_json_atomic(path, payload)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_clean_tracked_worktree()

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    authority_root = output_root / "authority"
    data_yaml = prepare_data_yaml(
        dataset_root,
        "formal",
        authority_root / "data",
    )
    initial_state = validate_initial_state_file(args.initial_state)
    arm = args.arm
    config_path = BPDD_CONFIG if arm == "bpdd" else EXPERT_CONFIG
    settings = build_settings(data_yaml, output_root, args.name, arm=arm)
    record = build_launch_record(
        source_identity=current_source_identity(),
        config_path=config_path,
        arm=arm,
        initial_state_path=initial_state,
        dataset=dataset_signature(dataset_root),
        settings=settings,
    )
    _write_authority(authority_root / f"{settings['name']}.json", record)

    print(json.dumps(record, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    trainer = LRSFDRBPDDTrainer(
        overrides=settings,
        initial_state_path=initial_state,
        experiment_seed=0,
    )
    recorder = RuntimeEvidenceRecorder()
    trainer.add_callback("on_train_epoch_start", recorder.reset)
    trainer.add_callback("on_train_batch_end", recorder.capture)
    trainer.add_callback("on_train_epoch_end", recorder.write)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
