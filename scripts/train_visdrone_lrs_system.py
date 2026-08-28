"""Launch one frozen VisDrone LRS system arm under Formal100 seed 0."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sync_experiment_checkpoint import write_json_atomic  # noqa: E402
from scripts.train_ace_fdr import require_clean_tracked_worktree  # noqa: E402
from scripts.train_rtdetr_fdr import (  # noqa: E402
    FORMAL_EPOCHS,
    FROZEN_SETTINGS,
    current_source_identity,
    prepare_data_yaml,
)
from src.lpr_protocol import dataset_signature  # noqa: E402
from src.rtdetr_lrs_system import (  # noqa: E402
    ARM_CONFIGS,
    TRAINER_TYPES,
    load_fdr_initial_state_artifact,
)


ARM_METHODS = {
    "g": "lrs_fdr_bpdd",
    "h": "lrs_fdr_fia",
    "i": "lrs_fdr_bpdd_fia",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one frozen VisDrone LRS system arm under Formal100 seed 0."
    )
    parser.add_argument("--arm", choices=tuple(ARM_METHODS), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_run_name(name: str) -> str:
    path = Path(name)
    if (
        not name
        or name in {".", ".."}
        or path.is_absolute()
        or "/" in name
        or "\\" in name
        or path.name != name
    ):
        raise ValueError("run name must be one non-empty safe path component")
    return name


def validate_initial_state_file(path: Path) -> Path:
    requested = Path(path)
    if requested.is_symlink() or not requested.is_file():
        raise FileNotFoundError(f"FDR initial state not found: {requested}")
    resolved = requested.resolve()
    load_fdr_initial_state_artifact(resolved)
    return resolved


def build_settings(
    arm: str,
    data_yaml: Path,
    output_root: Path,
    name: str | None = None,
) -> dict[str, Any]:
    if arm not in ARM_METHODS:
        raise ValueError(f"unknown LRS system arm: {arm}")
    run_name = validate_run_name(
        name if name is not None else f"formal-seed0-{ARM_METHODS[arm]}-v1"
    )
    return {
        **FROZEN_SETTINGS,
        "model": str(ARM_CONFIGS[arm].resolve()),
        "save_period": -1,
        "data": str(Path(data_yaml).resolve()),
        "epochs": FORMAL_EPOCHS,
        "seed": 0,
        "project": str(Path(output_root).resolve()),
        "name": run_name,
        "exist_ok": False,
    }


def build_launch_record(
    *,
    arm: str,
    source_identity: Mapping[str, Any],
    config_path: Path,
    initial_state_path: Path,
    dataset: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    if arm not in ARM_METHODS:
        raise ValueError(f"unknown LRS system arm: {arm}")
    config_path = Path(config_path).resolve()
    initial_state_path = Path(initial_state_path).resolve()
    return {
        "format_version": 1,
        "arm": arm,
        "method": ARM_METHODS[arm],
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


def write_authority(path: Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    payload = dict(record)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"launch authority already exists with different bytes: {path}"
            )
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
    settings = build_settings(args.arm, data_yaml, output_root, args.name)
    record = build_launch_record(
        arm=args.arm,
        source_identity=current_source_identity(),
        config_path=ARM_CONFIGS[args.arm],
        initial_state_path=initial_state,
        dataset=dataset_signature(dataset_root),
        settings=settings,
    )
    record_path = authority_root / f"{settings['name']}.json"
    write_authority(record_path, record)

    print(json.dumps(record, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    trainer = TRAINER_TYPES[args.arm](
        overrides=settings,
        initial_state_path=initial_state,
        experiment_seed=0,
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
