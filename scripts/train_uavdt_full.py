"""Launch revised assignment-consistent Full on UAVDT baseline settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_ace_fdr import require_clean_tracked_worktree  # noqa: E402
from scripts.train_rtdetr_fdr import current_source_identity  # noqa: E402
from scripts.train_visdrone_lrs_system import (  # noqa: E402
    validate_run_name,
    write_authority,
)
from src.rtdetr_lrs_system import (  # noqa: E402
    ARM_CONFIGS,
    TRAINER_TYPES,
    load_fdr_initial_state_artifact,
)


METHOD = "lrs_fdr_ac_bpdd_fia"
ARM = "i"
CONFIG = ARM_CONFIGS[ARM]
TRAINER = TRAINER_TYPES[ARM]
REPLACED_BASELINE_FIELDS = {
    "model",
    "data",
    "project",
    "name",
    "save_dir",
    "resume",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _ordinary_file(path: Path, label: str) -> Path:
    requested = Path(path)
    if requested.is_symlink() or not requested.is_file():
        raise FileNotFoundError(f"{label} not found as an ordinary file: {requested}")
    return requested.resolve()


def _load_yaml_mapping(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = _ordinary_file(path, label)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a YAML mapping")
    return resolved, dict(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train revised LRS-FDR+AC-BPDD+FIA Full on UAVDT."
    )
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--baseline-args", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_initial_state_file(path: Path) -> Path:
    resolved = _ordinary_file(path, "FDR initial state")
    load_fdr_initial_state_artifact(resolved)
    return resolved


def validate_data_yaml(path: Path) -> dict[str, Any]:
    resolved, payload = _load_yaml_mapping(path, "UAVDT data YAML")
    for split in ("train", "val"):
        if split not in payload or payload[split] in (None, "", [], {}):
            raise ValueError(f"UAVDT data YAML requires non-empty {split}")

    names = payload.get("names")
    if isinstance(names, list):
        normalized_names = [str(value) for value in names]
    elif isinstance(names, dict):
        if not names or any(not isinstance(key, int) for key in names):
            raise ValueError("UAVDT names mapping must use contiguous integer keys")
        expected = list(range(len(names)))
        if sorted(names) != expected:
            raise ValueError("UAVDT names mapping must use contiguous class ids")
        normalized_names = [str(names[index]) for index in expected]
    else:
        raise ValueError("UAVDT names must be a non-empty list or mapping")
    if not normalized_names or any(not name for name in normalized_names):
        raise ValueError("UAVDT names must be non-empty")

    nc = len(normalized_names)
    if "nc" in payload and int(payload["nc"]) != nc:
        raise ValueError("UAVDT declared nc conflicts with names")
    return {
        "path": str(resolved),
        "sha256": _file_sha256(resolved),
        "nc": nc,
        "names": normalized_names,
        "train": payload["train"],
        "val": payload["val"],
    }


def build_settings(
    baseline: Mapping[str, Any],
    *,
    data_yaml: Path,
    output_root: Path,
    name: str,
) -> dict[str, Any]:
    if not isinstance(baseline, Mapping):
        raise TypeError("baseline args must be a mapping")
    settings = {
        key: value
        for key, value in baseline.items()
        if key not in REPLACED_BASELINE_FIELDS
    }
    settings.update(
        {
            "model": str(CONFIG.resolve()),
            "data": str(Path(data_yaml).resolve()),
            "project": str(Path(output_root).resolve()),
            "name": validate_run_name(name),
            "exist_ok": False,
        }
    )
    return settings


def build_launch_record(
    *,
    source: Mapping[str, Any],
    baseline_path: Path,
    initial_state: Path,
    dataset: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_path = Path(baseline_path).resolve()
    initial_state = Path(initial_state).resolve()
    return {
        "format_version": 1,
        "arm": ARM,
        "method": METHOD,
        "source": dict(source),
        "config": {
            "path": str(CONFIG.resolve()),
            "sha256": _file_sha256(CONFIG.resolve()),
        },
        "baseline_args": {
            "path": str(baseline_path),
            "sha256": _file_sha256(baseline_path),
        },
        "initial_state": {
            "path": str(initial_state),
            "sha256": _file_sha256(initial_state),
        },
        "dataset": dict(dataset),
        "settings": dict(settings),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_clean_tracked_worktree()

    data_record = validate_data_yaml(args.data_yaml)
    data_yaml = Path(data_record["path"])
    baseline_path, baseline = _load_yaml_mapping(
        args.baseline_args, "baseline args"
    )
    initial_state = validate_initial_state_file(args.initial_state)
    output_root = args.output_root.resolve()
    epochs = int(baseline.get("epochs", 100))
    seed = int(baseline.get("seed", 0))
    default_name = f"uavdt-formal{epochs}-seed{seed}-{METHOD}-v1"
    settings = build_settings(
        baseline,
        data_yaml=data_yaml,
        output_root=output_root,
        name=args.name or default_name,
    )
    record = build_launch_record(
        source=current_source_identity(),
        baseline_path=baseline_path,
        initial_state=initial_state,
        dataset=data_record,
        settings=settings,
    )
    authority = output_root / "authority" / f"{settings['name']}.json"
    write_authority(authority, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    trainer = TRAINER(
        overrides=settings,
        initial_state_path=initial_state,
        experiment_seed=seed,
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
