"""Launch the frozen VisDrone LRS-GFDR-FIA arm under Formal100 seed 0."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from src.lrs_runtime_evidence import RuntimeEvidenceRecorder  # noqa: E402
from src.lpr_protocol import dataset_signature  # noqa: E402
from src.rtdetr_lrs_system import (  # noqa: E402
    LRS_GFDR_FIA_CONFIG,
    LRSGFDRFIATrainer,
    load_fdr_initial_state_artifact,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _safe_name(name: str) -> str:
    path = Path(name)
    if not name or name in {".", ".."} or path.is_absolute() or path.name != name:
        raise ValueError("run name must be one non-empty safe path component")
    return name


def _validate_initial_state(path: Path) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    load_fdr_initial_state_artifact(path)
    return path


def build_settings(data_yaml: Path, output_root: Path, name: str) -> dict[str, Any]:
    return {
        **FROZEN_SETTINGS,
        "model": str(LRS_GFDR_FIA_CONFIG.resolve()),
        "save_period": -1,
        "data": str(data_yaml.resolve()),
        "epochs": FORMAL_EPOCHS,
        "seed": 0,
        "project": str(output_root.resolve()),
        "name": _safe_name(name),
        "exist_ok": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name", default="formal-seed0-lrs_gfdr_fia-v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    authority_root = output_root / "authority"
    data_yaml = prepare_data_yaml(args.dataset_root.resolve(), "formal", authority_root / "data")
    initial_state = _validate_initial_state(args.initial_state)
    settings = build_settings(data_yaml, output_root, args.name)
    record = {
        "format_version": 1,
        "method_revision": "lrs-gfdr-fia-v1",
        "method": "lrs_gfdr_fia",
        "source": current_source_identity(),
        "config": {"path": str(LRS_GFDR_FIA_CONFIG.resolve()), "sha256": _sha256(LRS_GFDR_FIA_CONFIG)},
        "initial_state": {"path": str(initial_state), "sha256": _sha256(initial_state)},
        "dataset": dataset_signature(args.dataset_root.resolve()),
        "settings": settings,
    }
    write_json_atomic(authority_root / f"{settings['name']}.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    trainer = LRSGFDRFIATrainer(
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
