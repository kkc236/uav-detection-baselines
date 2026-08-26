"""Launch all-on gradient-decoupled Persistent DCF-FDR Formal100."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "rtdetr-l-persistent-gradient-dcf-fdr.yaml"
sys.path.insert(0, str(ROOT))

from scripts.sync_experiment_checkpoint import write_json_atomic  # noqa: E402
from scripts.train_ace_fdr import (  # noqa: E402
    require_clean_tracked_worktree,
    validate_initial_state_file,
)
from scripts.train_dcf_fdr import (  # noqa: E402
    build_launch_record as build_base_launch_record,
)
from scripts.train_rtdetr_fdr import (  # noqa: E402
    FORMAL_EPOCHS,
    FROZEN_SETTINGS,
    current_source_identity,
    prepare_data_yaml,
)
from src.lpr_protocol import dataset_signature  # noqa: E402
from src.persistent_dcf import (  # noqa: E402
    audit_persistent_dcf_state,
    persistent_dcf_state,
)
from src.transient_dcf import find_distribution_feedback_decoder  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train all-on gradient-decoupled DCF-FDR Formal100 seed0."
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
        "model": str(CONFIG.resolve()),
        "save_period": -1,
        "data": str(Path(data_yaml).resolve()),
        "epochs": FORMAL_EPOCHS,
        "seed": 0,
        "project": str(Path(output_root).resolve()),
        "name": name or "formal-seed0-persistent-gradient-dcf-fdr-v1",
        "exist_ok": False,
    }


def build_method_record() -> dict[str, object]:
    return {
        "kind": "persistent_gradient_dcf_v1",
        "scale": "1.0_all_epochs",
        "trainable": "all_epochs",
        "checkpoint_eligible_from_epoch": 1,
        "resume_policy": "restart_from_epoch_0",
    }


def build_launch_record(
    *,
    source_identity: Mapping[str, Any],
    initial_state_path: Path,
    dataset: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    record = build_base_launch_record(
        source_identity=source_identity,
        initial_state_path=initial_state_path,
        dataset=dataset,
        settings=settings,
    )
    record["method"] = "persistent_gradient_dcf_fdr"
    record["persistent_dcf"] = build_method_record()
    return record


def append_epoch_evidence(path: Path, trainer: Any) -> None:
    """Assert the all-on contract and append one canonical epoch row."""

    state = persistent_dcf_state(trainer.epoch + 1, trainer.epochs)
    record = audit_persistent_dcf_state(trainer.model, trainer.ema.ema, state)
    decoder = find_distribution_feedback_decoder(trainer.model)
    feedback_ids = {
        id(parameter) for parameter in decoder.distribution_feedback.parameters()
    }
    groups = trainer.gradient_parameter_groups()
    private_ids = {id(parameter) for parameter in groups["fdr_gradient_norm"]}
    common_ids = {id(parameter) for parameter in groups["gradient_norm"]}
    if not feedback_ids or not feedback_ids <= private_ids:
        raise RuntimeError("DCF parameters are missing from private FDR gradient group")
    if not feedback_ids.isdisjoint(common_ids):
        raise RuntimeError("DCF parameters leaked into common gradient group")
    record["private_gradient_group"] = True
    rows: list[dict[str, object]] = []
    if path.exists():
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    if any(row["paper_epoch"] == state.paper_epoch for row in rows):
        raise ValueError(f"duplicate paper epoch: {state.paper_epoch}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        )


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
    trainer.add_callback(
        "on_train_epoch_start",
        lambda current: append_epoch_evidence(
            Path(current.save_dir) / "persistent-dcf-state.jsonl", current
        ),
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
