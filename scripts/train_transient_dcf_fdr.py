"""Launch the frozen training-only Transient DCF-FDR Formal100 experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
TRANSIENT_DCF_FDR_CONFIG = ROOT / "configs" / "rtdetr-l-transient-dcf-fdr.yaml"
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
from src.transient_dcf import (  # noqa: E402
    TransientDCFState,
    apply_transient_dcf_state,
    find_distribution_feedback_decoder,
    transient_dcf_state,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Transient DCF-FDR under Formal100 seed0."
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
        "model": str(TRANSIENT_DCF_FDR_CONFIG.resolve()),
        "save_period": -1,
        "data": str(Path(data_yaml).resolve()),
        "epochs": FORMAL_EPOCHS,
        "seed": 0,
        "project": str(Path(output_root).resolve()),
        "name": name or "formal-seed0-transient-dcf-fdr-v1",
        "exist_ok": False,
    }


def build_schedule_record() -> dict[str, object]:
    return {
        "kind": "transient_dcf_v1",
        "full_through_ratio": "2/3",
        "withdrawal": "cosine",
        "off_from_ratio": "3/4",
        "formal_epochs": FORMAL_EPOCHS,
        "checkpoint_eligible_from_epoch": 75,
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
    record["method"] = "transient_dcf_fdr"
    record["schedule"] = build_schedule_record()
    return record


def _schedule_evidence_row(
    trainer: Any, state: TransientDCFState
) -> dict[str, int | float | bool]:
    live = find_distribution_feedback_decoder(trainer.model)
    ema = find_distribution_feedback_decoder(trainer.ema.ema)
    return {
        **state.to_dict(),
        "live_scale": live.distribution_feedback_scale,
        "ema_scale": ema.distribution_feedback_scale,
        "live_feedback_trainable": any(
            parameter.requires_grad
            for parameter in live.distribution_feedback.parameters()
        ),
    }


def append_schedule_evidence(
    evidence_path: Path, trainer: Any, state: TransientDCFState
) -> None:
    """Append one canonical row and refuse ambiguous repeated epochs."""

    evidence_path = Path(evidence_path)
    existing_rows: list[dict[str, object]] = []
    if evidence_path.exists():
        existing_rows = [
            json.loads(line)
            for line in evidence_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if any(row.get("paper_epoch") == state.paper_epoch for row in existing_rows):
        raise ValueError(f"duplicate paper epoch in schedule evidence: {state.paper_epoch}")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _schedule_evidence_row(trainer, state),
        sort_keys=True,
        separators=(",", ":"),
    )
    with evidence_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")


def configure_transient_epoch(trainer: Any, evidence_path: Path) -> None:
    """Apply the exact paper-epoch state before an epoch starts."""

    state = transient_dcf_state(trainer.epoch + 1, trainer.epochs)
    apply_transient_dcf_state(trainer.model, trainer.ema.ema, state)
    if state.checkpoint_eligible and not getattr(
        trainer, "transient_tail_best_reset", False
    ):
        trainer.best_fitness = None
        trainer.transient_tail_best_reset = True
    append_schedule_evidence(evidence_path, trainer, state)


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
            raise ValueError(
                f"launch authority already exists with different bytes: {record_path}"
            )
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
        lambda current: configure_transient_epoch(
            current,
            Path(current.save_dir) / "transient-dcf-schedule.jsonl",
        ),
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
