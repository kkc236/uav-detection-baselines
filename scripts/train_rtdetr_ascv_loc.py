from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ascv_loc_cli import build_parser, build_settings, validate_protocol_inputs
from src.ascv_loc_diagnostics import ASCVMechanismAccumulator
from src.ascv_loc_stage import ASCVStage


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = build_parser().parse_args()
    protocol = validate_protocol_inputs(args)
    import ultralytics

    if ultralytics.__version__ != protocol["environment"]["ultralytics"]:
        raise RuntimeError(
            f"ASCV_LOC_ULTRALYTICS_DRIFT: {ultralytics.__version__}"
        )
    from src.rtdetr_ascv_loc import (
        ASCVControlTrainer,
        ASCVLocTrainer,
        MATCHED_AMP_SCALE,
        MATCHED_BATCH_SIZE,
    )

    accumulator = ASCVMechanismAccumulator()
    trainer_class = ASCVLocTrainer if args.arm == "ascv" else ASCVControlTrainer
    trainer = trainer_class(
        overrides=build_settings(args),
        stage=args.stage,
        initial_state_path=args.initial_state,
    )

    def epoch_start(current) -> None:
        if args.arm != "ascv":
            return
        model = current.model.module if hasattr(current.model, "module") else current.model
        model.set_ascv_progress(int(current.epoch))

    def batch_end(current) -> None:
        if args.arm == "ascv":
            model = current.model.module if hasattr(current.model, "module") else current.model
            result = model.last_ascv_result
            if result is None:
                raise RuntimeError("ASCV_LOC_MISSING_BATCH_DIAGNOSTICS")
            accumulator.record(result)
        current.record_successful_batch()

    trainer.add_callback("on_train_epoch_start", epoch_start)
    trainer.add_callback("on_train_batch_end", batch_end)
    trainer.train()

    summary = {
        "schema_version": "ascv-loc-training-summary/v1",
        "stage": args.stage.value,
        "arm": args.arm,
        "protocol_manifest": str(args.protocol_manifest.resolve()),
        "protocol_source_commit": protocol["source_commit"],
        "initial_state": str(args.initial_state.resolve()),
        "seed": args.seed,
        "batch": MATCHED_BATCH_SIZE,
        "amp": True,
        "amp_scale": MATCHED_AMP_SCALE,
        "successful_batches": trainer.ascv_successful_batches,
        "optimizer_attempts": trainer.ascv_optimizer_attempts,
        "expected_successful_batches": trainer.ascv_policy.expected_successful_batches,
        "expected_optimizer_attempts": trainer.ascv_policy.expected_optimizer_attempts,
        "amp_scale_min": trainer.ascv_amp_scale_min,
        "amp_scale_max": trainer.ascv_amp_scale_max,
        "workers": 8,
        "internal_validation_bypass_count": trainer.internal_validation_bypass_count,
        **accumulator.summary(),
    }
    if torch.cuda.is_available():
        summary["cuda_peak_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
        summary["cuda_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 1024**2, 2)
    summary_path = Path(trainer.save_dir) / "ascv_training_summary.json"
    _atomic_json(summary_path, summary)

    runtime_failures = []
    if trainer.ascv_successful_batches != trainer.ascv_policy.expected_successful_batches:
        runtime_failures.append(
            f"successful_batches={trainer.ascv_successful_batches}, "
            f"expected={trainer.ascv_policy.expected_successful_batches}"
        )
    if trainer.ascv_optimizer_attempts != trainer.ascv_policy.expected_optimizer_attempts:
        runtime_failures.append(
            f"optimizer_attempts={trainer.ascv_optimizer_attempts}, "
            f"expected={trainer.ascv_policy.expected_optimizer_attempts}"
        )
    if trainer.ascv_amp_scale_min != MATCHED_AMP_SCALE or trainer.ascv_amp_scale_max != MATCHED_AMP_SCALE:
        runtime_failures.append("AMP scale was not fixed at 128")
    if runtime_failures:
        _atomic_json(
            Path(trainer.save_dir) / "runtime_invalid.json",
            {"decision": "INVALID", "failures": runtime_failures, **summary},
        )
        raise RuntimeError("ASCV_LOC_RUNTIME_INVALID: " + "; ".join(runtime_failures))

    if args.stage is ASCVStage.MECHANISM_500:
        passed, failures = accumulator.mechanism_gate()
        gate = {"decision": "GO" if passed else "ASCV_LOC_STOP", "failures": failures, **summary}
        _atomic_json(Path(trainer.save_dir) / "mechanism_gate.json", gate)
        if not passed:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
