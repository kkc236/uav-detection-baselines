from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ascv_loc_cli import build_parser, build_settings, validate_protocol_inputs
from src.ascv_loc_diagnostics import (
    ASCVMechanismAccumulator,
    validate_local_checkpoint_runtime,
)
from src.ascv_loc_protocol import sha256_file, source_bundle_sha256
from src.ascv_loc_stage import ASCVStage, allowed_observed_tensor_batch_sizes


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
    local_forward_calls = 0
    local_forward_call_histogram = {"1": 0, "2": 0}
    local_bn_preserved_batches = 0
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
        nonlocal local_forward_calls, local_bn_preserved_batches
        if args.arm == "ascv":
            model = current.model.module if hasattr(current.model, "module") else current.model
            result = model.last_ascv_result
            if result is None:
                raise RuntimeError("ASCV_LOC_MISSING_BATCH_DIAGNOSTICS")
            validate_local_checkpoint_runtime(
                stage=args.stage,
                calls=model.last_local_forward_calls,
                batchnorm_preserved=model.last_local_bn_preserved,
                non_tiny_pair_count=result.non_tiny_pair_count,
            )
            local_forward_calls += model.last_local_forward_calls
            local_forward_call_histogram[str(model.last_local_forward_calls)] += 1
            local_bn_preserved_batches += int(model.last_local_bn_preserved)
            accumulator.record(result)
        current.record_successful_batch()

    trainer.add_callback("on_train_epoch_start", epoch_start)
    trainer.add_callback("on_train_batch_end", batch_end)
    trainer.train()

    checkpoint_path = Path(trainer.last).resolve()
    checkpoint_record = {
        "kind": "last.pt",
        "path": checkpoint_path.as_posix(),
        "sha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() else "",
    }
    summary = {
        "schema_version": "ascv-loc-training-summary/v2",
        "stage": args.stage.value,
        "arm": args.arm,
        "protocol_manifest": str(args.protocol_manifest.resolve()),
        "protocol_manifest_sha256": sha256_file(args.protocol_manifest.resolve()),
        "protocol_source_commit": protocol["source_commit"],
        "source_repo_bundle_sha256": protocol["source"]["repo_bundle_sha256"],
        "source_upstream_bundle_sha256": source_bundle_sha256(protocol["source"]["upstream"]),
        "initial_state": str(args.initial_state.resolve()),
        "initial_state_sha256": sha256_file(args.initial_state.resolve()),
        "initial_state_common_fingerprint": protocol["initial_states"][str(args.seed)][
            "common_fingerprint"
        ],
        "data_sha256": sha256_file(args.data.resolve()),
        "subset_binding": {
            key: protocol["subset"][key]
            for key in ("count", "semantic_sha256", "file_sha256")
        },
        "seed": args.seed,
        "batch": int(trainer.batch_size),
        "observed_tensor_batch_sizes": sorted(trainer.ascv_observed_tensor_batch_sizes),
        "batch_canaries": trainer.ascv_batch_canaries,
        "amp": True,
        "amp_scale": MATCHED_AMP_SCALE,
        "successful_batches": trainer.ascv_successful_batches,
        "optimizer_attempts": trainer.ascv_optimizer_attempts,
        "expected_successful_batches": trainer.ascv_policy.expected_successful_batches,
        "expected_optimizer_attempts": trainer.ascv_policy.expected_optimizer_attempts,
        "amp_scale_min": trainer.ascv_amp_scale_min,
        "amp_scale_max": trainer.ascv_amp_scale_max,
        "workers": int(trainer.args.workers),
        "loader": trainer.ascv_loader_observation,
        "optimizer": trainer.ascv_optimizer_observation,
        "local_forward_calls": local_forward_calls,
        "local_forward_call_histogram": local_forward_call_histogram,
        "local_bn_preserved_batches": local_bn_preserved_batches,
        "internal_validation_bypass_count": trainer.internal_validation_bypass_count,
        "test_loader_is_none": trainer.test_loader is None,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": str(trainer.device),
        },
        "checkpoint": checkpoint_record,
        **accumulator.summary(),
    }
    if torch.cuda.is_available():
        summary["cuda_peak_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
        summary["cuda_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 1024**2, 2)
    runtime_failures = []
    if not checkpoint_path.is_file() or not checkpoint_record["sha256"]:
        runtime_failures.append("fixed last.pt checkpoint is missing")
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
    if summary["batch"] != MATCHED_BATCH_SIZE or summary["workers"] != 8:
        runtime_failures.append("observed batch/workers do not match 8/8")
    allowed_tensor_batches = allowed_observed_tensor_batch_sizes(args.stage)
    if set(summary["observed_tensor_batch_sizes"]) != set(allowed_tensor_batches):
        runtime_failures.append(
            "observed transformed tensor batches do not match the frozen "
            f"{sorted(allowed_tensor_batches)} contract"
        )
    if trainer.ascv_optimizer_observation.get("class") != "MuSGD":
        runtime_failures.append("observed optimizer is not MuSGD")
    if args.arm == "ascv":
        observed_local_batches = sum(local_forward_call_histogram.values())
        if observed_local_batches != trainer.ascv_successful_batches:
            runtime_failures.append("local-forward batch accounting mismatch")
        if args.stage is ASCVStage.PREFLIGHT_1 and local_forward_calls != 2:
            runtime_failures.append("preflight activation checkpoint did not recompute")
    elif local_forward_calls != 0 or any(local_forward_call_histogram.values()):
        runtime_failures.append("control arm executed local forwards")
    if runtime_failures:
        _atomic_json(
            Path(trainer.save_dir) / "runtime_invalid.json",
            {"decision": "INVALID", "failures": runtime_failures, **summary},
        )
        raise RuntimeError("ASCV_LOC_RUNTIME_INVALID: " + "; ".join(runtime_failures))

    summary_path = Path(trainer.save_dir) / "ascv_training_summary.json"
    _atomic_json(summary_path, summary)

    if args.stage is ASCVStage.MECHANISM_500:
        passed, failures = accumulator.mechanism_gate()
        gate = {
            **summary,
            "schema_version": "ascv-loc-mechanism-gate/v2",
            "decision": "GO" if passed else "ASCV_LOC_STOP",
            "failures": failures,
            "training_summary": {
                "path": summary_path.resolve().as_posix(),
                "sha256": sha256_file(summary_path),
            },
        }
        _atomic_json(Path(trainer.save_dir) / "mechanism_gate.json", gate)
        if not passed:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
