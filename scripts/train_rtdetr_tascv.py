from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tascv_cli import (
    build_parser,
    build_settings,
    validate_protocol_inputs,
)
from src.tascv_diagnostics import (
    TASCVMechanismAccumulator,
    validate_tascv_checkpoint_runtime,
)
from src.tascv_protocol import sha256_file, source_bundle_sha256
from src.tascv_stage import (
    TASCVStage,
    allowed_observed_tensor_batch_sizes,
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"))
            + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _run(args, protocol: dict | None = None) -> None:
    if protocol is None:
        protocol = validate_protocol_inputs(args)
    from src.rtdetr_tascv import (
        MATCHED_AMP_SCALE,
        MATCHED_BATCH_SIZE,
        TASCVControlTrainer,
        TASCVTrainer,
    )

    accumulator = TASCVMechanismAccumulator()
    mechanism_records: list[dict] = []
    local_forward_calls = 0
    local_forward_call_histogram = {"1": 0, "2": 0}
    local_bn_preserved_batches = 0
    trainer_class = (
        TASCVTrainer if args.arm == "tascv" else TASCVControlTrainer
    )
    trainer = trainer_class(
        overrides=build_settings(args),
        stage=args.stage,
        initial_state_path=args.initial_state,
    )
    if args.arm == "control":
        endpoint = protocol["control_allowlist"]["slots"][
            f"B:{args.stage.value}:{args.seed}"
        ]["fresh_target"]
    else:
        endpoint = protocol["treatment_endpoints"][
            f"T:{args.stage.value}:{args.seed}"
        ]
    expected_target = Path(endpoint["target_dir"]).resolve()
    if Path(trainer.save_dir).resolve() != expected_target:
        raise RuntimeError(
            "TASCV_SAVE_DIR_ENDPOINT_DRIFT: "
            f"actual={Path(trainer.save_dir).resolve()}, "
            f"expected={expected_target}"
        )
    if Path(trainer.last).resolve() != expected_target / "weights/last.pt":
        raise RuntimeError("TASCV_LAST_CHECKPOINT_ENDPOINT_DRIFT")

    def epoch_start(current) -> None:
        if args.arm != "tascv":
            return
        model = (
            current.model.module
            if hasattr(current.model, "module")
            else current.model
        )
        model.set_tascv_progress(current.epoch)

    def batch_end(current) -> None:
        nonlocal local_forward_calls, local_bn_preserved_batches
        if args.arm == "tascv":
            model = (
                current.model.module
                if hasattr(current.model, "module")
                else current.model
            )
            result = model.last_tascv_result
            if result is None:
                raise RuntimeError("TASCV_MISSING_BATCH_DIAGNOSTICS")
            validate_tascv_checkpoint_runtime(
                stage=args.stage,
                calls=model.last_local_forward_calls,
                batchnorm_preserved=model.last_local_bn_preserved,
            )
            calls = model.last_local_forward_calls
            local_forward_calls += calls
            local_forward_call_histogram[str(calls)] += 1
            local_bn_preserved_batches += int(
                model.last_local_bn_preserved
            )
            non_tiny_auxiliary = int(
                model.last_tascv_diagnostics[
                    "auxiliary_non_tiny_pair_count"
                ]
            )
            if non_tiny_auxiliary:
                raise RuntimeError(
                    "TASCV_NON_TINY_AUXILIARY_CONTRIBUTION_INVALID"
                )
            if args.stage is TASCVStage.TINY_MECHANISM_500:
                accumulator.record(
                    result,
                    auxiliary_non_tiny_pair_count=non_tiny_auxiliary,
                )
                mechanism_records.append(
                    {
                        "batch": len(mechanism_records) + 1,
                        "matched_pairs": result.matched_pair_count,
                        "auxiliary_tiny_pairs": (
                            result.auxiliary_tiny_pair_count
                        ),
                        "excluded_non_tiny_pairs": (
                            result.excluded_non_tiny_pair_count
                        ),
                        "auxiliary_non_tiny_pairs": 0,
                        "tiny_teacher_advantage_sum": float(
                            result.tiny_teacher_advantage_sum.detach().cpu()
                        ),
                        "tiny_teacher_win_count": (
                            result.tiny_teacher_win_count
                        ),
                    }
                )
        current.record_successful_batch()

    trainer.add_callback("on_train_epoch_start", epoch_start)
    trainer.add_callback("on_train_batch_end", batch_end)
    trainer.train()

    checkpoint_path = Path(trainer.last).resolve()
    checkpoint = {
        "kind": "last.pt",
        "path": checkpoint_path.as_posix(),
        "sha256": (
            sha256_file(checkpoint_path)
            if checkpoint_path.is_file()
            else ""
        ),
    }
    manifest_path = args.protocol_manifest.resolve()
    initial_path = args.initial_state.resolve()
    data_path = args.data.resolve()
    summary = {
        "schema_version": "tascv-training-summary/v1",
        "stage": args.stage.value,
        "arm": args.arm,
        "protocol_manifest": manifest_path.as_posix(),
        "protocol_manifest_sha256": sha256_file(manifest_path),
        "protocol_source_commit": protocol["runtime_source"]["commit"],
        "predecessor_evidence": (
            {
                "path": args.predecessor_evidence.resolve().as_posix(),
                "sha256": sha256_file(
                    args.predecessor_evidence.resolve()
                ),
            }
            if args.predecessor_evidence is not None
            else None
        ),
        "source_repo_bundle_sha256": protocol["runtime_source"][
            "repo_bundle_sha256"
        ],
        "source_upstream_bundle_sha256": source_bundle_sha256(
            protocol["runtime_source"]["upstream"]
        ),
        "approved_tascv_parent": protocol["approved_tascv_parent"],
        "r0_evaluation_anchor_sha256": protocol["r0_authority"][
            "evaluation_anchor_sha256"
        ],
        "control_slot": protocol["control_allowlist"]["slots"].get(
            f"B:{args.stage.value}:{args.seed}"
        ),
        "initial_state": initial_path.as_posix(),
        "initial_state_sha256": sha256_file(initial_path),
        "initial_state_common_fingerprint": protocol[
            "initial_states"
        ][str(args.seed)]["common_fingerprint"],
        "data": data_path.as_posix(),
        "data_sha256": sha256_file(data_path),
        "subset_binding": {
            key: protocol["subset"][key]
            for key in ("count", "semantic_sha256", "file_sha256")
        },
        "seed": args.seed,
        "batch": int(trainer.batch_size),
        "observed_tensor_batch_sizes": sorted(
            trainer.tascv_observed_tensor_batch_sizes
        ),
        "batch_canaries": trainer.tascv_batch_canaries,
        "amp": True,
        "amp_scale": MATCHED_AMP_SCALE,
        "amp_scale_min": trainer.tascv_amp_scale_min,
        "amp_scale_max": trainer.tascv_amp_scale_max,
        "successful_batches": trainer.tascv_successful_batches,
        "optimizer_attempts": trainer.tascv_optimizer_attempts,
        "expected_successful_batches": (
            trainer.tascv_policy.expected_successful_batches
        ),
        "expected_optimizer_attempts": (
            trainer.tascv_policy.expected_optimizer_attempts
        ),
        "workers": int(trainer.args.workers),
        "loader": trainer.tascv_loader_observation,
        "optimizer": trainer.tascv_optimizer_observation,
        "local_forward_calls": local_forward_calls,
        "local_forward_call_histogram": (
            local_forward_call_histogram
        ),
        "local_bn_preserved_batches": local_bn_preserved_batches,
        "internal_validation_bypass_count": (
            trainer.internal_validation_bypass_count
        ),
        "test_loader_is_none": trainer.test_loader is None,
        "auxiliary_non_tiny_pair_count": 0,
        "hardware": {
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "device": str(trainer.device),
        },
        "checkpoint": checkpoint,
        "mechanism_summary": (
            accumulator.summary()
            if args.stage is TASCVStage.TINY_MECHANISM_500
            else None
        ),
    }
    if torch.cuda.is_available():
        summary["cuda_peak_mib"] = round(
            torch.cuda.max_memory_allocated() / 1024**2,
            2,
        )
        summary["cuda_peak_reserved_mib"] = round(
            torch.cuda.max_memory_reserved() / 1024**2,
            2,
        )

    failures: list[str] = []
    if not checkpoint_path.is_file() or not checkpoint["sha256"]:
        failures.append("fixed last.pt checkpoint is missing")
    if (
        trainer.tascv_successful_batches
        != trainer.tascv_policy.expected_successful_batches
    ):
        failures.append("successful batch count drift")
    if (
        trainer.tascv_optimizer_attempts
        != trainer.tascv_policy.expected_optimizer_attempts
    ):
        failures.append("optimizer attempt count drift")
    if (
        trainer.tascv_amp_scale_min != MATCHED_AMP_SCALE
        or trainer.tascv_amp_scale_max != MATCHED_AMP_SCALE
    ):
        failures.append("AMP scale drift")
    if summary["batch"] != MATCHED_BATCH_SIZE or summary["workers"] != 8:
        failures.append("batch/workers drift")
    if set(summary["observed_tensor_batch_sizes"]) != set(
        allowed_observed_tensor_batch_sizes(args.stage)
    ):
        failures.append("observed tensor batch-size drift")
    if trainer.tascv_optimizer_observation.get("class") != "MuSGD":
        failures.append("optimizer class drift")
    if args.arm == "tascv":
        if (
            sum(local_forward_call_histogram.values())
            != trainer.tascv_successful_batches
        ):
            failures.append("local-forward accounting drift")
        if (
            args.stage is TASCVStage.PREFLIGHT_1
            and local_forward_calls != 2
        ):
            failures.append("preflight checkpoint recomputation missing")
    elif (
        local_forward_calls != 0
        or any(local_forward_call_histogram.values())
    ):
        failures.append("control executed local forwards")

    output_dir = Path(trainer.save_dir)
    if failures:
        _atomic_json(
            output_dir / "runtime_invalid.json",
            {
                **summary,
                "schema_version": "tascv-runtime-invalid/v1",
                "decision": "INVALID",
                "failures": failures,
            },
        )
        raise RuntimeError("TASCV_RUNTIME_INVALID: " + "; ".join(failures))
    if args.stage is TASCVStage.TINY_MECHANISM_500:
        records_path = output_dir / "tascv_mechanism_records.jsonl"
        _atomic_jsonl(records_path, mechanism_records)
        summary["mechanism_records"] = {
            "path": records_path.resolve().as_posix(),
            "sha256": sha256_file(records_path),
            "count": len(mechanism_records),
        }
    summary_path = output_dir / "tascv_training_summary.json"
    _atomic_json(summary_path, summary)


def main() -> None:
    args = build_parser().parse_args()
    protocol = validate_protocol_inputs(args)
    target = args.project.resolve() / args.name
    target_existed = target.exists()
    try:
        _run(args, protocol)
    except BaseException as error:
        normalized = target.as_posix().lower()
        if "test-dev" not in normalized and "test_dev" not in normalized:
            invalid_path = (
                args.project.resolve()
                / f"{args.name}.launch_invalid.json"
                if target_existed
                else target / "runtime_invalid.json"
            )
            if not invalid_path.exists():
                try:
                    _atomic_json(
                        invalid_path,
                        {
                            "schema_version": (
                                "tascv-runtime-invalid/v1"
                            ),
                            "decision": "INVALID",
                            "stage": args.stage.value,
                            "arm": args.arm,
                            "seed": args.seed,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                    )
                except OSError:
                    pass
        raise


if __name__ == "__main__":
    main()
