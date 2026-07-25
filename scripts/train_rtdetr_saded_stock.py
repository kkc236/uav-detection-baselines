from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.saded_stock_cli import (
    build_parser,
    build_settings,
    current_upstream_source_hashes,
    sha256_file,
    source_closure,
    validate_protocol_inputs,
)
from src.tascv_protocol import (
    EXPECTED_UPSTREAM_SOURCE_SHA256,
    FROZEN_OPTIMIZER_OBSERVATION,
)
from src.tascv_stage import TASCVStage


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _valid_sha(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def validate_runtime_summary(summary: dict) -> list[str]:
    failures: list[str] = []
    exact_values = {
        "schema_version": "saded-stock-training-summary/v1",
        "stage": "FORMAL_100",
        "arm": "stock_control",
        "seed": 0,
        "completed_epochs": 100,
        "batch": 8,
        "workers": 8,
        "successful_batches": 80_900,
        "optimizer_attempts": 10_556,
        "expected_successful_batches": 80_900,
        "expected_optimizer_attempts": 10_556,
        "amp": True,
        "amp_scale": 128.0,
        "amp_scale_min": 128.0,
        "amp_scale_max": 128.0,
        "local_forward_calls": 0,
        "local_forward_call_histogram": {"1": 0, "2": 0},
        "local_bn_preserved_batches": 0,
        "auxiliary_non_tiny_pair_count": 0,
        "internal_validation_bypass_count": 1,
        "test_loader_is_none": True,
        "source_unchanged": True,
        "data_unchanged": True,
        "initial_state_unchanged": True,
    }
    labels = {
        "completed_epochs": "completed epoch count drift",
        "successful_batches": "successful batch count drift",
        "optimizer_attempts": "optimizer attempt count drift",
        "internal_validation_bypass_count": (
            "internal validation count drift"
        ),
        "test_loader_is_none": "test loader was constructed",
        "source_unchanged": "source changed during training",
        "data_unchanged": "data changed during training",
        "initial_state_unchanged": "initial state changed during training",
    }
    amp_keys = {"amp", "amp_scale", "amp_scale_min", "amp_scale_max"}
    for key, expected in exact_values.items():
        if summary.get(key) != expected:
            if key in amp_keys:
                label = "AMP scale drift"
            else:
                label = labels.get(key, f"{key} drift")
            if label not in failures:
                failures.append(label)
    if summary.get("observed_tensor_batch_sizes") != [7, 8]:
        failures.append("observed tensor batch-size drift")
    if summary.get("loader") != {
        "trainer_batch_size": 8,
        "per_rank_batch_size": 8,
        "loader_batch_size": 8,
        "loader_num_workers": 8,
    }:
        failures.append("loader contract drift")
    if summary.get("optimizer") != FROZEN_OPTIMIZER_OBSERVATION:
        failures.append("optimizer observation drift")
    canaries = summary.get("batch_canaries")
    positions = (
        [
            (record.get("epoch"), record.get("batch"))
            for record in canaries
        ]
        if isinstance(canaries, list)
        and all(isinstance(record, dict) for record in canaries)
        else []
    )
    if positions != [(0, 1), (0, 2), (1, 810)]:
        failures.append("batch canary position drift")
    if not isinstance(canaries, list) or not all(
        _valid_sha(record.get("sha256"))
        for record in canaries
        if isinstance(record, dict)
    ) or len(canaries) != 3:
        failures.append("batch canary digest drift")
    checkpoint = summary.get("checkpoint", {})
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("kind") != "last.pt"
        or not _valid_sha(checkpoint.get("sha256"))
        or checkpoint.get("loadable") is not True
    ):
        failures.append("last checkpoint is invalid")
    return failures


def _run(args, protocol: dict) -> None:
    from src.rtdetr_tascv import (
        MATCHED_AMP_SCALE,
        TASCVControlTrainer,
    )

    source_before = source_closure(ROOT)
    upstream_before = current_upstream_source_hashes()
    data_before = sha256_file(Path(args.data).resolve())
    initial_before = sha256_file(Path(args.initial_state).resolve())
    trainer = TASCVControlTrainer(
        overrides=build_settings(args),
        stage=TASCVStage.FORMAL_100,
        initial_state_path=args.initial_state,
    )
    expected_target = Path(
        protocol["endpoint"]["target_dir"]
    ).resolve()
    if Path(trainer.save_dir).resolve() != expected_target:
        raise RuntimeError("SADED_STOCK_SAVE_DIR_ENDPOINT_DRIFT")
    if Path(trainer.last).resolve() != expected_target / "weights/last.pt":
        raise RuntimeError("SADED_STOCK_LAST_CHECKPOINT_ENDPOINT_DRIFT")

    def batch_end(current) -> None:
        current.record_successful_batch()

    trainer.add_callback("on_train_batch_end", batch_end)
    trainer.train()

    checkpoint_path = Path(trainer.last).resolve()
    checkpoint_loadable = False
    if checkpoint_path.is_file():
        try:
            loaded = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            checkpoint_loadable = isinstance(loaded, dict)
            del loaded
        except Exception:
            checkpoint_loadable = False
    source_after = source_closure(ROOT)
    upstream_after = current_upstream_source_hashes()
    data_after = sha256_file(Path(args.data).resolve())
    initial_after = sha256_file(Path(args.initial_state).resolve())
    summary = {
        "schema_version": "saded-stock-training-summary/v1",
        "stage": "FORMAL_100",
        "arm": "stock_control",
        "seed": args.seed,
        "completed_epochs": int(trainer.epoch) + 1,
        "protocol_manifest": Path(
            args.protocol_manifest
        ).resolve().as_posix(),
        "protocol_manifest_sha256": sha256_file(
            Path(args.protocol_manifest).resolve()
        ),
        "protocol_source_commit": protocol["runtime_source"]["commit"],
        "source_repo_bundle_sha256": protocol["runtime_source"][
            "repo_bundle_sha256"
        ],
        "source_upstream_bundle_sha256": protocol["runtime_source"][
            "upstream_bundle_sha256"
        ],
        "initial_state": Path(args.initial_state).resolve().as_posix(),
        "initial_state_sha256": initial_after,
        "initial_state_common_fingerprint": protocol["initial_state"][
            "common_fingerprint"
        ],
        "data": Path(args.data).resolve().as_posix(),
        "data_sha256": data_after,
        "batch": int(trainer.batch_size),
        "workers": int(trainer.args.workers),
        "successful_batches": trainer.tascv_successful_batches,
        "optimizer_attempts": trainer.tascv_optimizer_attempts,
        "expected_successful_batches": (
            trainer.tascv_policy.expected_successful_batches
        ),
        "expected_optimizer_attempts": (
            trainer.tascv_policy.expected_optimizer_attempts
        ),
        "observed_tensor_batch_sizes": sorted(
            trainer.tascv_observed_tensor_batch_sizes
        ),
        "batch_canaries": trainer.tascv_batch_canaries,
        "amp": True,
        "amp_scale": MATCHED_AMP_SCALE,
        "amp_scale_min": trainer.tascv_amp_scale_min,
        "amp_scale_max": trainer.tascv_amp_scale_max,
        "loader": trainer.tascv_loader_observation,
        "optimizer": trainer.tascv_optimizer_observation,
        "local_forward_calls": 0,
        "local_forward_call_histogram": {"1": 0, "2": 0},
        "local_bn_preserved_batches": 0,
        "auxiliary_non_tiny_pair_count": 0,
        "internal_validation_bypass_count": (
            trainer.internal_validation_bypass_count
        ),
        "test_loader_is_none": trainer.test_loader is None,
        "source_unchanged": (
            source_before == source_after
            == protocol["runtime_source"]["repo_files"]
            and upstream_before == upstream_after
            == EXPECTED_UPSTREAM_SOURCE_SHA256
        ),
        "data_unchanged": data_before == data_after,
        "initial_state_unchanged": initial_before == initial_after,
        "hardware": {
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "device": str(trainer.device),
        },
        "checkpoint": {
            "kind": "last.pt",
            "path": checkpoint_path.as_posix(),
            "sha256": (
                sha256_file(checkpoint_path)
                if checkpoint_path.is_file()
                else ""
            ),
            "loadable": checkpoint_loadable,
        },
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
    failures = validate_runtime_summary(summary)
    output_dir = Path(trainer.save_dir)
    if failures:
        _atomic_json(
            output_dir / "runtime_invalid.json",
            {
                **summary,
                "schema_version": "saded-stock-runtime-invalid/v1",
                "decision": "INVALID",
                "failures": failures,
            },
        )
        raise RuntimeError(
            "SADED_STOCK_RUNTIME_INVALID: " + "; ".join(failures)
        )
    _atomic_json(
        output_dir / "saded_stock_training_summary.json",
        summary,
    )


def main() -> None:
    args = build_parser().parse_args()
    protocol = validate_protocol_inputs(args)
    target = Path(args.project).resolve() / args.name
    try:
        _run(args, protocol)
    except BaseException as error:
        normalized = target.as_posix().lower()
        if "test-dev" not in normalized and "test_dev" not in normalized:
            invalid_path = target / "runtime_invalid.json"
            if not invalid_path.exists():
                try:
                    _atomic_json(
                        invalid_path,
                        {
                            "schema_version": (
                                "saded-stock-runtime-invalid/v1"
                            ),
                            "decision": "INVALID",
                            "stage": "FORMAL_100",
                            "arm": "stock_control",
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
