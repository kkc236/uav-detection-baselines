"""Independent runtime adjudication for T-ASCV staged training."""

from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path
import subprocess

from src.tascv_protocol import (
    APPROVED_TASCV_PARENT,
    EXPECTED_UPSTREAM_SOURCE_SHA256,
    FROZEN_OPTIMIZER_OBSERVATION,
    FROZEN_STAGE_CONTRACT,
    PROTOCOL_VERSION,
    reject_forbidden_path,
    repo_source_hashes,
    require_clean_repo,
    sha256_file,
    source_bundle_sha256,
    validate_r0_closure,
    validate_runtime_manifest,
)


_PAIR_FIELDS = (
    "protocol_manifest_sha256",
    "protocol_source_commit",
    "source_repo_bundle_sha256",
    "source_upstream_bundle_sha256",
    "approved_tascv_parent",
    "r0_evaluation_anchor_sha256",
    "control_slot",
    "initial_state_sha256",
    "initial_state_common_fingerprint",
    "data_sha256",
    "subset_binding",
    "seed",
    "batch_canaries",
)


def expected_artifact_paths(summary: dict) -> dict[str, Path]:
    manifest_path = reject_forbidden_path(
        summary.get("protocol_manifest", ""),
        context="T-ASCV endpoint manifest",
    )
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    stage = summary.get("stage")
    seed = summary.get("seed")
    if summary.get("arm") == "control":
        slot = manifest["control_allowlist"]["slots"][
            f"B:{stage}:{seed}"
        ]
        if slot.get("resolution") != "RUN_FRESH":
            raise ValueError("bound control has no fresh training endpoint")
        target_dir = reject_forbidden_path(
            slot["fresh_target"]["target_dir"],
            context="T-ASCV control endpoint",
        )
    elif summary.get("arm") == "tascv":
        target_dir = reject_forbidden_path(
            manifest["treatment_endpoints"][
                f"T:{stage}:{seed}"
            ]["target_dir"],
            context="T-ASCV treatment endpoint",
        )
    else:
        raise ValueError("unknown T-ASCV summary arm")
    return {
        "summary": target_dir / "tascv_training_summary.json",
        "checkpoint": target_dir / "weights/last.pt",
        "records": target_dir / "tascv_mechanism_records.jsonl",
    }


def _authority_failures(summary: dict) -> list[str]:
    try:
        manifest_path = reject_forbidden_path(
            summary.get("protocol_manifest", ""),
            context="T-ASCV adjudication manifest",
        )
        manifest, manifest_sha = validate_runtime_manifest(
            manifest_path
        )
        if manifest_sha != summary.get("protocol_manifest_sha256"):
            raise ValueError("manifest checksum")
        if manifest.get("schema_version") != PROTOCOL_VERSION:
            raise ValueError("manifest schema")
        source = manifest.get("runtime_source", {})
        if (
            summary.get("protocol_source_commit")
            != source.get("commit")
            or summary.get("source_repo_bundle_sha256")
            != source.get("repo_bundle_sha256")
            or summary.get("source_upstream_bundle_sha256")
            != source.get("upstream_bundle_sha256")
            or source.get("upstream") != EXPECTED_UPSTREAM_SOURCE_SHA256
            or source_bundle_sha256(source.get("upstream", {}))
            != source.get("upstream_bundle_sha256")
            or manifest.get("approved_tascv_parent")
            != APPROVED_TASCV_PARENT
            or summary.get("approved_tascv_parent")
            != APPROVED_TASCV_PARENT
        ):
            raise ValueError("source/parent binding")
        repo_root = Path(__file__).resolve().parents[1]
        require_clean_repo(repo_root)
        hashes = repo_source_hashes(repo_root)
        if (
            hashes != source.get("repo_files")
            or source_bundle_sha256(hashes)
            != source.get("repo_bundle_sha256")
        ):
            raise ValueError("repo source closure")
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if commit != source.get("commit"):
            raise ValueError("runtime commit")
        validate_r0_closure(manifest.get("r0_authority", {}))
        if (
            summary.get("r0_evaluation_anchor_sha256")
            != manifest["r0_authority"][
                "evaluation_anchor_sha256"
            ]
        ):
            raise ValueError("R0 summary binding")
        allowlist_record = manifest.get("control_allowlist", {})
        allowlist_path = reject_forbidden_path(
            allowlist_record.get("path", ""),
            context="T-ASCV adjudication allowlist",
        )
        if (
            not allowlist_path.is_file()
            or sha256_file(allowlist_path)
            != allowlist_record.get("sha256")
        ):
            raise ValueError("control allowlist checksum")
        allowlist = json.loads(
            allowlist_path.read_text(encoding="utf-8")
        )
        if allowlist.get("slots") != allowlist_record.get("slots"):
            raise ValueError("control allowlist content")
        initial_path = reject_forbidden_path(
            summary.get("initial_state", ""),
            context="T-ASCV adjudication initial state",
        )
        data_path = reject_forbidden_path(
            summary.get("data", ""),
            context="T-ASCV adjudication data",
        )
        if (
            not initial_path.is_file()
            or sha256_file(initial_path)
            != summary.get("initial_state_sha256")
            or not data_path.is_file()
            or sha256_file(data_path) != summary.get("data_sha256")
        ):
            raise ValueError("initial/data closure")
        seed = str(summary.get("seed"))
        expected_initial = manifest["initial_states"][seed]
        if (
            summary.get("initial_state_sha256")
            != expected_initial["sha256"]
            or summary.get("initial_state_common_fingerprint")
            != expected_initial["common_fingerprint"]
            or summary.get("subset_binding")
            != {
                key: manifest["subset"][key]
                for key in (
                    "count",
                    "semantic_sha256",
                    "file_sha256",
                )
            }
        ):
            raise ValueError("initial/subset manifest binding")
        slot_key = (
            f"B:{summary.get('stage')}:{summary.get('seed')}"
        )
        if summary.get("control_slot") != allowlist_record[
            "slots"
        ].get(slot_key):
            raise ValueError("control slot binding")
        if summary.get("arm") == "control":
            endpoint = allowlist_record["slots"][slot_key]
            if endpoint.get("resolution") != "RUN_FRESH":
                raise ValueError("non-fresh control training summary")
            target_dir = Path(
                endpoint["fresh_target"]["target_dir"]
            ).resolve()
        else:
            treatment_key = (
                f"T:{summary.get('stage')}:{summary.get('seed')}"
            )
            target_dir = Path(
                manifest["treatment_endpoints"][treatment_key][
                    "target_dir"
                ]
            ).resolve()
        checkpoint_path = reject_forbidden_path(
            summary.get("checkpoint", {}).get("path", ""),
            context="T-ASCV adjudication checkpoint",
        )
        if checkpoint_path != target_dir / "weights/last.pt":
            raise ValueError("fixed last.pt endpoint drift")
    except (
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as error:
        return [f"authority:{type(error).__name__}:{error}"]
    return []


def _runtime_failures(summary: dict, arm: str) -> list[str]:
    failures: list[str] = _authority_failures(summary)
    exact = {
        "schema_version": "tascv-training-summary/v1",
        "stage": "PREFLIGHT_1",
        "arm": arm,
        "seed": 0,
        "batch": 8,
        "observed_tensor_batch_sizes": [8],
        "amp": True,
        "amp_scale": 128.0,
        "amp_scale_min": 128.0,
        "amp_scale_max": 128.0,
        "successful_batches": 1,
        "optimizer_attempts": 1,
        "expected_successful_batches": 1,
        "expected_optimizer_attempts": 1,
        "workers": 8,
        "test_loader_is_none": True,
        "auxiliary_non_tiny_pair_count": 0,
        "internal_validation_bypass_count": 1,
    }
    for key, expected in exact.items():
        if summary.get(key) != expected:
            failures.append(f"{arm}:{key}")
    loader = summary.get("loader")
    if loader != {
        "trainer_batch_size": 8,
        "per_rank_batch_size": 8,
        "loader_batch_size": 8,
        "loader_num_workers": 8,
    }:
        failures.append(f"{arm}:loader")
    if summary.get("optimizer") != FROZEN_OPTIMIZER_OBSERVATION:
        failures.append(f"{arm}:optimizer")
    canaries = summary.get("batch_canaries")
    if (
        not isinstance(canaries, list)
        or len(canaries) != 1
        or canaries[0].get("epoch") != 0
        or canaries[0].get("batch") != 1
        or not isinstance(canaries[0].get("sha256"), str)
        or len(canaries[0]["sha256"]) != 64
    ):
        failures.append(f"{arm}:batch_canaries")
    expected_local = 2 if arm == "tascv" else 0
    if summary.get("local_forward_calls") != expected_local:
        failures.append(f"{arm}:local_forward_calls")
    expected_histogram = (
        {"1": 0, "2": 1}
        if arm == "tascv"
        else {"1": 0, "2": 0}
    )
    if summary.get("local_forward_call_histogram") != expected_histogram:
        failures.append(f"{arm}:local_forward_call_histogram")
    if summary.get("local_bn_preserved_batches") != (
        1 if arm == "tascv" else 0
    ):
        failures.append(f"{arm}:local_bn_preserved_batches")
    checkpoint = summary.get("checkpoint")
    if not isinstance(checkpoint, dict):
        failures.append(f"{arm}:checkpoint")
    else:
        try:
            path = Path(checkpoint["path"]).resolve()
            if (
                checkpoint.get("kind") != "last.pt"
                or not path.is_file()
                or sha256_file(path) != checkpoint.get("sha256")
            ):
                failures.append(f"{arm}:checkpoint")
        except (KeyError, OSError, TypeError, ValueError):
            failures.append(f"{arm}:checkpoint")
    return failures


def adjudicate_preflight(summaries: dict[str, dict]) -> dict:
    failures: list[str] = []
    if set(summaries) != {"control", "tascv"}:
        failures.append("preflight_requires_control_and_tascv")
        return {
            "schema_version": "tascv-preflight-adjudication/v1",
            "decision": "INVALID",
            "failures": failures,
        }
    control = summaries["control"]
    treatment = summaries["tascv"]
    failures.extend(_runtime_failures(control, "control"))
    failures.extend(_runtime_failures(treatment, "tascv"))
    for field in _PAIR_FIELDS:
        if control.get(field) != treatment.get(field):
            failures.append(f"pair:{field}")
    return {
        "schema_version": "tascv-preflight-adjudication/v1",
        "decision": "INVALID" if failures else "TASCV_PREFLIGHT_GO",
        "failures": failures,
        "protocol_manifest_sha256": control.get(
            "protocol_manifest_sha256"
        ),
        "protocol_source_commit": control.get(
            "protocol_source_commit"
        ),
        "summaries": {
            arm: {
                "checkpoint": summary.get("checkpoint"),
            }
            for arm, summary in summaries.items()
        },
    }


def validate_preflight_control_summary(summary: dict) -> None:
    failures = _runtime_failures(summary, "control")
    if failures:
        raise ValueError(
            "T-ASCV fresh preflight control is invalid: "
            + ", ".join(failures)
        )


def validate_paired_control_summary(
    summary: dict,
    *,
    stage: str,
    seed: int,
) -> None:
    if stage == "PREFLIGHT_1":
        validate_preflight_control_summary(summary)
        return
    failures = _authority_failures(summary)
    contract = FROZEN_STAGE_CONTRACT[stage]
    exact = {
        "schema_version": "tascv-training-summary/v1",
        "stage": stage,
        "arm": "control",
        "seed": seed,
        "batch": 8,
        "observed_tensor_batch_sizes": contract[
            "allowed_observed_tensor_batch_sizes"
        ],
        "amp": True,
        "amp_scale": 128.0,
        "amp_scale_min": 128.0,
        "amp_scale_max": 128.0,
        "successful_batches": contract[
            "expected_successful_batches"
        ],
        "optimizer_attempts": contract[
            "expected_optimizer_attempts"
        ],
        "expected_successful_batches": contract[
            "expected_successful_batches"
        ],
        "expected_optimizer_attempts": contract[
            "expected_optimizer_attempts"
        ],
        "workers": 8,
        "test_loader_is_none": True,
        "auxiliary_non_tiny_pair_count": 0,
        "local_forward_calls": 0,
        "local_forward_call_histogram": {"1": 0, "2": 0},
        "local_bn_preserved_batches": 0,
        "internal_validation_bypass_count": 1,
    }
    for key, expected in exact.items():
        if summary.get(key) != expected:
            failures.append(f"control:{key}")
    if summary.get("loader") != {
        "trainer_batch_size": 8,
        "per_rank_batch_size": 8,
        "loader_batch_size": 8,
        "loader_num_workers": 8,
    }:
        failures.append("control:loader")
    if summary.get("optimizer") != FROZEN_OPTIMIZER_OBSERVATION:
        failures.append("control:optimizer")
    if failures:
        raise ValueError(
            "T-ASCV paired stock control is invalid: "
            + ", ".join(failures)
        )


def replay_preflight_gate(gate: dict) -> dict:
    if (
        gate.get("schema_version")
        != "tascv-preflight-adjudication/v1"
        or gate.get("decision") != "TASCV_PREFLIGHT_GO"
    ):
        raise ValueError("not a T-ASCV preflight GO gate")
    bindings = gate.get("summary_bindings")
    if not isinstance(bindings, dict) or set(bindings) != {
        "control",
        "tascv",
    }:
        raise ValueError("T-ASCV preflight summary bindings drift")
    summaries: dict[str, dict] = {}
    for arm, binding in bindings.items():
        if not isinstance(binding, dict):
            raise ValueError("T-ASCV preflight summary binding invalid")
        path = reject_forbidden_path(
            binding.get("path", ""),
            context="T-ASCV preflight replay summary",
        )
        if (
            not path.is_file()
            or sha256_file(path) != binding.get("sha256")
        ):
            raise ValueError("T-ASCV preflight summary checksum drift")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("arm") != arm:
            raise ValueError("T-ASCV preflight summary arm drift")
        if path != expected_artifact_paths(summary)["summary"]:
            raise ValueError("T-ASCV preflight summary endpoint drift")
        summaries[arm] = summary
    replayed = adjudicate_preflight(summaries)
    for key, value in replayed.items():
        if gate.get(key) != value:
            raise ValueError("T-ASCV preflight gate replay drift")
    return gate


def _mechanism_runtime_failures(summary: dict) -> list[str]:
    failures: list[str] = _authority_failures(summary)
    exact = {
        "schema_version": "tascv-training-summary/v1",
        "stage": "TINY_MECHANISM_500",
        "arm": "tascv",
        "seed": 1,
        "batch": 8,
        "observed_tensor_batch_sizes": [7, 8],
        "amp": True,
        "amp_scale": 128.0,
        "amp_scale_min": 128.0,
        "amp_scale_max": 128.0,
        "successful_batches": 500,
        "optimizer_attempts": 106,
        "expected_successful_batches": 500,
        "expected_optimizer_attempts": 106,
        "workers": 8,
        "test_loader_is_none": True,
        "auxiliary_non_tiny_pair_count": 0,
        "internal_validation_bypass_count": 1,
    }
    for key, expected in exact.items():
        if summary.get(key) != expected:
            failures.append(f"runtime:{key}")
    if summary.get("loader") != {
        "trainer_batch_size": 8,
        "per_rank_batch_size": 8,
        "loader_batch_size": 8,
        "loader_num_workers": 8,
    }:
        failures.append("runtime:loader")
    if summary.get("optimizer") != FROZEN_OPTIMIZER_OBSERVATION:
        failures.append("runtime:optimizer")
    histogram = summary.get("local_forward_call_histogram")
    if (
        not isinstance(histogram, dict)
        or set(histogram) != {"1", "2"}
        or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < 0
            for value in histogram.values()
        )
        or sum(histogram.values()) != 500
        or summary.get("local_forward_calls")
        != histogram["1"] + 2 * histogram["2"]
    ):
        failures.append("runtime:local_forward_histogram")
    if summary.get("local_bn_preserved_batches") != 500:
        failures.append("runtime:local_bn_preserved_batches")
    canaries = summary.get("batch_canaries")
    if (
        not isinstance(canaries, list)
        or len(canaries) != 3
        or [
            (record.get("epoch"), record.get("batch"))
            for record in canaries
        ]
        != [(0, 1), (0, 2), (1, 82)]
        or any(
            not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            for record in canaries
        )
    ):
        failures.append("runtime:batch_canaries")
    checkpoint = summary.get("checkpoint")
    if not isinstance(checkpoint, dict):
        failures.append("runtime:checkpoint")
    else:
        try:
            path = Path(checkpoint["path"]).resolve()
            if (
                checkpoint.get("kind") != "last.pt"
                or not path.is_file()
                or sha256_file(path) != checkpoint.get("sha256")
            ):
                failures.append("runtime:checkpoint")
        except (KeyError, OSError, TypeError, ValueError):
            failures.append("runtime:checkpoint")
    records = summary.get("mechanism_records")
    if (
        not isinstance(records, dict)
        or records.get("count") != 500
        or not isinstance(records.get("path"), str)
        or not isinstance(records.get("sha256"), str)
        or len(records["sha256"]) != 64
    ):
        failures.append("runtime:mechanism_records")
    return failures


def _strict_record_count(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 0
    ):
        raise ValueError("invalid T-ASCV mechanism record count")
    return int(value)


def adjudicate_mechanism(
    summary: dict,
    records: list[dict],
) -> dict:
    invalid = _mechanism_runtime_failures(summary)
    expected_keys = {
        "batch",
        "matched_pairs",
        "auxiliary_tiny_pairs",
        "excluded_non_tiny_pairs",
        "auxiliary_non_tiny_pairs",
        "tiny_teacher_advantage_sum",
        "tiny_teacher_win_count",
    }
    normalized: list[dict[str, int | float]] = []
    if not isinstance(records, list) or len(records) != 500:
        invalid.append("records:count")
    else:
        for expected_batch, record in enumerate(records, start=1):
            try:
                if not isinstance(record, dict) or set(record) != expected_keys:
                    raise ValueError
                batch = _strict_record_count(record["batch"])
                matched = _strict_record_count(record["matched_pairs"])
                tiny = _strict_record_count(
                    record["auxiliary_tiny_pairs"]
                )
                excluded = _strict_record_count(
                    record["excluded_non_tiny_pairs"]
                )
                non_tiny = _strict_record_count(
                    record["auxiliary_non_tiny_pairs"]
                )
                wins = _strict_record_count(
                    record["tiny_teacher_win_count"]
                )
                advantage_value = record[
                    "tiny_teacher_advantage_sum"
                ]
                if (
                    isinstance(advantage_value, bool)
                    or not isinstance(advantage_value, Real)
                ):
                    raise ValueError
                advantage = float(advantage_value)
                if (
                    batch != expected_batch
                    or matched != tiny + excluded
                    or non_tiny != 0
                    or wins > tiny
                    or not math.isfinite(advantage)
                    or (tiny == 0 and (wins != 0 or advantage != 0))
                ):
                    raise ValueError
                normalized.append(
                    {
                        "batch": batch,
                        "matched": matched,
                        "tiny": tiny,
                        "excluded": excluded,
                        "advantage": advantage,
                        "wins": wins,
                    }
                )
            except (KeyError, TypeError, ValueError):
                invalid.append(f"records:batch{expected_batch}")
                break
    if invalid:
        return {
            "schema_version": "tascv-mechanism-adjudication/v1",
            "decision": "INVALID",
            "failures": invalid,
        }

    def summarize(items: list[dict[str, int | float]]) -> dict:
        tiny_count = sum(int(item["tiny"]) for item in items)
        advantage_total = sum(
            float(item["advantage"]) for item in items
        )
        win_count = sum(int(item["wins"]) for item in items)
        return {
            "batches": len(items),
            "matched_pairs": sum(
                int(item["matched"]) for item in items
            ),
            "tiny_pairs": tiny_count,
            "excluded_non_tiny_pairs": sum(
                int(item["excluded"]) for item in items
            ),
            "auxiliary_non_tiny_pairs": 0,
            "tiny_batches_with_pairs": sum(
                int(item["tiny"]) > 0 for item in items
            ),
            "tiny_teacher_advantage_mean": (
                advantage_total / tiny_count
                if tiny_count
                else None
            ),
            "tiny_teacher_win_rate": (
                win_count / tiny_count if tiny_count else None
            ),
        }

    tail = normalized[400:500]
    all_summary = summarize(normalized)
    tail_summary = summarize(tail)
    replayed_summary = {
        "all": all_summary,
        "tail": tail_summary,
        "tail_window": [401, 500],
    }
    if summary.get("mechanism_summary") != replayed_summary:
        return {
            "schema_version": "tascv-mechanism-adjudication/v1",
            "decision": "INVALID",
            "failures": ["evidence:mechanism_summary_drift"],
        }
    tiny_pairs = int(tail_summary["tiny_pairs"])
    tiny_batches = int(tail_summary["tiny_batches_with_pairs"])
    failures: list[str] = []
    if tiny_pairs < 100:
        failures.append("tail_tiny_pairs<100")
    if tiny_batches < 80:
        failures.append("tail_tiny_batches_with_pairs<80")
    if tiny_pairs:
        if float(tail_summary["tiny_teacher_advantage_mean"]) <= 0:
            failures.append("tiny_teacher_advantage_mean<=0")
        if float(tail_summary["tiny_teacher_win_rate"]) <= 0.5:
            failures.append("tiny_teacher_win_rate<=0.5")
    return {
        "schema_version": "tascv-mechanism-adjudication/v1",
        "decision": (
            "TASCV_STOP" if failures else "TASCV_MECHANISM_GO"
        ),
        "failures": failures,
        "protocol_manifest_sha256": summary.get(
            "protocol_manifest_sha256"
        ),
        "protocol_source_commit": summary.get(
            "protocol_source_commit"
        ),
        "tail_window": [401, 500],
        "tail": tail_summary,
    }


def replay_mechanism_gate(gate: dict) -> dict:
    if (
        gate.get("schema_version")
        != "tascv-mechanism-adjudication/v1"
        or gate.get("decision") != "TASCV_MECHANISM_GO"
    ):
        raise ValueError("not a T-ASCV mechanism GO gate")
    bindings = gate.get("summary_bindings")
    if not isinstance(bindings, dict) or set(bindings) != {"tascv"}:
        raise ValueError("T-ASCV mechanism summary binding drift")
    summary_binding = bindings["tascv"]
    summary_path = reject_forbidden_path(
        summary_binding.get("path", ""),
        context="T-ASCV mechanism replay summary",
    )
    if (
        not summary_path.is_file()
        or sha256_file(summary_path)
        != summary_binding.get("sha256")
    ):
        raise ValueError("T-ASCV mechanism summary checksum drift")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("stage") != "TINY_MECHANISM_500"
        or summary.get("arm") != "tascv"
    ):
        raise ValueError("T-ASCV mechanism summary identity drift")
    expected_paths = expected_artifact_paths(summary)
    if summary_path != expected_paths["summary"]:
        raise ValueError("T-ASCV mechanism summary endpoint drift")
    predecessor = summary.get("predecessor_evidence", {})
    predecessor_path = reject_forbidden_path(
        predecessor.get("path", ""),
        context="T-ASCV mechanism replay predecessor",
    )
    if (
        not predecessor_path.is_file()
        or sha256_file(predecessor_path)
        != predecessor.get("sha256")
    ):
        raise ValueError("T-ASCV mechanism predecessor checksum drift")
    replayed_preflight = replay_preflight_gate(
        json.loads(predecessor_path.read_text(encoding="utf-8"))
    )
    if (
        replayed_preflight.get("protocol_manifest_sha256")
        != summary.get("protocol_manifest_sha256")
        or replayed_preflight.get("protocol_source_commit")
        != summary.get("protocol_source_commit")
    ):
        raise ValueError("T-ASCV mechanism predecessor source drift")
    record_binding = gate.get("mechanism_records_binding")
    if (
        not isinstance(record_binding, dict)
        or record_binding != summary.get("mechanism_records")
    ):
        raise ValueError("T-ASCV mechanism record binding drift")
    records_path = reject_forbidden_path(
        record_binding.get("path", ""),
        context="T-ASCV mechanism replay records",
    )
    if (
        not records_path.is_file()
        or sha256_file(records_path)
        != record_binding.get("sha256")
        or records_path != expected_paths["records"]
    ):
        raise ValueError("T-ASCV mechanism record checksum drift")
    records = [
        json.loads(line)
        for line in records_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    replayed = adjudicate_mechanism(summary, records)
    for key, value in replayed.items():
        if gate.get(key) != value:
            raise ValueError("T-ASCV mechanism gate replay drift")
    return gate


__all__ = [
    "adjudicate_mechanism",
    "adjudicate_preflight",
    "expected_artifact_paths",
    "replay_mechanism_gate",
    "replay_preflight_gate",
    "validate_paired_control_summary",
    "validate_preflight_control_summary",
]
