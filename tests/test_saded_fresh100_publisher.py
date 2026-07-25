from __future__ import annotations

import os

import psutil
import pytest

from scripts.publish_saded_fresh100 import (
    PublicationConflictError,
    ScientificValidationError,
    discard_terminal_staging,
    build_terminal_manifest,
    claim_immutable_record,
    classify_lock_owner,
    classify_terminal_state,
    dirty_paths_are_allowed,
    ensure_no_opposite_terminal,
    is_release_not_found,
    load_json_object,
    release_tag_for_state,
    sha256_file,
    terminal_directory_name,
    validate_terminal_facts,
    validate_summary_bindings,
    validate_success_candidate,
    validate_process_baseline,
    verify_checksum_closure,
    verify_release_assets,
    verify_release_identity,
)


def test_complete_zero_is_success() -> None:
    assert (
        classify_terminal_state("TRAIN_COMPLETE", "0")
        == "SUCCESS_CANDIDATE"
    )


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [
        ("TRAIN_INVALID", "1"),
        ("TRAIN_COMPLETE", "7"),
        ("RUNNING", "9"),
    ],
)
def test_invalid_or_nonzero_is_invalid(
    status: str,
    exit_code: str,
) -> None:
    assert classify_terminal_state(status, exit_code) == "INVALID"


def test_running_is_not_terminal() -> None:
    assert classify_terminal_state("RUNNING", None) is None


def test_invalid_manifest_cannot_claim_success() -> None:
    manifest = build_terminal_manifest(
        run_id="final-saded-fresh100-c5c35374",
        terminal_state="INVALID",
        exit_code="9",
        artifacts={"train.log": "ABC"},
        validation_passed=False,
    )

    assert manifest["terminal_state"] == "INVALID"
    assert manifest["publish_as_success"] is False
    assert manifest["artifacts"] == {"train.log": "ABC"}


def test_manifest_rejects_nonterminal_state() -> None:
    with pytest.raises(ValueError, match="SUCCESS or INVALID"):
        build_terminal_manifest(
            run_id="final-saded-fresh100-c5c35374",
            terminal_state="RUNNING",
            exit_code=None,
            artifacts={},
            validation_passed=False,
        )


def test_success_manifest_requires_validation_proof() -> None:
    with pytest.raises(ValueError, match="validated"):
        build_terminal_manifest(
            run_id="final-saded-fresh100-c5c35374",
            terminal_state="SUCCESS",
            exit_code="0",
            artifacts={},
            validation_passed=False,
        )


def test_invalid_manifest_cannot_claim_validation_passed() -> None:
    with pytest.raises(ValueError, match="INVALID evidence"):
        build_terminal_manifest(
            run_id="final-saded-fresh100-c5c35374",
            terminal_state="INVALID",
            exit_code="1",
            artifacts={},
            validation_passed=True,
        )


def test_terminal_evidence_does_not_overwrite_progress_snapshot() -> None:
    assert terminal_directory_name("SUCCESS") == "terminal"
    assert terminal_directory_name("INVALID") == "invalid"


def test_failure_release_tag_is_explicitly_invalid() -> None:
    base = "saded-fresh100-seed0-c5c35374"
    assert release_tag_for_state(base, "SUCCESS") == base
    assert (
        release_tag_for_state(base, "INVALID")
        == "saded-fresh100-seed0-c5c35374-invalid"
    )


def test_success_candidate_rebinds_checkpoint_for_runtime_validator(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    protocol = tmp_path / "protocol_manifest.json"
    protocol.write_bytes(b"protocol")
    seen = {}

    def validator(summary):
        seen["checkpoint"] = summary["checkpoint"]
        return []

    summary = {
        "checkpoint": {
            "path": "/remote/last.pt",
            "expected_path": "/remote/last.pt",
        }
    }
    result = validate_success_candidate(
        summary=summary,
        checkpoint=checkpoint,
        protocol=protocol,
        expected_protocol_sha256=(
            "2EA88C7A30351B12A4DCFC06CDCE2AF6EAB18416176466C2500CB6EF74F745BF"
        ),
        runtime_validator=validator,
    )

    assert result["passed"] is True
    assert seen["checkpoint"]["path"] == checkpoint.as_posix()
    assert seen["checkpoint"]["expected_path"] == checkpoint.as_posix()
    assert summary["checkpoint"]["path"] == "/remote/last.pt"


def test_success_candidate_rejects_runtime_failure(tmp_path) -> None:
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    protocol = tmp_path / "protocol_manifest.json"
    protocol.write_bytes(b"protocol")

    with pytest.raises(RuntimeError, match="runtime validation failed"):
        validate_success_candidate(
            summary={"checkpoint": {}},
            checkpoint=checkpoint,
            protocol=protocol,
            expected_protocol_sha256=(
                "2EA88C7A30351B12A4DCFC06CDCE2AF6EAB18416176466C2500CB6EF74F745BF"
            ),
            runtime_validator=lambda _: ["batch drift"],
        )


def test_discard_terminal_staging_preserves_progress(tmp_path) -> None:
    progress = tmp_path / "progress"
    progress.mkdir()
    (progress / "latest.json").write_text("{}")
    staging = tmp_path / ".terminal-staging"
    staging.mkdir()
    (staging / "partial.pt").write_bytes(b"partial")

    discard_terminal_staging(tmp_path, "SUCCESS")

    assert not staging.exists()
    assert (progress / "latest.json").read_text() == "{}"


def test_summary_bindings_match_protocol() -> None:
    protocol = {
        "runtime_source": {
            "commit": "commit",
            "repo_bundle_sha256": "repo",
            "upstream_bundle_sha256": "upstream",
        },
        "data": {"sha256": "data"},
        "initial_state": {
            "sha256": "initial",
            "common_fingerprint": "fingerprint",
        },
    }
    summary = {
        "protocol_source_commit": "commit",
        "source_repo_bundle_sha256": "repo",
        "source_upstream_bundle_sha256": "upstream",
        "data_sha256": "data",
        "initial_state_sha256": "initial",
        "initial_state_common_fingerprint": "fingerprint",
    }

    assert validate_summary_bindings(summary, protocol)["passed"] is True


def test_summary_bindings_reject_data_drift() -> None:
    protocol = {
        "runtime_source": {
            "commit": "commit",
            "repo_bundle_sha256": "repo",
            "upstream_bundle_sha256": "upstream",
        },
        "data": {"sha256": "data"},
        "initial_state": {
            "sha256": "initial",
            "common_fingerprint": "fingerprint",
        },
    }
    summary = {
        "protocol_source_commit": "commit",
        "source_repo_bundle_sha256": "repo",
        "source_upstream_bundle_sha256": "upstream",
        "data_sha256": "wrong",
        "initial_state_sha256": "initial",
        "initial_state_common_fingerprint": "fingerprint",
    }

    with pytest.raises(RuntimeError, match="data_sha256"):
        validate_summary_bindings(summary, protocol)


def test_lock_probe_is_read_only_and_checks_process_identity() -> None:
    process = psutil.Process(os.getpid())
    record = {
        "pid": process.pid,
        "create_time": process.create_time(),
        "run_id": "run",
        "script_sha256": "ABC",
    }

    assert classify_lock_owner(
        record,
        expected_run_id="run",
        expected_script_sha256="ABC",
    ) == "MATCH"
    assert classify_lock_owner(
        {**record, "create_time": record["create_time"] - 1},
        expected_run_id="run",
        expected_script_sha256="ABC",
    ) == "STALE"
    assert classify_lock_owner(
        record,
        expected_run_id="different",
        expected_script_sha256="ABC",
    ) == "CONFLICT"


def test_release_verification_checks_digest_and_identity(tmp_path) -> None:
    asset = tmp_path / "artifact.bin"
    asset.write_bytes(b"artifact")
    release = {
        "isDraft": True,
        "isPrerelease": False,
        "targetCommitish": "abc123",
        "assets": [
            {
                "name": asset.name,
                "size": asset.stat().st_size,
                "state": "uploaded",
                    "digest": (
                        "sha256:"
                        "C7C5C1D70C5DEC4416AB6158AFD0B223EF40C29B"
                        "1DC1F97ED9428B94D4CADB1C".lower()
                    ),
            }
        ],
    }

    verify_release_identity(
        release,
        target_commit="abc123",
        prerelease=False,
        allow_draft=True,
    )
    verify_release_assets(release, [asset])


def test_release_verification_rejects_wrong_digest(tmp_path) -> None:
    asset = tmp_path / "artifact.bin"
    asset.write_bytes(b"artifact")
    release = {
        "assets": [
            {
                "name": asset.name,
                "size": asset.stat().st_size,
                "state": "uploaded",
                "digest": "sha256:" + "0" * 64,
            }
        ]
    }

    with pytest.raises(PublicationConflictError, match="asset mismatch"):
        verify_release_assets(release, [asset])


def test_terminal_claim_is_idempotent_but_not_reversible(tmp_path) -> None:
    claim = tmp_path / "terminal_observation.json"
    payload = {"status": "TRAIN_INVALID", "exit_code": "1"}

    assert claim_immutable_record(claim, payload) == payload
    assert claim_immutable_record(claim, payload) == payload
    with pytest.raises(PublicationConflictError, match="immutable record"):
        claim_immutable_record(
            claim,
            {"status": "TRAIN_COMPLETE", "exit_code": "0"},
        )


def test_opposite_terminal_directory_is_a_conflict(tmp_path) -> None:
    (tmp_path / "invalid").mkdir()

    with pytest.raises(PublicationConflictError, match="opposite"):
        ensure_no_opposite_terminal(tmp_path, "SUCCESS")


def test_success_download_must_repeat_terminal_facts() -> None:
    validate_terminal_facts(
        "TRAIN_COMPLETE",
        "0",
        "SUCCESS_CANDIDATE",
    )
    with pytest.raises(ScientificValidationError, match="terminal facts"):
        validate_terminal_facts(
            "RUNNING",
            None,
            "SUCCESS_CANDIDATE",
        )


def test_only_expected_terminal_paths_are_allowed_dirty() -> None:
    status = (
        "?? docs/evidence/saded_fresh100_seed0/terminal/a.json\n"
        "A  docs/evidence/saded_fresh100_seed0/terminal/b.json"
    )
    assert dirty_paths_are_allowed(
        status,
        "docs/evidence/saded_fresh100_seed0/terminal",
    )
    assert not dirty_paths_are_allowed(
        status + "\n M scripts/publish_saded_fresh100.py",
        "docs/evidence/saded_fresh100_seed0/terminal",
    )


def test_release_absence_only_accepts_explicit_not_found() -> None:
    assert is_release_not_found("release not found")
    assert is_release_not_found("HTTP 404: Not Found")
    assert not is_release_not_found("HTTP 503: Service Unavailable")
    assert not is_release_not_found("authentication failed")


def test_json_object_loader_rejects_non_object_root(tmp_path) -> None:
    path = tmp_path / "payload.json"
    path.write_text("[]")

    with pytest.raises(ScientificValidationError, match="JSON object"):
        load_json_object(path, scientific=True)


def test_checksum_closure_rejects_mutated_evidence(tmp_path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("original")
    (tmp_path / "checksums.sha256").write_text(
        f"{sha256_file(artifact)}  artifact.txt\n"
    )
    verify_checksum_closure(tmp_path)

    artifact.write_text("mutated")
    with pytest.raises(PublicationConflictError, match="checksum"):
        verify_checksum_closure(tmp_path)


def test_release_assets_reject_unexpected_remote_asset(tmp_path) -> None:
    asset = tmp_path / "expected.bin"
    asset.write_bytes(b"expected")
    release = {
        "assets": [
            {
                "name": asset.name,
                "size": asset.stat().st_size,
                "state": "uploaded",
                "digest": f"sha256:{sha256_file(asset).lower()}",
            },
            {
                "name": "unexpected.bin",
                "size": 1,
                "state": "uploaded",
                "digest": "sha256:" + "0" * 64,
            },
        ]
    }

    with pytest.raises(PublicationConflictError, match="asset mismatch"):
        verify_release_assets(release, [asset])


def test_process_baseline_requires_exact_run_and_pid_set() -> None:
    identity = {
        "pid": 1,
        "boot_id": "boot",
        "start_ticks": "10",
        "command": "driver",
    }
    baseline = {
        "run_id": "run",
        "processes": {
            "1": identity,
            "2": {**identity, "pid": 2, "command": "trainer"},
        },
    }
    assert set(
        validate_process_baseline(
            baseline,
            expected_run_id="run",
            expected_pids={1, 2},
        )
    ) == {1, 2}
    with pytest.raises(PublicationConflictError, match="baseline"):
        validate_process_baseline(
            {**baseline, "run_id": "other"},
            expected_run_id="run",
            expected_pids={1, 2},
        )
