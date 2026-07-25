from __future__ import annotations

import pytest

from scripts.publish_saded_fresh100 import (
    build_terminal_manifest,
    classify_terminal_state,
    release_tag_for_state,
    terminal_directory_name,
)


def test_complete_zero_is_success() -> None:
    assert classify_terminal_state("TRAIN_COMPLETE", "0") == "SUCCESS"


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
