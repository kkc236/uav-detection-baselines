"""Publish terminal evidence for the authoritative Fresh-100 seed-0 run."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _require_terminal_state(terminal_state: str) -> None:
    if terminal_state not in {"SUCCESS", "INVALID"}:
        raise ValueError("terminal_state must be SUCCESS or INVALID")


def classify_terminal_state(
    status: str | None,
    exit_code: str | None,
) -> str | None:
    """Map remote driver state to a fail-closed publication state."""

    if status == "TRAIN_COMPLETE" and exit_code == "0":
        return "SUCCESS"
    if status == "TRAIN_INVALID":
        return "INVALID"
    if exit_code not in (None, "", "0"):
        return "INVALID"
    return None


def build_terminal_manifest(
    *,
    run_id: str,
    terminal_state: str,
    exit_code: str | None,
    artifacts: Mapping[str, str],
) -> dict[str, Any]:
    """Build a deterministic manifest that cannot mislabel invalid evidence."""

    _require_terminal_state(terminal_state)
    return {
        "schema_version": "saded-fresh100-publication/v1",
        "run_id": run_id,
        "terminal_state": terminal_state,
        "exit_code": exit_code,
        "publish_as_success": terminal_state == "SUCCESS",
        "artifacts": dict(sorted(artifacts.items())),
    }


def terminal_directory_name(terminal_state: str) -> str:
    """Return the isolated evidence directory for a terminal state."""

    _require_terminal_state(terminal_state)
    return "terminal" if terminal_state == "SUCCESS" else "invalid"


def release_tag_for_state(base_tag: str, terminal_state: str) -> str:
    """Return a tag whose spelling cannot hide invalid evidence."""

    _require_terminal_state(terminal_state)
    return base_tag if terminal_state == "SUCCESS" else f"{base_tag}-invalid"
