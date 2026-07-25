from __future__ import annotations

from pathlib import Path

import pytest

from scripts.resolve_saded_controls import _reject_candidate_path


@pytest.mark.parametrize(
    "value",
    (
        "/tmp/test-dev/candidate.json",
        "/tmp/results/candidate.json",
        "/tmp/metrics/candidate.json",
        "/tmp/val_annotations/candidate.json",
    ),
)
def test_resolver_rejects_forbidden_path_before_read(value: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        _reject_candidate_path(Path(value))


def test_resolver_allows_provenance_sidecar_path() -> None:
    _reject_candidate_path(Path("/home/control/provenance.json"))
