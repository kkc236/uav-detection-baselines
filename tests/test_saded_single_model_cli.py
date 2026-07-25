from __future__ import annotations

from pathlib import Path

import pytest

from scripts.adjudicate_saded_single_model import (
    adjudicate,
    build_parser,
)


def test_cli_rejects_test_dev_in_any_path_before_reading(tmp_path):
    ordinary = tmp_path / "ordinary"
    forbidden = tmp_path / "test-dev" / "output"
    args = build_parser().parse_args(
        [
            "--tascv-protocol",
            str(ordinary),
            "--tascv-gate",
            str(ordinary),
            "--tascv-adjudication-anchor",
            str(ordinary),
            "--r0-input-manifest",
            str(ordinary),
            "--r0-route-root",
            str(ordinary),
            "--r0-evaluation",
            str(ordinary),
            "--checkpoint",
            str(ordinary),
            "--output",
            str(forbidden),
        ]
    )

    with pytest.raises(ValueError, match="test-dev"):
        adjudicate(args)
