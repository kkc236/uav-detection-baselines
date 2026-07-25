from __future__ import annotations

from scripts.adjudicate_tascv import build_parser


def test_adjudicator_cli_exposes_only_evidence_paths() -> None:
    parser = build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert options == {
        "-h",
        "--help",
        "--stage",
        "--summary",
        "--records",
        "--output",
    }
    assert "--threshold" not in options
    assert "--metric" not in options
