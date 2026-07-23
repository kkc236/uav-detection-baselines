import pytest


def test_cli_rejects_scientific_overrides():
    from scripts.run_sbr_g0 import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["s0", "--checkpoint", "x", "--data", "d", "--output", "o", "--conf", "0.2"])


def test_g0bc_requires_matching_pass_gate(tmp_path):
    from scripts.run_sbr_g0 import validate_prior_gate

    gate = tmp_path / "g0_gate.json"
    gate.write_text('{"status":"SBR_G0A_FAIL","source_hash":"s","checkpoint_hash":"c","dataset_signature":"d","protocol_hash":"p"}', encoding="utf-8")
    with pytest.raises(ValueError):
        validate_prior_gate(gate, {"source_hash":"s","checkpoint_hash":"c","dataset_signature":"d","protocol_hash":"p"})
