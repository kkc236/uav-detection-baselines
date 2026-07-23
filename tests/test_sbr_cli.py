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


def test_effective_gain_uses_arm_a_640_letterbox():
    from scripts.run_sbr_g0 import arm_a_effective_gain

    assert arm_a_effective_gain(1000, 500) == 0.64
    assert arm_a_effective_gain(320, 240) == 1.0


def test_d_c_gate_requires_reduction_and_singletons():
    from scripts.run_sbr_g0 import evaluate_g0bc_gate

    c = {"mAP50-95": 0.50, "AP-tiny-SBR": 0.30, "AP-large-SBR": 0.40}
    d = {"mAP50-95": 0.499, "AP-tiny-SBR": 0.299, "AP-large-SBR": 0.399}
    diagnostics = {
        "internal_boundary_fp": {"C": 20, "D": 16},
        "duplicate_detections": {"C": 10, "D": 10},
        "singleton_preservation": 1.0,
        "boundary_target_recall": {"C": 0.8, "D": 0.799},
    }
    assert evaluate_g0bc_gate(c, d, diagnostics)[1] == "SBR_G0BC_PASS"
    diagnostics["singleton_preservation"] = 0.99
    assert evaluate_g0bc_gate(c, d, diagnostics)[1] == "SBR_G0BC_FAIL"


def test_prior_gate_rejects_not_run_adjudicator(tmp_path):
    from scripts.run_sbr_g0 import validate_prior_gate

    out = tmp_path
    (out / "g0_gate.json").write_text(
        '{"status":"SBR_G0A_PASS","source_hash":"s","checkpoint_hash":"c","dataset_signature":"d","protocol_hash":"p"}',
        encoding="utf-8",
    )
    (out / "independent_adjudication.json").write_text('{"status":"NOT_RUN"}', encoding="utf-8")
    with pytest.raises(ValueError):
        validate_prior_gate(out / "g0_gate.json", {"source_hash":"s","checkpoint_hash":"c","dataset_signature":"d","protocol_hash":"p"})


def test_g0bc_evidence_must_match_gate_directory(tmp_path):
    from scripts.run_sbr_g0 import build_parser, run

    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    gate = gate_dir / "g0_gate.json"
    gate.write_text('{"status":"SBR_G0A_FAIL"}', encoding="utf-8")
    args = build_parser().parse_args([
        "g0-b", "--checkpoint", str(tmp_path / "x"), "--data", str(tmp_path / "d"),
        "--output", str(tmp_path / "o"), "--gate", str(gate), "--evidence", str(tmp_path / "other"),
    ])
    with pytest.raises(ValueError):
        run(args)
