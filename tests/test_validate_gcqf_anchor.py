from scripts.validate_gcqf_anchor import anchor_gate, build_parser


def _metrics(map_value, tiny, recall, medium, large):
    return {
        "mAP50-95": map_value,
        "AP-tiny-SBR": tiny,
        "tiny_recall": recall,
        "AP-medium-SBR": medium,
        "AP-large-SBR": large,
    }


def test_anchor_cli_is_a_pretraining_stage():
    args = build_parser().parse_args(
        [
            "--cache",
            "val/manifest.json",
            "--data",
            "visdrone.yaml",
            "--output",
            "anchor",
        ]
    )

    assert args.expected_images == 548
    assert args.stage == "G0-A"


def test_anchor_must_be_strong_before_gcqf_training():
    gate = anchor_gate(
        global_metrics=_metrics(0.20, 0.08, 0.50, 0.25, 0.30),
        anchor_metrics=_metrics(0.21, 0.095, 0.53, 0.249, 0.297),
        protected_exact=True,
    )

    assert gate["map_anchor_gain"] is True
    assert gate["tiny_anchor_gain"] is True
    assert gate["large_anchor_budget"] is True
    assert gate["advance_to_training"] is True


def test_anchor_gate_stops_weak_or_large_degrading_route():
    gate = anchor_gate(
        global_metrics=_metrics(0.20, 0.08, 0.50, 0.25, 0.30),
        anchor_metrics=_metrics(0.203, 0.09, 0.53, 0.249, 0.29),
        protected_exact=True,
    )

    assert gate["map_anchor_gain"] is False
    assert gate["large_anchor_budget"] is False
    assert gate["advance_to_training"] is False
