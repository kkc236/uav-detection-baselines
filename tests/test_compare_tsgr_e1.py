from copy import deepcopy

from scripts.compare_tsgr_e1 import _json_sha256, build_report


def _signed(payload: dict) -> dict:
    value = deepcopy(payload)
    value["signature"] = _json_sha256(value)
    return value


def _run(seed: int, arm: str, final: float, tail: float) -> dict:
    config = None
    ordinary = []
    if arm == "tsgr-p2":
        config = {
            "lambda_p2": 0.1,
            "lambda_ebc": 0.0,
            "lambda_quality": 0.0,
            "query_injection_enabled": False,
            "quality_gated_p2": False,
            "quality_weighted_ebc": False,
            "learnable_fusion_gamma": False,
            "p2_c2_grad_scale": 0.1,
            "contribution_separated_aux_gradients": True,
        }
        ordinary = [300]
    hashes = [f"{arm}-{seed}-{epoch}" for epoch in (8, 9, 10)]
    return _signed(
        {
            "stage": "e1",
            "arm": arm,
            "seed": seed,
            "controlled_amp": {"skipped_attempts": 0},
            "ebc_config": config,
            "protocol": {
                "signature": f"protocol-{seed}",
                "experiment_signature": "experiment",
                "initial_state_sha256": f"initial-{seed}",
                "subset": {"sha256": "subset"},
            },
            "optimizer_evidence": {
                "p2_entry_count_max": 0,
                "ordinary_query_count_values": ordinary,
            },
            "results": {
                "epochs": 10,
                "final_map50_95": final,
                "tail3_map50_95": tail,
                "final_ap_tiny": 0.0,
                "final_recall_tiny": 0.0,
            },
            "epoch_checkpoints": [{"sha256": value} for value in hashes],
        }
    )


def _diagnostics(seed: int, arm: str, coverage: float, rank: float) -> list[dict]:
    return [
        _signed(
            {
                "checkpoint_sha256": f"{arm}-{seed}-{epoch}",
                "stock_query": {
                    "stock_top300_coverage": coverage,
                    "normalized_best_rank_mean": rank,
                },
            }
        )
        for epoch in (8, 9, 10)
    ]


def test_e1_comparator_requires_joint_metric_coverage_and_rank_signal():
    controls = {seed: _run(seed, "control", 0.10 + seed * 0.01, 0.09 + seed * 0.01) for seed in range(3)}
    methods = {seed: _run(seed, "tsgr-p2", 0.11 + seed * 0.01, 0.10 + seed * 0.01) for seed in range(3)}
    control_diagnostics = {seed: _diagnostics(seed, "control", 0.40, 0.30) for seed in range(3)}
    method_diagnostics = {seed: _diagnostics(seed, "tsgr-p2", 0.45, 0.25) for seed in range(3)}

    report = build_report(controls, methods, control_diagnostics, method_diagnostics)

    assert report["classification"] == "TSGR_E1_PASS"
    assert report["passed"] is True
    assert report["aggregate"]["final_wins"] == 3
    assert report["aggregate"]["coverage_wins"] == 3
    assert report["aggregate"]["rank_wins"] == 3

    failed_diagnostics = deepcopy(method_diagnostics)
    for records in failed_diagnostics.values():
        for record in records:
            unsigned = dict(record)
            unsigned.pop("signature")
            unsigned["stock_query"]["normalized_best_rank_mean"] = 0.35
            record.clear()
            record.update(_signed(unsigned))
    failed = build_report(controls, methods, control_diagnostics, failed_diagnostics)
    assert failed["classification"] == "TSGR_E1_FAIL"
    assert failed["conditions"]["rank_wins_at_least_2_of_3"] is False
