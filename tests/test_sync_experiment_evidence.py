from scripts.sync_experiment_checkpoint import LIGHTWEIGHT_ARTIFACTS


def test_final_sqda_gate_evidence_is_published() -> None:
    assert {
        "exact-manual-validation.json",
        "frozen-stock-audit.json",
        "final-gate-decision.json",
    } <= set(LIGHTWEIGHT_ARTIFACTS)
