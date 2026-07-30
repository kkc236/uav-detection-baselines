from scripts.sync_experiment_checkpoint import LIGHTWEIGHT_ARTIFACTS, collect_lightweight_artifacts


def test_final_sqda_gate_evidence_is_published() -> None:
    assert {
        "exact-manual-validation.json",
        "frozen-stock-audit.json",
        "final-gate-decision.json",
        "evaluation-inventory/candidate-inventory.json",
    } <= set(LIGHTWEIGHT_ARTIFACTS)


def test_candidate_inventory_is_copied_with_its_relative_path(tmp_path) -> None:
    run_dir = tmp_path / "run"
    destination = tmp_path / "results"
    inventory = run_dir / "evaluation-inventory" / "candidate-inventory.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_text('{"selected_checkpoint": null}', encoding="utf-8")

    collect_lightweight_artifacts(run_dir, destination, {"completed_epoch": 3})

    assert (destination / "evaluation-inventory" / "candidate-inventory.json").is_file()
