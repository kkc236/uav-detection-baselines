from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "continue_bpdd_formal_server.sh"


def test_formal_continuation_preserves_fresh_paired_protocol() -> None:
    assert SCRIPT.is_file()
    text = SCRIPT.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "formal-seed0-fdr-bpdd-v1" in text
    assert "formal-seed0-fdr_bpdd-bpdd-v1" in text
    assert "--variant fdr_bpdd" in text
    assert "--stage formal" in text
    assert "--initial-state \"${INITIAL}\"" in text
    assert "--resume" not in text
    assert 'test "$(wc -l < "${FDR_RUN}/bpdd-epochs.jsonl")" -eq 100' in text
    assert 'test ! -e "${BPDD_RUN}"' in text


def test_formal_continuation_records_failure_and_single_owner() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "flock -n 9" in text
    assert "formal-continuation.lock" in text
    assert "formal_engineering_failed" in text
    assert "formal_pair_training_complete" in text
