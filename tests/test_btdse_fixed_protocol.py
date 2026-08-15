from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_btdse_server_runner_never_changes_the_paper_batch():
    script = (ROOT / "scripts" / "run_btdse_4090.sh").read_text(encoding="utf-8")

    assert 'BATCH="${BATCH:-8}"' in script
    assert '[[ "$BATCH" == "8" ]]' in script
    assert "MIN_BATCH" not in script
    assert "reducing batch" not in script
    assert "Fixed protocol violation detected" in script
