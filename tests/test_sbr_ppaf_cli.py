from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_importing_router_core_does_not_import_evaluator():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import src.sbr_ppaf; "
                "assert 'src.sbr_metrics' not in sys.modules"
            ),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
