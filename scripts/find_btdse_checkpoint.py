from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint_recovery import find_resume_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Find the newest valid resumable BTD-SE checkpoint.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    checkpoint = find_resume_checkpoint(args.run_dir)
    if checkpoint is None:
        return 1
    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
