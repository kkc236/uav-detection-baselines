from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training_monitor import monitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live monitor for Ultralytics training results.")
    parser.add_argument("--run", default="runs/detect/runs/baselines/scratch-yolo-100ep")
    parser.add_argument("--total-epochs", type=int, default=100)
    parser.add_argument("--interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    monitor(Path(args.run), total_epochs=args.total_epochs, interval=args.interval)


if __name__ == "__main__":
    main()
