from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visdrone import prepare_visdrone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and convert VisDrone to YOLO format.")
    parser.add_argument("--dataset-dir", default="datasets/VisDrone")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_visdrone(Path(args.dataset_dir), tuple(args.splits))


if __name__ == "__main__":
    main()
