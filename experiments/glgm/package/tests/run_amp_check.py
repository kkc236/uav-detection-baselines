#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ultralytics import RTDETR
from ultralytics.utils.checks import check_amp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    model = RTDETR(args.checkpoint).model.cuda()
    print(f"AMP_CHECK_RESULT={check_amp(model)}")


if __name__ == "__main__":
    main()
