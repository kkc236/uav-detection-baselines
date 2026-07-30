from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rtdetr_sqda_sgc import BASELINE_SHA256, sha256_file
from src.sqda_preflight import validate_visdrone_dataset, write_dataset_yaml
from src.visdrone import download_with_resume, prepare_visdrone


BASELINE_URLS = (
    "https://gh-proxy.com/https://github.com/kkc236/uav-detection-baselines/releases/download/"
    "rtdetr-l-btdse-matched-baseline-live/matched-baseline-best-epoch-0100.pt",
    "https://github.com/kkc236/uav-detection-baselines/releases/download/"
    "rtdetr-l-btdse-matched-baseline-live/matched-baseline-best-epoch-0100.pt",
)
BASELINE_BYTES = 66_262_262


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and hard-validate immutable SQDA-SGC server inputs."
    )
    parser.add_argument("--root", type=Path, default=Path("/root/data/uav"))
    parser.add_argument("--report", type=Path)
    return parser


def _download_verified_baseline(destination: Path) -> dict:
    if destination.is_file():
        actual_sha = sha256_file(destination)
        if destination.stat().st_size == BASELINE_BYTES and actual_sha == BASELINE_SHA256:
            return {
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": actual_sha,
                "source": "existing",
            }
        destination.unlink()
    errors = []
    for url in BASELINE_URLS:
        try:
            download_with_resume(url, destination)
            actual_sha = sha256_file(destination)
            if destination.stat().st_size != BASELINE_BYTES or actual_sha != BASELINE_SHA256:
                raise RuntimeError(
                    f"identity mismatch bytes={destination.stat().st_size}, sha256={actual_sha}"
                )
            return {
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": actual_sha,
                "source": url,
            }
        except Exception as error:
            errors.append(f"{url}: {type(error).__name__}: {error}")
            destination.unlink(missing_ok=True)
    raise RuntimeError("all baseline sources failed: " + " | ".join(errors))


def prepare(root: Path) -> dict:
    root = root.expanduser().resolve()
    dataset_root = root / "datasets" / "VisDrone"
    checkpoint = root / "checkpoints" / "matched-baseline-best-epoch-0100.pt"
    dataset_yaml = root / "protocols" / "tsgr-p2-e1" / "source-VisDrone-full.yaml"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    baseline_report = _download_verified_baseline(checkpoint)
    prepare_visdrone(dataset_root, ("train", "val"))
    dataset_report = validate_visdrone_dataset(dataset_root)
    write_dataset_yaml(dataset_root, dataset_yaml)
    return {
        "baseline": baseline_report,
        "dataset": dataset_report,
        "dataset_yaml": {
            "path": str(dataset_yaml),
            "sha256": sha256_file(dataset_yaml),
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    report = prepare(args.root)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    destination = (
        args.report.expanduser().resolve()
        if args.report
        else args.root.expanduser().resolve() / "logs" / "input-preflight.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
