from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml


PROTOCOL_VERSION = "ascv-loc-protocol/v1"
ULTRALYTICS_VERSION = "8.4.90"
SUBSET_FRACTION = 0.10
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _reject_forbidden_path(path: Path) -> None:
    value = path.resolve().as_posix().lower()
    if "test-dev" in value or "test_dev" in value:
        raise ValueError(f"test-dev path is forbidden during ASCV-Loc development: {path}")


def _source_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def prepare_protocol(
    *,
    checkpoint: Path,
    dataset_root: Path,
    dataset_yaml: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict:
    checkpoint = checkpoint.resolve()
    dataset_root = dataset_root.resolve()
    dataset_yaml = dataset_yaml.resolve()
    output_dir = output_dir.resolve()
    for path in (checkpoint, dataset_root, dataset_yaml, output_dir, repo_root):
        _reject_forbidden_path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not dataset_yaml.is_file():
        raise FileNotFoundError(dataset_yaml)
    train_root = dataset_root / "images" / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(train_root)

    images = sorted(
        path.resolve()
        for path in train_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError("ASCV-Loc protocol found no training images")
    ranked = sorted(
        images,
        key=lambda path: (
            hashlib.sha256(
                f"{PROTOCOL_VERSION}|{path.relative_to(dataset_root).as_posix()}".encode("utf-8")
            ).hexdigest(),
            path.relative_to(dataset_root).as_posix(),
        ),
    )
    subset_count = max(1, (len(ranked) + 9) // 10)
    selected = ranked[:subset_count]
    train_list = output_dir / "train_10pct_hash.txt"
    _atomic_write(train_list, "".join(f"{path.as_posix()}\n" for path in selected))
    full_train_list = output_dir / "train_full.txt"
    _atomic_write(full_train_list, "".join(f"{path.as_posix()}\n" for path in images))

    source_yaml = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(source_yaml, dict) or not isinstance(source_yaml.get("names"), (dict, list)):
        raise ValueError("dataset YAML must define class names")
    train_only_data = {
        "path": dataset_root.as_posix(),
        "train": train_list.as_posix(),
        # Ultralytics requires a val key while resolving a detection dataset.
        # It points to the train subset and is never loaded by ASCVLocTrainer.
        "val": train_list.as_posix(),
        "names": source_yaml["names"],
    }
    train_yaml = output_dir / "train_only.yaml"
    _atomic_write(train_yaml, yaml.safe_dump(train_only_data, sort_keys=False, allow_unicode=True))
    full_train_data = {
        **train_only_data,
        "train": full_train_list.as_posix(),
        "val": full_train_list.as_posix(),
    }
    full_train_yaml = output_dir / "train_full_only.yaml"
    _atomic_write(full_train_yaml, yaml.safe_dump(full_train_data, sort_keys=False, allow_unicode=True))

    manifest = {
        "schema_version": PROTOCOL_VERSION,
        "source_commit": _source_commit(repo_root.resolve()),
        "ultralytics_version": ULTRALYTICS_VERSION,
        "checkpoint": {
            "path": checkpoint.as_posix(),
            "sha256": sha256_file(checkpoint),
        },
        "dataset": {
            "root": dataset_root.as_posix(),
            "source_yaml": dataset_yaml.as_posix(),
            "source_yaml_sha256": sha256_file(dataset_yaml),
            "train_image_count": len(images),
        },
        "subset": {
            "algorithm": "sha256(protocol|dataset-relative-path), ascending",
            "rounding": "ceil(N*0.10)",
            "fraction": SUBSET_FRACTION,
            "count": len(selected),
            "train_list": train_list.as_posix(),
            "train_list_sha256": sha256_file(train_list),
        },
        "train_only_yaml": {
            "path": train_yaml.as_posix(),
            "sha256": sha256_file(train_yaml),
        },
        "full_train": {
            "count": len(images),
            "train_list": full_train_list.as_posix(),
            "train_list_sha256": sha256_file(full_train_list),
        },
        "full_train_only_yaml": {
            "path": full_train_yaml.as_posix(),
            "sha256": sha256_file(full_train_yaml),
        },
        "forbidden_data": ["test-dev", "test_dev"],
    }
    manifest_path = output_dir / "protocol_manifest.json"
    _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare immutable ASCV-Loc train-only inputs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_protocol(
        checkpoint=args.checkpoint,
        dataset_root=args.dataset_root,
        dataset_yaml=args.dataset_yaml,
        output_dir=args.output,
        repo_root=args.repo_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
