#!/usr/bin/env python3
"""Validate, cross-check, extract, and convert VisDrone train/val archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import yaml

from prepare_visdrone import NAMES, convert_split


ARCHIVES = {
    "VisDrone2019-DET-train.zip": ("VisDrone2019-DET-train", "train", 6471),
    "VisDrone2019-DET-val.zip": ("VisDrone2019-DET-val", "val", 548),
}
SOURCE_PREFIX = "https://github.com/ultralytics/assets/releases/download/v0.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-snapshot", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member_target(root: Path, member: ZipInfo) -> Path:
    name = member.filename.replace("\\", "/")
    if name.startswith("/") or name.startswith("../") or "/../" in name:
        raise RuntimeError(f"unsafe archive member: {member.filename}")
    target = (root / name).resolve()
    if target != root and root not in target.parents:
        raise RuntimeError(f"archive member escapes destination: {member.filename}")
    file_type = (member.external_attr >> 16) & 0o170000
    if file_type == 0o120000:
        raise RuntimeError(f"archive contains a symbolic link: {member.filename}")
    return target


def validate_archive(path: Path, root: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with ZipFile(path) as archive:
        members = archive.infolist()
        for member in members:
            safe_member_target(root, member)
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"CRC failure in {path}: {corrupt}")
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "member_count": len(members),
        "source_url": f"{SOURCE_PREFIX}/{path.name}",
    }


def cross_check_images(root: Path, reference: Path) -> dict[str, object]:
    result: dict[str, object] = {"reference": str(reference), "splits": {}}
    total = 0
    for _, (source_name, split, _) in ARCHIVES.items():
        source_images = root / source_name / "images"
        reference_images = reference / source_name / "images"
        if not reference_images.is_dir():
            result["splits"][split] = {"matched": 0, "reference_missing": True}
            continue
        matched = 0
        for candidate in sorted(reference_images.glob("*.jpg")):
            source = source_images / candidate.name
            if not source.is_file():
                raise RuntimeError(f"reference image missing from archive: {candidate.name}")
            if sha256_file(source) != sha256_file(candidate):
                raise RuntimeError(f"cross-source image hash mismatch: {candidate.name}")
            matched += 1
        result["splits"][split] = {"matched": matched, "reference_missing": False}
        total += matched
    if total == 0:
        raise RuntimeError("cross-source check found no complete reference images")
    result["total_matched"] = total
    return result


def main() -> None:
    args = parse_args()
    archives_dir = args.archives_dir.expanduser().resolve()
    root = args.root.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing to reuse non-empty dataset root: {root}")
    root.mkdir(parents=True, exist_ok=True)

    archive_records = []
    for filename in ARCHIVES:
        archive_records.append(validate_archive(archives_dir / filename, root))
    for filename in ARCHIVES:
        with ZipFile(archives_dir / filename) as archive:
            archive.extractall(root)

    cross_source = None
    if args.reference_snapshot is not None:
        cross_source = cross_check_images(root, args.reference_snapshot.expanduser().resolve())

    counts: dict[str, dict[str, int]] = {}
    annotation_filter: dict[str, dict[str, int]] = {}
    for _, (source_name, split, expected) in ARCHIVES.items():
        annotation_filter[split] = convert_split(root, source_name, split)
        image_count = len(list((root / "images" / split).glob("*.jpg")))
        label_count = len(list((root / "labels" / split).glob("*.txt")))
        if image_count != expected or label_count != expected:
            raise RuntimeError(
                f"unexpected {split} counts: images={image_count}, labels={label_count}, expected={expected}"
            )
        counts[split] = {"images": image_count, "labels": label_count}

    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(NAMES)},
    }
    data_path = root / "data.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    receipt = {
        "schema": "glgm-visdrone-archive-source-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archives": archive_records,
        "cross_source": cross_source,
        "counts": counts,
        "annotation_filter": annotation_filter,
        "data_yaml": str(data_path),
    }
    receipt_path = root / "SOURCE_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
