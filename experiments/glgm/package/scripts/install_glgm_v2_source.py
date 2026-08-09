#!/usr/bin/env python3
"""Install the audited GLGM-v2 source overlay into an Ultralytics checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OVERLAY = (
    PACKAGE_ROOT / "source-overlay" / "ultralytics" / "nn" / "modules" / "glgm_v2.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def insert_once(text: str, anchor: str, replacement: str, path: Path) -> str:
    if replacement in text:
        return text
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"expected one patch anchor in {path}, found {count}: {anchor!r}"
        )
    return text.replace(anchor, replacement, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_dir.resolve()
    modules = repo / "ultralytics" / "nn" / "modules"
    init_path = modules / "__init__.py"
    tasks_path = repo / "ultralytics" / "nn" / "tasks.py"
    for path in (OVERLAY, init_path, tasks_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    before = {
        str(path.relative_to(repo)): sha256(path) for path in (init_path, tasks_path)
    }
    target_module = modules / "glgm_v2.py"
    target_module.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OVERLAY, target_module)

    init_text = init_path.read_text(encoding="utf-8")
    init_text = insert_once(
        init_text,
        ")\nfrom .conv import (",
        ")\nfrom .glgm_v2 import GLGMLite\nfrom .conv import (",
        init_path,
    )
    init_text = insert_once(
        init_text, '    "GLGM",\n', '    "GLGM",\n    "GLGMLite",\n', init_path
    )
    init_path.write_text(init_text, encoding="utf-8")

    tasks_text = tasks_path.read_text(encoding="utf-8")
    if "    GLGMLite,\n" not in tasks_text:
        count = tasks_text.count("    GLGM,\n")
        if count != 2:
            raise RuntimeError(
                f"expected two GLGM anchors in {tasks_path}, found {count}"
            )
        tasks_text = tasks_text.replace("    GLGM,\n", "    GLGM,\n    GLGMLite,\n")
    tasks_path.write_text(tasks_text, encoding="utf-8")

    after_paths = (init_path, tasks_path, target_module)
    receipt = {
        "schema": "glgm-v2-source-install-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_dir": str(repo),
        "before_sha256": before,
        "after_sha256": {
            str(path.relative_to(repo)): sha256(path) for path in after_paths
        },
        "overlay_sha256": sha256(OVERLAY),
    }
    args.receipt.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.receipt.resolve().write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
