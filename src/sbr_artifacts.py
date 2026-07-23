"""Audited SBR-RTDETR artifact and dataset helpers.

All serialization in this module is deterministic and fail-closed.  The
runner is intentionally thin; these helpers are also useful to independent
adjudicators and tests.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence


def _check_value(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}")
    if isinstance(value, Mapping):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(f"JSON object key must be string at {path}")
            _check_value(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check_value(item, f"{path}[{i}]")


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical UTF-8 JSON (sorted keys, compact separators)."""
    _check_value(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _atomic_replace(path: Path, writer: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    tmp = Path(name)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_bytes(path: Path | str, data: bytes) -> Path:
    payload = bytes(data)
    _atomic_replace(Path(path), lambda p: p.write_bytes(payload))
    return Path(path)


def atomic_write_json(path: Path | str, value: Any) -> Path:
    payload = canonical_json_bytes(value)
    return atomic_write_bytes(path, payload)


def atomic_write_jsonl_gz(path: Path | str, rows: Iterable[Any]) -> Path:
    data = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)

    def write(target: Path) -> None:
        with target.open("wb") as raw:
            # Explicit mtime keeps gzip bytes stable across runs.
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
                gz.write(data)

    _atomic_replace(Path(path), write)
    return Path(path)


def ensure_empty_output(path: Path | str) -> Path:
    p = Path(path)
    if p.exists():
        if not p.is_dir():
            raise FileExistsError(f"output exists and is not a directory: {p}")
        if any(p.iterdir()):
            raise FileExistsError(f"output directory must be empty: {p}")
    else:
        p.mkdir(parents=True)
    return p


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksums(expected: Mapping[str, str], actual: Mapping[str, Any]) -> None:
    """Verify hash map; values may be bytes, paths, or hexadecimal strings."""
    for name, digest in expected.items():
        if name not in actual:
            raise ValueError(f"missing checksum target: {name}")
        value = actual[name]
        if isinstance(value, (str, os.PathLike)) and Path(value).exists():
            got = sha256_file(value)
        elif isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value):
            got = value.lower()
        else:
            got = sha256_bytes(value)
        if got.lower() != str(digest).lower():
            raise ValueError(f"checksum mismatch for {name}")


def validate_raw_view_cache(path: Path | str, *, expected_sha256: str | None = None) -> list[dict[str, Any]]:
    """Read a compressed raw-view cache and validate deterministic JSON rows."""
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if expected_sha256 and sha256_file(p) != expected_sha256:
        raise ValueError("raw-view cache checksum mismatch")
    try:
        with gzip.open(p, "rt", encoding="utf-8", newline="") as fh:
            for no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"raw-view row {no} is not an object")
                _check_value(value)
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid raw-view cache: {p}") from exc
    return rows


def write_checksums(path: Path | str, files: Sequence[Path | str], *, root: Path | None = None) -> Path:
    root = Path(root) if root is not None else None
    rows = []
    for file in sorted((Path(f) for f in files), key=lambda p: p.as_posix()):
        label = file.relative_to(root).as_posix() if root is not None else file.name
        rows.append(f"{sha256_file(file)}  {label}")
    return atomic_write_bytes(path, ("\n".join(rows) + ("\n" if rows else "")).encode("utf-8"))


def _yaml_load(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to parse dataset YAML") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError("dataset YAML must contain a mapping")
    return data


def _resolve_dataset_root(yaml_path: Path, config: Mapping[str, Any]) -> Path:
    root_value = config.get("path", ".")
    root = Path(root_value)
    if not root.is_absolute():
        root = yaml_path.parent / root
    return root.resolve()


def _split_image_dir(root: Path, yaml_path: Path, config: Mapping[str, Any], split: str) -> Path:
    value = config.get(split)
    if value is None:
        raise ValueError(f"dataset YAML lacks {split!r} path")
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("multi-path splits are not supported for deterministic SBR runs")
        value = value[0]
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def _parse_label(path: Path, width: int, height: int, *, ignore: bool = False) -> tuple[list[list[float]], list[int]]:
    boxes: list[list[float]] = []
    classes: list[int] = []
    if not path.exists():
        return boxes, classes
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if ignore:
            if len(fields) < 4:
                raise ValueError(f"invalid ignore label {path}:{line_no}")
            vals = fields[:4]
            cls = 0
        else:
            if len(fields) < 5:
                raise ValueError(f"invalid label {path}:{line_no}")
            cls = int(fields[0])
            vals = fields[1:5]
            if cls < 0:
                raise ValueError(f"negative class in {path}:{line_no}")
        try:
            cx, cy, wn, hn = (float(x) for x in vals)
        except ValueError:
            raise ValueError(f"non-numeric label {path}:{line_no}") from None
        if not all(math.isfinite(v) for v in (cx, cy, wn, hn)) or wn <= 0 or hn <= 0:
            raise ValueError(f"illegal label geometry {path}:{line_no}")
        x1, y1 = (cx - wn / 2) * width, (cy - hn / 2) * height
        x2, y2 = (cx + wn / 2) * width, (cy + hn / 2) * height
        box = [max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"empty label box {path}:{line_no}")
        boxes.append(box)
        classes.append(cls)
    return boxes, classes


def _content_manifest(root: Path, split: str) -> tuple[list[str], str]:
    paths: list[Path] = []
    for folder in (root / "images" / split, root / "labels" / split, root / "labels_ignore" / split):
        if folder.exists():
            paths.extend(p for p in folder.rglob("*") if p.is_file())
    lines = []
    for p in sorted(paths, key=lambda x: x.relative_to(root).as_posix()):
        rel = p.relative_to(root).as_posix()
        lines.append(f"{sha256_file(p)}  {rel}")
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    return lines, sha256_bytes(payload)


def load_dataset(yaml_path: Path | str, *, split: str = "val") -> dict[str, Any]:
    """Load sorted image records and normalized YOLO labels in pixel xyxy."""
    yaml_file = Path(yaml_path).resolve()
    config = _yaml_load(yaml_file)
    root = _resolve_dataset_root(yaml_file, config)
    image_dir = _split_image_dir(root, yaml_file, config, split)
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_paths = sorted((p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts), key=lambda p: p.relative_to(image_dir).as_posix())
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to inspect dataset images") from exc
    records = []
    labels_root = root / "labels" / split
    ignores_root = root / "labels_ignore" / split
    for p in image_paths:
        with Image.open(p) as im:
            width, height = im.size
        rel = p.relative_to(image_dir).as_posix()
        label = labels_root / Path(rel).with_suffix(".txt")
        ignore = ignores_root / Path(rel).with_suffix(".txt")
        gt, classes = _parse_label(label, width, height)
        ign, _ = _parse_label(ignore, width, height, ignore=True)
        records.append({"path": p, "relative_path": rel, "width": width, "height": height, "gt_boxes": gt, "gt_classes": classes, "ignore_boxes": ign})
    manifest, signature = _content_manifest(root, split)
    return {
        "yaml_path": yaml_file,
        "yaml_hash": sha256_file(yaml_file),
        "root": root,
        "split": split,
        "images": records,
        "image_list": [r["relative_path"] for r in records],
        "image_count": len(records),
        "content_manifest": manifest,
        "dataset_signature": signature,
    }


def git_provenance(repo: Path | str = ".") -> dict[str, Any]:
    repo = Path(repo)
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""
    tracked_status = run("status", "--porcelain", "--untracked-files=no")
    all_status = run("status", "--porcelain", "--untracked-files=all")
    tree_hash = hashlib.sha256()
    try:
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo, stderr=subprocess.DEVNULL)
        for raw in sorted((x for x in tracked.split(b"\0") if x), key=lambda x: x):
            path = repo / raw.decode("utf-8")
            if path.is_file():
                tree_hash.update(raw + b"\0" + sha256_file(path).encode("ascii") + b"\0")
    except Exception:
        pass
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "clean_tracked": not bool(tracked_status),
        "untracked": any(line.startswith("??") for line in all_status.splitlines()),
        "source_tree_hash": tree_hash.hexdigest(),
    }


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    for name, module in (("python", None), ("numpy", "numpy"), ("torch", "torch"), ("ultralytics", "ultralytics")):
        if module is None:
            import sys
            info[name] = sys.version.split()[0]
        else:
            try:
                mod = __import__(module)
                info[name] = getattr(mod, "__version__", "unknown")
            except Exception:
                info[name] = None
    try:
        import torch
        info["cuda"] = {"available": bool(torch.cuda.is_available()), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception:
        info["cuda"] = {"available": False, "device": None}
    return info


def protocol_signature(protocol: Any) -> str:
    value = protocol.to_dict() if hasattr(protocol, "to_dict") else protocol
    return sha256_bytes(canonical_json_bytes(value))


# Descriptive aliases used by runner/adjudicator integrations.
canonical_json_dumps = canonical_json
atomic_write_gzip_jsonl = atomic_write_jsonl_gz
dataset_content_signature = lambda root, split="val": _content_manifest(Path(root), split)[1]
validate_finite_legal = _check_value
atomic_json_dump = atomic_write_json
atomic_jsonl_gzip = atomic_write_jsonl_gz
dataset_signature = dataset_content_signature
source_fingerprint = git_provenance
collect_environment = environment_info


__all__ = [
    "canonical_json_bytes", "canonical_json", "atomic_write_bytes", "atomic_write_json",
    "atomic_write_jsonl_gz", "ensure_empty_output", "sha256_bytes", "sha256_file",
    "verify_checksums", "write_checksums", "load_dataset", "git_provenance",
    "environment_info", "protocol_signature",
    "canonical_json_dumps", "atomic_write_gzip_jsonl", "dataset_content_signature",
    "validate_finite_legal", "validate_raw_view_cache",
    "atomic_json_dump", "atomic_jsonl_gzip", "dataset_signature",
    "source_fingerprint", "collect_environment",
]
