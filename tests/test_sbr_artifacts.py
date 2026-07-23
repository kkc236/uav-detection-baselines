from pathlib import Path
import gzip
import json

import pytest


def test_canonical_json_and_atomic_writers(tmp_path: Path):
    from src.sbr_artifacts import canonical_json_bytes, atomic_write_json, atomic_write_jsonl_gz

    assert canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    out = tmp_path / "x.json"
    atomic_write_json(out, {"b": 1, "a": 2})
    assert out.read_bytes() == b'{"a":2,"b":1}'
    rows = tmp_path / "rows.jsonl.gz"
    atomic_write_jsonl_gz(rows, [{"z": 1}, {"a": 2}])
    with gzip.open(rows, "rt", encoding="utf-8") as fh:
        assert fh.read() == '{"z":1}\n{"a":2}\n'


def test_nonempty_output_and_hash_mismatch_fail_closed(tmp_path: Path):
    from src.sbr_artifacts import ensure_empty_output, sha256_bytes, verify_checksums

    folder = tmp_path / "out"
    ensure_empty_output(folder)
    (folder / "existing").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ensure_empty_output(folder)
    with pytest.raises(ValueError):
        verify_checksums({"file": "00"}, {"file": b"not-zero"})
    assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_dataset_loader_signature_and_labels(tmp_path: Path):
    from PIL import Image
    from src.sbr_artifacts import load_dataset

    root = tmp_path / "data"
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "labels_ignore" / "val").mkdir(parents=True)
    Image.new("RGB", (100, 50)).save(root / "images" / "val" / "b.jpg")
    Image.new("RGB", (100, 50)).save(root / "images" / "val" / "a.jpg")
    (root / "labels" / "val" / "a.txt").write_text("2 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    (root / "labels_ignore" / "val" / "a.txt").write_text("0.5 0.5 0.1 0.2\n", encoding="utf-8")
    yaml = tmp_path / "data.yaml"
    yaml.write_text("path: %s\nval: images/val\n" % root.as_posix(), encoding="utf-8")
    ds = load_dataset(yaml, split="val")
    assert [x["path"].name for x in ds["images"]] == ["a.jpg", "b.jpg"]
    assert ds["images"][0]["gt_boxes"][0] == pytest.approx([40, 15, 60, 35])
    assert ds["images"][0]["ignore_boxes"][0] == pytest.approx([45, 20, 55, 30])
    assert ds["image_count"] == 2 and len(ds["dataset_signature"]) == 64


def test_git_provenance_records_untracked_and_tree_hash():
    from src.sbr_artifacts import git_provenance

    info = git_provenance(Path.cwd())
    assert "untracked" in info
    assert "source_tree_hash" in info
    assert len(info["source_tree_hash"]) == 64
