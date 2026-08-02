from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import torch

from src.iber_cache import (
    CACHE_FORMAT_VERSION,
    DEFAULT_SHARD_SIZE,
    DESIGN_VERSION,
    REQUIRED_RECORD_TENSORS,
    CacheViolation,
    image_rgb_for_probe,
    load_evidence_cache,
    write_evidence_cache,
)


EXPECTED_TENSORS = (
    "hidden",
    "stock_boxes",
    "stock_scores",
    "f3",
    "image_rgb",
    "target_edges",
    "match_source",
    "match_target",
)
HASH_FIELDS = (
    "baseline_sha256",
    "dataset_sha256",
    "category_sha256",
    "subset_sha256",
    "runtime_amendment_sha256",
)


def _execute_marker_and_return_tensor(marker_path: str) -> torch.Tensor:
    Path(marker_path).write_text("executed", encoding="utf-8")
    return torch.zeros((300, 8), dtype=torch.float16)


class _UnsupportedPicklePayload:
    def __init__(self, marker_path: Path) -> None:
        self.marker_path = marker_path

    def __reduce__(self):
        return (_execute_marker_and_return_tensor, (str(self.marker_path),))


def _authority() -> dict[str, str]:
    return {
        "baseline_sha256": "A" * 64,
        "dataset_sha256": "B" * 64,
        "category_sha256": "C" * 64,
        "subset_sha256": "D" * 64,
        "source_commit": "e" * 40,
        "runtime_amendment_sha256": "F" * 64,
    }


def _record(index: int, image_id: str, value: int = 0) -> dict[str, object]:
    image = torch.full((3, 640, 640), value, dtype=torch.uint8)
    return {
        "index": index,
        "image_id": image_id,
        "hidden": torch.full((300, 8), float(value), dtype=torch.float16),
        "stock_boxes": torch.full((300, 4), 0.4, dtype=torch.float32),
        "stock_scores": torch.full((300, 10), float(value), dtype=torch.float16),
        "f3": torch.full((4, 8, 8), float(value), dtype=torch.float16),
        "image_rgb": image,
        "target_edges": torch.tensor([[0.3, 0.3, 0.5, 0.5]], dtype=torch.float32),
        "match_source": torch.tensor([0], dtype=torch.long),
        "match_target": torch.tensor([0], dtype=torch.long),
    }


def _write_small_cache(root: Path):
    return write_evidence_cache(
        root,
        train_records=[_record(0, "images/train/a.jpg", 1)],
        val_records=[_record(0, "images/val/b.jpg", 2)],
        authority=_authority(),
    )


def _rewrite_manifest(root: Path, mutate) -> None:
    path = root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _load_builder_module():
    path = Path("scripts/cache_iber_evidence.py")
    spec = importlib.util.spec_from_file_location("cache_iber_evidence_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_constants_and_tensor_schema_are_exactly_frozen() -> None:
    assert CACHE_FORMAT_VERSION == 1
    assert DESIGN_VERSION == "iber-be-v1.0"
    assert DEFAULT_SHARD_SIZE == 16
    assert REQUIRED_RECORD_TENSORS == EXPECTED_TENSORS


def test_roundtrip_is_split_isolated_contiguous_and_hashed(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    train = [_record(i, f"images/train/{i}.jpg", i) for i in range(3)]
    val = [_record(i, f"images/val/{i}.jpg", 10 + i) for i in range(2)]

    manifest = write_evidence_cache(
        root,
        train_records=train,
        val_records=val,
        authority=_authority(),
        shard_size=2,
    )
    loaded = load_evidence_cache(root, expected_authority=_authority())

    assert manifest.complete is True
    assert manifest.shard_size == 2
    assert manifest.split_counts == {"train": 3, "val": 2}
    assert [item.split for item in manifest.shards] == ["train", "train", "val"]
    assert [item.start_index for item in manifest.shards] == [0, 2, 0]
    assert [item.end_index for item in manifest.shards] == [1, 2, 1]
    assert all(item.bytes > 0 for item in manifest.shards)
    assert all(re.fullmatch(r"[0-9A-F]{64}", item.sha256) for item in manifest.shards)
    assert [r["index"] for r in loaded.records["train"]] == [0, 1, 2]
    assert [r["index"] for r in loaded.records["val"]] == [0, 1]
    assert set(loaded.records) == {"train", "val"}
    torch.testing.assert_close(loaded.records["train"][2]["hidden"], train[2]["hidden"])


def test_record_schema_rgb_dtype_shape_contiguity_and_probe_conversion_are_exact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    record = _record(0, "images/train/a.jpg")
    image = torch.arange(3 * 640 * 640, dtype=torch.int64).remainder(256)
    record["image_rgb"] = image.reshape(3, 640, 640).to(torch.uint8)
    write_evidence_cache(
        root,
        train_records=[record],
        val_records=[_record(0, "images/val/b.jpg")],
        authority=_authority(),
    )
    loaded = load_evidence_cache(root, expected_authority=_authority())
    actual = loaded.records["train"][0]

    assert set(actual) == {"index", "image_id", *EXPECTED_TENSORS}
    rgb = actual["image_rgb"]
    assert rgb.dtype is torch.uint8
    assert rgb.shape == (3, 640, 640)
    assert rgb.is_contiguous()
    expected = rgb.float().div(255)
    converted = image_rgb_for_probe(actual)
    assert converted.dtype is torch.float32
    torch.testing.assert_close(converted, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda r: r.update(image_rgb=r["image_rgb"].float()), "image_rgb"),
        (lambda r: r.update(image_rgb=torch.zeros(640, 640, 3, dtype=torch.uint8)), "image_rgb"),
        (lambda r: r.update(image_rgb=r["image_rgb"][:, :, ::2]), "image_rgb"),
        (lambda r: r.update(box_l1=torch.zeros(300, 4)), "schema"),
        (lambda r: r.pop("f3"), "f3|schema"),
    ],
)
def test_writer_rejects_dtype_shape_noncontiguous_and_schema_drift(
    tmp_path: Path, mutation, match: str
) -> None:
    record = _record(0, "images/train/a.jpg")
    mutation(record)
    with pytest.raises(CacheViolation, match=match):
        write_evidence_cache(
            tmp_path / "cache",
            train_records=[record],
            val_records=[_record(0, "images/val/b.jpg")],
            authority=_authority(),
        )


@pytest.mark.parametrize("field", (*HASH_FIELDS, "source_commit"))
def test_authority_normalizes_on_write_and_rejects_every_drift(
    tmp_path: Path, field: str
) -> None:
    root = tmp_path / field
    mixed = {name: value.lower() for name, value in _authority().items()}
    manifest = write_evidence_cache(
        root,
        train_records=[_record(0, "images/train/a.jpg")],
        val_records=[_record(0, "images/val/b.jpg")],
        authority=mixed,
    )
    assert all(manifest.authority[name].isupper() for name in HASH_FIELDS)
    assert manifest.authority["source_commit"].islower()

    changed = dict(_authority())
    changed[field] = ("0" * 64) if field != "source_commit" else ("0" * 40)
    with pytest.raises(CacheViolation, match=field):
        load_evidence_cache(root, expected_authority=changed)


@pytest.mark.parametrize("missing", (*HASH_FIELDS, "source_commit"))
def test_authority_requires_all_exact_fields(tmp_path: Path, missing: str) -> None:
    authority = _authority()
    authority.pop(missing)
    with pytest.raises(CacheViolation, match="authority"):
        write_evidence_cache(
            tmp_path / "cache",
            train_records=[_record(0, "images/train/a.jpg")],
            val_records=[_record(0, "images/val/b.jpg")],
            authority=authority,
        )


@pytest.mark.parametrize("mode", ["gap", "duplicate", "cross_split"])
def test_writer_rejects_index_gaps_and_global_image_overlap(
    tmp_path: Path, mode: str
) -> None:
    train = [_record(0, "images/train/a.jpg"), _record(1, "images/train/b.jpg")]
    val = [_record(0, "images/val/c.jpg")]
    if mode == "gap":
        train[1]["index"] = 2
    elif mode == "duplicate":
        train[1]["image_id"] = train[0]["image_id"]
    else:
        val[0]["image_id"] = train[0]["image_id"]
    with pytest.raises(CacheViolation, match="index|overlap"):
        write_evidence_cache(
            tmp_path / "cache",
            train_records=train,
            val_records=val,
            authority=_authority(),
        )


def test_manifest_is_published_last_and_failure_leaves_no_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"

    def fail_save(*_args, **_kwargs):
        raise RuntimeError("injected shard failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="injected"):
        _write_small_cache(root)
    assert not (root / "manifest.json").exists()
    assert not (root / "manifest.json.tmp").exists()


def test_writer_refuses_nonempty_cache_root(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    (root / "owned.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        _write_small_cache(root)


@pytest.mark.parametrize("corruption", ["bytes", "sha256"])
def test_loader_rejects_shard_byte_or_sha_corruption(
    tmp_path: Path, corruption: str
) -> None:
    root = tmp_path / "cache"
    manifest = _write_small_cache(root)
    shard = root.joinpath(*manifest.shards[0].path.split("/"))
    if corruption == "bytes":
        shard.write_bytes(shard.read_bytes() + b"corrupt")
    else:
        _rewrite_manifest(root, lambda p: p["shards"][0].update(sha256="0" * 64))
    with pytest.raises(CacheViolation, match="bytes|sha256"):
        load_evidence_cache(root, expected_authority=_authority())


@pytest.mark.parametrize("unsafe", ["../escape.pt", "/absolute.pt", "C:/escape.pt", "shards\\x.pt"])
def test_loader_rejects_unsafe_non_posix_shard_paths(tmp_path: Path, unsafe: str) -> None:
    root = tmp_path / "cache"
    _write_small_cache(root)
    _rewrite_manifest(root, lambda p: p["shards"][0].update(path=unsafe))
    with pytest.raises(CacheViolation, match="path"):
        load_evidence_cache(root, expected_authority=_authority())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(complete=False),
        lambda p: p.update(format_version=999),
        lambda p: p.update(design_version="wrong"),
        lambda p: p.update(split_counts={"train": 2, "val": 1}),
        lambda p: p["shards"][0].update(start_index=4),
        lambda p: p["shards"][0].update(end_index=9),
        lambda p: p["shards"][0].update(count=7),
        lambda p: p["shards"][0].update(split="other"),
    ],
)
def test_loader_rejects_manifest_and_shard_metadata_drift(
    tmp_path: Path, mutation
) -> None:
    root = tmp_path / "cache"
    _write_small_cache(root)
    _rewrite_manifest(root, mutation)
    with pytest.raises(CacheViolation):
        load_evidence_cache(root, expected_authority=_authority())


def test_loader_validates_every_shard_before_exposing_any_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    manifest = write_evidence_cache(
        root,
        train_records=[_record(0, "images/train/a.jpg"), _record(1, "images/train/b.jpg")],
        val_records=[_record(0, "images/val/c.jpg")],
        authority=_authority(),
        shard_size=1,
    )
    last = root.joinpath(*manifest.shards[-1].path.split("/"))
    last.write_bytes(last.read_bytes() + b"late corruption")
    loads = 0
    original = torch.load

    def count_load(*args, **kwargs):
        nonlocal loads
        loads += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", count_load)
    with pytest.raises(CacheViolation, match="bytes|sha256"):
        load_evidence_cache(root, expected_authority=_authority())
    assert loads == 0


def test_loader_rejects_unsupported_pickle_global_without_executing_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    manifest = _write_small_cache(root)
    shard = root.joinpath(*manifest.shards[0].path.split("/"))
    artifact = torch.load(shard, map_location="cpu", weights_only=True)
    marker = tmp_path / "payload-executed.txt"
    artifact["records"][0]["hidden"] = _UnsupportedPicklePayload(marker)
    torch.save(artifact, shard)

    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["shards"][0]["bytes"] = shard.stat().st_size
    payload["shards"][0]["sha256"] = hashlib.sha256(
        shard.read_bytes()
    ).hexdigest().upper()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CacheViolation, match="load|pickle|shard"):
        load_evidence_cache(root, expected_authority=_authority())
    assert not marker.exists()


def test_loader_source_forces_weights_only_without_fallback() -> None:
    source = Path("src/iber_cache.py").read_text(encoding="utf-8")
    assert "weights_only=True" in source
    assert "weights_only=False" not in source


def test_cache_sources_contain_no_trajectory_or_old_box_names() -> None:
    paths = (Path("src/iber_cache.py"), Path("scripts/cache_iber_evidence.py"))
    forbidden = ("trajectory", "box_l1", "box_l2", "ITBER", "itber_cache", "rtdetr_itber")
    for path in paths:
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        lowered = source.lower()
        assert not any(name.lower() in lowered for name in forbidden), path


def test_builder_cli_arguments_defaults_and_fixed_contract() -> None:
    module = _load_builder_module()
    args = module._parse_args(
        [
            "--baseline-checkpoint", "baseline.pt",
            "--dataset-root", "dataset",
            "--output-root", "cache",
        ]
    )
    assert args.baseline_checkpoint == Path("baseline.pt")
    assert args.dataset_root == Path("dataset")
    assert args.output_root == Path("cache")
    assert (args.batch, args.workers, args.device) == (8, 8, "0")
    assert module.IMAGE_SIZE == 640
    assert module.SHARD_SIZE == 16
    assert module.TRAIN_COUNT == 647
    assert module.VAL_COUNT == 548


def test_builder_source_locks_subset_rgb_forward_matcher_and_context_cleanup() -> None:
    path = Path("scripts/cache_iber_evidence.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert "select_hashed_subset" in source
    assert "subset_signature" in source
    assert "EXPECTED_SUBSET_SHA256" in source
    assert "FrozenIBERAdapter.from_detector" in source
    assert "with FrozenIBERAdapter.from_detector" in source
    assert ".forward_evidence(images)" in source
    assert ".criterion.matcher(" in source
    exact_rgb = ".mul(255).round().clamp(0, 255).to(torch.uint8).cpu()"
    assert exact_rgb in source
    assert "image_rgb_for_probe" not in source
    assert any(isinstance(node.func, ast.Attribute) and node.func.attr == "forward_evidence" for node in calls)
