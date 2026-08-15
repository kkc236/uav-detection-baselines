"""Run the frozen detached C0/C1/Q RT-DETR learnable quality probe."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import secrets
import sys
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_rtdetr_quality_oracle import (  # noqa: E402
    BATCH_SIZE,
    CONFIDENCE,
    IMAGE_SIZE,
    MAX_DET,
    NUM_CLASSES,
    STOCK_AUTHORITY,
    VAL_COUNT,
    WORKERS,
    _assert_cuda0_detector,
    _assert_cuda0_tensor,
    _assert_detector_isolated,
    _assert_full_dataset_authority,
    _assert_stock_authority,
    _batch_target,
    _build_validation_loader,
    _device,
    _execution_environment,
    _file_sha256,
    _load_detector,
    _prediction_record,
    _read_canonical_json,
    _source_commit,
    _state_sha256,
    _write_image_list_create_only,
    _write_or_validate_canonical_json,
)
from src.iber_evaluation import compute_detection_metrics  # noqa: E402
from src.iber_protocol import (  # noqa: E402
    EXPECTED_BASELINE_SHA256,
    EXPECTED_DATASET_SHA256,
    EXPECTED_SUBSET_SHA256,
)
from src.lpr_protocol import (  # noqa: E402
    CATEGORY_NAMES,
    category_mapping_sha256,
    select_hashed_subset,
    subset_signature,
)
from src.rtdetr_quality_oracle import (  # noqa: E402
    DEV_COUNT,
    EXPECTED_DEV_SHA256,
    flattened_topk,
    ordered_path_sha256,
    same_class_iou_quality,
    select_internal_dev,
)
from src.rtdetr_quality_probe import (  # noqa: E402
    C1QualityProbe,
    PROBE_ALPHA,
    QQualityProbe,
    c1_features,
    evaluate_internal_probe_gate,
    quality_probe_loss,
    rerank_with_predicted_quality,
    top_pair_mask,
)


PROBE_TRAIN_COUNT = 518
TRAIN_SUBSET_COUNT = 647
HIDDEN_DIM = 256
EPOCHS = 20
TOP_PAIRS = 600
SEED = 0
LR0 = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
SHARD_SIZE = 32
EXPECTED_PROBE_TRAIN_SHA256 = (
    "1E46817FFFBDBCBA0BA1675CA6142ABABBD6147394AA1D0F10B57F0ECAF7236D"
)
ORACLE_DECISION_PATH = (
    REPOSITORY_ROOT / "evidence" / "quality-oracle" / "quality-oracle-decision.json"
)
EXPECTED_ORACLE_DECISION_SHA256 = (
    "F2DBABDD4638896D3D9C727CCC659D86173DD639AF476709C8F415F0E2EEE199"
)

_ordered_path_sha256 = ordered_path_sha256


class CapturedBatch(NamedTuple):
    stock: torch.Tensor
    boxes: torch.Tensor
    logits: torch.Tensor
    hidden: torch.Tensor


def _oracle_decision_authority() -> dict[str, Any]:
    path = ORACLE_DECISION_PATH.resolve()
    digest = _file_sha256(path)
    if digest != EXPECTED_ORACLE_DECISION_SHA256:
        raise RuntimeError("quality oracle decision sha256 mismatch")
    payload = _read_canonical_json(path)
    if (
        payload.get("format_version") != 1
        or payload.get("status") != "passed"
        or payload.get("finite") is not True
        or payload.get("selected_alpha") != PROBE_ALPHA
        or payload.get("deltas")
        != {"map": "0.15571345572052406", "ap75": "0.14920384179689443"}
    ):
        raise RuntimeError("quality oracle decision did not pass the frozen Gate")
    return {
        "sha256": digest,
        "status": payload["status"],
        "selected_alpha": payload["selected_alpha"],
        "map_delta": payload["deltas"]["map"],
        "ap75_delta": payload["deltas"]["ap75"],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser.parse_args(argv)


def _split_probe_paths(
    subset: Sequence[Path], dev: Sequence[Path], *, root: Path
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    root = Path(root).resolve()
    subset_paths = tuple(Path(path).resolve() for path in subset)
    dev_paths = tuple(Path(path).resolve() for path in dev)
    if len(subset_paths) != TRAIN_SUBSET_COUNT or len(set(subset_paths)) != len(subset_paths):
        raise RuntimeError("fixed subset must contain exactly 647 unique paths")
    if len(dev_paths) != DEV_COUNT or len(set(dev_paths)) != len(dev_paths):
        raise RuntimeError("internal dev must contain exactly 129 unique paths")
    subset_set = set(subset_paths)
    if not set(dev_paths) <= subset_set:
        raise RuntimeError("internal dev is not a subset of the fixed 647 images")
    train_paths = tuple(path for path in subset_paths if path not in set(dev_paths))
    if len(train_paths) != PROBE_TRAIN_COUNT or set(train_paths) & set(dev_paths):
        raise RuntimeError("probe train/dev partition is invalid")
    for path in (*train_paths, *dev_paths):
        path.relative_to(root)
    return train_paths, dev_paths


def _prepare_probe_paths(
    dataset_root: Path, cache_root: Path
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    root = Path(dataset_root).resolve()
    all_train = tuple(sorted((root / "images" / "train").glob("*.jpg")))
    if len(all_train) != 6471:
        raise RuntimeError(f"training image count mismatch: {len(all_train)}")
    subset = tuple(select_hashed_subset(all_train, root=root, fraction=0.10))
    if len(subset) != TRAIN_SUBSET_COUNT:
        raise RuntimeError("fixed subset count mismatch")
    if subset_signature(subset, root=root) != EXPECTED_SUBSET_SHA256:
        raise RuntimeError("fixed subset authority mismatch")
    dev = tuple(select_internal_dev(subset, root=root))
    train, dev = _split_probe_paths(subset, dev, root=root)
    if ordered_path_sha256(dev, root=root) != EXPECTED_DEV_SHA256:
        raise RuntimeError("internal-dev ordered hash mismatch")
    if ordered_path_sha256(train, root=root) != EXPECTED_PROBE_TRAIN_SHA256:
        raise RuntimeError("probe-train ordered hash mismatch")
    list_path = Path(cache_root).resolve().parent / f"{Path(cache_root).name}-subset.txt"
    _write_image_list_create_only(list_path, subset)
    return subset, train, dev


def _schema_sha256() -> str:
    payload = {
        "identity": "rtdetr-learnable-quality-probe-v1",
        "image_size": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "workers": WORKERS,
        "confidence": CONFIDENCE,
        "max_det": MAX_DET,
        "nms": False,
        "subset_count": TRAIN_SUBSET_COUNT,
        "probe_train_count": PROBE_TRAIN_COUNT,
        "dev_count": DEV_COUNT,
        "val_count": VAL_COUNT,
        "num_classes": NUM_CLASSES,
        "hidden_dim": HIDDEN_DIM,
        "epochs": EPOCHS,
        "top_pairs": TOP_PAIRS,
        "alpha": PROBE_ALPHA,
        "seed": SEED,
        "optimizer": {
            "name": "Ultralytics MuSGD",
            "lr": LR0,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "nesterov": True,
            "muon": 0.2,
            "sgd": 1.0,
        },
        "classes": list(CATEGORY_NAMES),
        "category_sha256": category_mapping_sha256(CATEGORY_NAMES),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest().upper()


def _build_authority(
    baseline: Path,
    dataset_root: Path,
    train_paths: Sequence[Path],
    dev_paths: Sequence[Path],
) -> dict[str, str]:
    oracle_authority = _oracle_decision_authority()
    baseline_sha = _file_sha256(Path(baseline).resolve())
    if baseline_sha != EXPECTED_BASELINE_SHA256:
        raise RuntimeError("baseline authority mismatch")
    root = Path(dataset_root).resolve()
    authority = {
        "baseline_sha256": baseline_sha,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "subset_sha256": EXPECTED_SUBSET_SHA256,
        "probe_train_sha256": ordered_path_sha256(train_paths, root=root),
        "dev_sha256": ordered_path_sha256(dev_paths, root=root),
        "schema_sha256": _schema_sha256(),
        "oracle_decision_sha256": oracle_authority["sha256"],
        "source_commit": _source_commit(),
    }
    if authority["probe_train_sha256"] != EXPECTED_PROBE_TRAIN_SHA256:
        raise RuntimeError("probe-train authority mismatch")
    if authority["dev_sha256"] != EXPECTED_DEV_SHA256:
        raise RuntimeError("internal-dev authority mismatch")
    environment_raw = json.dumps(
        _execution_environment(), sort_keys=True, separators=(",", ":")
    ).encode()
    authority["environment_sha256"] = hashlib.sha256(environment_raw).hexdigest().upper()
    return authority


def _raw_decoder_batch(detector: Any, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    detector.model[-1].export = False
    with torch.inference_mode():
        stock, auxiliary = detector.predict(images)
    if not isinstance(auxiliary, tuple) or len(auxiliary) != 5:
        raise RuntimeError("RT-DETR auxiliary decoder tuple is invalid")
    decoder_boxes, decoder_logits, _, _, _ = auxiliary
    boxes = decoder_boxes[-1].detach().float()
    logits = decoder_logits[-1].detach().float()
    with torch.inference_mode():
        reconstructed = detector.model[-1].postprocess(boxes, logits.sigmoid())
    if not torch.equal(stock, reconstructed):
        raise RuntimeError("stock output reconstruction mismatch")
    if boxes.ndim != 3 or boxes.shape[-1] != 4 or logits.ndim != 3:
        raise RuntimeError("decoder evidence shape is invalid")
    if logits.shape[:2] != boxes.shape[:2] or logits.shape[-1] != NUM_CLASSES:
        raise RuntimeError("decoder logits disagree with boxes/classes")
    if not bool(torch.isfinite(boxes).all() and torch.isfinite(logits).all()):
        raise RuntimeError("decoder evidence contains non-finite values")
    if stock.requires_grad or boxes.requires_grad or logits.requires_grad:
        raise RuntimeError("detector evidence is attached to gradients")
    return stock.detach(), boxes, logits


def _capture_hidden_batch(detector: Any, images: torch.Tensor) -> CapturedBatch:
    head = detector.model[-1]
    eval_index = int(head.decoder.eval_idx)
    score_head = head.dec_score_head[eval_index]
    captures: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("score-head hook input is invalid")
        captures.append(args[0].detach())
        return None

    handle = score_head.register_forward_pre_hook(capture)
    try:
        stock, boxes, logits = _raw_decoder_batch(detector, images)
    finally:
        handle.remove()
    if len(captures) != 1:
        raise RuntimeError(f"score-head hook fired {len(captures)} times")
    hidden = captures[0].detach().float()
    if hidden.shape[:2] != boxes.shape[:2]:
        raise RuntimeError("captured decoder hidden shape is invalid")
    with torch.inference_mode():
        reproduced_logits = score_head(hidden).detach().float()
    if not torch.equal(reproduced_logits, logits):
        raise RuntimeError("captured hidden does not reproduce final logits")
    if not bool(torch.isfinite(hidden).all()) or hidden.requires_grad:
        raise RuntimeError("captured hidden is invalid")
    _assert_detector_isolated(detector)
    return CapturedBatch(stock, boxes, logits, hidden)


def _prove_hook_neutrality(detector: Any, images: torch.Tensor) -> dict[str, Any]:
    state_before = _state_sha256(detector)
    stock, boxes, logits = _raw_decoder_batch(detector, images)
    hooked = _capture_hidden_batch(detector, images)
    for name, expected, actual in (
        ("stock", stock, hooked.stock),
        ("boxes", boxes, hooked.boxes),
        ("logits", logits, hooked.logits),
    ):
        if expected.shape != actual.shape or expected.dtype != actual.dtype or not torch.equal(expected, actual):
            raise RuntimeError(f"hook changed {name}")
    if _state_sha256(detector) != state_before:
        raise RuntimeError("detector state changed during hook canary")
    _assert_detector_isolated(detector)
    return {"hook_calls": 1, "output_neutral": True}


def _canonical_image_id(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError("record image_id is not canonical POSIX")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError("record image_id is not a safe relative path")
    return path.as_posix()


def _validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "image_id",
        "boxes",
        "logits",
        "hidden",
        "quality",
        "target_boxes",
        "target_classes",
    }
    if set(record) != expected_keys:
        raise RuntimeError("probe cache record schema mismatch")
    image_id = _canonical_image_id(record["image_id"])
    tensors = {name: record[name] for name in expected_keys - {"image_id"}}
    if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        raise RuntimeError("probe cache record contains a non-tensor field")
    boxes = tensors["boxes"].detach().cpu().contiguous()
    logits = tensors["logits"].detach().cpu().contiguous()
    hidden = tensors["hidden"].detach().cpu().contiguous()
    quality = tensors["quality"].detach().cpu().contiguous()
    target_boxes = tensors["target_boxes"].detach().cpu().contiguous()
    target_classes = tensors["target_classes"].detach().cpu().contiguous()
    if boxes.dtype != torch.float32 or boxes.shape != (MAX_DET, 4):
        raise RuntimeError("cached boxes must be float32 [300,4]")
    if logits.dtype != torch.float32 or logits.shape != (MAX_DET, NUM_CLASSES):
        raise RuntimeError("cached logits must be float32 [300,10]")
    if hidden.dtype != torch.float32 or hidden.shape != (MAX_DET, HIDDEN_DIM):
        raise RuntimeError("cached hidden must be float32 [300,256]")
    if quality.dtype != torch.float32 or quality.shape != (MAX_DET, NUM_CLASSES):
        raise RuntimeError("cached quality must be float32 [300,10]")
    if target_boxes.dtype != torch.float32 or target_boxes.ndim != 2 or target_boxes.shape[1:] != (4,):
        raise RuntimeError("cached target_boxes must be float32 [N,4]")
    if target_classes.dtype != torch.int64 or target_classes.shape != (target_boxes.shape[0],):
        raise RuntimeError("cached target_classes must be int64 [N]")
    if target_classes.numel() and bool(((target_classes < 0) | (target_classes >= NUM_CLASSES)).any()):
        raise RuntimeError("cached target class is outside the category mapping")
    for name, value in tensors.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"cached {name} contains non-finite values")
    recomputed = same_class_iou_quality(boxes, target_boxes, target_classes, NUM_CLASSES)
    if not torch.equal(recomputed, quality):
        raise RuntimeError("cached quality target does not reproduce exactly")
    return {
        "image_id": image_id,
        "boxes": boxes.clone(),
        "logits": logits.clone(),
        "hidden": hidden.clone(),
        "quality": quality.clone(),
        "target_boxes": target_boxes.clone(),
        "target_classes": target_classes.clone(),
    }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_create_bytes(path: Path, raw: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    _atomic_create_bytes(path, buffer.getvalue())


def _safe_file_bytes(path: Path) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"unsafe or missing immutable file: {path}")
    return path.read_bytes()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def _load_torch_bytes(raw: bytes, *, label: str) -> Any:
    try:
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError(f"{label} is corrupt or cannot load") from error


def _write_cache_stage(
    root: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    authority: Mapping[str, str],
    split: str,
    external_digest_path: Path,
) -> dict[str, Any]:
    root = Path(root)
    normalized = [_validate_record(record) for record in records]
    ids = [record["image_id"] for record in normalized]
    if len(set(ids)) != len(ids):
        raise RuntimeError("cache record identities are not unique")
    if root.exists() and (root / "manifest.json").is_file():
        _load_cache_stage(
            root,
            authority=authority,
            split=split,
            external_digest_path=external_digest_path,
            expected_ids=tuple(ids),
        )
        return _read_canonical_json(external_digest_path)
    if root.is_symlink():
        raise RuntimeError("cache stage root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    intent = {
        "format_version": 1,
        "authority": dict(authority),
        "split": split,
        "count": len(normalized),
        "image_ids": ids,
        "shard_size": SHARD_SIZE,
    }
    _write_or_validate_canonical_json(root / "intent.json", intent)
    shard_metadata: list[dict[str, Any]] = []
    for start in range(0, len(normalized), SHARD_SIZE):
        index = start // SHARD_SIZE
        shard_path = root / f"shard-{index:04d}.pt"
        chunk = normalized[start : start + SHARD_SIZE]
        if not shard_path.exists():
            _atomic_torch_save(shard_path, chunk)
        raw = _safe_file_bytes(shard_path)
        loaded = _load_torch_bytes(raw, label=shard_path.name)
        if not isinstance(loaded, list):
            raise RuntimeError("cache shard payload is invalid")
        checked = [_validate_record(record) for record in loaded]
        if [record["image_id"] for record in checked] != [record["image_id"] for record in chunk]:
            raise RuntimeError("cache shard identities differ during resume")
        shard_metadata.append(
            {
                "path": shard_path.name,
                "count": len(checked),
                "bytes": len(raw),
                "sha256": _sha256_bytes(raw),
            }
        )
    manifest = {**intent, "shards": shard_metadata}
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        _atomic_create_bytes(manifest_path, _canonical_json_bytes(manifest))
    elif _safe_file_bytes(manifest_path) != _canonical_json_bytes(manifest):
        raise RuntimeError("immutable cache manifest differs")
    manifest_raw = _safe_file_bytes(manifest_path)
    digest = {
        "format_version": 1,
        "authority": dict(authority),
        "split": split,
        "count": len(normalized),
        "manifest_sha256": _sha256_bytes(manifest_raw),
    }
    _write_or_validate_canonical_json(external_digest_path, digest)
    return digest


def _load_cache_stage(
    root: Path,
    *,
    authority: Mapping[str, str],
    split: str,
    external_digest_path: Path,
    expected_ids: Sequence[str],
) -> list[dict[str, Any]]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("cache stage root is missing or unsafe")
    digest = _read_canonical_json(external_digest_path)
    manifest_raw = _safe_file_bytes(root / "manifest.json")
    if digest != {
        "format_version": 1,
        "authority": dict(authority),
        "split": split,
        "count": len(expected_ids),
        "manifest_sha256": _sha256_bytes(manifest_raw),
    }:
        raise RuntimeError("cache external digest authority mismatch")
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("cache manifest is corrupt") from error
    if manifest_raw != _canonical_json_bytes(manifest):
        raise RuntimeError("cache manifest is not canonical")
    if (
        manifest.get("authority") != dict(authority)
        or manifest.get("split") != split
        or manifest.get("image_ids") != list(expected_ids)
        or manifest.get("count") != len(expected_ids)
    ):
        raise RuntimeError("cache manifest authority mismatch")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise RuntimeError("cache manifest shards are invalid")
    allowed = {"intent.json", "manifest.json", *(item.get("path") for item in shards if isinstance(item, dict))}
    actual = {path.name for path in root.iterdir()}
    if actual != allowed:
        raise RuntimeError("cache stage contains missing or unexpected files")
    records: list[dict[str, Any]] = []
    for item in shards:
        if not isinstance(item, dict) or set(item) != {"path", "count", "bytes", "sha256"}:
            raise RuntimeError("cache shard metadata is invalid")
        raw = _safe_file_bytes(root / item["path"])
        if len(raw) != item["bytes"] or _sha256_bytes(raw) != item["sha256"]:
            raise RuntimeError("cache shard sha256 or byte count mismatch")
        loaded = _load_torch_bytes(raw, label=item["path"])
        if not isinstance(loaded, list) or len(loaded) != item["count"]:
            raise RuntimeError("cache shard record count mismatch")
        records.extend(_validate_record(record) for record in loaded)
    if [record["image_id"] for record in records] != list(expected_ids):
        raise RuntimeError("cache record identity/order mismatch")
    return records


def _relative_id(path: Path, *, root: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _extract_records(
    detector: torch.nn.Module,
    loader: Any,
    validator: Any,
    *,
    dataset_root: Path,
    device: torch.device,
    expected_count: int,
) -> list[dict[str, Any]]:
    if device != torch.device("cuda:0"):
        raise RuntimeError("probe extraction requires cuda:0")
    _assert_cuda0_detector(detector)
    _assert_detector_isolated(detector)
    state_before = _state_sha256(detector)
    records: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = validator.preprocess(raw_batch)
        images = batch["img"]
        _assert_cuda0_tensor(images, label="preprocessed input")
        captured = _capture_hidden_batch(detector, images)
        if captured.boxes.shape[1:] != (MAX_DET, 4):
            raise RuntimeError("production box shape mismatch")
        if captured.logits.shape[1:] != (MAX_DET, NUM_CLASSES):
            raise RuntimeError("production logit shape mismatch")
        if captured.hidden.shape[1:] != (MAX_DET, HIDDEN_DIM):
            raise RuntimeError("production hidden shape mismatch")
        image_ids = batch.get("im_file")
        if not isinstance(image_ids, Sequence) or isinstance(image_ids, (str, bytes)):
            raise RuntimeError("validator batch is missing image identities")
        for image_index, image_id in enumerate(image_ids):
            target_boxes, target_classes = _batch_target(batch, image_index)
            boxes = captured.boxes[image_index].detach().float().cpu().contiguous().clone()
            logits = captured.logits[image_index].detach().float().cpu().contiguous().clone()
            hidden = captured.hidden[image_index].detach().float().cpu().contiguous().clone()
            quality = same_class_iou_quality(
                boxes, target_boxes, target_classes, NUM_CLASSES
            ).detach().float().cpu().contiguous()
            records.append(
                _validate_record(
                    {
                        "image_id": _relative_id(Path(str(image_id)), root=dataset_root),
                        "boxes": boxes,
                        "logits": logits,
                        "hidden": hidden,
                        "quality": quality,
                        "target_boxes": target_boxes,
                        "target_classes": target_classes,
                    }
                )
            )
    if len(records) != expected_count:
        raise RuntimeError(
            f"probe evidence count mismatch: expected={expected_count}, actual={len(records)}"
        )
    if _state_sha256(detector) != state_before:
        raise RuntimeError("detector state changed during extraction")
    _assert_detector_isolated(detector)
    return records


def _reorder_records(
    records: Sequence[Mapping[str, Any]], paths: Sequence[Path], *, root: Path
) -> list[dict[str, Any]]:
    by_id = {record["image_id"]: record for record in records}
    expected = [_relative_id(path, root=root) for path in paths]
    if len(by_id) != len(records) or set(by_id) != set(expected):
        raise RuntimeError("extracted record identity set mismatch")
    return [dict(by_id[image_id]) for image_id in expected]


def _top_pair_probe_loss(
    predicted_logits: torch.Tensor,
    target_quality: torch.Tensor,
    stock_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    mask = top_pair_mask(stock_probabilities.detach(), topk=TOP_PAIRS)
    selected_per_image = int(mask.sum().item() // mask.shape[0])
    if selected_per_image != TOP_PAIRS:
        raise RuntimeError("top-pair selection count mismatch")
    return (
        quality_probe_loss(
            predicted_logits[mask], target_quality.detach()[mask], stock_probabilities.detach()[mask]
        ),
        selected_per_image,
    )


def _select_checkpoint(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not history:
        raise RuntimeError("checkpoint history is empty")
    checked: list[dict[str, Any]] = []
    for item in history:
        metrics = item.get("metrics")
        epoch = item.get("epoch")
        if (
            type(epoch) is not int
            or epoch < 1
            or not isinstance(metrics, Mapping)
            or any(
                name not in metrics
                or not isinstance(metrics[name], (int, float))
                or isinstance(metrics[name], bool)
                or not math.isfinite(float(metrics[name]))
                for name in ("map", "ap75")
            )
        ):
            raise RuntimeError("checkpoint metrics are invalid")
        checked.append(dict(item))
    return max(
        checked,
        key=lambda item: (
            float(item["metrics"]["map"]),
            float(item["metrics"]["ap75"]),
            -int(item["epoch"]),
        ),
    )


def _stack_records(records: Sequence[Mapping[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.stack([record[name] for record in records]).to(device)
        for name in ("boxes", "logits", "hidden", "quality")
    }


def _probe_forward(
    model: torch.nn.Module,
    arm: str,
    boxes: torch.Tensor,
    logits: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    features = c1_features(boxes, logits, num_classes=NUM_CLASSES)
    if arm == "c1":
        return model(features)
    if arm == "q":
        return model(features, hidden.detach())
    raise ValueError("trainable arm must be c1 or q")


def _model_for_arm(arm: str, device: torch.device) -> torch.nn.Module:
    feature_dim = 10 + NUM_CLASSES
    if arm == "c1":
        return C1QualityProbe(feature_dim=feature_dim).to(device)
    if arm == "q":
        return QQualityProbe(feature_dim=feature_dim, hidden_dim=HIDDEN_DIM).to(device)
    raise ValueError("trainable arm must be c1 or q")


def _build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    from ultralytics.optim.muon import MuSGD

    muon_parameters = [parameter for parameter in model.parameters() if parameter.ndim == 2]
    non_muon_parameters = [parameter for parameter in model.parameters() if parameter.ndim != 2]
    if not muon_parameters or not non_muon_parameters:
        raise RuntimeError("quality probe MuSGD parameter grouping is incomplete")
    return MuSGD(
        [
            {
                "params": muon_parameters,
                "use_muon": True,
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": non_muon_parameters,
                "use_muon": False,
                "weight_decay": 0.0,
            },
        ],
        lr=LR0,
        momentum=MOMENTUM,
        weight_decay=0.0,
        nesterov=True,
        muon=0.2,
        sgd=1.0,
    )


def _evaluate_probe_records(
    records: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    model: torch.nn.Module | None,
    device: torch.device,
) -> dict[str, float]:
    if not records:
        raise RuntimeError("probe evaluation records are empty")
    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []
    if model is not None:
        model.eval()
    with torch.inference_mode():
        for start in range(0, len(records), BATCH_SIZE):
            chunk = records[start : start + BATCH_SIZE]
            tensors = _stack_records(chunk, device)
            if arm == "c0":
                output = flattened_topk(
                    tensors["boxes"], tensors["logits"].sigmoid(),
                    num_classes=NUM_CLASSES, max_det=MAX_DET,
                )
            else:
                if model is None:
                    raise RuntimeError("trainable probe evaluation requires a model")
                quality_logits = _probe_forward(
                    model, arm, tensors["boxes"], tensors["logits"], tensors["hidden"]
                )
                output = rerank_with_predicted_quality(
                    tensors["boxes"], tensors["logits"], quality_logits,
                    num_classes=NUM_CLASSES, max_det=MAX_DET,
                )
            for index, record in enumerate(chunk):
                predictions.append(_prediction_record(output[index]))
                targets.append(
                    {
                        "boxes": record["target_boxes"].detach().float().cpu(),
                        "classes": record["target_classes"].detach().long().cpu(),
                    }
                )
    return compute_detection_metrics(predictions, targets, image_size=IMAGE_SIZE)


def _set_deterministic() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _checkpoint_paths(report_root: Path, arm: str, epoch: int) -> tuple[Path, Path]:
    root = Path(report_root) / "checkpoints" / arm
    root.mkdir(parents=True, exist_ok=True)
    return root / f"epoch-{epoch:02d}.pt", root / f"epoch-{epoch:02d}.json"


def _checkpoint_payload(
    *,
    arm: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    authority: Mapping[str, str],
    permutation_sha256: str,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "arm": arm,
        "epoch": epoch,
        "authority": dict(authority),
        "permutation_sha256": permutation_sha256,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = _load_torch_bytes(_safe_file_bytes(path), label=path.name)
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint payload is invalid")
    return payload


def _checkpoint_sidecar(
    path: Path,
    *,
    arm: str,
    epoch: int,
    authority: Mapping[str, str],
    metrics: Mapping[str, float],
    permutation_sha256: str,
) -> dict[str, Any]:
    raw = _safe_file_bytes(path)
    return {
        "format_version": 1,
        "arm": arm,
        "epoch": epoch,
        "authority": dict(authority),
        "permutation_sha256": permutation_sha256,
        "checkpoint": {"path": path.name, "bytes": len(raw), "sha256": _sha256_bytes(raw)},
        "metrics": dict(metrics),
    }


def _validate_checkpoint_pair(
    checkpoint: Path,
    sidecar: Path,
    *,
    arm: str,
    epoch: int,
    authority: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _read_canonical_json(sidecar)
    raw = _safe_file_bytes(checkpoint)
    expected_file = {"path": checkpoint.name, "bytes": len(raw), "sha256": _sha256_bytes(raw)}
    if (
        metadata.get("format_version") != 1
        or metadata.get("arm") != arm
        or metadata.get("epoch") != epoch
        or metadata.get("authority") != dict(authority)
        or metadata.get("checkpoint") != expected_file
    ):
        raise RuntimeError("checkpoint sidecar authority mismatch")
    payload = _load_torch_bytes(raw, label=checkpoint.name)
    if (
        not isinstance(payload, dict)
        or payload.get("arm") != arm
        or payload.get("epoch") != epoch
        or payload.get("authority") != dict(authority)
        or payload.get("permutation_sha256") != metadata.get("permutation_sha256")
    ):
        raise RuntimeError("checkpoint payload authority mismatch")
    return payload, metadata


def _permutation(epoch: int, count: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(SEED + epoch)
    return torch.randperm(count, generator=generator)


def _permutation_sha256(permutation: torch.Tensor) -> str:
    return hashlib.sha256(permutation.numpy().astype("<i8", copy=False).tobytes()).hexdigest().upper()


def _train_arm(
    arm: str,
    train_records: Sequence[Mapping[str, Any]],
    dev_records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    report_root: Path,
    authority: Mapping[str, str],
) -> tuple[torch.nn.Module, list[dict[str, Any]]]:
    if len(train_records) != PROBE_TRAIN_COUNT or len(dev_records) != DEV_COUNT:
        raise RuntimeError("probe training split count mismatch")
    _set_deterministic()
    model = _model_for_arm(arm, device)
    optimizer = _build_optimizer(model)
    history: list[dict[str, Any]] = []
    next_epoch = 1
    for epoch in range(1, EPOCHS + 1):
        checkpoint, sidecar = _checkpoint_paths(report_root, arm, epoch)
        if checkpoint.exists() and sidecar.exists():
            payload, metadata = _validate_checkpoint_pair(
                checkpoint, sidecar, arm=arm, epoch=epoch, authority=authority
            )
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            history.append(metadata)
            next_epoch = epoch + 1
            continue
        if sidecar.exists() and not checkpoint.exists():
            raise RuntimeError("checkpoint sidecar exists without checkpoint")
        if checkpoint.exists() and not sidecar.exists():
            payload = _load_checkpoint(checkpoint)
            if (
                payload.get("arm") != arm
                or payload.get("epoch") != epoch
                or payload.get("authority") != dict(authority)
            ):
                raise RuntimeError("orphan checkpoint authority mismatch")
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            metrics = _evaluate_probe_records(
                dev_records, arm=arm, model=model, device=device
            )
            metadata = _checkpoint_sidecar(
                checkpoint,
                arm=arm,
                epoch=epoch,
                authority=authority,
                metrics=metrics,
                permutation_sha256=payload["permutation_sha256"],
            )
            _write_or_validate_canonical_json(sidecar, metadata)
            history.append(metadata)
            next_epoch = epoch + 1
            continue
        if any(
            any(path.exists() for path in _checkpoint_paths(report_root, arm, later))
            for later in range(epoch + 1, EPOCHS + 1)
        ):
            raise RuntimeError("checkpoint sequence is not contiguous")
        next_epoch = epoch
        break
    if next_epoch > EPOCHS:
        return model, history

    for epoch in range(next_epoch, EPOCHS + 1):
        model.train()
        permutation = _permutation(epoch, len(train_records))
        permutation_sha = _permutation_sha256(permutation)
        for start in range(0, len(train_records), BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE].tolist()
            batch = _stack_records([train_records[index] for index in indices], device)
            optimizer.zero_grad(set_to_none=True)
            predicted = _probe_forward(
                model, arm, batch["boxes"], batch["logits"], batch["hidden"]
            )
            loss, _ = _top_pair_probe_loss(
                predicted, batch["quality"], batch["logits"].sigmoid()
            )
            loss.backward()
            optimizer.step()
        metrics = _evaluate_probe_records(dev_records, arm=arm, model=model, device=device)
        checkpoint, sidecar = _checkpoint_paths(report_root, arm, epoch)
        payload = _checkpoint_payload(
            arm=arm,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            authority=authority,
            permutation_sha256=permutation_sha,
        )
        _atomic_torch_save(checkpoint, payload)
        metadata = _checkpoint_sidecar(
            checkpoint,
            arm=arm,
            epoch=epoch,
            authority=authority,
            metrics=metrics,
            permutation_sha256=permutation_sha,
        )
        _write_or_validate_canonical_json(sidecar, metadata)
        history.append(metadata)
    return model, history


def _load_selected_model(
    arm: str,
    selected: Mapping[str, Any],
    *,
    device: torch.device,
    report_root: Path,
    authority: Mapping[str, str],
) -> torch.nn.Module:
    epoch = int(selected["epoch"])
    checkpoint, sidecar = _checkpoint_paths(report_root, arm, epoch)
    payload, _ = _validate_checkpoint_pair(
        checkpoint, sidecar, arm=arm, epoch=epoch, authority=authority
    )
    _set_deterministic()
    model = _model_for_arm(arm, device)
    model.load_state_dict(payload["model"])
    return model.eval()


def _official_gate(*, c0: Mapping[str, float], q: Mapping[str, float]) -> dict[str, Any]:
    for metrics in (c0, q):
        if any(
            name not in metrics
            or not isinstance(metrics[name], (int, float))
            or isinstance(metrics[name], bool)
            or not math.isfinite(float(metrics[name]))
            for name in ("map", "ap75")
        ):
            raise RuntimeError("official probe metrics are invalid")
    map_delta = Decimal(str(q["map"])) - Decimal(str(c0["map"]))
    ap75_delta = Decimal(str(q["ap75"])) - Decimal(str(c0["ap75"]))
    passed = map_delta > 0 and ap75_delta > 0
    return {
        "status": "passed" if passed else "scientific_failed",
        "deltas": {"map": str(map_delta), "ap75": str(ap75_delta)},
        "thresholds": {"map": "0", "ap75": "0", "strict": True},
    }


def _run_official_if_authorized(
    internal_decision: Mapping[str, Any],
    *,
    open_official: Callable[[], Any],
    evaluate: Callable[[Any], Any],
) -> Any | None:
    if internal_decision.get("status") != "passed":
        return None
    records = open_official()
    return evaluate(records)


def _ensure_report_root(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError("report root must not be a symlink")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise RuntimeError("report root must be a directory")


def _expected_ids(paths: Sequence[Path], *, root: Path) -> tuple[str, ...]:
    return tuple(_relative_id(path, root=root) for path in paths)


def _extract_internal_cache(
    *,
    baseline: Path,
    dataset_root: Path,
    cache_root: Path,
    report_root: Path,
    subset_paths: Sequence[Path],
    train_paths: Sequence[Path],
    dev_paths: Sequence[Path],
    authority: Mapping[str, str],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_root = cache_root / "probe_train"
    dev_root = cache_root / "internal_dev"
    train_digest = report_root / "probe-train-cache-authority.json"
    dev_digest = report_root / "internal-dev-cache-authority.json"
    train_ids = _expected_ids(train_paths, root=dataset_root)
    dev_ids = _expected_ids(dev_paths, root=dataset_root)
    if (train_root / "manifest.json").is_file() and (dev_root / "manifest.json").is_file():
        return (
            _load_cache_stage(
                train_root, authority=authority, split="probe_train",
                external_digest_path=train_digest, expected_ids=train_ids,
            ),
            _load_cache_stage(
                dev_root, authority=authority, split="internal_dev",
                external_digest_path=dev_digest, expected_ids=dev_ids,
            ),
            _read_canonical_json(report_root / "hook-neutrality-report.json"),
        )
    detector = _load_detector(baseline, device)
    subset_list = cache_root.parent / f"{cache_root.name}-subset.txt"
    loader, validator = _build_validation_loader(
        dataset_root,
        baseline,
        device,
        image_source=subset_list,
        split_name="fixed-10pct-subset",
        save_dir=cache_root.parent / f".{cache_root.name}-validator-subset",
        expected_count=TRAIN_SUBSET_COUNT,
    )
    first_raw = next(iter(loader))
    first_batch = validator.preprocess(first_raw)
    _assert_cuda0_tensor(first_batch["img"], label="preprocessed input")
    proof = _prove_hook_neutrality(detector, first_batch["img"])
    proof_report = {"format_version": 1, "authority": dict(authority), **proof}
    _write_or_validate_canonical_json(report_root / "hook-neutrality-report.json", proof_report)
    extracted = _extract_records(
        detector,
        loader,
        validator,
        dataset_root=dataset_root,
        device=device,
        expected_count=TRAIN_SUBSET_COUNT,
    )
    ordered = _reorder_records(extracted, subset_paths, root=dataset_root)
    train_set = set(train_ids)
    train_records = [record for record in ordered if record["image_id"] in train_set]
    dev_records = [record for record in ordered if record["image_id"] not in train_set]
    train_records = _reorder_records(train_records, train_paths, root=dataset_root)
    dev_records = _reorder_records(dev_records, dev_paths, root=dataset_root)
    _write_cache_stage(
        train_root, records=train_records, authority=authority,
        split="probe_train", external_digest_path=train_digest,
    )
    _write_cache_stage(
        dev_root, records=dev_records, authority=authority,
        split="internal_dev", external_digest_path=dev_digest,
    )
    return train_records, dev_records, proof_report


def _extract_official_cache(
    *,
    baseline: Path,
    dataset_root: Path,
    cache_root: Path,
    report_root: Path,
    authority: Mapping[str, str],
    device: torch.device,
) -> list[dict[str, Any]]:
    _assert_full_dataset_authority(dataset_root)
    val_paths = tuple(sorted((dataset_root / "images" / "val").glob("*.jpg")))
    if len(val_paths) != VAL_COUNT:
        raise RuntimeError("official validation image count mismatch")
    expected_ids = _expected_ids(val_paths, root=dataset_root)
    stage_root = cache_root / "official_val"
    digest_path = report_root / "official-val-cache-authority.json"
    if (stage_root / "manifest.json").is_file():
        return _load_cache_stage(
            stage_root, authority=authority, split="official_val",
            external_digest_path=digest_path, expected_ids=expected_ids,
        )
    detector = _load_detector(baseline, device)
    loader, validator = _build_validation_loader(
        dataset_root,
        baseline,
        device,
        image_source=dataset_root / "images" / "val",
        split_name="official-val",
        save_dir=cache_root.parent / f".{cache_root.name}-validator-val",
        expected_count=VAL_COUNT,
    )
    records = _extract_records(
        detector,
        loader,
        validator,
        dataset_root=dataset_root,
        device=device,
        expected_count=VAL_COUNT,
    )
    records = _reorder_records(records, val_paths, root=dataset_root)
    _write_cache_stage(
        stage_root, records=records, authority=authority,
        split="official_val", external_digest_path=digest_path,
    )
    return records


def _selected_checkpoint_report(selected: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "epoch": selected["epoch"],
        "metrics": selected["metrics"],
        "checkpoint": selected["checkpoint"],
        "permutation_sha256": selected["permutation_sha256"],
    }


def _run(args: argparse.Namespace) -> int:
    baseline = Path(args.baseline_checkpoint).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    report_root = Path(args.report_root).resolve()
    _ensure_report_root(report_root)
    subset_paths, train_paths, dev_paths = _prepare_probe_paths(dataset_root, cache_root)
    authority = _build_authority(baseline, dataset_root, train_paths, dev_paths)
    device = _device(args.device)
    train_records, dev_records, hook_report = _extract_internal_cache(
        baseline=baseline,
        dataset_root=dataset_root,
        cache_root=cache_root,
        report_root=report_root,
        subset_paths=subset_paths,
        train_paths=train_paths,
        dev_paths=dev_paths,
        authority=authority,
        device=device,
    )
    c0_internal = _evaluate_probe_records(
        dev_records, arm="c0", model=None, device=device
    )
    _, c1_history = _train_arm(
        "c1", train_records, dev_records,
        device=device, report_root=report_root, authority=authority,
    )
    _, q_history = _train_arm(
        "q", train_records, dev_records,
        device=device, report_root=report_root, authority=authority,
    )
    if len(c1_history) != EPOCHS or len(q_history) != EPOCHS:
        raise RuntimeError("both probe arms must complete exactly 20 epochs")
    selected_c1 = _select_checkpoint(c1_history)
    selected_q = _select_checkpoint(q_history)
    c1_model = _load_selected_model(
        "c1", selected_c1, device=device, report_root=report_root, authority=authority
    )
    q_model = _load_selected_model(
        "q", selected_q, device=device, report_root=report_root, authority=authority
    )
    c1_internal = _evaluate_probe_records(
        dev_records, arm="c1", model=c1_model, device=device
    )
    q_internal = _evaluate_probe_records(
        dev_records, arm="q", model=q_model, device=device
    )
    internal_decision = evaluate_internal_probe_gate(
        controls={"c0": c0_internal, "c1": c1_internal}, q=q_internal
    )
    selection_report = {
        "format_version": 1,
        "authority": authority,
        "split": {
            "train_count": PROBE_TRAIN_COUNT,
            "dev_count": DEV_COUNT,
            "probe_train_sha256": EXPECTED_PROBE_TRAIN_SHA256,
            "dev_sha256": EXPECTED_DEV_SHA256,
        },
        "c0": c0_internal,
        "c1": _selected_checkpoint_report(selected_c1),
        "q": _selected_checkpoint_report(selected_q),
        "q_selected_metrics": q_internal,
        "gate": internal_decision,
        "hook_neutrality": hook_report,
    }
    _write_or_validate_canonical_json(
        report_root / "internal-selection-report.json", selection_report
    )
    internal_payload = {
        "format_version": 1,
        "authority": authority,
        **internal_decision,
    }
    _write_or_validate_canonical_json(
        report_root / "internal-quality-probe-decision.json", internal_payload
    )
    verified_internal = _read_canonical_json(
        report_root / "internal-quality-probe-decision.json"
    )

    def open_official() -> list[dict[str, Any]]:
        return _extract_official_cache(
            baseline=baseline,
            dataset_root=dataset_root,
            cache_root=cache_root,
            report_root=report_root,
            authority=authority,
            device=device,
        )

    def evaluate_official(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        c0 = _evaluate_probe_records(records, arm="c0", model=None, device=device)
        _assert_stock_authority(c0)
        c1 = _evaluate_probe_records(records, arm="c1", model=c1_model, device=device)
        q = _evaluate_probe_records(records, arm="q", model=q_model, device=device)
        return {"c0": c0, "c1": c1, "q": q, "gate": _official_gate(c0=c0, q=q)}

    official = _run_official_if_authorized(
        verified_internal, open_official=open_official, evaluate=evaluate_official
    )
    if official is None:
        final_decision = {
            "format_version": 1,
            "authority": authority,
            "stage": "internal",
            **internal_decision,
        }
    else:
        _write_or_validate_canonical_json(
            report_root / "official-quality-probe-report.json",
            {
                "format_version": 1,
                "authority": authority,
                "split": {"count": VAL_COUNT, "detector_passes": 1},
                **official,
            },
        )
        final_decision = {
            "format_version": 1,
            "authority": authority,
            "stage": "official",
            **official["gate"],
            "authorizes_detector_30epoch": official["gate"]["status"] == "passed",
        }
    _write_or_validate_canonical_json(
        report_root / "quality-probe-decision.json", final_decision
    )
    inventory = {
        "format_version": 1,
        "authority": authority,
        "environment": _execution_environment(),
        "baseline": {
            "path": str(baseline),
            "bytes": baseline.stat().st_size,
            "sha256": _file_sha256(baseline),
        },
        "reports": {
            path.name: _file_sha256(path)
            for path in sorted(report_root.glob("*.json"))
            if path.name != "environment-hash-inventory.json"
        },
    }
    _write_or_validate_canonical_json(
        report_root / "environment-hash-inventory.json", inventory
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return _run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
