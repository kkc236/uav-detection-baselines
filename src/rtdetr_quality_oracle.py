"""Mathematical core for the frozen RT-DETR quality-reranking oracle."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch


ALPHA_GRID = (0.25, 0.5, 1.0, 2.0)
DEV_COUNT = 129
DEV_SPLIT_SALT = b"rtdetr-quality-oracle-dev-v1\0"
EXPECTED_DEV_SHA256 = (
    "FCF8749BAADBA8BDDF5870F472BDE1E937156AFBCEEFDA9F96FED21FA6BB0514"
)
MAP_GAIN_THRESHOLD = Decimal("0.0050")

_AUTHORIZED_PATH_COUNT = 647
_VAL_COUNT = 548
_CACHE_FORMAT_VERSION = 1
_AUTHORITY_FIELDS = (
    "baseline_sha256",
    "dataset_sha256",
    "subset_sha256",
    "runtime_amendment_sha256",
    "source_commit",
    "schema_sha256",
    "dev_sha256",
)
_RECORD_FIELDS = (
    "image_id",
    "boxes",
    "logits",
    "target_boxes",
    "target_classes",
)

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


class QualityOracleCacheViolation(ValueError):
    """Raised when immutable quality-oracle evidence is unsafe or has drifted."""


def _require_positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_floating_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating-point tensor")


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def same_class_iou_quality(
    boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Return the maximum same-class IoU for every query and class."""
    if not isinstance(boxes, torch.Tensor):
        raise TypeError("boxes must be a tensor")
    if not isinstance(target_boxes, torch.Tensor):
        raise TypeError("target_boxes must be a tensor")
    if not isinstance(target_classes, torch.Tensor):
        raise TypeError("target_classes must be a tensor")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [Q, 4]")
    if target_boxes.ndim != 2 or target_boxes.shape[1] != 4:
        raise ValueError("target_boxes must have shape [N, 4]")
    if target_classes.ndim != 1:
        raise ValueError("target_classes must have shape [N]")
    if target_boxes.shape[0] != target_classes.shape[0]:
        raise ValueError(
            "target_boxes and target_classes must contain the same number of targets"
        )
    _require_positive_int(num_classes, "num_classes")
    _require_floating_tensor(boxes, "boxes")
    _require_floating_tensor(target_boxes, "target_boxes")
    if target_classes.dtype not in _INTEGER_DTYPES:
        raise TypeError("target_classes must contain integer class indices")
    if boxes.device != target_boxes.device or boxes.device != target_classes.device:
        raise ValueError("boxes, target_boxes, and target_classes must share a device")
    _require_finite(boxes, "boxes")
    _require_finite(target_boxes, "target_boxes")
    if not bool(((boxes >= 0) & (boxes <= 1)).all()):
        raise ValueError("boxes must be normalized to [0, 1]")
    if not bool(((target_boxes >= 0) & (target_boxes <= 1)).all()):
        raise ValueError("target_boxes must be normalized to [0, 1]")
    if target_classes.numel() and not bool(
        ((target_classes >= 0) & (target_classes < num_classes)).all()
    ):
        raise ValueError("target_classes must be in [0, num_classes)")

    compute_dtype = torch.promote_types(boxes.dtype, target_boxes.dtype)
    if compute_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    quality = torch.zeros(
        (boxes.shape[0], num_classes), dtype=compute_dtype, device=boxes.device
    )
    if target_boxes.shape[0] == 0:
        return quality

    query_boxes = boxes.detach().to(dtype=compute_dtype)
    ground_truth = target_boxes.detach().to(dtype=compute_dtype)
    query_center, query_size = query_boxes.split(2, dim=-1)
    target_center, target_size = ground_truth.split(2, dim=-1)
    query_lower = query_center - query_size / 2
    query_upper = query_center + query_size / 2
    target_lower = target_center - target_size / 2
    target_upper = target_center + target_size / 2

    intersection = (
        torch.minimum(query_upper[:, None], target_upper[None])
        - torch.maximum(query_lower[:, None], target_lower[None])
    ).clamp_min(0).prod(dim=-1)
    query_area = query_size.prod(dim=-1)
    target_area = target_size.prod(dim=-1)
    union = query_area[:, None] + target_area[None] - intersection
    iou = torch.where(union > 0, intersection / union, torch.zeros_like(union))
    iou = torch.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0, 1)

    class_indices = target_classes.detach().to(dtype=torch.long)
    for class_index in range(num_classes):
        selected = class_indices == class_index
        if bool(selected.any()):
            quality[:, class_index] = iou[:, selected].amax(dim=1)
    return quality


def flattened_topk(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    num_classes: int,
    max_det: int = 300,
) -> torch.Tensor:
    """Apply Ultralytics 8.4.90's flattened query-by-class Top-K."""
    if not isinstance(boxes, torch.Tensor):
        raise TypeError("boxes must be a tensor")
    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be a tensor")
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [B, Q, 4]")
    if scores.ndim != 3:
        raise ValueError("scores must have shape [B, Q, C]")
    _require_positive_int(num_classes, "num_classes")
    _require_positive_int(max_det, "max_det")
    if scores.shape != (*boxes.shape[:2], num_classes):
        raise ValueError("scores must have shape [B, Q, num_classes]")
    _require_floating_tensor(boxes, "boxes")
    _require_floating_tensor(scores, "scores")
    if boxes.device != scores.device:
        raise ValueError("boxes and scores must share a device")
    if boxes.dtype != scores.dtype:
        raise ValueError("boxes and scores must share a dtype")
    _require_finite(boxes, "boxes")
    _require_finite(scores, "scores")
    if max_det > scores.shape[1] * num_classes:
        raise ValueError("max_det cannot exceed the flattened score count")

    selected_scores, index = scores.flatten(1).topk(max_det)
    query_index = torch.div(index, num_classes, rounding_mode="floor")
    selected_boxes = boxes.gather(
        dim=1,
        index=query_index.unsqueeze(-1).expand(-1, -1, 4).long(),
    )
    class_index = (index - query_index * num_classes)[..., None].float()
    return torch.cat(
        [selected_boxes, selected_scores[..., None], class_index], dim=-1
    )


def oracle_topk(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    qualities: torch.Tensor,
    alpha: float,
    num_classes: int,
    max_det: int = 300,
) -> torch.Tensor:
    """Rerank sigmoid class scores by perfect same-class IoU quality."""
    if isinstance(alpha, bool) or alpha not in ALPHA_GRID:
        raise ValueError(f"alpha must be one of ALPHA_GRID={ALPHA_GRID}")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a tensor")
    if not isinstance(qualities, torch.Tensor):
        raise TypeError("qualities must be a tensor")
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, Q, C]")
    if qualities.shape != logits.shape:
        raise ValueError("qualities and logits must have identical shapes")
    if logits.shape[-1] != num_classes:
        raise ValueError("logits class dimension must equal num_classes")
    _require_floating_tensor(logits, "logits")
    _require_floating_tensor(qualities, "qualities")
    if logits.device != qualities.device:
        raise ValueError("logits and qualities must share a device")
    if logits.dtype != qualities.dtype:
        raise ValueError("logits and qualities must share a dtype")
    _require_finite(logits, "logits")
    _require_finite(qualities, "qualities")
    if not bool(((qualities >= 0) & (qualities <= 1)).all()):
        raise ValueError("qualities must be in [0, 1]")

    reranked_scores = logits.sigmoid() * qualities.pow(alpha)
    return flattened_topk(
        boxes,
        reranked_scores,
        num_classes=num_classes,
        max_det=max_det,
    )


def _canonical_paths(
    paths: Sequence[Path], *, root: Path
) -> tuple[tuple[Path, str], ...]:
    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise TypeError("paths must be a sequence of Path values")
    try:
        resolved_root = root.resolve()
    except OSError as error:
        raise ValueError("root path is invalid") from error

    canonical: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for path in paths:
        if not isinstance(path, Path):
            raise TypeError("every path must be a Path")
        try:
            relative = path.resolve().relative_to(resolved_root).as_posix()
        except (OSError, ValueError) as error:
            raise ValueError(f"path is outside root: {path}") from error
        if relative in {"", "."} or any(character in relative for character in "\0\r\n"):
            raise ValueError(f"path is malformed: {path}")
        try:
            relative.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(f"path is not valid UTF-8: {path}") from error
        if relative in seen:
            raise ValueError("paths must be unique under root")
        seen.add(relative)
        canonical.append((path, relative))
    return tuple(canonical)


def select_internal_dev(
    paths: Sequence[Path], *, root: Path
) -> tuple[Path, ...]:
    """Select the frozen development partition from 647 authorized paths."""
    canonical = _canonical_paths(paths, root=root)
    if len(canonical) != _AUTHORIZED_PATH_COUNT:
        raise ValueError(f"paths must contain exactly {_AUTHORIZED_PATH_COUNT} entries")
    ranked = sorted(
        canonical,
        key=lambda item: (
            hashlib.sha256(
                DEV_SPLIT_SALT + item[1].encode("utf-8")
            ).digest(),
            item[1],
        ),
    )
    return tuple(path for path, _ in ranked[:DEV_COUNT])


def ordered_path_sha256(paths: Sequence[Path], *, root: Path) -> str:
    """Hash ordered relative POSIX paths with canonical UTF-8/LF framing."""
    canonical = _canonical_paths(paths, root=root)
    payload = "".join(f"{relative}\n" for _, relative in canonical).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _normalize_authority(authority: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(authority, Mapping) or set(authority) != set(_AUTHORITY_FIELDS):
        raise QualityOracleCacheViolation("authority schema mismatch")
    normalized: dict[str, str] = {}
    for name in _AUTHORITY_FIELDS:
        value = authority[name]
        length = 40 if name == "source_commit" else 64
        if (
            not isinstance(value, str)
            or len(value) != length
            or any(character not in "0123456789abcdefABCDEF" for character in value)
        ):
            raise QualityOracleCacheViolation(f"invalid authority {name}")
        normalized[name] = value.lower() if name == "source_commit" else value.upper()
    if normalized["dev_sha256"] != EXPECTED_DEV_SHA256:
        raise QualityOracleCacheViolation("invalid authority dev_sha256")
    return normalized


def _validate_cache_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping) or set(record) != set(_RECORD_FIELDS):
        raise QualityOracleCacheViolation("record schema mismatch")
    image_id = record["image_id"]
    if not isinstance(image_id, str) or not image_id:
        raise QualityOracleCacheViolation("record image_id is invalid")

    boxes = record["boxes"]
    logits = record["logits"]
    target_boxes = record["target_boxes"]
    target_classes = record["target_classes"]
    tensors = (boxes, logits, target_boxes, target_classes)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise QualityOracleCacheViolation("record evidence must be tensors")
    if boxes.shape != (300, 4) or logits.shape != (300, 10):
        raise QualityOracleCacheViolation("record production tensor shape mismatch")
    if (
        target_boxes.ndim != 2
        or target_boxes.shape[1] != 4
        or target_classes.ndim != 1
        or target_boxes.shape[0] != target_classes.shape[0]
    ):
        raise QualityOracleCacheViolation("record target tensor shape mismatch")
    if any(value.dtype != torch.float32 for value in (boxes, logits, target_boxes)):
        raise QualityOracleCacheViolation(
            "boxes, logits, and target_boxes must have dtype torch.float32"
        )
    if target_classes.dtype != torch.int64:
        raise QualityOracleCacheViolation(
            "target_classes must have dtype torch.int64"
        )
    if any(value.device.type != "cpu" for value in tensors):
        raise QualityOracleCacheViolation("record tensors must be on CPU")
    if any(value.requires_grad for value in tensors):
        raise QualityOracleCacheViolation("record tensors must be detached")
    if any(not value.is_contiguous() for value in tensors):
        raise QualityOracleCacheViolation("record tensors must be contiguous")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise QualityOracleCacheViolation("record tensors must be finite")
    if not bool(((boxes >= 0) & (boxes <= 1)).all()):
        raise QualityOracleCacheViolation("boxes must be normalized to [0, 1]")
    if not bool(((target_boxes >= 0) & (target_boxes <= 1)).all()):
        raise QualityOracleCacheViolation("target_boxes must be normalized to [0, 1]")
    if target_classes.numel() and not bool(
        ((target_classes >= 0) & (target_classes <= 9)).all()
    ):
        raise QualityOracleCacheViolation("target_classes must be in [0, 9]")
    return {name: record[name] for name in _RECORD_FIELDS}


def _validate_cache_splits(
    dev: Sequence[Mapping[str, Any]], val: Sequence[Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    if isinstance(dev, (str, bytes)) or not isinstance(dev, Sequence):
        raise QualityOracleCacheViolation("dev split must be a record sequence")
    if isinstance(val, (str, bytes)) or not isinstance(val, Sequence):
        raise QualityOracleCacheViolation("val split must be a record sequence")
    if len(dev) != DEV_COUNT:
        raise QualityOracleCacheViolation(f"dev split must contain exactly {DEV_COUNT} records")
    if len(val) != _VAL_COUNT:
        raise QualityOracleCacheViolation(f"val split must contain exactly {_VAL_COUNT} records")
    validated = {
        "dev": [_validate_cache_record(record) for record in dev],
        "val": [_validate_cache_record(record) for record in val],
    }
    identifiers = {
        split: [record["image_id"] for record in records]
        for split, records in validated.items()
    }
    for split, image_ids in identifiers.items():
        if len(set(image_ids)) != len(image_ids):
            raise QualityOracleCacheViolation(f"{split} image IDs must be unique")
    overlap = set(identifiers["dev"]) & set(identifiers["val"])
    if overlap:
        raise QualityOracleCacheViolation(
            f"dev/validation image overlap: {min(overlap)}"
        )
    return validated


def _save_fsynced(path: Path, payload: Any) -> None:
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_quality_oracle_cache(
    root: Path,
    *,
    dev: list[dict[str, Any]],
    val: list[dict[str, Any]],
    authority: Mapping[str, str],
) -> dict[str, Any]:
    """Create a complete immutable quality-oracle cache and manifest."""
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite cache root: {root}")
    normalized_authority = _normalize_authority(authority)
    validated = _validate_cache_splits(dev, val)
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite cache root: {root}") from error

    artifacts: list[dict[str, Any]] = []
    for split in ("dev", "val"):
        path = root / f"{split}.pt"
        _save_fsynced(
            path,
            {
                "format_version": _CACHE_FORMAT_VERSION,
                "split": split,
                "records": validated[split],
            },
        )
        artifacts.append(
            {
                "split": split,
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )

    manifest = {
        "format_version": _CACHE_FORMAT_VERSION,
        "complete": True,
        "authority": normalized_authority,
        "split_counts": {"dev": DEV_COUNT, "val": _VAL_COUNT},
        "artifacts": artifacts,
    }
    manifest_path = root / "manifest.json"
    with manifest_path.open("xb") as stream:
        stream.write(_canonical_json(manifest))
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def _load_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualityOracleCacheViolation(f"manifest load failed: {error}") from error
    if not isinstance(manifest, dict):
        raise QualityOracleCacheViolation("manifest schema mismatch")
    try:
        canonical = _canonical_json(manifest)
    except (TypeError, ValueError) as error:
        raise QualityOracleCacheViolation("manifest contains unsafe values") from error
    if raw != canonical:
        raise QualityOracleCacheViolation("manifest is not canonical")
    return manifest


def load_quality_oracle_cache(
    root: Path, *, authority: Mapping[str, str]
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Verify and safely load a complete immutable quality-oracle cache."""
    root = Path(root)
    expected_authority = _normalize_authority(authority)
    manifest = _load_manifest(root)
    expected_manifest_fields = {
        "format_version",
        "complete",
        "authority",
        "split_counts",
        "artifacts",
    }
    if set(manifest) != expected_manifest_fields:
        raise QualityOracleCacheViolation("manifest schema mismatch")
    if (
        type(manifest["format_version"]) is not int
        or manifest["format_version"] != _CACHE_FORMAT_VERSION
    ):
        raise QualityOracleCacheViolation("manifest format version mismatch")
    if manifest["complete"] is not True:
        raise QualityOracleCacheViolation("manifest is not complete")

    actual_authority = manifest["authority"]
    try:
        normalized_actual = _normalize_authority(actual_authority)
    except (TypeError, QualityOracleCacheViolation) as error:
        raise QualityOracleCacheViolation("manifest authority schema mismatch") from error
    if actual_authority != normalized_actual:
        raise QualityOracleCacheViolation("manifest authority is not canonical")
    if normalized_actual != expected_authority:
        differing = [
            name
            for name in _AUTHORITY_FIELDS
            if normalized_actual[name] != expected_authority[name]
        ]
        raise QualityOracleCacheViolation("authority mismatch: " + ",".join(differing))
    if manifest["split_counts"] != {"dev": DEV_COUNT, "val": _VAL_COUNT}:
        raise QualityOracleCacheViolation("split count mismatch")

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        raise QualityOracleCacheViolation("artifact manifest mismatch")
    try:
        entries = {path.name for path in root.iterdir()}
    except OSError as error:
        raise QualityOracleCacheViolation(f"cache root is unreadable: {error}") from error
    if entries != {"dev.pt", "val.pt", "manifest.json"}:
        raise QualityOracleCacheViolation("cache root contents mismatch")

    expected_paths = {"dev": "dev.pt", "val": "val.pt"}
    artifacts_by_split: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "split",
            "path",
            "bytes",
            "sha256",
        }:
            raise QualityOracleCacheViolation("artifact schema mismatch")
        split = artifact["split"]
        if (
            not isinstance(split, str)
            or split not in expected_paths
            or split in artifacts_by_split
        ):
            raise QualityOracleCacheViolation("artifact split mismatch")
        if artifact["path"] != expected_paths[split]:
            raise QualityOracleCacheViolation("artifact path mismatch")
        if type(artifact["bytes"]) is not int or artifact["bytes"] <= 0:
            raise QualityOracleCacheViolation("artifact byte count mismatch")
        sha256 = artifact["sha256"]
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or sha256 != sha256.upper()
            or any(character not in "0123456789ABCDEF" for character in sha256)
        ):
            raise QualityOracleCacheViolation("artifact sha256 schema mismatch")
        artifacts_by_split[split] = artifact
    if set(artifacts_by_split) != set(expected_paths):
        raise QualityOracleCacheViolation("artifact split mismatch")

    for split in ("dev", "val"):
        artifact = artifacts_by_split[split]
        path = root / expected_paths[split]
        if path.is_symlink() or not path.is_file():
            raise QualityOracleCacheViolation("artifact is missing or unsafe")
        try:
            size = path.stat().st_size
            digest = _file_sha256(path)
        except OSError as error:
            raise QualityOracleCacheViolation(
                f"artifact preflight failed: {error}"
            ) from error
        if size != artifact["bytes"] or digest != artifact["sha256"]:
            raise QualityOracleCacheViolation("artifact bytes or sha256 mismatch")

    raw_records: dict[str, list[Mapping[str, Any]]] = {}
    for split in ("dev", "val"):
        path = root / expected_paths[split]
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise QualityOracleCacheViolation(f"artifact load failed: {error}") from error
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"format_version", "split", "records"}
            or type(payload["format_version"]) is not int
            or payload["format_version"] != _CACHE_FORMAT_VERSION
            or payload["split"] != split
            or not isinstance(payload["records"], list)
        ):
            raise QualityOracleCacheViolation("artifact schema mismatch")
        raw_records[split] = payload["records"]

    validated = _validate_cache_splits(raw_records["dev"], raw_records["val"])
    return {split: tuple(validated[split]) for split in ("dev", "val")}


def select_alpha(
    metrics_by_alpha: Mapping[float, Mapping[str, float]],
) -> float:
    """Select the frozen alpha by map, AP75, AP50, then smaller alpha."""
    if not isinstance(metrics_by_alpha, Mapping):
        raise TypeError("metrics_by_alpha must be a mapping")
    if (
        len(metrics_by_alpha) != len(ALPHA_GRID)
        or any(type(alpha) is not float for alpha in metrics_by_alpha)
        or set(metrics_by_alpha) != set(ALPHA_GRID)
    ):
        raise ValueError(f"metrics_by_alpha must contain exactly ALPHA_GRID={ALPHA_GRID}")
    ranking: dict[float, tuple[float, float, float, float]] = {}
    for alpha in ALPHA_GRID:
        metrics = metrics_by_alpha[alpha]
        if not isinstance(metrics, Mapping):
            raise TypeError(f"metrics for alpha={alpha} must be a mapping")
        values: list[float] = []
        for name in ("map", "ap75", "ap50"):
            if name not in metrics:
                raise ValueError(f"metrics for alpha={alpha} must contain {name}")
            value = metrics[name]
            if isinstance(value, bool):
                raise ValueError(f"metric {name} for alpha={alpha} must be finite")
            try:
                finite_value = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"metric {name} for alpha={alpha} must be finite"
                ) from error
            if not math.isfinite(finite_value):
                raise ValueError(f"metric {name} for alpha={alpha} must be finite")
            values.append(finite_value)
        ranking[alpha] = (*values, -alpha)
    return max(ALPHA_GRID, key=ranking.__getitem__)


def _decimal_metric(value: float, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not decimal_value.is_finite():
        raise ValueError(f"{name} must be finite")
    return decimal_value


def decide_quality_oracle(
    *,
    stock_map: float,
    stock_ap75: float,
    oracle_map: float,
    oracle_ap75: float,
) -> dict[str, Any]:
    """Apply the exact mAP and strict AP75 scientific gate."""
    observed = {
        "stock_map": _decimal_metric(stock_map, "stock_map"),
        "stock_ap75": _decimal_metric(stock_ap75, "stock_ap75"),
        "oracle_map": _decimal_metric(oracle_map, "oracle_map"),
        "oracle_ap75": _decimal_metric(oracle_ap75, "oracle_ap75"),
    }
    deltas = {
        "map": observed["oracle_map"] - observed["stock_map"],
        "ap75": observed["oracle_ap75"] - observed["stock_ap75"],
    }
    thresholds = {"map": MAP_GAIN_THRESHOLD, "ap75": Decimal("0")}
    passed = deltas["map"] >= thresholds["map"] and deltas["ap75"] > thresholds["ap75"]
    return {
        "status": "passed" if passed else "scientific_failed",
        "finite": True,
        "observed": {name: format(value, "f") for name, value in observed.items()},
        "deltas": {name: format(value, "f") for name, value in deltas.items()},
        "thresholds": {
            name: format(value, "f") for name, value in thresholds.items()
        },
    }
