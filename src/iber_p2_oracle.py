"""Deterministic stride-4 boundary evidence oracle for IBER-BE."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


P2_NORMAL_OFFSETS_PX = (-12, -8, -4, 0, 4, 8, 12)
P2_TANGENT_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
ORACLE_EPOCHS = 20
ORACLE_SEED = 10_000
ORACLE_BATCH_SIZE = 1024
P2_TINY_DIRECTION_THRESHOLD = 0.624866
P2_SMALL_DIRECTION_THRESHOLD = 0.634066
P2_ORACLE_CACHE_VERSION = 1
_AUTHORITY_FIELDS = (
    "baseline_sha256",
    "dataset_sha256",
    "subset_sha256",
    "runtime_amendment_sha256",
    "source_commit",
    "schema_sha256",
)
_RECORD_FIELDS = (
    "image_id",
    "profiles",
    "hidden",
    "geometry",
    "labels",
    "valid",
    "buckets",
)


class P2OracleCacheViolation(ValueError):
    """Raised when immutable P2 oracle evidence drifts or is corrupted."""


def enable_p2_oracle_determinism() -> None:
    """Lock deterministic CPU/CUDA kernels before oracle model construction."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    half_size = boxes[..., 2:].mul(0.5)
    return torch.cat((boxes[..., :2] - half_size, boxes[..., :2] + half_size), dim=-1)


def sample_p2_edge_profiles(
    p2: torch.Tensor,
    stock_boxes_cxcywh: torch.Tensor,
    *,
    image_size: int = 640,
) -> torch.Tensor:
    """Sample tangent-averaged P2 normal profiles around four stock-box edges."""
    if p2.ndim != 4 or stock_boxes_cxcywh.ndim != 3:
        raise ValueError("P2 must be BCHW and stock boxes must be BQ4")
    if p2.shape[0] != stock_boxes_cxcywh.shape[0] or stock_boxes_cxcywh.shape[-1] != 4:
        raise ValueError("P2 and stock box batch dimensions must agree")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    batch, channels = p2.shape[:2]
    queries = stock_boxes_cxcywh.shape[1]
    normal_count = len(P2_NORMAL_OFFSETS_PX)
    if queries == 0:
        return p2.new_empty((batch, 0, 4, normal_count, channels))

    boxes = _cxcywh_to_xyxy(stock_boxes_cxcywh.to(dtype=p2.dtype))
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    tangent = p2.new_tensor(P2_TANGENT_FRACTIONS)
    offsets = p2.new_tensor(P2_NORMAL_OFFSETS_PX).div(float(image_size))

    vertical_y = y1[..., None] + tangent * (y2 - y1)[..., None]
    horizontal_x = x1[..., None] + tangent * (x2 - x1)[..., None]
    tangent_count = tangent.numel()

    def vertical(x: torch.Tensor) -> torch.Tensor:
        sample_x = x[..., None, None] + offsets[:, None]
        sample_x = sample_x.expand(batch, queries, normal_count, tangent_count)
        sample_y = vertical_y[..., None, :].expand_as(sample_x)
        return torch.stack((sample_x, sample_y), dim=-1)

    def horizontal(y: torch.Tensor) -> torch.Tensor:
        sample_y = y[..., None, None] + offsets[:, None]
        sample_y = sample_y.expand(batch, queries, normal_count, tangent_count)
        sample_x = horizontal_x[..., None, :].expand_as(sample_y)
        return torch.stack((sample_x, sample_y), dim=-1)

    grid = torch.stack((vertical(x1), vertical(x2), horizontal(y1), horizontal(y2)), dim=2)
    grid = grid.mul(2).sub(1).reshape(batch, queries * 4 * normal_count, tangent_count, 2)
    sampled = F.grid_sample(
        p2,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    sampled = sampled.reshape(batch, channels, queries, 4, normal_count, tangent_count)
    return sampled.mean(dim=-1).permute(0, 2, 3, 4, 1).contiguous()


def correction_direction_targets(
    stock_edges: torch.Tensor,
    target_edges: torch.Tensor,
    *,
    image_size: int = 640,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return binary target-minus-stock signs and a stable one-pixel validity mask."""
    if stock_edges.shape != target_edges.shape or stock_edges.shape[-1] != 4:
        raise ValueError("stock and target edges must have the same trailing dimension four")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    correction = target_edges - stock_edges
    labels = correction.gt(0).to(dtype=stock_edges.dtype)
    width = (stock_edges[..., 2] - stock_edges[..., 0]).clamp_min(1e-6)
    height = (stock_edges[..., 3] - stock_edges[..., 1]).clamp_min(1e-6)
    scale = torch.stack((width, height, width, height), dim=-1)
    normalized = (correction / (0.05 * scale + 1e-6)).clamp(-1, 1)
    valid = normalized.abs().gt(1e-6)
    return labels, valid


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _normalize_authority(authority: Mapping[str, str]) -> dict[str, str]:
    if set(authority) != set(_AUTHORITY_FIELDS):
        raise P2OracleCacheViolation("authority schema mismatch")
    normalized: dict[str, str] = {}
    for name in _AUTHORITY_FIELDS:
        value = authority[name]
        length = 40 if name == "source_commit" else 64
        if not isinstance(value, str) or len(value) != length or any(
            character not in "0123456789abcdefABCDEF" for character in value
        ):
            raise P2OracleCacheViolation(f"invalid authority {name}")
        normalized[name] = value.lower() if name == "source_commit" else value.upper()
    return normalized


def _validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if set(record) != set(_RECORD_FIELDS):
        raise P2OracleCacheViolation("record schema mismatch")
    image_id = record["image_id"]
    if not isinstance(image_id, str) or not image_id:
        raise P2OracleCacheViolation("record image_id is invalid")
    tensors = {name: record[name] for name in _RECORD_FIELDS if name != "image_id"}
    if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise P2OracleCacheViolation("record evidence must be tensors")
    profiles = tensors["profiles"]
    hidden = tensors["hidden"]
    geometry = tensors["geometry"]
    labels = tensors["labels"]
    valid = tensors["valid"]
    buckets = tensors["buckets"]
    count = profiles.shape[0] if profiles.ndim == 4 else -1
    if (
        profiles.ndim != 4
        or profiles.shape[1:3] != (4, len(P2_NORMAL_OFFSETS_PX))
        or hidden.ndim != 2
        or geometry.ndim != 2
        or labels.shape != (count, 4)
        or valid.shape != (count, 4)
        or buckets.shape != (count,)
        or hidden.shape[0] != count
        or geometry.shape[0] != count
    ):
        raise P2OracleCacheViolation("record tensor shape mismatch")
    if valid.dtype is not torch.bool or buckets.dtype is not torch.long:
        raise P2OracleCacheViolation("record mask dtype mismatch")
    if any(value.device.type != "cpu" for value in tensors.values()):
        raise P2OracleCacheViolation("record tensors must be detached CPU evidence")
    if any(value.requires_grad for value in tensors.values()):
        raise P2OracleCacheViolation("record tensors must be detached")
    if any(not torch.isfinite(value).all() for value in (profiles, hidden, geometry, labels)):
        raise P2OracleCacheViolation("record evidence is nonfinite")
    return {"image_id": image_id, **{name: value.contiguous() for name, value in tensors.items()}}


def _save_create_only(path: Path, payload: Any) -> None:
    handle, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_p2_oracle_cache(
    root: Path,
    *,
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    authority: Mapping[str, str],
) -> dict[str, Any]:
    """Write two create-only evidence artifacts and publish their manifest last."""
    root = Path(root)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty cache root: {root}")
    normalized_authority = _normalize_authority(authority)
    selected = {
        "train": [_validate_record(record) for record in train],
        "val": [_validate_record(record) for record in val],
    }
    train_ids = {record["image_id"] for record in selected["train"]}
    val_ids = {record["image_id"] for record in selected["val"]}
    overlap = train_ids & val_ids
    if overlap:
        raise P2OracleCacheViolation(f"train/validation image overlap: {min(overlap)}")
    root.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for split in ("train", "val"):
        path = root / f"{split}.pt"
        _save_create_only(
            path,
            {
                "format_version": P2_ORACLE_CACHE_VERSION,
                "split": split,
                "records": selected[split],
            },
        )
        artifacts.append(
            {
                "split": split,
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "format_version": P2_ORACLE_CACHE_VERSION,
        "complete": True,
        "authority": normalized_authority,
        "split_counts": {name: len(records) for name, records in selected.items()},
        "artifacts": artifacts,
    }
    manifest_path = root / "manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def load_p2_oracle_cache(
    root: Path,
    *,
    authority: Mapping[str, str],
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Preflight every artifact hash, then load safe tensor-only evidence."""
    root = Path(root)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise P2OracleCacheViolation(f"manifest: {error}") from error
    expected_authority = _normalize_authority(authority)
    actual_authority = manifest.get("authority")
    if actual_authority != expected_authority:
        differing = [
            name for name in _AUTHORITY_FIELDS
            if not isinstance(actual_authority, Mapping) or actual_authority.get(name) != expected_authority[name]
        ]
        raise P2OracleCacheViolation("authority mismatch: " + ",".join(differing))
    if manifest.get("format_version") != P2_ORACLE_CACHE_VERSION or manifest.get("complete") is not True:
        raise P2OracleCacheViolation("manifest version or completeness mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        raise P2OracleCacheViolation("artifact manifest mismatch")
    for artifact in artifacts:
        path = root / str(artifact.get("path"))
        if path.parent != root or not path.is_file():
            raise P2OracleCacheViolation("unsafe or missing artifact path")
        if path.stat().st_size != artifact.get("bytes") or _sha256(path) != artifact.get("sha256"):
            raise P2OracleCacheViolation("artifact bytes or sha256 mismatch")
    result: dict[str, tuple[dict[str, Any], ...]] = {}
    for artifact in artifacts:
        path = root / artifact["path"]
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (RuntimeError, EOFError, OSError) as error:
            raise P2OracleCacheViolation(f"artifact load failed: {error}") from error
        split = artifact["split"]
        if (
            not isinstance(payload, Mapping)
            or payload.get("format_version") != P2_ORACLE_CACHE_VERSION
            or payload.get("split") != split
            or not isinstance(payload.get("records"), list)
        ):
            raise P2OracleCacheViolation("artifact schema mismatch")
        result[split] = tuple(_validate_record(record) for record in payload["records"])
    if {name: len(records) for name, records in result.items()} != manifest.get("split_counts"):
        raise P2OracleCacheViolation("split count mismatch")
    return result


class _P2DirectionOracle(nn.Module):
    def __init__(
        self,
        profile_dim: int,
        hidden_dim: int,
        geometry_dim: int,
        *,
        use_profile: bool,
        use_context: bool,
    ) -> None:
        super().__init__()
        self.use_profile = bool(use_profile)
        self.use_context = bool(use_context)
        self.edge = nn.Embedding(4, 8)
        input_width = 8
        if self.use_profile:
            self.profile = nn.Sequential(nn.LayerNorm(profile_dim), nn.Linear(profile_dim, 128), nn.SiLU())
            input_width += 128
        context_width = 0
        if self.use_context:
            self.context = nn.Sequential(nn.LayerNorm(hidden_dim + geometry_dim), nn.Linear(hidden_dim + geometry_dim, 64), nn.SiLU())
            context_width = 64
        self.classifier = nn.Sequential(nn.Linear(input_width + context_width, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(
        self,
        profiles: torch.Tensor,
        hidden: torch.Tensor,
        geometry: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        features = [self.edge(edge_ids)]
        if self.use_profile:
            features.insert(0, self.profile(profiles.flatten(1)))
        if self.use_context:
            features.append(self.context(torch.cat((hidden, geometry), dim=-1)))
        return self.classifier(torch.cat(features, dim=-1)).squeeze(-1)


def _flatten_records(records: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("P2 oracle split must be non-empty")
    values: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("profiles", "hidden", "geometry", "labels", "valid", "buckets")
    }
    for raw in records:
        record = _validate_record(raw)
        for name in values:
            values[name].append(record[name])
    objects = {name: torch.cat(parts) for name, parts in values.items()}
    count = objects["profiles"].shape[0]
    return {
        "profiles": objects["profiles"].reshape(count * 4, *objects["profiles"].shape[2:]).float(),
        "hidden": objects["hidden"][:, None, :].expand(-1, 4, -1).reshape(count * 4, -1).float(),
        "geometry": objects["geometry"][:, None, :].expand(-1, 4, -1).reshape(count * 4, -1).float(),
        "labels": objects["labels"].reshape(-1).float(),
        "valid": objects["valid"].reshape(-1),
        "buckets": objects["buckets"][:, None].expand(-1, 4).reshape(-1),
        "edge_ids": torch.arange(4).repeat(count),
    }


def _validation_metrics(model: nn.Module, data: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, float | int]:
    model.eval()
    with torch.no_grad():
        logits = model(
            data["profiles"].to(device),
            data["hidden"].to(device),
            data["geometry"].to(device),
            data["edge_ids"].to(device),
        ).cpu()
    predicted = logits.ge(0)
    target = data["labels"].ge(0.5)
    valid = data["valid"]
    metrics: dict[str, float | int] = {"valid_edges": int(valid.sum())}
    for name, bucket in (("tiny", 0), ("small", 1)):
        mask = valid & data["buckets"].eq(bucket)
        if not mask.any():
            raise ValueError(f"P2 oracle validation has no valid {name} edges")
        metrics[f"{name}_direction_accuracy"] = float(predicted[mask].eq(target[mask]).float().mean())
        metrics[f"{name}_valid_edges"] = int(mask.sum())
    return metrics


def _train_one(
    train: Mapping[str, torch.Tensor],
    val: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    use_profile: bool,
    use_context: bool,
) -> dict[str, Any]:
    torch.manual_seed(ORACLE_SEED)
    model = _P2DirectionOracle(
        int(train["profiles"].shape[1] * train["profiles"].shape[2]),
        int(train["hidden"].shape[1]),
        int(train["geometry"].shape[1]),
        use_profile=use_profile,
        use_context=use_context,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    valid_indices = torch.nonzero(train["valid"], as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise ValueError("P2 oracle training has no valid edges")
    history: list[dict[str, float | int]] = []
    for epoch in range(1, ORACLE_EPOCHS + 1):
        model.train()
        generator = torch.Generator().manual_seed(ORACLE_SEED + epoch)
        order = valid_indices[torch.randperm(valid_indices.numel(), generator=generator)]
        losses: list[float] = []
        for start in range(0, order.numel(), ORACLE_BATCH_SIZE):
            index = order[start : start + ORACLE_BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                train["profiles"][index].to(device),
                train["hidden"][index].to(device),
                train["geometry"][index].to(device),
                train["edge_ids"][index].to(device),
            )
            loss = F.binary_cross_entropy_with_logits(logits, train["labels"][index].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": math.fsum(losses) / len(losses)})
    return {"history": history, "validation": _validation_metrics(model, val, device)}


def _majority_baseline(
    train: Mapping[str, torch.Tensor],
    val: Mapping[str, torch.Tensor],
) -> dict[str, float | int]:
    predicted = torch.zeros_like(val["labels"], dtype=torch.bool)
    for bucket in (0, 1, 2):
        for edge in range(4):
            train_mask = train["valid"] & train["buckets"].eq(bucket) & train["edge_ids"].eq(edge)
            positive = bool(train["labels"][train_mask].mean().ge(0.5)) if train_mask.any() else False
            val_mask = val["buckets"].eq(bucket) & val["edge_ids"].eq(edge)
            predicted[val_mask] = positive
    target = val["labels"].ge(0.5)
    metrics: dict[str, float | int] = {"valid_edges": int(val["valid"].sum())}
    for name, bucket in (("tiny", 0), ("small", 1)):
        mask = val["valid"] & val["buckets"].eq(bucket)
        if not mask.any():
            raise ValueError(f"P2 oracle validation has no valid {name} edges")
        metrics[f"{name}_direction_accuracy"] = float(predicted[mask].eq(target[mask]).float().mean())
        metrics[f"{name}_valid_edges"] = int(mask.sum())
    return metrics


def train_p2_oracles(
    cache: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Train fixed P2-only and context oracles and evaluate final epoch only."""
    enable_p2_oracle_determinism()
    if set(cache) != {"train", "val"}:
        raise ValueError("P2 oracle cache must contain exactly train and val")
    train = _flatten_records(cache["train"])
    val = _flatten_records(cache["val"])
    report = {
        "selection": "final_epoch_only",
        "evaluated_epoch": ORACLE_EPOCHS,
        "seed": ORACLE_SEED,
        "majority_baseline": _majority_baseline(train, val),
        "p2_only": _train_one(train, val, device=device, use_profile=True, use_context=False),
        "context_only": _train_one(train, val, device=device, use_profile=False, use_context=True),
        "context": _train_one(train, val, device=device, use_profile=True, use_context=True),
    }
    flattened = [
        value
        for mode in ("p2_only", "context_only", "context")
        for row in report[mode]["history"]
        for value in (row["loss"],)
    ]
    flattened.extend(
        float(report[mode]["validation"][metric])
        for mode in ("p2_only", "context_only", "context")
        for metric in ("tiny_direction_accuracy", "small_direction_accuracy")
    )
    if any(not math.isfinite(value) for value in flattened):
        raise FloatingPointError("nonfinite P2 oracle report")
    return report


def decide_p2_viability(report: Mapping[str, object]) -> dict[str, Any]:
    """Apply the predeclared final-epoch held-out evidence thresholds exactly."""
    if report.get("selection") != "final_epoch_only" or report.get("evaluated_epoch") != ORACLE_EPOCHS:
        raise ValueError("P2 oracle decision requires the frozen final epoch")
    try:
        validation = report["context"]["validation"]  # type: ignore[index]
        tiny = float(validation["tiny_direction_accuracy"])
        small = float(validation["small_direction_accuracy"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("P2 oracle report is incomplete") from error
    thresholds = {
        "tiny_direction_accuracy": P2_TINY_DIRECTION_THRESHOLD,
        "small_direction_accuracy": P2_SMALL_DIRECTION_THRESHOLD,
    }
    finite = math.isfinite(tiny) and math.isfinite(small)
    passed = finite and tiny >= thresholds["tiny_direction_accuracy"] and small >= thresholds["small_direction_accuracy"]
    return {
        "status": "passed" if passed else "scientific_failed",
        "thresholds": thresholds,
        "observed": {"tiny_direction_accuracy": tiny, "small_direction_accuracy": small},
        "finite": finite,
    }


__all__ = [
    "ORACLE_EPOCHS",
    "ORACLE_SEED",
    "P2OracleCacheViolation",
    "P2_NORMAL_OFFSETS_PX",
    "P2_SMALL_DIRECTION_THRESHOLD",
    "P2_TANGENT_FRACTIONS",
    "P2_TINY_DIRECTION_THRESHOLD",
    "correction_direction_targets",
    "decide_p2_viability",
    "enable_p2_oracle_determinism",
    "load_p2_oracle_cache",
    "sample_p2_edge_profiles",
    "train_p2_oracles",
    "write_p2_oracle_cache",
]
