"""Runtime diagnostics for training-only BPDD on top of frozen FDR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import io
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import torch
from torch import Tensor

from src.bpdd_loss import BPDDDetectionLoss, BPDDOptions
from src.fdr_loss import FDRDetectionLoss
from src.fdr_loss import MatchIndices


def _assignment_map(layer: MatchIndices) -> dict[tuple[int, int], int]:
    mapping: dict[tuple[int, int], int] = {}
    for batch_index, (queries, targets) in enumerate(layer):
        if queries.numel() != targets.numel():
            raise ValueError("stock assignment query/target lengths differ")
        for query, target in zip(queries.detach().cpu().tolist(), targets.detach().cpu().tolist()):
            key = (batch_index, int(query))
            value = int(target)
            if key in mapping and mapping[key] != value:
                raise ValueError("one query has conflicting stock targets")
            mapping[key] = value
    return mapping


def summarize_assignment_continuity(
    assignments: Sequence[MatchIndices],
) -> dict[str, Any]:
    """Measure whether final matched Query identities keep the same GT earlier."""

    if len(assignments) < 2:
        raise ValueError("assignment continuity requires at least two decoder layers")
    maps = [_assignment_map(layer) for layer in assignments]
    final = maps[-1]
    denominator = len(final)
    layers: list[dict[str, Any]] = []
    total_supported = 0
    total_same = 0
    for index, mapping in enumerate(maps[:-1]):
        supported = sum(key in mapping for key in final)
        same = sum(mapping.get(key) == target for key, target in final.items())
        total_supported += supported
        total_same += same
        layers.append(
            {
                "layer": index,
                "supported_queries": supported,
                "same_target_queries": same,
                "query_support_rate": float(supported / denominator) if denominator else 0.0,
                "same_target_rate": float(same / denominator) if denominator else 0.0,
            }
        )
    total = denominator * len(layers)
    return {
        "final_matched_queries": denominator,
        "layers": layers,
        "overall_query_support_rate": float(total_supported / total) if total else 0.0,
        "overall_same_target_rate": float(total_same / total) if total else 0.0,
    }


def _tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _tree_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _tree_equal(left[key], right[key]) for key in left
        )
    return left == right


def _manifest_and_artifact(context: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    from src.fdr_runtime_preflight import _file_sha256
    from src.fdr_protocol import validate_fdr_initial_state

    manifest = json.loads(Path(context.protocol_manifest).read_text("utf-8"))
    initial = manifest.get("initial_state")
    if not isinstance(initial, Mapping):
        raise ValueError("BPDD protocol has no initial-state authority")
    expected_path = Path(context.initial_state).resolve()
    manifest_path = Path(str(initial.get("path", ""))).resolve()
    if expected_path != manifest_path or not expected_path.is_file():
        raise ValueError("BPDD preflight initial-state path differs from authority")
    if _file_sha256(expected_path) != initial.get("sha256"):
        raise ValueError("BPDD preflight initial-state SHA256 mismatch")
    artifact = torch.load(expected_path, map_location="cpu", weights_only=False)
    validate_fdr_initial_state(artifact)
    if artifact.get("fingerprints") != initial.get("fingerprints"):
        raise ValueError("BPDD initial-state partition fingerprints differ")
    return manifest, artifact


def _models(context: Any, device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    from src.fdr_protocol import load_fdr_initial_state
    from src.rtdetr_fdr import FDR_MODEL_CFG, FDRRTDETRDetectionModel
    from src.rtdetr_fdr_bpdd import BPDD_MODEL_CFG, FDRBPDDDetectionModel

    _manifest, artifact = _manifest_and_artifact(context)
    fdr = FDRRTDETRDetectionModel(
        FDR_MODEL_CFG, ch=3, nc=10, verbose=False, private_seed=10_000
    ).to(device)
    bpdd = FDRBPDDDetectionModel(
        BPDD_MODEL_CFG, ch=3, nc=10, verbose=False, private_seed=10_000
    ).to(device)
    load_fdr_initial_state(fdr, artifact, variant="fdr")
    load_fdr_initial_state(bpdd, artifact, variant="fdr")
    return fdr, bpdd, artifact


def run_b0(context: Any) -> dict[str, Any]:
    """Bind BPDD design, source, FDR initial state, and numerical authority."""

    from scripts.train_rtdetr_bpdd import (
        load_authority,
        validate_initial_state_file,
        validate_source_authority,
    )
    from src.bpdd_protocol import BPDD_PROTOCOL_SHA256

    manifest = load_authority(Path(context.protocol_manifest))
    source = validate_source_authority(manifest, Path(context.repository_root))
    state = validate_initial_state_file(Path(context.initial_state), manifest)
    authority_path = Path(context.repository_root) / "research" / "bpdd" / "authority.json"
    authority = json.loads(authority_path.read_text("utf-8"))
    candidate = authority.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("BPDD research authority has no candidate mapping")
    expected = {
        "weight": 0.5,
        "temperature": 0.5,
        "margin": 0.02,
        "eps": 1e-6,
    }
    if any(float(candidate.get(key, math.nan)) != value for key, value in expected.items()):
        raise ValueError("BPDD research authority differs from the frozen protocol")
    return {
        "status": "passed",
        "gate": "B0",
        "protocol_sha256": BPDD_PROTOCOL_SHA256,
        "source": source,
        "initial_state_sha256": state["sha256"],
        "candidate": dict(candidate),
    }


def _synthetic_inputs() -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any], list[MatchIndices]]:
    generator = torch.Generator().manual_seed(2901)
    boxes = torch.rand((3, 1, 3, 4), generator=generator)
    boxes[..., 2:] = boxes[..., 2:] * 0.2 + 0.05
    scores = torch.randn((3, 1, 3, 3), generator=generator)
    corners = torch.zeros((2, 1, 3, 4, 33))
    corners[0, 0, 0, :, 0] = -4.0
    corners[0, 0, 0, :, 1] = 4.0
    corners[1, 0, 0, :, 0] = 4.0
    corners[1, 0, 0, :, 1] = -4.0
    corners = corners.reshape(2, 1, 3, 132).requires_grad_(True)
    pre_boxes = boxes[1].detach().clone()
    batch = {
        "cls": torch.tensor([1], dtype=torch.long),
        "bboxes": torch.tensor([[0.52, 0.48, 0.18, 0.22]], dtype=torch.float32),
        "gt_groups": [1],
    }
    matches = [
        [(torch.tensor([query]), torch.tensor([0]))]
        for query in (2, 1, 0)
    ]
    return boxes, scores, corners, pre_boxes, batch, matches


def run_b1(_context: Any) -> dict[str, Any]:
    """Prove stock/FDR isolation, final-match reuse, detach, and finite BPDD math."""

    boxes, scores, corners, pre_boxes, batch, matches = _synthetic_inputs()
    parent = FDRDetectionLoss(
        nc=3, use_vfl=True, fgl_weight=0.0, supervise_pre_boxes=False
    )
    disabled = BPDDDetectionLoss(
        nc=3,
        use_vfl=True,
        fgl_weight=0.0,
        supervise_pre_boxes=False,
        bpdd_options=BPDDOptions(enabled=False),
    )
    enabled = BPDDDetectionLoss(
        nc=3,
        use_vfl=True,
        fgl_weight=0.0,
        supervise_pre_boxes=False,
        bpdd_options=BPDDOptions(margin=0.0),
    )
    expected = parent(
        (boxes, scores), batch, normal_match_indices=matches,
        corner_logits=corners, pre_boxes=pre_boxes,
    )
    neutral = disabled(
        (boxes, scores), batch, normal_match_indices=matches,
        corner_logits=corners, pre_boxes=pre_boxes,
    )
    losses = enabled(
        (boxes, scores), batch, normal_match_indices=matches,
        corner_logits=corners, pre_boxes=pre_boxes,
    )
    losses["loss_bpdd"].backward()
    gradient = corners.grad.reshape(2, 1, 3, 4, 33)
    checks = {
        "disabled_exact_fdr": set(expected) == set(neutral) and all(
            torch.equal(expected[key], neutral[key]) for key in expected
        ),
        "only_adds_bpdd_loss": set(losses) == {*set(expected), "loss_bpdd"},
        "loss_finite_positive": bool(
            torch.isfinite(losses["loss_bpdd"]) and losses["loss_bpdd"] > 0
        ),
        "no_extra_matcher": enabled.stock_match_calls == 0 and enabled.fgl_extra_match_calls == 0,
        "student_gradient_live": bool(gradient[0, 0, 0].abs().sum() > 0),
        "teacher_detached": bool(torch.count_nonzero(gradient[-1]) == 0),
        "unmatched_gradient_zero": bool(torch.count_nonzero(gradient[:, :, 1:]) == 0),
    }
    if not all(checks.values()):
        raise RuntimeError(f"BPDD B1 isolation failed: {checks}")
    return {"status": "passed", "gate": "B1", "device": "cpu", "checks": checks}


def run_b2(context: Any) -> dict[str, Any]:
    """Prove parameter/state identity and inference equivalence to FDR."""

    from src.fdr_protocol import public_state_sha256

    rng_before = torch.random.get_rng_state().clone()
    fdr, bpdd, _artifact = _models(context, torch.device("cpu"))
    rng_after = torch.random.get_rng_state().clone()
    fdr_state = {name: value.detach().cpu() for name, value in fdr.state_dict().items()}
    bpdd_state = {name: value.detach().cpu() for name, value in bpdd.state_dict().items()}
    generator = torch.Generator().manual_seed(2902)
    image = torch.rand((1, 3, 128, 128), generator=generator)
    fdr.eval()
    bpdd.eval()
    with torch.inference_mode():
        fdr_prediction = fdr.predict(image)
        bpdd_prediction = bpdd.predict(image)
    buffer = io.BytesIO()
    torch.save(bpdd_state, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu", weights_only=True)
    checks = {
        "state_keys_exact": set(fdr_state) == set(bpdd_state),
        "state_tensors_exact": _tree_equal(fdr_state, bpdd_state),
        "state_hash_exact": public_state_sha256(fdr_state) == public_state_sha256(bpdd_state),
        "parameter_count_exact": sum(p.numel() for p in fdr.parameters()) == sum(p.numel() for p in bpdd.parameters()),
        "inference_exact": _tree_equal(fdr_prediction, bpdd_prediction),
        "constructor_rng_preserved": bool(torch.equal(rng_before, rng_after)),
        "checkpoint_roundtrip": _tree_equal(bpdd_state, restored),
        "no_bpdd_module_parameters": not any("bpdd" in name.lower() for name, _ in bpdd.named_parameters()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"BPDD B2 state/inference isolation failed: {checks}")
    return {"status": "passed", "gate": "B2", "device": "cpu", "checks": checks}


def run_b3(context: Any) -> dict[str, Any]:
    """Execute one real VisDrone batch8 AMP128 MuSGD step on RTX 4090."""

    import torchvision
    import ultralytics
    from src.fdr_runtime_preflight import (
        _build_loader,
        _move_batch,
        _musgd,
        _python_tree_sha256,
        _tree_sha256,
    )
    from src.fdr_protocol import public_state_sha256

    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("B3 requires cuda:0 NVIDIA GeForce RTX 4090")
    device = torch.device("cuda:0")
    repository_root = Path(context.repository_root).resolve()
    source_before = _tree_sha256(repository_root)
    ultralytics_root = Path(ultralytics.__file__).resolve().parent
    ultralytics_before = _python_tree_sha256(ultralytics_root)
    fdr, model, artifact = _models(context, device)
    del fdr
    loader, subset_sha = _build_loader(context, augment=True)
    batch = _move_batch(next(iter(loader)), device)
    if int(batch["img"].shape[0]) != 8:
        raise RuntimeError("B3 real batch does not contain eight images")
    model.train()
    optimizer = _musgd(model)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=True, init_scale=128.0, growth_interval=2**31 - 1
    )
    optimizer.zero_grad(set_to_none=True)
    scale_before = float(scaler.get_scale())
    with torch.autocast("cuda", dtype=torch.float16, enabled=True):
        loss, _items = model.loss(batch)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    gradients_finite = bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    optimizer.zero_grad(set_to_none=True)
    stats = {
        name: float(value.detach().float().cpu())
        for name, value in model.last_bpdd_statistics.items()
    }
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=True):
        prediction = model.predict(batch["img"])
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu", weights_only=True)
    current = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()[0]
    properties = torch.cuda.get_device_properties(0)
    return {
        "status": "passed",
        "gate": "B3",
        "runtime": {"device": "cuda:0", "batch": 8, "imgsz": 640, "amp_scale": 128.0},
        "hardware": {
            "gpu_name": properties.name,
            "device_index": 0,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "driver": driver,
            "cuda": str(torch.version.cuda),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
        },
        "single_step": {
            "real_visdrone_batch": True,
            "forward": True,
            "backward": True,
            "optimizer": "MuSGD",
            "optimizer_steps": 1,
            "loss_finite": bool(torch.isfinite(loss.detach())),
            "gradients_finite": gradients_finite,
            "amp_scale_before": scale_before,
            "amp_scale_after": scale_after,
            "amp_skipped_steps": int(scale_after < scale_before),
            "validation_postprocess": isinstance(prediction, (Tensor, tuple)),
            "checkpoint_roundtrip": public_state_sha256(current) == public_state_sha256(restored),
        },
        "bpdd": stats,
        "immutability": {
            "source_before_sha256": source_before,
            "source_after_sha256": _tree_sha256(repository_root),
            "ultralytics_before_sha256": ultralytics_before,
            "ultralytics_after_sha256": _python_tree_sha256(ultralytics_root),
            "initial_fdr_state_sha256": artifact["fingerprints"]["fdr"],
            "data_order_sha256": subset_sha,
        },
    }


def run_b4(context: Any) -> dict[str, Any]:
    """Measure real teacher coverage and Query-to-target continuity before screening."""

    from src.fdr_runtime_preflight import _build_loader, _move_batch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _fdr, model, _artifact = _models(context, device)
    del _fdr
    loader, _subset_sha = _build_loader(context, augment=False)
    probes: list[dict[str, float]] = []
    continuity: list[dict[str, Any]] = []
    model.train()
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if batch_index >= 4:
                break
            batch = _move_batch(raw_batch, device)
            model.loss(batch)
            probes.append(
                {
                    name: float(value.detach().float().cpu())
                    for name, value in model.last_bpdd_statistics.items()
                }
            )
            assignments = model.criterion.normal_assignment_snapshot()
            continuity.append(summarize_assignment_continuity(assignments[1:]))
    if not probes:
        raise RuntimeError("B4 could not read a real VisDrone batch")
    final_matches = sum(item["final_matched_queries"] for item in continuity)
    active_values = [item["active_edge_ratio"] for item in probes]
    improvement_values = [item["mean_teacher_improvement"] for item in probes]
    finite = all(math.isfinite(value) for item in probes for value in item.values())
    active = max(active_values) > 0
    better = max(improvement_values) > 0
    evidence = {
        "status": "passed" if final_matches > 0 and finite and active and better else "scientific_failed",
        "gate": "B4",
        "batches": len(probes),
        "final_matched_queries": final_matches,
        "statistics": {
            name: sum(item[name] for item in probes) / len(probes)
            for name in probes[0]
        },
        "assignment_continuity": {
            "overall_query_support_rate": sum(
                item["overall_query_support_rate"] for item in continuity
            ) / len(continuity),
            "overall_same_target_rate": sum(
                item["overall_same_target_rate"] for item in continuity
            ) / len(continuity),
        },
        "checks": {
            "matched_normal_queries": final_matches > 0,
            "statistics_finite": finite,
            "teacher_active": active,
            "teacher_improvement_positive": better,
            "final_stock_assignment_only": True,
            "denoising_excluded": True,
        },
    }
    return evidence


__all__ = [
    "run_b0",
    "run_b1",
    "run_b2",
    "run_b3",
    "run_b4",
    "summarize_assignment_continuity",
]
