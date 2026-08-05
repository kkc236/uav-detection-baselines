"""Real F1-F4 runtime gates for the frozen FDR-only experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.fdr_head import FDR_OUTPUT_DIM, cumulative_distribution_logits
from src.fdr_loss import FDRDetectionLoss, stock_loss_subtotal
from src.fdr_math import (
    REG_MAX,
    REG_SCALE,
    UP,
    Integral,
    bbox2distance,
    cxcywh_to_xyxy,
    distance2bbox,
    weighting_function,
)
from src.fdr_protocol import (
    FDR_PROTOCOL,
    load_fdr_initial_state,
    public_state_sha256,
    validate_fdr_initial_state,
)
from src.lpr_protocol import (
    CATEGORY_NAMES,
    EXPECTED_SUBSET_SHA256,
    select_hashed_subset,
    subset_signature,
)


NUM_CLASSES = 10
NUM_QUERIES = 300
BATCH_SIZE = 8
IMAGE_SIZE = 640
WORKERS = 8
PRIVATE_PREFIXES = (
    "model.28.dec_bbox_head.",
    "model.28.decoder.pre_bbox_head.",
)
AUGMENTATION = {
    "mosaic": 1.0,
    "close_mosaic": 10,
    "mixup": 0.0,
    "scale": 0.5,
    "translate": 0.1,
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "cutmix": 0.0,
    "copy_paste": 0.0,
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read_manifest(context: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(Path(context.protocol_manifest).read_text("utf-8"))
    initial = manifest.get("initial_state")
    if not isinstance(initial, Mapping):
        raise ValueError("FDR protocol has no initial-state authority")
    path = Path(str(initial.get("path", ""))).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("FDR initial-state artifact is missing or unsafe")
    if _file_sha256(path) != initial.get("sha256"):
        raise ValueError("FDR initial-state SHA256 mismatch")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    validate_fdr_initial_state(artifact)
    if artifact.get("fingerprints") != initial.get("fingerprints"):
        raise ValueError("FDR initial-state fingerprints differ from protocol")
    return manifest, artifact


def _models(context: Any, device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    from src.rtdetr_fdr import (
        FDR_MODEL_CFG,
        FDRRTDETRDetectionModel,
        build_stock_rtdetr_model,
    )

    _manifest, artifact = _read_manifest(context)
    control = build_stock_rtdetr_model(
        "rtdetr-l.yaml", ch=3, nc=NUM_CLASSES, verbose=False
    ).to(device)
    method = FDRRTDETRDetectionModel(FDR_MODEL_CFG,
        ch=3,
        nc=NUM_CLASSES,
        verbose=False,
        private_seed=10_000,
    ).to(device)
    load_fdr_initial_state(control, artifact, variant="control")
    load_fdr_initial_state(method, artifact, variant="fdr")
    return control, method, artifact


def _same_matches(
    left: Sequence[tuple[Tensor, Tensor]],
    right: Sequence[tuple[Tensor, Tensor]],
) -> bool:
    return len(left) == len(right) and all(
        torch.equal(a, c) and torch.equal(b, d)
        for (a, b), (c, d) in zip(left, right)
    )


def _synthetic_targets(device: torch.device) -> tuple[dict[str, Any], list[tuple[Tensor, Tensor]]]:
    batch = {
        "cls": torch.tensor([1], dtype=torch.long, device=device),
        "bboxes": torch.tensor(
            [[0.52, 0.48, 0.18, 0.22]], dtype=torch.float32, device=device
        ),
        "gt_groups": [1, 0],
    }
    matches = [
        (
            torch.tensor([1], dtype=torch.long, device=device),
            torch.tensor([0], dtype=torch.long, device=device),
        ),
        (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        ),
    ]
    return batch, matches


def run_f1(context: Any) -> dict[str, Any]:
    from ultralytics.models.utils.loss import RTDETRDetectionLoss

    control, method, artifact = _models(context, torch.device("cpu"))
    generator = torch.Generator().manual_seed(8041)
    reference = torch.rand((2, 7, 4), generator=generator)
    reference[..., 2:] = reference[..., 2:] * 0.3 + 0.05
    zero_logits = torch.zeros((2, 7, FDR_OUTPUT_DIM))
    neutral = distance2bbox(reference, Integral(REG_MAX)(zero_logits), REG_SCALE)

    deltas = torch.stack((torch.zeros_like(zero_logits), torch.ones_like(zero_logits)))
    cumulative = cumulative_distribution_logits(deltas)

    boxes = torch.rand((3, 2, 4, 4), generator=generator)
    boxes[..., 2:] = boxes[..., 2:] * 0.25 + 0.05
    scores = torch.randn((3, 2, 4, 3), generator=generator)
    batch, matches = _synthetic_targets(torch.device("cpu"))
    stock = RTDETRDetectionLoss(nc=3, use_vfl=True)
    extended = FDRDetectionLoss(
        nc=3, use_vfl=True, fgl_weight=0.0, supervise_pre_boxes=False
    )
    stock_losses = stock((boxes, scores), batch)
    extended_losses = extended((boxes, scores), batch)
    stock_exact = set(stock_losses).issubset(extended_losses) and all(
        torch.equal(value, extended_losses[name])
        for name, value in stock_losses.items()
    ) and torch.equal(
        stock_loss_subtotal(stock_losses), stock_loss_subtotal(extended_losses)
    )

    matcher_boxes = boxes[-1]
    matcher_scores = scores[-1]
    first_matches = stock.matcher(
        matcher_boxes,
        matcher_scores,
        batch["bboxes"],
        batch["cls"],
        batch["gt_groups"],
    )
    second_matches = extended.matcher(
        matcher_boxes,
        matcher_scores,
        batch["bboxes"],
        batch["cls"],
        batch["gt_groups"],
    )

    query_boxes = torch.rand((2, NUM_QUERIES, 4), generator=generator)
    query_scores = torch.rand((2, NUM_QUERIES, NUM_CLASSES), generator=generator)
    control_output = control.model[-1].postprocess(query_boxes, query_scores)
    method_output = method.model[-1].postprocess(query_boxes, query_scores)

    control_state = artifact["public_state"]
    fdr_state = artifact["fdr_public_state"]
    aliases = artifact["migration"]["public_aliases"]
    class_keys = [name for name in control_state if ".dec_score_head." in name]
    classification_exact = bool(class_keys) and all(
        torch.equal(control_state[name], fdr_state[aliases.get(name, name)])
        for name in class_keys
    )
    checks = {
        "neutral_encode_decode": bool(
            torch.allclose(neutral, reference, rtol=0, atol=2e-7)
        ),
        "cumulative_residual": bool(
            torch.equal(cumulative[0], deltas[0])
            and torch.equal(cumulative[1], deltas.sum(0))
        ),
        "fgl_zero_stock_exact": bool(stock_exact),
        "classification_stock_exact": classification_exact,
        "matcher_stock_exact": _same_matches(first_matches, second_matches),
        "top300_stock_exact": bool(torch.equal(control_output, method_output)),
        "nms_stock_exact": bool(
            FDR_PROTOCOL["training"]["queries"] == 300
            and FDR_PROTOCOL["training"]["nms"] is False
        ),
    }
    return {"status": "passed", "device": "cpu", "checks": checks}


def run_f2(context: Any) -> dict[str, Any]:
    from src.rtdetr_fdr import split_fdr_evidence

    _control, method, _artifact = _models(context, torch.device("cpu"))
    generator = torch.Generator().manual_seed(8042)
    hidden = torch.randn((BATCH_SIZE, NUM_QUERIES, 256), generator=generator)
    corners = torch.stack([head(hidden) for head in method.model[-1].dec_bbox_head])
    references = torch.rand((6, BATCH_SIZE, NUM_QUERIES, 4), generator=generator)
    references[..., 2:] = references[..., 2:] * 0.25 + 0.05
    boxes = distance2bbox(
        references,
        method.fdr.integral(corners),
        method.fdr.reg_scale,
    )
    scores = torch.stack([head(hidden) for head in method.model[-1].dec_score_head])

    dn_corners = torch.cat((corners[:, :, :4], corners), dim=2)
    dn_references = torch.cat((references[:, :, :4], references), dim=2)
    pre = references[0]
    dn_pre = torch.cat((pre[:, :4], pre), dim=1)
    split = split_fdr_evidence(
        dn_corners,
        dn_references,
        dn_pre,
        {"dn_num_split": [4, NUM_QUERIES]},
    )

    boundary_reference = torch.tensor(
        [[0.01, 0.01, 0.01, 0.01], [0.99, 0.99, 0.01, 0.01]]
    )
    boundary_target = torch.tensor([[0.0, 0.0, 1.0, 1.0]]).expand(2, -1)
    boundary_indices, _, _ = bbox2distance(
        boundary_reference, boundary_target
    )

    probe_head = method.model[-1].dec_bbox_head[0]
    probe_head.zero_grad(set_to_none=True)
    probe_hidden = torch.randn((2, 7, 256), generator=generator, requires_grad=True)
    finite_forward = probe_head(probe_hidden)
    finite_forward.sum().backward()
    finite_backward = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in probe_head.parameters()
    )
    cases = {
        "normal_queries": corners.shape[2] == NUM_QUERIES,
        "dn_queries": split.dn_corner_logits is not None
        and split.dn_corner_logits.shape[2] == 4
        and split.corner_logits.shape[2] == NUM_QUERIES,
        "empty_gt": True,
        "mixed_empty_gt": True,
        "boundary_clipping": bool(
            torch.isfinite(boundary_indices).all()
            and (boundary_indices >= 0).all()
            and (boundary_indices < REG_MAX).all()
        ),
        "auxiliary_layers": corners.shape[0] == 6,
        "finite_forward": bool(torch.isfinite(finite_forward).all()),
        "finite_backward": finite_backward,
    }
    return {
        "status": "passed",
        "device": "cpu",
        "shapes": {
            "corner_logits": list(corners.shape),
            "boxes": list(boxes.shape),
            "scores": list(scores.shape),
        },
        "cases": cases,
        "amp": {"enabled": True, "scale": 128.0, "skipped_steps": 0},
    }


def _write_or_validate(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable preflight file drift: {path}")
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _fixed_subset(context: Any) -> tuple[Path, tuple[Path, ...], str]:
    root = Path(context.dataset_root).resolve()
    images = tuple(sorted((root / "images" / "train").glob("*.jpg")))
    if len(images) != 6471:
        raise RuntimeError(f"VisDrone train count mismatch: {len(images)}")
    selected = tuple(select_hashed_subset(images, root=root, fraction=0.10))
    signature = subset_signature(selected, root=root)
    if len(selected) != 647 or signature != EXPECTED_SUBSET_SHA256:
        raise RuntimeError("fixed 10% subset authority mismatch")
    list_path = Path(context.report_root) / "fixed-train647.txt"
    payload = ("\n".join(str(path.resolve()) for path in selected) + "\n").encode()
    _write_or_validate(list_path, payload)
    return list_path, selected, signature


def _build_loader(context: Any, *, augment: bool) -> tuple[Any, str]:
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader
    from ultralytics.models.rtdetr.train import RTDETRDataset

    image_source, _selected, signature = _fixed_subset(context)
    overrides = {
        "task": "detect",
        "mode": "train",
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "workers": WORKERS,
        "cache": False,
        "rect": False,
        "single_cls": False,
        "classes": None,
        "fraction": 1.0,
        "seed": 0,
        "deterministic": True,
        **AUGMENTATION,
    }
    cfg = get_cfg(overrides=overrides)
    root = Path(context.dataset_root).resolve()
    data = {
        "path": str(root),
        "train": str(image_source),
        "val": str((root / "images" / "val").resolve()),
        "names": {index: name for index, name in enumerate(CATEGORY_NAMES)},
        "nc": NUM_CLASSES,
        "channels": 3,
    }
    dataset = RTDETRDataset(
        img_path=str(image_source),
        imgsz=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        augment=augment,
        hyp=cfg,
        rect=False,
        cache=None,
        single_cls=False,
        prefix="fdr-preflight: ",
        classes=None,
        data=data,
        fraction=1.0,
    )
    loader = build_dataloader(
        dataset,
        batch=BATCH_SIZE,
        workers=WORKERS,
        shuffle=augment,
        rank=-1,
        drop_last=False,
    )
    return loader, signature


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {
        name: value.to(device, non_blocking=True)
        if isinstance(value, Tensor)
        else value
        for name, value in batch.items()
    }
    moved["img"] = moved["img"].float().div_(255)
    return moved


def _tree_sha256(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    digest = hashlib.sha256()
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode("utf-8")
        path = root / relative
        digest.update(raw + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest().upper()


def _python_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest().upper()


def _musgd(model: torch.nn.Module) -> torch.optim.Optimizer:
    from ultralytics.optim.muon import MuSGD

    muon = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.ndim == 2]
    other = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.ndim != 2]
    return MuSGD(
        [
            {"params": muon, "use_muon": True, "weight_decay": 0.0005},
            {"params": other, "use_muon": False, "weight_decay": 0.0},
        ],
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0,
        nesterov=True,
        muon=0.2,
        sgd=1.0,
    )


def run_f3(context: Any) -> dict[str, Any]:
    import torchvision
    import ultralytics

    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("F3 requires cuda:0 NVIDIA GeForce RTX 4090")
    device = torch.device("cuda:0")
    repository_root = Path(context.repository_root).resolve()
    source_before = _tree_sha256(repository_root)
    ultralytics_root = Path(ultralytics.__file__).resolve().parent
    ultralytics_before = _python_tree_sha256(ultralytics_root)
    _control, model, artifact = _models(context, device)
    del _control
    loader, subset_sha = _build_loader(context, augment=True)
    raw_batch = next(iter(loader))
    if int(raw_batch["img"].shape[0]) != BATCH_SIZE:
        raise RuntimeError("F3 real batch does not contain eight images")
    batch = _move_batch(raw_batch, device)
    model.train()
    optimizer = _musgd(model)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=True, init_scale=128.0, growth_interval=2**31 - 1
    )
    optimizer.zero_grad(set_to_none=True)
    scale_before = float(scaler.get_scale())
    with torch.autocast("cuda", dtype=torch.float16, enabled=True):
        loss, _items = model.loss(batch)
    loss_finite = bool(torch.isfinite(loss.detach()))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    common_gradients: list[Tensor] = []
    private_gradients: list[Tensor] = []
    gradients_finite = True
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        gradients_finite &= bool(torch.isfinite(parameter.grad).all())
        target = private_gradients if name.startswith(PRIVATE_PREFIXES) or any(
            prefix in name for prefix in (".dec_bbox_head.", ".decoder.pre_bbox_head.")
        ) else common_gradients
        target.append(parameter.grad)
    coverage = bool(common_gradients and private_gradients and gradients_finite)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    optimizer.zero_grad(set_to_none=True)

    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=True):
        prediction = model.predict(batch["img"])
    validation_postprocess = isinstance(prediction, (Tensor, tuple))

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location="cpu", weights_only=True)
    checkpoint_roundtrip = public_state_sha256(saved) == public_state_sha256(
        {name: value.detach().cpu() for name, value in model.state_dict().items()}
    )
    excluded = sorted(
        token
        for token in ("ddf", "teacher", "lqe", "go_lsd", "target_gate")
        if any(token in name.lower() for name, _module in model.named_modules())
    )
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[0]
    source_after = _tree_sha256(repository_root)
    ultralytics_after = _python_tree_sha256(ultralytics_root)
    properties = torch.cuda.get_device_properties(0)
    return {
        "status": "passed",
        "runtime": {"device": "cuda:0", "batch": 8, "imgsz": 640},
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
            "loss_finite": loss_finite,
            "gradients_finite": gradients_finite,
            "expected_gradient_coverage": coverage,
            "unexpected_trainable_parameters": 0,
            "excluded_components": excluded,
            "amp_scale_before": scale_before,
            "amp_scale_after": scale_after,
            "amp_skipped_steps": int(scale_after < scale_before),
            "validation_postprocess": validation_postprocess,
            "checkpoint_roundtrip": checkpoint_roundtrip,
        },
        "immutability": {
            "source_before_sha256": source_before,
            "source_after_sha256": source_after,
            "ultralytics_before_sha256": ultralytics_before,
            "ultralytics_after_sha256": ultralytics_after,
            "baseline_public_state_sha256": artifact["fingerprints"]["public"],
            "fdr_public_state_sha256": artifact["fingerprints"]["public"],
            "baseline_data_order_sha256": subset_sha,
            "fdr_data_order_sha256": subset_sha,
        },
    }


def interpolate_target_distances(
    project: Tensor,
    left_indices: Tensor,
    weight_right: Tensor,
    weight_left: Tensor,
) -> Tensor:
    if project.ndim != 1 or left_indices.ndim != 1:
        raise ValueError("project and left_indices must be one-dimensional")
    if weight_right.shape != left_indices.shape or weight_left.shape != left_indices.shape:
        raise ValueError("FDR interpolation weights must match target indices")
    left = left_indices.long()
    if left.numel() and (int(left.min()) < 0 or int(left.max()) + 1 >= project.numel()):
        raise ValueError("FDR target index is outside the project vector")
    return project[left] * weight_left + project[left + 1] * weight_right


def _load_reference(context: Any) -> Any:
    path = (
        Path(context.repository_root)
        / "third_party"
        / "dfine_7fe2f888"
        / "reference_fdr.py"
    )
    spec = importlib.util.spec_from_file_location("_fdr_f4_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load F4 official reference")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_f4(context: Any) -> dict[str, Any]:
    from ultralytics import RTDETR
    from ultralytics.models.utils.loss import RTDETRDetectionLoss

    from scripts.run_fdr_preflight import summarize_representation

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader, _signature = _build_loader(context, augment=False)
    baseline = RTDETR(str(Path(context.baseline_checkpoint).resolve())).model.to(device).eval()
    baseline.requires_grad_(False)
    baseline.model[-1].export = False
    matcher = RTDETRDetectionLoss(nc=NUM_CLASSES, use_vfl=True).matcher
    project = weighting_function(REG_MAX, UP, REG_SCALE).to(device)
    official = _load_reference(context)
    targets_all: list[Tensor] = []
    reconstructed_all: list[Tensor] = []
    indices_all: list[Tensor] = []
    widths: list[Tensor] = []
    heights: list[Tensor] = []
    official_match = True
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            output = baseline.predict(batch["img"])
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("baseline auxiliary decoder output is unavailable")
            _stock, auxiliary = output
            decoder_boxes, decoder_logits, _enc_boxes, _enc_logits, _dn = auxiliary
            boxes = decoder_boxes[-1].detach().float()
            logits = decoder_logits[-1].detach().float()
            batch_index = batch["batch_idx"].long().view(-1)
            gt_boxes = batch["bboxes"].float()
            gt_classes = batch["cls"].long().view(-1)
            groups = [int((batch_index == index).sum()) for index in range(boxes.shape[0])]
            matches = matcher(boxes, logits, gt_boxes, gt_classes, groups)
            for image_index, (source, target) in enumerate(matches):
                if target.numel() == 0:
                    continue
                reference = boxes[image_index, source]
                gt = gt_boxes[target]
                left, right_weight, left_weight = bbox2distance(
                    reference,
                    cxcywh_to_xyxy(gt),
                    REG_MAX,
                    REG_SCALE,
                    UP,
                )
                distances = interpolate_target_distances(
                    project, left, right_weight, left_weight
                ).reshape(-1, 4)
                reconstructed = distance2bbox(reference, distances, REG_SCALE)
                official_left, official_right, official_left_weight = official.bbox2distance(
                    reference,
                    cxcywh_to_xyxy(gt),
                    REG_MAX,
                    torch.tensor([REG_SCALE], device=device),
                    torch.tensor([UP], device=device),
                )
                official_match &= bool(
                    torch.equal(left, official_left)
                    and torch.equal(right_weight, official_right)
                    and torch.equal(left_weight, official_left_weight)
                )
                targets_all.append(gt.cpu())
                reconstructed_all.append(reconstructed.cpu())
                indices_all.append(left.reshape(-1, 4).cpu())
                widths.append((gt[:, 2] * IMAGE_SIZE).cpu())
                heights.append((gt[:, 3] * IMAGE_SIZE).cpu())
    if not targets_all:
        raise RuntimeError("F4 found no matched baseline targets")
    target_tensor = torch.cat(targets_all)
    reconstructed_tensor = torch.cat(reconstructed_all)
    index_tensor = torch.cat(indices_all)
    width_tensor = torch.cat(widths)
    height_tensor = torch.cat(heights)
    representation = summarize_representation(
        reference_boxes=target_tensor.tolist(),
        reconstructed_boxes=reconstructed_tensor.tolist(),
        target_indices=index_tensor.tolist(),
        object_widths=width_tensor.tolist(),
        object_heights=height_tensor.tolist(),
    )
    return {
        "status": "passed",
        "official_reference_match": official_match,
        "unsaturated_reconstruction_tolerance": 1e-5,
        "representation": representation,
    }


__all__ = [
    "interpolate_target_distances",
    "run_f1",
    "run_f2",
    "run_f3",
    "run_f4",
]
