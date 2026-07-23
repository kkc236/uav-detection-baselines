"""Pure NumPy evaluator for the frozen SBR-RTDETR protocol.

The evaluator deliberately has no dependency on a detector or on Ultralytics.
Coordinates are half-open ``xyxy`` pixels and are validated fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

IOU_THRESHOLDS = np.arange(0.50, 0.951, 0.05, dtype=float)
MAX_DET = 300
CONF_THRESHOLD = 0.001
SIZE_BINS = {
    "tiny": (0.0, 16.0, True, True),
    "small": (16.0, 32.0, False, True),
    "medium": (32.0, 96.0, False, True),
    "large": (96.0, float("inf"), False, False),
}


@dataclass(frozen=True)
class Prediction:
    box: tuple[float, float, float, float]
    score: float
    cls: int
    source: int = 0
    query: int = 0


@dataclass(frozen=True)
class Target:
    box: tuple[float, float, float, float]
    cls: int
    effective_gain: float = 1.0


def _arr_boxes(value: Any, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.size == 0:
        return np.empty((0, 4), dtype=float)
    if a.ndim == 1 and a.shape == (4,):
        a = a.reshape(1, 4)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N,4)")
    if not np.isfinite(a).all() or (a[:, 2:] < a[:, :2]).any() or (a < 0).any():
        raise ValueError(f"{name} contains non-finite or illegal xyxy coordinates")
    return a


def _arr_1d(value: Any, n: int, name: str, dtype: Any) -> np.ndarray:
    a = np.asarray(value, dtype=dtype).reshape(-1)
    if a.size != n:
        raise ValueError(f"{name} length must be {n}")
    return a


def _validate(
    pred_boxes: Any,
    pred_scores: Any,
    pred_classes: Any,
    gt_boxes: Any,
    gt_classes: Any,
    ignore_boxes: Any,
    pred_source: Any,
    pred_query: Any,
    conf_threshold: float,
) -> tuple[np.ndarray, ...]:
    pb = _arr_boxes(pred_boxes, "pred_boxes")
    gb = _arr_boxes(gt_boxes, "gt_boxes")
    ib = _arr_boxes(ignore_boxes if ignore_boxes is not None else [], "ignore_boxes")
    ps = _arr_1d(pred_scores, len(pb), "pred_scores", float)
    pc = _arr_1d(pred_classes, len(pb), "pred_classes", int)
    gc = _arr_1d(gt_classes, len(gb), "gt_classes", int)
    src = _arr_1d(pred_source if pred_source is not None else np.zeros(len(pb)), len(pb), "pred_source", int)
    qry = _arr_1d(pred_query if pred_query is not None else np.arange(len(pb)), len(pb), "pred_query", int)
    if not np.isfinite(ps).all() or (ps < 0).any() or (ps > 1).any():
        raise ValueError("pred_scores must be finite values in [0,1]")
    if (pc < 0).any() or (gc < 0).any():
        raise ValueError("class ids must be non-negative")
    if not np.isfinite(conf_threshold) or conf_threshold < 0 or conf_threshold > 1:
        raise ValueError("conf_threshold must be in [0,1]")
    return pb, ps, pc, gb, gc, ib, src, qry


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Vectorized IoU matrix."""
    a = _arr_boxes(boxes1, "boxes1")
    b = _arr_boxes(boxes2, "boxes2")
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.maximum(rb - lt, 0.0), axis=2)
    area_a = np.prod(np.maximum(a[:, 2:] - a[:, :2], 0.0), axis=1)
    area_b = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0.0), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _ioa_prediction_ignore(pred_boxes: np.ndarray, ignore_boxes: np.ndarray) -> np.ndarray:
    if len(pred_boxes) == 0 or len(ignore_boxes) == 0:
        return np.zeros(len(pred_boxes), dtype=bool)
    lt = np.maximum(pred_boxes[:, None, :2], ignore_boxes[None, :, :2])
    rb = np.minimum(pred_boxes[:, None, 2:], ignore_boxes[None, :, 2:])
    inter = np.prod(np.maximum(rb - lt, 0.0), axis=2)
    area = np.prod(np.maximum(pred_boxes[:, 2:] - pred_boxes[:, :2], 0.0), axis=1)
    ioa = np.divide(inter, area[:, None], out=np.zeros_like(inter), where=area[:, None] > 0)
    return np.any(ioa >= 0.50, axis=1)


def _sqrt_effective_area(gt_boxes: np.ndarray, gain: float | np.ndarray) -> np.ndarray:
    g = np.asarray(gain, dtype=float)
    if g.ndim == 0:
        g = np.full(len(gt_boxes), float(g))
    elif g.shape == (2,):
        if not np.isfinite(g).all() or (g <= 0).any():
            raise ValueError("effective_gain must be positive and finite")
        wh = np.maximum(gt_boxes[:, 2:] - gt_boxes[:, :2], 0.0)
        return np.sqrt(wh[:, 0] * g[0] * wh[:, 1] * g[1])
    if g.shape != (len(gt_boxes),) or not np.isfinite(g).all() or (g <= 0).any():
        raise ValueError("effective_gain must be positive and match gt count")
    wh = np.maximum(gt_boxes[:, 2:] - gt_boxes[:, :2], 0.0)
    return np.sqrt(wh[:, 0] * wh[:, 1]) * g


def _in_bin(radius: np.ndarray, name: str) -> np.ndarray:
    if name not in SIZE_BINS:
        raise ValueError(f"unknown size bin {name}")
    lo, hi, lo_inclusive, hi_inclusive = SIZE_BINS[name]
    left = radius >= lo if lo_inclusive else radius > lo
    right = radius <= hi if hi_inclusive else radius > lo
    return left & right


def compute_ap(recall: Sequence[float], precision: Sequence[float]) -> float:
    """Ultralytics ``compute_ap`` compatible 101-point interpolation."""
    r = np.asarray(recall, dtype=float).reshape(-1)
    p = np.asarray(precision, dtype=float).reshape(-1)
    if len(r) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], np.clip(r, 0, 1), [1.0]))
    mpre = np.concatenate(([1.0], np.clip(p, 0, 1), [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def _ap_from_records(records: list[tuple[float, int, int, int]], gt_count: Mapping[int, int]) -> float:
    """Records are ``(score, class, tp, fp)`` after neutral predictions are removed."""
    by_cls: dict[int, list[tuple[float, int, int]]] = {}
    for rec in records:
        by_cls.setdefault(int(rec[1]), []).append((rec[0], rec[2], rec[3]))
    aps = []
    for cls, n_gt in gt_count.items():
        if n_gt <= 0:
            continue
        rows = sorted(by_cls.get(int(cls), []), key=lambda x: -x[0])
        if not rows:
            aps.append(0.0)
            continue
        tp = np.cumsum([r[1] for r in rows], dtype=float)
        fp = np.cumsum([r[2] for r in rows], dtype=float)
        aps.append(compute_ap(tp / max(float(n_gt), 1.0), tp / np.maximum(tp + fp, 1e-12)))
    return float(np.mean(aps)) if aps else 0.0


def _prepare_predictions(
    pb: np.ndarray,
    ps: np.ndarray,
    pc: np.ndarray,
    src: np.ndarray,
    qry: np.ndarray,
    conf_threshold: float,
    max_det: int,
) -> tuple[np.ndarray, ...]:
    keep = ps >= conf_threshold
    idx = np.flatnonzero(keep)
    # Stable deterministic order: confidence, source, query, original index.
    idx = np.array(sorted(idx.tolist(), key=lambda i: (-float(ps[i]), int(src[i]), int(qry[i]), int(i))), dtype=int)
    if max_det < 1:
        raise ValueError("max_det must be positive")
    idx = idx[:max_det]
    return pb[idx], ps[idx], pc[idx], src[idx], qry[idx], idx


def _evaluate_threshold(
    pb: np.ndarray,
    ps: np.ndarray,
    pc: np.ndarray,
    neutral_global: np.ndarray,
    gb: np.ndarray,
    gc: np.ndarray,
    selected_gt: np.ndarray,
    iou: np.ndarray,
    threshold: float,
) -> tuple[dict[str, int], list[tuple[float, int, int, int]]]:
    matched = np.zeros(len(gb), dtype=bool)
    records: list[tuple[float, int, int, int]] = []
    tp = fp = neutralized = 0
    for i in range(len(pb)):
        if neutral_global[i]:
            neutralized += 1
            continue
        same = (gc == pc[i]) & selected_gt & ~matched & (iou[i] >= threshold)
        if np.any(same):
            candidates = np.flatnonzero(same)
            # Highest IoU, then lowest GT index for deterministic ties.
            j = int(candidates[np.argmax(iou[i, candidates])])
            matched[j] = True
            tp += 1
            records.append((float(ps[i]), int(pc[i]), 1, 0))
            continue
        # COCO-style area range: a prediction matching an out-of-range GT
        # is ignored at this threshold only when IoU reaches this threshold.
        out = (gc == pc[i]) & ~selected_gt & (iou[i] >= threshold)
        if np.any(out):
            neutralized += 1
            continue
        fp += 1
        records.append((float(ps[i]), int(pc[i]), 0, 1))
    fn = int(np.count_nonzero(selected_gt & ~matched))
    return {"tp": tp, "fp": fp, "fn": fn, "neutralized": neutralized, "predictions": len(pb)}, records


def evaluate_sbr(
    pred_boxes: Any,
    pred_scores: Any,
    pred_classes: Any,
    gt_boxes: Any,
    gt_classes: Any,
    *,
    ignore_boxes: Any = None,
    effective_gain: float | Sequence[float] = 1.0,
    max_det: int = 300,
    conf_threshold: float = 0.001,
    pred_source: Any = None,
    pred_query: Any = None,
    size_bin: str | None = None,
) -> dict[str, Any]:
    """Evaluate one image (or one already aggregated image record).

    ``size_bin`` may restrict output to one bin; by default all four explicit
    ``*-SBR`` metrics are returned.
    """
    if not isinstance(max_det, (int, np.integer)) or int(max_det) != MAX_DET:
        raise ValueError(f"max_det is frozen at {MAX_DET}")
    if not isinstance(conf_threshold, (float, np.floating)) or float(conf_threshold) != CONF_THRESHOLD:
        raise ValueError(f"conf_threshold is frozen at {CONF_THRESHOLD}")
    pb, ps, pc, gb, gc, ib, src, qry = _validate(
        pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, ignore_boxes,
        pred_source, pred_query, conf_threshold,
    )
    pb, ps, pc, src, qry, _ = _prepare_predictions(pb, ps, pc, src, qry, conf_threshold, int(max_det))
    neutral_global = _ioa_prediction_ignore(pb, ib)
    iou = box_iou(pb, gb)
    gain = np.asarray(effective_gain, dtype=float)
    radius = _sqrt_effective_area(gb, gain)
    bins = [size_bin] if size_bin is not None else list(SIZE_BINS)
    out: dict[str, Any] = {
        "iou_thresholds": IOU_THRESHOLDS.copy(),
        "counts": {},
        "per_threshold": {},
    }
    all_classes = sorted(set(gc.tolist()))

    for bname in bins:
        selected = _in_bin(radius, bname)
        per_t: dict[float, Any] = {}
        aps = []
        gt_count = {int(c): int(np.count_nonzero(selected & (gc == c))) for c in all_classes}
        for t in IOU_THRESHOLDS:
            counts, recs = _evaluate_threshold(pb, ps, pc, neutral_global, gb, gc, selected, iou, float(t))
            counts["gt"] = int(np.count_nonzero(selected))
            per_t[float(round(float(t), 2))] = counts
            aps.append(_ap_from_records(recs, gt_count))
        out["per_threshold"][bname] = per_t
        out[f"AP-{bname}-SBR"] = float(np.mean(aps))
        out[f"AP-{bname}"] = float(np.mean(aps))
        out[f"AP50-{bname}-SBR"] = float(aps[0])
        out[f"AP75-{bname}-SBR"] = float(aps[5])

    # Overall metrics and canonical summary counts at IoU=.50.
    selected_all = np.ones(len(gb), dtype=bool)
    gt_count_all = {int(c): int(np.count_nonzero(gc == c)) for c in all_classes}
    overall_aps = []
    for t in IOU_THRESHOLDS:
        counts, recs = _evaluate_threshold(pb, ps, pc, neutral_global, gb, gc, selected_all, iou, float(t))
        if round(float(t), 2) == 0.50:
            out["counts"] = counts
        out["counts"].setdefault("predictions", len(pb))
        overall_aps.append(_ap_from_records(recs, gt_count_all))
    out["AP50"] = float(overall_aps[0])
    out["AP75"] = float(overall_aps[5])
    out["mAP50-95"] = float(np.mean(overall_aps))
    # Convenience aliases expected by older callers.
    out["AP50-95"] = out["mAP50-95"]
    if size_bin is not None:
        for b in SIZE_BINS:
            out.setdefault(f"AP-{b}-SBR", 0.0)
            out.setdefault(f"AP-{b}", 0.0)
    tiny = out["per_threshold"].get("tiny", {}).get(0.5)
    if tiny is not None:
        out["tiny_recall"] = float(tiny["tp"] / tiny["gt"]) if tiny["gt"] else 0.0
    return out


def tiny_recall(
    pred_boxes: Any,
    pred_scores: Any,
    pred_classes: Any,
    gt_boxes: Any,
    gt_classes: Any,
    *,
    ignore_boxes: Any = None,
    effective_gain: float | Sequence[float] = 1.0,
    max_det: int = 300,
    conf_threshold: float = 0.001,
    pred_source: Any = None,
    pred_query: Any = None,
) -> dict[str, Any]:
    """Tiny-target micro recall at IoU=.50 using class-aware one-to-one matching."""
    m = evaluate_sbr(
        pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes,
        ignore_boxes=ignore_boxes, effective_gain=effective_gain,
        max_det=max_det, conf_threshold=conf_threshold,
        pred_source=pred_source, pred_query=pred_query, size_bin="tiny",
    )
    c = m["per_threshold"]["tiny"][0.5]
    # Report class-level numerators/denominators for auditing.  Reusing the
    # same evaluator keeps matching order and ignore handling identical.
    gc_arr = np.asarray(gt_classes, dtype=int).reshape(-1)
    per_class: dict[int, dict[str, float | int]] = {}
    for cls in sorted(set(gc_arr.tolist())):
        mask = gc_arr == cls
        sub = evaluate_sbr(
            pred_boxes, pred_scores, pred_classes,
            np.asarray(gt_boxes, dtype=float).reshape((-1, 4))[mask],
            gc_arr[mask],
            ignore_boxes=ignore_boxes, effective_gain=np.asarray(effective_gain)[mask]
            if np.asarray(effective_gain).ndim and np.asarray(effective_gain).shape != (2,)
            else effective_gain,
            max_det=max_det, conf_threshold=conf_threshold,
            pred_source=pred_source, pred_query=pred_query, size_bin="tiny",
        )
        sc = sub["per_threshold"]["tiny"][0.5]
        per_class[int(cls)] = {
            "matched": int(sc["tp"]),
            "targets": int(sc["gt"]),
            "recall": float(sc["tp"] / sc["gt"]) if sc["gt"] else 0.0,
        }
    return {
        "recall": float(c["tp"] / c["gt"]) if c["gt"] else 0.0,
        "matched": int(c["tp"]),
        "targets": int(c["gt"]),
        "false_positives": int(c["fp"]),
        "neutralized": int(c["neutralized"]),
        "per_class": per_class,
    }


def evaluate_dataset(images: Sequence[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Evaluate multiple image records with pooled class confidence curves."""
    if not images:
        return evaluate_sbr([], [], [], [], [], **kwargs)
    max_det = kwargs.pop("max_det", MAX_DET)
    conf_threshold = kwargs.pop("conf_threshold", CONF_THRESHOLD)
    if not isinstance(max_det, (int, np.integer)) or int(max_det) != MAX_DET:
        raise ValueError(f"max_det is frozen at {MAX_DET}")
    if not isinstance(conf_threshold, (float, np.floating)) or float(conf_threshold) != CONF_THRESHOLD:
        raise ValueError(f"conf_threshold is frozen at {CONF_THRESHOLD}")
    bins = list(SIZE_BINS)
    pooled: dict[str, dict[float, list[tuple[float, int, int, int]]]] = {
        b: {float(round(t, 2)): [] for t in IOU_THRESHOLDS} for b in bins
    }
    pooled_global: dict[float, list[tuple[float, int, int, int]]] = {float(round(t, 2)): [] for t in IOU_THRESHOLDS}
    gt_counts: dict[str, dict[int, int]] = {b: {} for b in bins}
    gt_global: dict[int, int] = {}
    sum_counts = {k: 0 for k in ("tp", "fp", "fn", "neutralized", "predictions")}
    bin_counts: dict[str, dict[float, dict[str, int]]] = {
        b: {float(round(t, 2)): {k: 0 for k in ("tp", "fp", "fn", "neutralized", "predictions", "gt")} for t in IOU_THRESHOLDS}
        for b in bins
    }
    for image in images:
        args = dict(kwargs)
        args.update({k: image[k] for k in ("pred_boxes", "pred_scores", "pred_classes", "gt_boxes", "gt_classes")})
        ib = image.get("ignore_boxes", args.pop("ignore_boxes", None))
        eg = image.get("effective_gain", args.pop("effective_gain", 1.0))
        src = image.get("pred_source", args.pop("pred_source", None))
        qry = image.get("pred_query", args.pop("pred_query", None))
        pb, ps, pc, gb, gc, ign, src, qry = _validate(
            args["pred_boxes"], args["pred_scores"], args["pred_classes"], args["gt_boxes"], args["gt_classes"],
            ib, src, qry, conf_threshold,
        )
        pb, ps, pc, src, qry, _ = _prepare_predictions(pb, ps, pc, src, qry, conf_threshold, int(max_det))
        neutral = _ioa_prediction_ignore(pb, ign)
        iou = box_iou(pb, gb)
        radius = _sqrt_effective_area(gb, eg)
        classes = sorted(set(gc.tolist()))
        for c in classes:
            gt_global[int(c)] = gt_global.get(int(c), 0) + int(np.count_nonzero(gc == c))
        for b in bins:
            sel = _in_bin(radius, b)
            for c in classes:
                gt_counts[b][int(c)] = gt_counts[b].get(int(c), 0) + int(np.count_nonzero(sel & (gc == c)))
        for t in IOU_THRESHOLDS:
            tk = float(round(float(t), 2))
            c, records = _evaluate_threshold(pb, ps, pc, neutral, gb, gc, np.ones(len(gb), dtype=bool), iou, float(t))
            pooled_global[tk].extend(records)
            if tk == 0.50:
                for k in sum_counts:
                    sum_counts[k] += c[k]
            for b in bins:
                sel = _in_bin(radius, b)
                c, records = _evaluate_threshold(pb, ps, pc, neutral, gb, gc, sel, iou, float(t))
                pooled[b][tk].extend(records)
                for k in bin_counts[b][tk]:
                    bin_counts[b][tk][k] += c.get(k, 0)
                bin_counts[b][tk]["gt"] += int(np.count_nonzero(sel))
    out: dict[str, Any] = {"counts": sum_counts, "per_threshold": {}}
    overall_aps = [_ap_from_records(pooled_global[float(round(float(t), 2))], gt_global) for t in IOU_THRESHOLDS]
    out["AP50"], out["AP75"], out["mAP50-95"] = overall_aps[0], overall_aps[5], float(np.mean(overall_aps))
    out["AP50-95"] = out["mAP50-95"]
    for b in bins:
        aps = [_ap_from_records(pooled[b][float(round(float(t), 2))], gt_counts[b]) for t in IOU_THRESHOLDS]
        out[f"AP-{b}-SBR"], out[f"AP-{b}"] = float(np.mean(aps)), float(np.mean(aps))
        out[f"AP50-{b}-SBR"], out[f"AP75-{b}-SBR"] = float(aps[0]), float(aps[5])
        out["per_threshold"][b] = bin_counts[b]
    tiny_c = bin_counts["tiny"][0.50]
    out["tiny_recall"] = float(tiny_c["tp"] / tiny_c["gt"]) if tiny_c["gt"] else 0.0
    return out


# Backwards-compatible descriptive alias.
evaluate = evaluate_sbr
