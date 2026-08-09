#!/usr/bin/env python3
"""Generate the non-checkpoint GLGM Screen30 final evidence package."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
METRICS = ("precision", "recall", "f1", "ap50", "ap75", "map50_95")
LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "ap50": "AP50",
    "ap75": "AP75",
    "map50_95": "mAP50-95",
}
CONTROL_COLOR = "#0072B2"
GLGM_COLOR = "#D55E00"
POSITIVE_COLOR = "#009E73"
NEGATIVE_COLOR = "#CC3311"
GRID_COLOR = "#D0D0D0"


def load_json(name: str) -> dict:
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def read_training_csv(name: str) -> list[dict[str, float]]:
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in raw.items()})
    return rows


def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> None:
    with (ROOT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assert_finite(value, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value at {path}: {value}")


def comparison_rows(comparison: dict) -> list[dict]:
    rows = []
    for key in METRICS:
        metric = comparison["metrics"][key]
        rows.append(
            {
                "metric": key,
                "control": metric["control"],
                "glgm": metric["glgm"],
                "absolute_delta": metric["absolute_delta"],
                "percentage_point_delta": metric["percentage_point_delta"],
                "relative_percent": metric["relative_percent"],
            }
        )
    return rows


def plot_core_metrics(best: dict) -> None:
    values_control = [best["metrics"][key]["control"] for key in METRICS]
    values_glgm = [best["metrics"][key]["glgm"] for key in METRICS]
    x = list(range(len(METRICS)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    control_bars = ax.bar(
        [v - width / 2 for v in x], values_control, width, label="Control", color=CONTROL_COLOR
    )
    glgm_bars = ax.bar(
        [v + width / 2 for v in x], values_glgm, width, label="GLGM", color=GLGM_COLOR
    )
    ax.set_title("Independent evaluation of best checkpoints")
    ax.set_ylabel("Metric (%)")
    ax.set_xticks(x, [LABELS[key] for key in METRICS])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)
    for bars in (control_bars, glgm_bars):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{bar.get_height() * 100:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    ax.set_ylim(0, max(values_control + values_glgm) * 1.16)
    fig.tight_layout()
    fig.savefig(ROOT / "core-metrics-best.png", dpi=180)
    plt.close(fig)


def plot_metric_deltas(best: dict) -> None:
    deltas = [best["metrics"][key]["percentage_point_delta"] for key in METRICS]
    colors = [POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR for value in deltas]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bars = ax.bar(range(len(METRICS)), deltas, color=colors)
    ax.axhline(0, color="#555555", linewidth=0.9)
    ax.set_title("GLGM delta against Control (best checkpoints)")
    ax.set_ylabel("Absolute delta (percentage points)")
    ax.set_xticks(range(len(METRICS)), [LABELS[key] for key in METRICS])
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    span = max(abs(value) for value in deltas)
    ax.set_ylim(-span * 1.35, span * 1.35)
    for bar, value in zip(bars, deltas, strict=True):
        offset = span * 0.06
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (offset if value >= 0 else -offset),
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(ROOT / "metric-deltas-best.png", dpi=180)
    plt.close(fig)


def plot_training_curves(control: list[dict], glgm: list[dict]) -> None:
    fields = (
        ("metrics/mAP50(B)", "Training-time AP50"),
        ("metrics/mAP50-95(B)", "Training-time mAP50-95"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, (field, title) in zip(axes, fields, strict=True):
        ax.plot(
            [row["epoch"] for row in control],
            [row[field] for row in control],
            color=CONTROL_COLOR,
            linewidth=2,
            label="Control",
        )
        ax.plot(
            [row["epoch"] for row in glgm],
            [row[field] for row in glgm],
            color=GLGM_COLOR,
            linewidth=2,
            label="GLGM",
        )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric (%)")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlim(1, 30)
        ax.grid(color=GRID_COLOR, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "training-curves.png", dpi=180)
    plt.close(fig)


def plot_per_class(best: dict) -> None:
    entries = sorted(
        best["per_class_delta"], key=lambda row: row["delta"]["map50_95"]
    )
    names = [row["name"] for row in entries]
    values = [row["delta"]["map50_95"] * 100 for row in entries]
    colors = [POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR for value in values]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    bars = ax.barh(names, values, color=colors)
    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.set_title("Per-class mAP50-95 delta (best checkpoints)")
    ax.set_xlabel("GLGM - Control (percentage points)")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    span = max(abs(value) for value in values)
    ax.set_xlim(-span * 1.35, span * 1.35)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + (span * 0.035 if value >= 0 else -span * 0.035),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(ROOT / "per-class-map50-95-delta.png", dpi=180)
    plt.close(fig)


def plot_efficiency(control: dict, glgm: dict) -> None:
    fields = (
        ("parameters", "Parameters", 1e6, "million"),
        ("mean_ms", "FP16 latency", 1.0, "ms/image"),
        ("peak_allocated_vram_bytes", "Peak allocated VRAM", 1024**2, "MiB"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, (field, title, divisor, unit) in zip(axes, fields, strict=True):
        values = [control[field] / divisor, glgm[field] / divisor]
        bars = ax.bar(["Control", "GLGM"], values, color=[CONTROL_COLOR, GLGM_COLOR])
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(values) * 1.18)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(ROOT / "efficiency-comparison.png", dpi=180)
    plt.close(fig)


def build_training_rows(control: list[dict], glgm: list[dict]) -> list[dict]:
    if len(control) != len(glgm):
        raise ValueError("Paired training curves have different lengths")
    rows = []
    for left, right in zip(control, glgm, strict=True):
        if left["epoch"] != right["epoch"]:
            raise ValueError("Paired training curves have mismatched epochs")
        rows.append(
            {
                "epoch": int(left["epoch"]),
                "control_precision": left["metrics/precision(B)"],
                "glgm_precision": right["metrics/precision(B)"],
                "control_recall": left["metrics/recall(B)"],
                "glgm_recall": right["metrics/recall(B)"],
                "control_ap50": left["metrics/mAP50(B)"],
                "glgm_ap50": right["metrics/mAP50(B)"],
                "control_map50_95": left["metrics/mAP50-95(B)"],
                "glgm_map50_95": right["metrics/mAP50-95(B)"],
            }
        )
    return rows


def main() -> None:
    best = load_json("comparison-best.json")
    last = load_json("comparison-last.json")
    control_benchmark = load_json("control-best-benchmark.json")
    glgm_benchmark = load_json("glgm-best-benchmark.json")
    preflight = load_json("paired_preflight_manifest.json")
    data_audit = load_json("visdrone-audit-pre-eval.json")
    control_training = read_training_csv("control-results.csv")
    glgm_training = read_training_csv("glgm-results.csv")
    for payload in (
        best,
        last,
        control_benchmark,
        glgm_benchmark,
        control_training,
        glgm_training,
    ):
        assert_finite(payload)

    best_rows = comparison_rows(best)
    last_rows = comparison_rows(last)
    write_csv(
        "core-metrics-best.csv",
        list(best_rows[0]),
        best_rows,
    )
    write_csv(
        "core-metrics-last.csv",
        list(last_rows[0]),
        last_rows,
    )

    per_class_rows = []
    for row in best["per_class_delta"]:
        per_class_rows.append(
            {
                "class_id": row["id"],
                "class_name": row["name"],
                **{f"{key}_delta": row["delta"][key] for key in METRICS},
            }
        )
    write_csv(
        "per-class-delta-best.csv",
        list(per_class_rows[0]),
        per_class_rows,
    )

    training_rows = build_training_rows(control_training, glgm_training)
    write_csv("training-curves.csv", list(training_rows[0]), training_rows)

    efficiency_rows = []
    for metric, unit in (
        ("parameters", "count"),
        ("mean_ms", "ms/image"),
        ("p50_ms", "ms/image"),
        ("p95_ms", "ms/image"),
        ("fps", "images/s"),
        ("peak_allocated_vram_bytes", "bytes"),
    ):
        control_value = control_benchmark[metric]
        glgm_value = glgm_benchmark[metric]
        efficiency_rows.append(
            {
                "metric": metric,
                "unit": unit,
                "control": control_value,
                "glgm": glgm_value,
                "absolute_delta": glgm_value - control_value,
                "relative_percent": (glgm_value - control_value) / control_value * 100,
            }
        )
    write_csv("efficiency.csv", list(efficiency_rows[0]), efficiency_rows)

    positive_classes = [
        row["name"]
        for row in best["per_class_delta"]
        if row["delta"]["map50_95"] > 0
    ]
    negative_classes = [
        row["name"]
        for row in best["per_class_delta"]
        if row["delta"]["map50_95"] < 0
    ]
    final_report = {
        "schema": "glgm-screen30-final-report-v1",
        "created_utc": best["created_utc"],
        "authority": "/home/ubuntu/glgm/work/screen30-seed0-v5",
        "experiment_complete": True,
        "strict_pair": best["strict_pair"],
        "protocol_sha256": best["paired_training_protocol_sha256"],
        "checkpoint_policy": {
            "primary": "best",
            "robustness_check": "last",
            "independent_evaluation": True,
        },
        "dataset": {
            "name": "VisDrone",
            "train_images": data_audit["splits"]["train"]["images"],
            "val_images": data_audit["splits"]["val"]["images"],
            "val_boxes": data_audit["splits"]["val"]["boxes"],
            "train_inventory_sha256": data_audit["splits"]["train"]["inventory_sha256"],
            "val_inventory_sha256": data_audit["splits"]["val"]["inventory_sha256"],
        },
        "protocol": {
            "model": "RT-DETR-X",
            "epochs": 30,
            "imgsz": 640,
            "batch": 4,
            "seed": 0,
            "device": "NVIDIA GeForce RTX 4090 GPU 0",
            "sequential_same_device": True,
            "control_parameters": preflight["control_parameters"],
            "glgm_parameters": preflight["glgm_parameters"],
            "parameter_delta": preflight["parameter_delta"],
            "parameter_delta_percent": preflight["parameter_delta_percent"],
            "public_initialization_sha256": preflight["public_state_sha256"],
            "control_training_seconds": control_training[-1]["time"],
            "glgm_training_seconds": glgm_training[-1]["time"],
            "all_sixty_training_rows_finite": True,
        },
        "best_checkpoint_comparison": best["metrics"],
        "last_checkpoint_comparison": last["metrics"],
        "benchmark": {
            "control": control_benchmark,
            "glgm": glgm_benchmark,
            "comparison": {row["metric"]: row for row in efficiency_rows},
        },
        "per_class_best": {
            "positive_map50_95_classes": positive_classes,
            "negative_map50_95_classes": negative_classes,
            "rows": best["per_class_delta"],
        },
        "verdict": {
            "status": "not_supported",
            "statement": "This single-seed Screen30 does not support GLGM as an effective improvement over Control.",
            "evidence": [
                "Best-checkpoint mAP50-95 improved by only 0.0184 percentage points.",
                "Best-checkpoint Precision, Recall, F1, and AP50 all decreased.",
                "Parameters increased by 9.8614 percent, while mean FP16 latency increased.",
                "Only 4 of 10 classes improved in per-class mAP50-95.",
            ],
        },
        "limitations": [
            "One random seed does not establish statistical significance.",
            "AP-tiny and AP-small were not emitted by the locked evaluator and are not inferred.",
            "CUDA deterministic warn-only operations prevent a bitwise reproducibility claim.",
            "Benchmark differences should be confirmed with repeated independent runs.",
        ],
        "publication": {
            "contains_model_checkpoint": False,
            "checkpoint_paths_and_hashes_in_raw_receipts_are_metadata_only": True,
        },
    }
    assert_finite(final_report)
    (ROOT / "FINAL_REPORT.json").write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    plot_core_metrics(best)
    plot_metric_deltas(best)
    plot_training_curves(control_training, glgm_training)
    plot_per_class(best)
    plot_efficiency(control_benchmark, glgm_benchmark)

    generated = [
        "FINAL_REPORT.json",
        "core-metrics-best.csv",
        "core-metrics-last.csv",
        "efficiency.csv",
        "per-class-delta-best.csv",
        "training-curves.csv",
        "core-metrics-best.png",
        "metric-deltas-best.png",
        "training-curves.png",
        "per-class-map50-95-delta.png",
        "efficiency-comparison.png",
    ]
    for name in generated:
        if not (ROOT / name).is_file() or (ROOT / name).stat().st_size == 0:
            raise RuntimeError(f"Missing generated artifact: {name}")


if __name__ == "__main__":
    main()
