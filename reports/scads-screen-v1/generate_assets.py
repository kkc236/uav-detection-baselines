"""Generate reproducible analysis assets from the immutable SCADS Gate report."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPORT_DIR = Path(__file__).resolve().parent
CORE_METRICS = (
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1", "f1"),
    ("AP50", "ap50"),
    ("AP75", "ap75"),
    ("mAP50-95", "map"),
    ("AP-tiny", "ap_tiny"),
    ("AP-small", "ap_small"),
)


def f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 2.0 * precision * recall / denominator if denominator else 0.0


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text_lf(path: Path, content: str, encoding: str) -> None:
    with path.open("w", encoding=encoding, newline="\n") as stream:
        stream.write(content)


def core_rows(report: dict) -> list[dict]:
    exact = report["exact_metrics"]
    values = {variant: dict(exact[variant]) for variant in ("fdr", "scads")}
    for variant in values:
        values[variant]["f1"] = f1(values[variant]["precision"], values[variant]["recall"])
    rows = []
    for label, key in CORE_METRICS:
        baseline = float(values["fdr"][key])
        scads = float(values["scads"][key])
        delta = scads - baseline
        rows.append(
            {
                "metric": label,
                "fdr": f"{baseline:.12f}",
                "scads": f"{scads:.12f}",
                "absolute_delta": f"{delta:.12f}",
                "relative_delta_pct": f"{100.0 * delta / baseline:.6f}" if baseline else "",
                "fdr_pct": f"{100.0 * baseline:.6f}",
                "scads_pct": f"{100.0 * scads:.6f}",
                "delta_percentage_points": f"{100.0 * delta:.6f}",
            }
        )
    return rows


def training_rows(report: dict) -> list[dict]:
    training = report["training_metrics"]
    rows = []
    for window in ("final", "tail3"):
        for metric in ("precision", "recall", "map50", "map75", "map"):
            baseline = float(training[window]["fdr"][metric])
            scads = float(training[window]["scads"][metric])
            rows.append(
                {
                    "window": window,
                    "epochs": ",".join(str(value) for value in training[window]["epochs"]),
                    "metric": metric,
                    "fdr": f"{baseline:.12f}",
                    "scads": f"{scads:.12f}",
                    "absolute_delta": f"{scads - baseline:.12f}",
                }
            )
    for variant in ("fdr", "scads"):
        rows.append(
            {
                "window": "best",
                "epochs": str(training["best"][variant]["epoch"]),
                "metric": f"map:{variant}",
                "fdr": f"{float(training['best'][variant]['map']):.12f}" if variant == "fdr" else "",
                "scads": f"{float(training['best'][variant]['map']):.12f}" if variant == "scads" else "",
                "absolute_delta": "",
            }
        )
    return rows


def efficiency_rows(report: dict) -> list[dict]:
    efficiency = report["efficiency"]
    metrics = (
        "parameters",
        "checkpoint_bytes",
        "latency_ms_median",
        "latency_ms_p95",
        "fps_from_median",
    )
    rows = []
    for metric in metrics:
        baseline = float(efficiency["fdr"][metric])
        scads = float(efficiency["scads"][metric])
        delta = scads - baseline
        rows.append(
            {
                "metric": metric,
                "fdr": f"{baseline:.9f}",
                "scads": f"{scads:.9f}",
                "absolute_delta": f"{delta:.9f}",
                "relative_delta_pct": f"{100.0 * delta / baseline:.6f}" if baseline else "",
            }
        )
    return rows


def mechanism_rows(report: dict) -> list[dict]:
    representation = report["representation"]
    rows = []
    for scale in ("tiny", "small", "other", "overall"):
        for variant, section in (
            ("fdr_fixed_base", representation["fdr_fixed_base"]),
            ("scads_adaptive", representation["scads_adaptive"]),
            ("scads_oracle", representation["scads_oracle"]),
        ):
            item = section["overall"] if scale == "overall" else section["by_scale"][scale]
            rows.append(
                {
                    "section": variant,
                    "scale": scale,
                    "metric": "edge_saturation_rate",
                    "value": f"{float(item['edge_saturation_rate']):.12f}",
                }
            )
            rows.append(
                {
                    "section": variant,
                    "scale": scale,
                    "metric": "object_saturation_rate",
                    "value": f"{float(item['object_saturation_rate']):.12f}",
                }
            )
    route = representation["route"]
    for metric in (
        "accuracy",
        "balanced_accuracy",
        "entropy_mean",
        "effective_up_scale_range",
        "wide_overflow_rate",
    ):
        rows.append(
            {
                "section": "route",
                "scale": "overall",
                "metric": metric,
                "value": f"{float(route[metric]):.12f}",
            }
        )
    rows.append(
        {
            "section": "gate",
            "scale": "tiny",
            "metric": "tiny_saturation_relative_reduction",
            "value": f"{float(representation['tiny_saturation_relative_reduction']):.12f}",
        }
    )
    return rows


def experiment_rows(report: dict) -> list[dict]:
    evaluation = report["evaluation"]
    values = (
        ("dataset", "VisDrone"),
        ("stage", "screen"),
        ("epochs_per_arm", 30),
        ("training_subset_images", 647),
        ("validation_images", 548),
        ("validation_instances", 38759),
        ("gpu", "NVIDIA GeForce RTX 4090"),
        ("image_size", evaluation["imgsz"]),
        ("evaluation_batch", evaluation["batch"]),
        ("confidence_threshold", evaluation["conf"]),
        ("max_detections", evaluation["max_det"]),
        ("nms", evaluation["nms"]),
        ("seed", evaluation["seed"]),
        ("published_checkpoints", 60),
    )
    return [{"field": field, "value": value} for field, value in values]


def plot_core_metrics(rows: list[dict], destination: Path) -> None:
    labels = [row["metric"] for row in rows]
    fdr_values = np.array([float(row["fdr_pct"]) for row in rows])
    scads_values = np.array([float(row["scads_pct"]) for row in rows])
    panels = ((slice(0, 3), "Detection balance", "Percent (%)"), (slice(3, None), "Average precision", "Percent (%)"))
    colors = ("#5B6573", "#0F8B8D")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#C8CDD3",
            "axes.labelcolor": "#30353B",
            "xtick.color": "#30353B",
            "ytick.color": "#30353B",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2), dpi=220, gridspec_kw={"width_ratios": [0.85, 1.35]})
    fig.patch.set_facecolor("white")
    width = 0.34
    for axis, (selection, title, ylabel) in zip(axes, panels, strict=True):
        panel_labels = labels[selection]
        left = fdr_values[selection]
        right = scads_values[selection]
        x = np.arange(len(panel_labels))
        first = axis.bar(x - width / 2, left, width, color=colors[0], label="FDR baseline")
        second = axis.bar(x + width / 2, right, width, color=colors[1], label="FDR + SCADS")
        axis.set_title(title, loc="left", pad=12)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, panel_labels)
        axis.grid(axis="y", color="#E3E6E9", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        maximum = max(float(right.max()), float(left.max()))
        axis.set_ylim(0, maximum * 1.22)
        axis.bar_label(first, fmt="%.3f", padding=3, fontsize=8, color="#3F464E")
        axis.bar_label(second, fmt="%.3f", padding=3, fontsize=8, color="#176A6B")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", bbox_to_anchor=(0.965, 0.965), frameon=False, ncols=2)
    fig.suptitle("VisDrone Screen30: FDR baseline vs. FDR + SCADS", x=0.06, y=0.975, ha="left", fontsize=15, fontweight="semibold")
    fig.text(0.06, 0.025, "Epoch 30 | Input 640 | Validation: 548 images, 38,759 instances | Values shown in percent", color="#5B6573", fontsize=9)
    fig.subplots_adjust(left=0.065, right=0.97, top=0.86, bottom=0.14, wspace=0.24)
    fig.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_readme(report: dict, rows: list[dict], destination: Path) -> None:
    by_metric = {row["metric"]: row for row in rows}
    gate = report["gate"]
    efficiency = report["efficiency"]
    gate_hash = hashlib.sha256((REPORT_DIR / "gate-report.json").read_bytes()).hexdigest().upper()
    content = f"""# SCADS/FDR VisDrone Screen30 实验报告

本目录包含 FDR 与 SCADS 严格配对实验的不可变 Gate 报告，以及由该报告生成的可复现分析数据和图表。

## 核心结果

SCADS 在统一独立评估中的全部检测指标均有提升。主要增益来自 Recall（+{float(by_metric['Recall']['delta_percentage_points']):.3f} 个百分点）、AP75（+{float(by_metric['AP75']['delta_percentage_points']):.3f} 个百分点）和 mAP50-95（+{float(by_metric['mAP50-95']['delta_percentage_points']):.3f} 个百分点）。F1 根据报告中的 Precision 和 Recall，按 `2PR/(P+R)` 计算。

| 指标 | FDR | FDR + SCADS | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
"""
    for label, _key in CORE_METRICS:
        row = by_metric[label]
        content += f"| {label} | {float(row['fdr_pct']):.3f}% | {float(row['scads_pct']):.3f}% | {float(row['delta_percentage_points']):+.3f} pp | {float(row['relative_delta_pct']):+.2f}% |\n"
    content += f"""

![FDR 与 SCADS 核心指标对比](core-metrics-comparison.png)

## Gate 结论

- Gate 通过：`{str(gate['passed']).lower()}`
- Formal100 可执行：`{str(report['formal100_eligible']).lower()}`
- 通过门检项：`{sum(bool(value) for value in gate['checks'].values())}/{len(gate['checks'])}`
- 未通过项：`tiny_saturation_reduction_ge_50pct`
- 实测 tiny 饱和率相对下降：`{100.0 * float(report['representation']['tiny_saturation_relative_reduction']):.2f}%`

因此实验停止在 Screen30，未启动 Formal100。

## 效率代价

| 指标 | FDR | FDR + SCADS | 变化 |
|---|---:|---:|---:|
| 参数量 | {int(efficiency['fdr']['parameters']):,} | {int(efficiency['scads']['parameters']):,} | {int(efficiency['delta']['parameters']):+,} |
| 中位延迟 | {float(efficiency['fdr']['latency_ms_median']):.3f} ms | {float(efficiency['scads']['latency_ms_median']):.3f} ms | {float(efficiency['delta']['latency_ms_median']):+.3f} ms |
| 推理速度 | {float(efficiency['fdr']['fps_from_median']):.3f} FPS | {float(efficiency['scads']['fps_from_median']):.3f} FPS | {float(efficiency['scads']['fps_from_median']) - float(efficiency['fdr']['fps_from_median']):+.3f} FPS |

## 文件说明

- `gate-report.json`：完整且不可变的评估器输出。
- `core-metrics.csv`：Precision、Recall、F1、AP50、AP75、mAP50-95、AP-tiny 和 AP-small。
- `training-windows.csv`：最后一轮和最后三轮平均训练指标。
- `efficiency-metrics.csv`：参数量、检查点大小、延迟和 FPS。
- `mechanism-metrics.csv`：饱和率、路由、oracle 和重建相关机制指标。
- `experiment-context.csv`：数据集、硬件和评估设置。
- `core-metrics-comparison.png`：核心指标配对对比图。
- `generate_assets.py`：CSV、README 和图表的确定性生成脚本。

## 复现

```bash
python reports/scads-screen-v1/generate_assets.py reports/scads-screen-v1/gate-report.json
```

Gate JSON SHA-256：`{gate_hash}`。
"""
    write_text_lf(destination, content, encoding="utf-8")


def write_checksums(destination: Path) -> None:
    names = (
        "gate-report.json",
        "core-metrics.csv",
        "training-windows.csv",
        "efficiency-metrics.csv",
        "mechanism-metrics.csv",
        "experiment-context.csv",
        "core-metrics-comparison.png",
        "README.md",
        "generate_assets.py",
    )
    lines = []
    for name in names:
        digest = hashlib.sha256((REPORT_DIR / name).read_bytes()).hexdigest().upper()
        lines.append(f"{digest}  {name}")
    write_text_lf(destination, "\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    source = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPORT_DIR / "gate-report.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    target = REPORT_DIR / "gate-report.json"
    if source != target.resolve():
        shutil.copy2(source, target)

    core = core_rows(report)
    write_csv(
        REPORT_DIR / "core-metrics.csv",
        ["metric", "fdr", "scads", "absolute_delta", "relative_delta_pct", "fdr_pct", "scads_pct", "delta_percentage_points"],
        core,
    )
    write_csv(
        REPORT_DIR / "training-windows.csv",
        ["window", "epochs", "metric", "fdr", "scads", "absolute_delta"],
        training_rows(report),
    )
    write_csv(
        REPORT_DIR / "efficiency-metrics.csv",
        ["metric", "fdr", "scads", "absolute_delta", "relative_delta_pct"],
        efficiency_rows(report),
    )
    write_csv(
        REPORT_DIR / "mechanism-metrics.csv",
        ["section", "scale", "metric", "value"],
        mechanism_rows(report),
    )
    write_csv(
        REPORT_DIR / "experiment-context.csv",
        ["field", "value"],
        experiment_rows(report),
    )
    plot_core_metrics(core, REPORT_DIR / "core-metrics-comparison.png")
    write_readme(report, core, REPORT_DIR / "README.md")
    write_checksums(REPORT_DIR / "SHA256SUMS.txt")


if __name__ == "__main__":
    main()
