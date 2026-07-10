from __future__ import annotations

import csv
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable


METRIC_COLUMNS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]


def read_latest_result(csv_path: Path) -> dict[str, float] | None:
    if not csv_path.exists():
        return None

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return None

    latest: dict[str, float] = {}
    for key, value in rows[-1].items():
        if key is None or value is None or value == "":
            continue
        try:
            latest[key] = float(value)
        except ValueError:
            continue
    return latest


def estimate_eta_seconds(latest: dict[str, float] | None, *, total_epochs: int) -> float | None:
    if not latest:
        return None
    epoch = latest.get("epoch", 0.0)
    elapsed = latest.get("time", 0.0)
    if epoch <= 0 or elapsed <= 0 or epoch >= total_epochs:
        return 0.0 if epoch >= total_epochs else None
    return (elapsed / epoch) * (total_epochs - epoch)


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def get_gpu_status() -> str:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "GPU: unavailable"

    if not output:
        return "GPU: unavailable"

    first = output.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 4:
        return f"GPU: {first}"
    name, used, total, util = parts[:4]
    return f"GPU: {name} | {used}/{total} MiB | util {util}%"


def get_training_processes() -> list[str]:
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'train_' } | "
        "ForEach-Object { \"$($_.ProcessId) $($_.CommandLine)\" }"
    )
    try:
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", command],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return []

    return [line.strip() for line in output.splitlines() if line.strip()]


def _fmt(value: float | None, digits: int = 5) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def render_status(run_dir: Path, *, total_epochs: int) -> str:
    csv_path = run_dir / "results.csv"
    latest = read_latest_result(csv_path)
    eta = estimate_eta_seconds(latest, total_epochs=total_epochs)
    processes = get_training_processes()

    lines = [
        "Training Monitor",
        "=" * 72,
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Run:  {run_dir}",
        get_gpu_status(),
        f"Process: {'RUNNING' if processes else 'not found'}",
    ]
    if processes:
        lines.extend([f"  {line[:150]}" for line in processes[:3]])

    lines.append("-" * 72)
    if latest is None:
        lines.append("results.csv not ready yet.")
        return "\n".join(lines)

    epoch = int(latest.get("epoch", 0))
    elapsed = latest.get("time", 0.0)
    lines.extend(
        [
            f"Epoch: {epoch}/{total_epochs}",
            f"Elapsed: {format_seconds(elapsed)}",
            f"ETA: {format_seconds(eta)}",
            "",
            "Metrics",
            f"  precision: {_fmt(latest.get('metrics/precision(B)'))}",
            f"  recall:    {_fmt(latest.get('metrics/recall(B)'))}",
            f"  mAP50:     {_fmt(latest.get('metrics/mAP50(B)'))}",
            f"  mAP50-95:  {_fmt(latest.get('metrics/mAP50-95(B)'))}",
            "",
            "Loss",
            f"  train box/giou: {_fmt(latest.get('train/box_loss', latest.get('train/giou_loss')))}",
            f"  train cls:      {_fmt(latest.get('train/cls_loss'))}",
            f"  train dfl/l1:   {_fmt(latest.get('train/dfl_loss', latest.get('train/l1_loss')))}",
            f"  val box/giou:   {_fmt(latest.get('val/box_loss', latest.get('val/giou_loss')))}",
            f"  val cls:        {_fmt(latest.get('val/cls_loss'))}",
            f"  val dfl/l1:     {_fmt(latest.get('val/dfl_loss', latest.get('val/l1_loss')))}",
        ]
    )
    return "\n".join(lines)


def monitor(run_dir: Path, *, total_epochs: int, interval: int) -> None:
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print(render_status(run_dir, total_epochs=total_epochs), flush=True)
        try:
            import time

            time.sleep(interval)
        except KeyboardInterrupt:
            break
