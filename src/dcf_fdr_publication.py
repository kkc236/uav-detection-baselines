"""Validate, compare, and stage Clean FDR versus DCF-FDR Formal100 evidence."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


EXPECTED_EPOCHS = 100
EXPECTED_SOURCE_COMMIT = "ec4e2a463db7a53f7c4c9c4bc9edabdf5c39f40b"
EXPECTED_INITIAL_STATE_SHA256 = (
    "51aab2eb3fb7d123501c69c7b8dc90ff3ea0b9344a108edeef2c7d6dcdbb742d"
)
METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)
_FAILURE_PATTERN = re.compile(
    r"traceback|cuda out of memory|no space left|non[- ]finite|nan loss",
    re.IGNORECASE,
)


class PublicationGateError(RuntimeError):
    """Raised before remote mutation when Formal100 evidence is invalid."""


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    output_root: Path
    run_name: str

    def __post_init__(self) -> None:
        if self.arm not in {"clean", "dcf"}:
            raise ValueError(f"unknown arm: {self.arm}")

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.run_name

    @property
    def authority_path(self) -> Path:
        return Path(self.output_root) / "authority" / f"{self.run_name}.json"

    @property
    def train_log(self) -> Path:
        return Path(self.output_root) / f"train-{self.arm}.log"


@dataclass(frozen=True)
class ValidatedArm:
    spec: ArmSpec
    rows: tuple[Mapping[str, float | int], ...]
    artifacts: Mapping[str, Path]
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class StagedEvidence:
    root: Path
    manifest_path: Path
    bundle_path: Path
    comparison: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        staging = Path(stream.name)
    os.replace(staging, path)


def _read_results(path: Path) -> tuple[Mapping[str, float | int], ...]:
    if path.is_symlink() or not path.is_file():
        raise PublicationGateError(f"missing results.csv: {path}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            raw_rows = list(csv.DictReader(stream))
        rows: list[Mapping[str, float | int]] = []
        for raw in raw_rows:
            row: dict[str, float | int] = {"epoch": int(float(str(raw["epoch"])))}
            for key in METRIC_KEYS:
                row[key] = float(str(raw[key]))
            rows.append(row)
    except (OSError, KeyError, TypeError, ValueError, csv.Error) as error:
        raise PublicationGateError(f"unreadable results.csv: {path}: {error}") from error
    if len(rows) != EXPECTED_EPOCHS:
        raise PublicationGateError(
            f"results.csv must contain exactly {EXPECTED_EPOCHS} epochs: "
            f"{path} has {len(rows)}"
        )
    epochs = [int(row["epoch"]) for row in rows]
    valid_sequences = (list(range(EXPECTED_EPOCHS)), list(range(1, EXPECTED_EPOCHS + 1)))
    if epochs not in valid_sequences:
        raise PublicationGateError(f"results.csv epochs are not continuous: {path}")
    return tuple(rows)


def _read_authority(spec: ArmSpec) -> Mapping[str, Any]:
    path = spec.authority_path
    if path.is_symlink() or not path.is_file():
        raise PublicationGateError(f"missing authority JSON: {path}")
    try:
        authority = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicationGateError(f"unreadable authority JSON: {path}: {error}") from error
    expected_method = "clean_fdr" if spec.arm == "clean" else "dcf_fdr"
    if authority.get("method") != expected_method:
        raise PublicationGateError(f"wrong method in authority JSON: {path}")
    source_commit = str(authority.get("source", {}).get("git_commit", "")).lower()
    if source_commit != EXPECTED_SOURCE_COMMIT:
        raise PublicationGateError(f"wrong source commit in authority JSON: {path}")
    initial_sha = str(authority.get("initial_state", {}).get("sha256", "")).lower()
    if initial_sha != EXPECTED_INITIAL_STATE_SHA256:
        raise PublicationGateError(f"wrong initial-state SHA-256 in authority JSON: {path}")
    settings = authority.get("settings", {})
    if (
        int(settings.get("epochs", -1)) != EXPECTED_EPOCHS
        or int(settings.get("seed", -1)) != 0
        or settings.get("name") != spec.run_name
    ):
        raise PublicationGateError(f"wrong Formal100 settings in authority JSON: {path}")
    return authority


def _validate_checkpoint(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise PublicationGateError(f"missing checkpoint: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or not any(
            key in payload and payload[key] is not None for key in ("model", "ema")
        ):
            raise ValueError("checkpoint has neither model nor EMA state")
    except Exception as error:
        raise PublicationGateError(f"unreadable checkpoint: {path}: {error}") from error
    finally:
        if "payload" in locals():
            del payload


def _validate_log(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise PublicationGateError(f"missing training log: {path}")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise PublicationGateError(f"unreadable training log: {path}: {error}") from error
    match = _FAILURE_PATTERN.search(text)
    if match:
        raise PublicationGateError(
            f"training log contains terminal failure marker {match.group(0)!r}: {path}"
        )


def _row_payload(row: Mapping[str, float | int]) -> dict[str, float | int]:
    return {"epoch": int(row["epoch"]), **{key: float(row[key]) for key in METRIC_KEYS}}


def _summary(arm: str, rows: Sequence[Mapping[str, float | int]]) -> dict[str, Any]:
    best = max(rows, key=lambda row: float(row[METRIC_KEYS[-1]]))
    peaks = {
        key: _row_payload(max(rows, key=lambda row, metric=key: float(row[metric])))
        for key in METRIC_KEYS
    }
    return {
        "arm": arm,
        "completed_epochs": len(rows),
        "best_by_map50_95": _row_payload(best),
        "latest": _row_payload(rows[-1]),
        "metric_peaks": peaks,
    }


def validate_arm(spec: ArmSpec) -> ValidatedArm:
    run_dir = spec.run_dir
    artifacts = {
        "args.yaml": run_dir / "args.yaml",
        "authority.json": spec.authority_path,
        "best.pt": run_dir / "weights" / "best.pt",
        "last.pt": run_dir / "weights" / "last.pt",
        "results.csv": run_dir / "results.csv",
        "train.log": spec.train_log,
    }
    for label in ("args.yaml",):
        path = artifacts[label]
        if path.is_symlink() or not path.is_file():
            raise PublicationGateError(f"missing {label}: {path}")
    rows = _read_results(artifacts["results.csv"])
    _read_authority(spec)
    _validate_log(artifacts["train.log"])
    _validate_checkpoint(artifacts["best.pt"])
    _validate_checkpoint(artifacts["last.pt"])
    return ValidatedArm(
        spec=spec,
        rows=rows,
        artifacts=artifacts,
        summary=_summary(spec.arm, rows),
    )


def build_comparison(
    clean: ValidatedArm, dcf: ValidatedArm
) -> tuple[dict[str, Any], list[dict[str, float | int]]]:
    if clean.spec.arm != "clean" or dcf.spec.arm != "dcf":
        raise ValueError("comparison requires clean then dcf")
    clean_epochs = [int(row["epoch"]) for row in clean.rows]
    dcf_epochs = [int(row["epoch"]) for row in dcf.rows]
    if clean_epochs != dcf_epochs:
        raise PublicationGateError("Clean and DCF epoch sequences do not align")
    aligned: list[dict[str, float | int]] = []
    for clean_row, dcf_row in zip(clean.rows, dcf.rows):
        row: dict[str, float | int] = {"epoch": int(clean_row["epoch"])}
        for key in METRIC_KEYS:
            clean_value = float(clean_row[key])
            dcf_value = float(dcf_row[key])
            row[f"clean/{key}"] = clean_value
            row[f"dcf/{key}"] = dcf_value
            row[f"delta/{key}"] = dcf_value - clean_value
        aligned.append(row)
    clean_best = clean.summary["best_by_map50_95"]
    dcf_best = dcf.summary["best_by_map50_95"]
    best_delta = {
        key: float(dcf_best[key]) - float(clean_best[key]) for key in METRIC_KEYS
    }
    report = {
        "format_version": 1,
        "primary_metric": METRIC_KEYS[-1],
        "decision": (
            "passed_nonnegative"
            if float(dcf_best[METRIC_KEYS[-1]]) >= float(clean_best[METRIC_KEYS[-1]])
            else "failed_negative"
        ),
        "clean": clean.summary,
        "dcf": dcf.summary,
        "best_delta": best_delta,
    }
    return report, aligned


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _gzip_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as raw, destination.open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
            shutil.copyfileobj(raw, compressed)


def _normalized_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _build_bundle(root: Path, destination: Path) -> None:
    members = sorted(
        path for path in root.rglob("*") if path.is_file() and path != destination
    )
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in members:
                    archive.add(
                        path,
                        arcname=path.relative_to(root).as_posix(),
                        recursive=False,
                        filter=_normalized_tarinfo,
                    )


def _results_markdown(report: Mapping[str, Any]) -> str:
    clean = report["clean"]["best_by_map50_95"]
    dcf = report["dcf"]["best_by_map50_95"]
    lines = [
        "# Clean FDR vs DCF-FDR Formal100",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Primary metric: `{report['primary_metric']}`",
        "",
        "| Arm | Epoch | Precision | Recall | AP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Clean FDR | {clean['epoch']} | {clean[METRIC_KEYS[0]]:.5f} | "
            f"{clean[METRIC_KEYS[1]]:.5f} | {clean[METRIC_KEYS[2]]:.5f} | "
            f"{clean[METRIC_KEYS[3]]:.5f} |"
        ),
        (
            f"| DCF-FDR | {dcf['epoch']} | {dcf[METRIC_KEYS[0]]:.5f} | "
            f"{dcf[METRIC_KEYS[1]]:.5f} | {dcf[METRIC_KEYS[2]]:.5f} | "
            f"{dcf[METRIC_KEYS[3]]:.5f} |"
        ),
        "",
        "All values are read from the untouched Formal100 CSV files. The decision",
        "uses unrounded best mAP50-95 values; unfavorable results remain published.",
        "",
    ]
    return "\n".join(lines)


def stage_evidence(
    clean: ValidatedArm, dcf: ValidatedArm, root: Path
) -> StagedEvidence:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    comparison, aligned = build_comparison(clean, dcf)
    for arm in (clean, dcf):
        destination = root / arm.spec.arm
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(arm.artifacts["results.csv"], destination / "results.csv")
        shutil.copy2(arm.artifacts["args.yaml"], destination / "args.yaml")
        shutil.copy2(arm.artifacts["authority.json"], destination / "authority.json")
        _gzip_copy(arm.artifacts["train.log"], destination / "train.log.gz")
        write_json_atomic(destination / "summary.json", arm.summary)
    _write_csv(root / "aligned-epochs.csv", aligned)
    write_json_atomic(root / "comparison.json", comparison)
    (root / "RESULTS.md").write_text(_results_markdown(comparison), encoding="utf-8")

    manifest_path = root / "artifact-manifest.json"
    bundle_path = root / "lightweight-evidence.tar.gz"
    artifact_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path not in {manifest_path, bundle_path}
    )
    manifest = {
        "format_version": 1,
        "experiment": "clean_dcf_fdr_formal100_seed0",
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256,
        "decision": comparison["decision"],
        "artifacts": {
            path.relative_to(root).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        },
        "release_assets": {},
    }
    write_json_atomic(manifest_path, manifest)
    _build_bundle(root, bundle_path)
    return StagedEvidence(
        root=root,
        manifest_path=manifest_path,
        bundle_path=bundle_path,
        comparison=comparison,
    )

