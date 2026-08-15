"""Independently evaluate one exact-final BPDD Formal100 arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bpdd_formal_evaluation import (  # noqa: E402
    CachedScaleRTDETRValidator,
    SCALE_NAMES,
    load_exact_final_checkpoint,
    summarize_native_box_metrics,
    summarize_scale_metrics,
    write_create_only_json,
)
from src.bpdd_protocol import BPDD_PROTOCOL  # noqa: E402
from src.lpr_protocol import CATEGORY_NAMES, dataset_signature  # noqa: E402


EVALUATION_PROTOCOL = {
    "imgsz": 640,
    "batch": 8,
    "workers": 8,
    "conf": 0.001,
    "max_det": 300,
    "nms": False,
}


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON authority is not an object: {path}")
    return payload


def _read_publication(path: str | Path) -> dict[str, Any]:
    publication_path = Path(path)
    if publication_path.suffix.lower() != ".jsonl":
        return _read_json(publication_path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        publication_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"publication ledger line {line_number} is not an object")
        rows.append(payload)
    try:
        epochs = [int(row["completed_epoch"]) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("publication ledger has invalid epochs") from error
    if epochs != list(range(1, 101)):
        raise ValueError("publication ledger must contain continuous epochs 1-100")
    return rows[-1]


def _validate_run_manifest(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "bpdd-run.json")
    identity = manifest.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Formal100 run manifest is missing run identity")
    if (
        manifest.get("format_version") != 1
        or identity.get("stage") != "formal"
        or identity.get("seed") != 0
        or identity.get("variant") not in {"fdr", "fdr_bpdd"}
        or not isinstance(identity.get("run_id"), str)
        or not manifest.get("data")
    ):
        raise ValueError("Formal100 run identity is invalid")
    return manifest


def _validate_publication(
    publication: Mapping[str, Any],
    *,
    run_identity: Mapping[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    remote_checkpoint = publication.get("checkpoint")
    if not isinstance(remote_checkpoint, Mapping):
        raise ValueError("publication checkpoint metadata is missing")
    if publication.get("completed_epoch") != 100:
        raise ValueError("publication epoch must be completed_epoch=100")
    if publication.get("verified") is not True:
        raise ValueError("publication asset is not verified")
    for field in ("run_id", "variant", "stage"):
        if publication.get(field) != run_identity.get(field):
            raise ValueError(f"publication identity mismatch for {field}")
    expected_sha = str(remote_checkpoint.get("sha256", "")).upper()
    if len(expected_sha) != 64:
        raise ValueError("publication checkpoint SHA256 is invalid")
    if int(remote_checkpoint.get("bytes", -1)) != checkpoint.stat().st_size:
        raise ValueError("publication checkpoint byte count mismatch")
    asset_name = remote_checkpoint.get("asset_name")
    release_url = publication.get("release_url")
    if not asset_name or not release_url:
        raise ValueError("publication remote asset binding is incomplete")
    return {
        "expected_sha256": expected_sha,
        "remote_published": True,
        "remote_asset": f"{str(release_url).rstrip('/')}#{asset_name}",
        "remote_asset_id": int(remote_checkpoint.get("asset_id", -1)),
        "remote_asset_name": str(asset_name),
        "remote_asset_bytes": int(remote_checkpoint["bytes"]),
    }


def run_official_validation(model: Any, *, data: Path, save_dir: Path) -> dict[str, Any]:
    """Run one native val pass and derive all metrics from that same pass."""

    validator = CachedScaleRTDETRValidator(
        save_dir=save_dir,
        args={
            "model": str(data),
            "data": str(data),
            "task": "detect",
            "mode": "val",
            "split": "val",
            **EVALUATION_PROTOCOL,
            "device": "0",
            "cache": False,
            "half": False,
            "rect": False,
            "plots": False,
            "save_json": False,
            "save_txt": False,
            "verbose": False,
        },
    )
    validator(model=model)
    processed = len(validator.scale_targets)
    if processed != 548 or len(validator.scale_predictions) != processed:
        raise RuntimeError(
            f"Formal100 official val processed {processed} images instead of 548"
        )
    native = summarize_native_box_metrics(validator.metrics.box, CATEGORY_NAMES)
    scales = summarize_scale_metrics(
        validator.scale_predictions,
        validator.scale_targets,
        class_count=len(CATEGORY_NAMES),
    )
    return {
        **native,
        **scales,
        "processed_images": processed,
        "prediction_passes": 1,
    }


def evaluate_formal_checkpoint(
    *,
    run_dir: str | Path,
    checkpoint: str | Path,
    publication_manifest: str | Path,
    dataset_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    output_path = Path(output).resolve()
    manifest = _validate_run_manifest(run)
    identity = dict(manifest["run_identity"])
    publication = _read_publication(publication_manifest)
    remote = _validate_publication(
        publication,
        run_identity=identity,
        checkpoint=checkpoint_path,
    )
    dataset = dataset_signature(Path(dataset_root).resolve())
    expected_dataset_sha = str(BPDD_PROTOCOL["dataset"]["sha256"]).upper()
    if str(dataset.get("sha256", "")).upper() != expected_dataset_sha:
        raise ValueError("Formal100 dataset SHA256 does not match BPDD authority")
    loaded = load_exact_final_checkpoint(
        checkpoint_path,
        expected_sha256=remote["expected_sha256"],
    )
    if loaded.metadata["kind"] != "exact-final-ema":
        raise ValueError("Formal100 independent evaluation requires checkpoint EMA")
    data_path = Path(str(manifest["data"]))
    validation = run_official_validation(
        loaded.model,
        data=data_path,
        save_dir=output_path.parent / "validator",
    )
    report = {
        "format_version": 1,
        "evaluation_identity": {
            **identity,
            "data": str(manifest["data"]),
            "dataset_sha256": str(dataset["sha256"]).upper(),
        },
        "checkpoint": {
            **loaded.metadata,
            **{key: value for key, value in remote.items() if key != "expected_sha256"},
        },
        "evaluation_protocol": dict(EVALUATION_PROTOCOL),
        **validation,
    }
    write_create_only_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--publication-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_formal_checkpoint(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        publication_manifest=args.publication_manifest,
        dataset_root=args.dataset_root,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
