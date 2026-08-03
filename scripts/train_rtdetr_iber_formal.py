"""Train the frozen seed0 full-model IBER-BE formal100 experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sync_experiment_checkpoint import (
    prune_local_epoch_checkpoints,
    validate_token_file,
)
from src.github_checkpoint_sync import checkpoint_metadata, sha256_file
from src.iber_formal_protocol import (
    FORMAL_DESIGN_VERSION,
    FORMAL_EPOCHS,
    FORMAL_FROZEN_PROTOCOL,
    build_formal_settings,
    validate_formal_initial_state,
    validate_formal_manifest,
)
from src.iber_formal_publication import (
    FormalPublicationConfig,
    FormalPublicationIdentity,
    FormalPublicationLedger,
    pending_epoch_checkpoints,
    publish_with_retry,
)
from src.lpr_protocol import (
    EXPECTED_SOURCE_SHA256,
    current_environment,
    dataset_signature,
    environment_violations,
    source_violations,
)
from src.rtdetr_iber_formal import IBERFullTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train frozen seed0 full-model signed IBER-BE for 100 epochs."
    )
    parser.add_argument("--seed", type=int, choices=(0,), required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "iber-be-formal")
    parser.add_argument("--name", default="formal-seed0-iber-be-v1")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--repo", default="kkc236/uav-detection-baselines")
    parser.add_argument(
        "--repo-url", default="https://github.com/kkc236/uav-detection-baselines.git"
    )
    parser.add_argument("--tag", default="iber-be-v1-rtdetr-l-formal-live")
    parser.add_argument("--source-branch", default="codex/iber-be")
    parser.add_argument("--results-branch", default="iber-be-v1-results")
    parser.add_argument(
        "--results-repo",
        type=Path,
        default=Path.home() / "uav-training-results-iber-be-formal",
    )
    parser.add_argument("--asset-prefix", default="iber-be-v1.0-formal-seed0-b3")
    parser.add_argument("--retain", type=int, default=3)
    return parser


def build_settings(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    if int(args.seed) != 0:
        raise ValueError("formal IBER-BE is frozen to seed0")
    return build_formal_settings(
        manifest,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def publication_authority(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "repo": args.repo,
        "repo_url": args.repo_url,
        "source_branch": args.source_branch,
        "tag": args.tag,
        "results_branch": args.results_branch,
        "results_repo": str(args.results_repo.resolve()),
        "asset_prefix": args.asset_prefix,
        "retain": int(args.retain),
        "token_file": str(args.token_file.resolve()),
    }


def runtime_authority(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    initial_state_sha256: str,
) -> dict[str, Any]:
    return {
        "design_version": FORMAL_DESIGN_VERSION,
        "stage": "formal",
        "seed": 0,
        "epochs": FORMAL_EPOCHS,
        "protocol": FORMAL_FROZEN_PROTOCOL,
        "manifest": manifest,
        "initial_state": str(args.initial_state.resolve()),
        "initial_state_sha256": initial_state_sha256.upper(),
        "protocol_sha256": sha256_file(args.protocol_manifest).upper(),
        "source_commit": _source_commit(),
        "publication": publication_authority(args),
    }


def validate_resume_authority(
    args: argparse.Namespace,
    authority: dict[str, Any],
) -> None:
    if args.resume is None:
        return
    checkpoint = args.resume.resolve()
    if not checkpoint.is_file() or checkpoint.parent.name != "weights":
        raise ValueError("resume checkpoint must exist inside the formal run weights directory")
    runtime_path = checkpoint.parent.parent / "iber_formal_protocol.json"
    if not runtime_path.is_file():
        raise ValueError("resume authority manifest is missing")
    actual = json.loads(runtime_path.read_text(encoding="utf-8"))
    if actual != authority:
        raise ValueError("resume authority does not match the immutable formal run")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    train_args = payload.get("train_args") if isinstance(payload, Mapping) else None
    if not isinstance(train_args, Mapping):
        raise ValueError("resume checkpoint training authority is missing")
    for name, expected in authority["protocol"].items():
        if name in {"amp_scale", "query_count"}:
            continue
        if train_args.get(name) != expected:
            raise ValueError(
                f"resume checkpoint {name} mismatch: "
                f"expected={expected!r}, actual={train_args.get(name)!r}"
            )
    expected_data = authority["manifest"]["data"]["formal"]["path"]
    if train_args.get("data") != expected_data:
        raise ValueError("resume checkpoint data mismatch")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("formal diagnostic scalar must contain one value")
        value = value.detach().float().cpu().item()
    result = float(value)
    return result if math.isfinite(result) else None


def _rms(value: torch.Tensor | None) -> float | None:
    if value is None or value.numel() == 0:
        return None
    data = value.detach().float()
    if not bool(torch.isfinite(data).all()):
        raise FloatingPointError("non-finite formal IBER diagnostic tensor")
    return float(data.square().mean().sqrt().cpu())


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def write_formal_diagnostics(trainer) -> dict[str, Any] | None:
    epoch = int(trainer.epoch + 1)
    if epoch > int(trainer.args.epochs):
        return None
    model = _unwrap(trainer.model)
    output = getattr(model, "last_iber_output", None)
    losses = getattr(model, "last_iber_losses", {})
    if output is None or any(losses.get(name) is None for name in ("box_l1", "box_giou")):
        raise RuntimeError("formal IBER private activity is unavailable")
    stock = trainer.tloss.detach().float().reshape(-1).cpu().tolist()
    metrics = trainer.validator.metrics.box
    norms = getattr(trainer, "last_gradient_norms", {})
    row = {
        "epoch": epoch,
        "map": float(metrics.map),
        "map50": float(metrics.map50),
        "map75": float(metrics.map75),
        "loss_giou": stock[0] if len(stock) > 0 else None,
        "loss_class": stock[1] if len(stock) > 1 else None,
        "loss_bbox": stock[2] if len(stock) > 2 else None,
        "loss_iber_l1": _float(losses.get("box_l1")),
        "loss_iber_giou": _float(losses.get("box_giou")),
        "gate_rms": _rms(None if output is None else output.gates),
        "residual_rms": _rms(None if output is None else output.residuals),
        "f3_boundary_rms": _rms(None if output is None else output.f3_boundary_evidence),
        "rgb_boundary_rms": _rms(None if output is None else output.rgb_boundary_evidence),
        "gradient_norm": _float(norms.get("gradient_norm")),
        "iber_gradient_norm": _float(norms.get("iber_gradient_norm")),
        "cuda_peak_mib": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 2)
            if torch.cuda.is_available()
            else 0.0
        ),
    }
    path = Path(trainer.save_dir) / "iber_formal_diagnostics.jsonl"
    rows = (
        [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if path.exists()
        else []
    )
    epochs = [item.get("epoch") for item in rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise ValueError(f"formal diagnostic ledger is not contiguous: {epochs}")
    if epoch <= len(rows):
        if rows[epoch - 1] != row:
            raise ValueError(f"changed formal diagnostic replay for epoch {epoch}")
        return row
    if epoch != len(rows) + 1:
        raise ValueError(f"formal diagnostic gap before epoch {epoch}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(item, sort_keys=True, allow_nan=False) + "\n"
            for item in [*rows, row]
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return row


def _publication_config(
    trainer,
    args: argparse.Namespace,
    identity: FormalPublicationIdentity,
) -> FormalPublicationConfig:
    return FormalPublicationConfig(
        repo=args.repo,
        repo_url=args.repo_url,
        source_branch=args.source_branch,
        tag=args.tag,
        run_name=Path(trainer.save_dir).name,
        token_file=args.token_file.resolve(),
        results_repo=args.results_repo.resolve(),
        identity=identity,
        results_branch=args.results_branch,
        asset_prefix=args.asset_prefix,
        retain=int(args.retain),
    )


def publish_current_epoch(
    trainer,
    *,
    args: argparse.Namespace,
    identity: FormalPublicationIdentity,
) -> dict[str, Any]:
    checkpoint = Path(trainer.save_dir) / "weights" / f"epoch{trainer.epoch}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"exact formal epoch checkpoint is missing: {checkpoint}")
    record = publish_with_retry(
        trainer.save_dir,
        checkpoint,
        _publication_config(trainer, args, identity),
    )
    expected = int(trainer.epoch + 1)
    if record.get("verified") is not True or int(record.get("completed_epoch", -1)) != expected:
        raise RuntimeError("formal epoch was not remotely verified")
    prune_local_epoch_checkpoints(checkpoint.parent, retain=int(args.retain))
    trainer.iber_formal_publication_record = record
    return record


def recover_pending_publications(
    trainer,
    *,
    args: argparse.Namespace,
    identity: FormalPublicationIdentity,
) -> list[dict[str, Any]]:
    """Repair the latest ledger snapshot and publish saved epoch gaps on resume."""
    run_dir = Path(trainer.save_dir)
    weights = run_dir / "weights"
    config = _publication_config(trainer, args, identity)
    ledger = FormalPublicationLedger(run_dir / "publication-ledger.jsonl", identity)
    records = ledger.records()
    recovered: list[dict[str, Any]] = []

    if records:
        latest_epoch = int(records[-1]["completed_epoch"])
        latest_checkpoint = None
        for path in weights.glob("epoch*.pt"):
            if checkpoint_metadata(path).completed_epoch == latest_epoch:
                latest_checkpoint = path
                break
        if latest_checkpoint is None:
            raise FileNotFoundError(
                f"latest verified formal checkpoint epoch {latest_epoch} is missing"
            )
        recovered.append(
            publish_with_retry(run_dir, latest_checkpoint, config)
        )

    for completed_epoch, checkpoint in pending_epoch_checkpoints(weights, ledger):
        record = publish_with_retry(run_dir, checkpoint, config)
        if (
            record.get("verified") is not True
            or int(record.get("completed_epoch", -1)) != completed_epoch
        ):
            raise RuntimeError(
                f"formal recovery did not verify completed epoch {completed_epoch}"
            )
        recovered.append(record)
    if recovered:
        prune_local_epoch_checkpoints(weights, retain=int(args.retain))
    return recovered


def _validate_launch(manifest: dict[str, Any]) -> None:
    validate_formal_manifest(manifest)
    violations = environment_violations(current_environment())
    if violations:
        raise ValueError(f"formal environment mismatch: {violations}")
    drift = source_violations()
    if drift:
        raise ValueError(f"formal Ultralytics source mismatch: {drift}")
    signature = dataset_signature(Path(manifest["dataset_root"]))
    if signature.get("sha256") != manifest["dataset"]["sha256"]:
        raise ValueError("formal dataset SHA-256 mismatch")


def main() -> None:
    args = build_parser().parse_args()
    args.token_file = args.token_file.resolve()
    args.results_repo = args.results_repo.resolve()
    validate_token_file(args.token_file)
    manifest = json.loads(args.protocol_manifest.read_text(encoding="utf-8"))
    _validate_launch(manifest)
    initial_artifact = torch.load(args.initial_state, map_location="cpu", weights_only=False)
    validate_formal_initial_state(initial_artifact)
    initial_sha = sha256_file(args.initial_state)
    authority = runtime_authority(args, manifest, initial_sha)
    validate_resume_authority(args, authority)
    identity = FormalPublicationIdentity(
        source_commit=authority["source_commit"],
        protocol_sha256=sha256_file(args.protocol_manifest),
        initial_state_sha256=initial_sha,
    )
    trainer = IBERFullTrainer(
        overrides=build_settings(args, manifest),
        experiment_seed=0,
        initial_state_path=args.initial_state,
    )
    trainer.add_callback(
        "on_train_start",
        lambda current: _atomic_json(
            Path(current.save_dir) / "iber_formal_protocol.json", authority
        ),
    )
    trainer.add_callback(
        "on_train_epoch_start",
        lambda _current: torch.cuda.reset_peak_memory_stats()
        if torch.cuda.is_available()
        else None,
    )
    trainer.add_callback(
        "on_train_start",
        lambda current: recover_pending_publications(
            current, args=args, identity=identity
        ),
    )
    trainer.add_callback("on_model_save", write_formal_diagnostics)
    trainer.add_callback(
        "on_model_save",
        lambda current: publish_current_epoch(current, args=args, identity=identity),
    )
    trainer.train()


if __name__ == "__main__":
    main()
